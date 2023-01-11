import os
import torch
import functools
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl

from lib.pointgroup_ops.functions import pointgroup_ops
from lib.utils.bbox import get_3d_box_batch
from lib.utils.nn_distance import nn_distance

from model.common import ResidualBlock, UBlock


class GTDetector(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        in_channel = cfg.model.use_color * 3 + cfg.model.use_normal * 3 + cfg.model.use_coords * 3 + cfg.model.use_multiview * 128
        m = cfg.model.m
        D = 3
        blocks = cfg.model.blocks
        block_reps = cfg.model.block_reps


        self.mode = cfg.train.score_mode


        block = ResidualBlock
        sp_norm = functools.partial(ME.MinkowskiBatchNorm, eps=1e-4, momentum=0.1)

        #### backbone
        self.backbone = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, m, kernel_size=3, bias=False, dimension=D),
            UBlock([m * c for c in blocks], sp_norm, block_reps, block),
            sp_norm(m),
            ME.MinkowskiReLU(inplace=True)
        )


    @staticmethod
    def get_batch_offsets(batch_idxs, batch_size):
        """
        :param batch_idxs: (N), int
        :param batch_size: int
        :return: batch_offsets: (batch_size + 1)
        """
        batch_offsets = torch.zeros(batch_size + 1).int().cuda()
        for i in range(batch_size):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets


    def get_object_assignments(self, data_dict):
        _, ind1, _, _ = nn_distance(data_dict["proposal_center_batched"], data_dict["center_label"], l1=True)

        data_dict["object_assignment"] = ind1

        return data_dict

    def convert_stack_to_batch(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        max_num_proposal = self.cfg.model.max_num_proposal
        data_dict["proposal_feats_batched"] = torch.zeros(batch_size, max_num_proposal, self.cfg.model.m).type_as(
            data_dict["proposal_feats"])
        data_dict["proposal_bbox_batched"] = torch.zeros(batch_size, max_num_proposal, 8, 3).type_as(
            data_dict["proposal_feats"])
        data_dict["proposal_center_batched"] = torch.zeros(batch_size, max_num_proposal, 3).type_as(
            data_dict["proposal_feats"])
        data_dict["proposal_sem_cls_batched"] = torch.zeros(batch_size, max_num_proposal).type_as(
            data_dict["proposal_feats"])
        data_dict["proposal_scores_batched"] = torch.zeros(batch_size, max_num_proposal).type_as(
            data_dict["proposal_feats"])
        data_dict["proposal_batch_mask"] = torch.zeros(batch_size, max_num_proposal).type_as(
            data_dict["proposal_feats"])

        proposal_bbox = data_dict["proposal_crop_bbox"].detach().cpu().numpy()
        proposal_bbox = get_3d_box_batch(proposal_bbox[:, :3], proposal_bbox[:, 3:6],
                                         proposal_bbox[:, 6])  # (nProposals, 8, 3)
        proposal_bbox_tensor = torch.tensor(proposal_bbox).type_as(data_dict["proposal_feats"])

        for b in range(batch_size):
            proposal_batch_idx = torch.nonzero(data_dict["proposals_batchId"] == b).squeeze(-1)
            pred_num = len(proposal_batch_idx)
            pred_num = pred_num if pred_num < max_num_proposal else max_num_proposal

            # NOTE proposals should be truncated if more than max_num_proposal proposals are predicted
            data_dict["proposal_feats_batched"][b, :pred_num, :] = data_dict["proposal_feats"][proposal_batch_idx][
                                                                   :pred_num]
            data_dict["proposal_bbox_batched"][b, :pred_num, :, :] = proposal_bbox_tensor[proposal_batch_idx][:pred_num]
            data_dict["proposal_center_batched"][b, :pred_num, :] = data_dict["proposal_crop_bbox"][proposal_batch_idx,
                                                                    :3][:pred_num]
            data_dict["proposal_sem_cls_batched"][b, :pred_num] = data_dict["proposal_crop_bbox"][
                                                                      proposal_batch_idx, 7][:pred_num]
            data_dict["proposal_scores_batched"][b, :pred_num] = data_dict["proposal_objectness_scores"][
                                                                     proposal_batch_idx][:pred_num]
            data_dict["proposal_batch_mask"][b, :pred_num] = 1

            # NOTE all proposal data in batch should be shuffled to prevent overfitting
            rearrange_ids = torch.randperm(max_num_proposal)
            data_dict["proposal_feats_batched"][b] = data_dict["proposal_feats_batched"][b][rearrange_ids]
            data_dict["proposal_bbox_batched"][b] = data_dict["proposal_bbox_batched"][b][rearrange_ids]
            data_dict["proposal_center_batched"][b] = data_dict["proposal_center_batched"][b][rearrange_ids]
            data_dict["proposal_sem_cls_batched"][b] = data_dict["proposal_sem_cls_batched"][b][rearrange_ids]
            data_dict["proposal_scores_batched"][b] = data_dict["proposal_scores_batched"][b][rearrange_ids]
            data_dict["proposal_batch_mask"][b] = data_dict["proposal_batch_mask"][b][rearrange_ids]

        # compute GT object assignments
        if self.cfg.general.task != "test":
            data_dict = self.get_object_assignments(data_dict)

        return data_dict

    def forward(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        x = ME.SparseTensor(features=data_dict["voxel_feats"], coordinates=data_dict["voxel_locs"].int())

        # backbone
        out = self.backbone(x)
        pt_feats = out.features[data_dict["p2v_map"].long()]  # (N, m)

        # instance_ids = torch.unique(data_dict["gt_proposals_idx"][:, 0])
        num_proposals = len(data_dict["gt_proposals_offset"]) - 1
        gt_proposal_features = torch.empty(size=(num_proposals, pt_feats.shape[1]), device=self.device)

        batch_idxs = data_dict["locs_scaled"][:, 0].int()
        proposals_batchId_all = batch_idxs[data_dict["gt_proposals_idx"][:, 1].long()].int()

        proposals_batchId = proposals_batchId_all[data_dict["gt_proposals_offset"][:-1].long()]
        sem_labels = torch.empty(size=(num_proposals, ), device=self.device)

        for idx in range(num_proposals):
            start_idx = data_dict["gt_proposals_offset"][idx]
            end_idx = data_dict["gt_proposals_offset"][idx+1]
            proposal_info = data_dict["gt_proposals_idx"][start_idx:end_idx]
            proposal_point_mask = proposal_info[:, 1].long()
            instance_id = proposal_info[:, 0]
            proposal_features = torch.mean(pt_feats[proposal_point_mask], dim=0)
            gt_proposal_features[idx] = proposal_features
            sem_labels[idx] = data_dict["sem_labels"][proposal_point_mask][0]
            if sem_labels[idx] == -1:
                sem_labels[idx] = 19

        data_dict["proposals_batchId"] = proposals_batchId
        data_dict["proposal_feats"] = gt_proposal_features
        data_dict["proposal_objectness_scores"] = torch.ones(size=(num_proposals,), dtype=torch.int32, device=self.device)



        if self.cfg.model.crop_bbox:
            proposal_crop_bbox = torch.zeros(num_proposals, 9).cuda()  # (nProposals, center+size+heading+label)
            proposal_crop_bbox[:, :3] = data_dict["instances_bboxes_tmp"][:, :3]
            proposal_crop_bbox[:, 3:6] = data_dict["instances_bboxes_tmp"][:, 3:6]
            proposal_crop_bbox[:, 7] = sem_labels
            proposal_crop_bbox[:, 8] = torch.ones(size=(num_proposals,), dtype=torch.int32, device=self.device)
            data_dict["proposal_crop_bbox"] = proposal_crop_bbox

        return data_dict


    def feed(self, data_dict, epoch=0):
        data_dict["epoch"] = epoch

        if self.cfg.model.use_coords:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), 1)

        data_dict["voxel_feats"] = pointgroup_ops.voxelization(data_dict["feats"], data_dict["v2p_map"],
                                                               self.cfg.data.mode)  # (M, C), float, cuda

        data_dict = self.forward(data_dict)

        data_dict = self.convert_stack_to_batch(data_dict)

        return data_dict



