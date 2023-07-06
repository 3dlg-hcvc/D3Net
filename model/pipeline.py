import os, json
import torch
import random
import pytorch_lightning as pl
from data.scannet.model_util_scannet_d3net import ScannetDatasetConfig
from model.pointgroup import PointGroup
from model.listener import ListenerNet
from lib.grounding.eval_helper import get_eval

from lib.det.ap_helper import APCalculator
from lib.evaluation.multi3drefer_evaluator import Multi3DReferEvaluator
from lib.grounding.loss_helper import get_loss as get_grounding_loss
from lib.utils.eval_helper_multi3drefer import *
from macro import *


class PipelineNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.init_random_seed()

        self.no_detection = cfg.model.no_detection
        self.no_grounding = cfg.model.no_grounding

        self.detector = PointGroup(cfg)
        self.val_test_step_outputs = []


        if not self.no_grounding:
            self.listener = ListenerNet(cfg)

        self.DC = ScannetDatasetConfig(cfg)

        self.num_class = cfg.model.num_bbox_class
        self.num_proposal = cfg.model.max_num_proposal

        self.use_lang_classifier = cfg.model.use_lang_classifier

        self.final_output = {}
        self.mem_hash = {}

        self.post_dict = {
            "remove_empty_box": False, 
            "use_3d_nms": True, 
            "nms_iou": 0.25,
            "use_old_type_nms": False, 
            "cls_nms": True, 
            "per_class_proposal": True,
            "conf_thresh": 0.09,
            "dataset_config": self.DC
        }
        # self.ap_calculator = APCalculator(0.5, self.DC.class2type)
        self.evaluator = Multi3DReferEvaluator(verbose=False, metric_name="f1")

        
    def init_random_seed(self):
        print("=> setting random seed...")
        if self.cfg.general.manual_seed:
            random.seed(self.cfg.general.manual_seed)
            np.random.seed(self.cfg.general.manual_seed)
            torch.manual_seed(self.cfg.general.manual_seed)
            torch.cuda.manual_seed_all(self.cfg.general.manual_seed)

    def training_step(self, data_dict, idx):
        if self.global_step % self.cfg.model.clear_cache_steps == 0:
            torch.cuda.empty_cache()

        # forward pass
        data_dict = self.detector.feed(data_dict, self.current_epoch)

        if not USE_GT:
            _, data_dict = self.detector.parse_feed_ret(data_dict)
            data_dict = self.detector.loss(data_dict, self.current_epoch)
        data_dict = self.listener(data_dict)

        _, data_dict = get_grounding_loss(
            data_dict,
            use_oracle=self.no_detection,
            grounding=not self.no_grounding,
            use_lang_classifier=self.use_lang_classifier,
            use_rl=False
        )

        if not USE_GT:
            loss = data_dict["total_loss"][0] + data_dict["ref_loss"] + data_dict["lang_loss"]
        else:
            loss = data_dict["ref_loss"] + data_dict["lang_loss"]

        # unpack
        if not USE_GT:
            log_dict = {
                "loss": loss,

                "detect_loss": data_dict["total_loss"][0],
                "grounding_loss": data_dict["ref_loss"],
                "lobjcls_loss": data_dict["lang_loss"],

                # "ref_acc_mean": data_dict["ref_acc_mean"],
                # "ref_iou_mean": data_dict["ref_iou_mean"],
                # "best_ious_mean": data_dict["best_ious_mean"],
                #
                # "ref_iou_rate_0.25": data_dict["ref_iou_rate_0.25"],
                # "ref_iou_rate_0.5": data_dict["ref_iou_rate_0.5"],

                "lang_acc": data_dict["lang_acc"],
            }
        else:
            log_dict = {
                "loss": loss,

                "grounding_loss": data_dict["ref_loss"],
                "lobjcls_loss": data_dict["lang_loss"],

                # "ref_acc_mean": data_dict["ref_acc_mean"],
                # "ref_iou_mean": data_dict["ref_iou_mean"],
                # "best_ious_mean": data_dict["best_ious_mean"],
                #
                # "ref_iou_rate_0.25": data_dict["ref_iou_rate_0.25"],
                # "ref_iou_rate_0.5": data_dict["ref_iou_rate_0.5"],

                "lang_acc": data_dict["lang_acc"],
            }

        # log
        for key, value in log_dict.items():
            ctg = "loss" if "loss" in key else "score"
            self.log("train_{}/{}".format(ctg, key), value, on_step=True)
        return loss

    # def on_validation_start(self) -> None:
    #     self.final_output = {}
    #     self.mem_hash = {}

    def _parse_pred_results(self, data_dict):
        batch_size, lang_chunk_size = data_dict["ann_id"].shape

        pred_aabb_score_masks = (
                torch.sigmoid(data_dict["cluster_ref"]) >= 0.1
        ).reshape(shape=(batch_size, lang_chunk_size, -1))

        pred_results = {}
        for i in range(batch_size):
            for j in range(lang_chunk_size):
                pred_bbox_corners = data_dict["proposal_bbox_batched"]
                min_max_bound = torch.stack((pred_bbox_corners.min(2)[0], pred_bbox_corners.max(2)[0]), dim=2)
                pred_aabbs = min_max_bound[i][pred_aabb_score_masks[i, j]]
                pred_results[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                     data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound": pred_aabbs.cpu().numpy()
                }
        return pred_results


    def validation_step(self, data_dict, idx, dataloader_idx=0):
        if self.global_step % self.cfg.model.clear_cache_steps == 0:
            torch.cuda.empty_cache()

        data_dict = self.detector.feed(data_dict, self.current_epoch)
        if not USE_GT:
            _, data_dict = self.detector.parse_feed_ret(data_dict)
            data_dict = self.detector.loss(data_dict, self.current_epoch)

        data_dict = self.listener(data_dict)

        # _, data_dict = get_grounding_loss(
        #     data_dict,
        #     use_oracle=self.no_detection,
        #     grounding=not self.no_grounding,
        #     use_lang_classifier=self.use_lang_classifier,
        #     use_rl=False
        # )

        self.val_test_step_outputs.append((self._parse_pred_results(data_dict), self._parse_gt(data_dict)))

    def _parse_gt(self, data_dict):
        batch_size, num_proposals = data_dict["cluster_ref"].shape
        gts = {}
        chunk_size = batch_size // data_dict["proposal_bbox_batched"].shape[0]
        box_masks = data_dict["multi_ref_box_label_list"].reshape(batch_size, num_proposals)

        gt_bboxes_list = data_dict["gt_bbox"]

        # gt_target_obj_id_masks = data_dict["gt_target_obj_id_mask"].permute(1, 0)
        for i in range(batch_size // chunk_size):
            # aabb_start_idx = data_dict["aabb_count_offsets"][i]
            # aabb_end_idx = data_dict["aabb_count_offsets"][i + 1]
            single_mask = box_masks[i]

            gt_bboxes = gt_bboxes_list[i // chunk_size][single_mask]
            gt_bboxes_bound = torch.stack((gt_bboxes.min(1)[0], gt_bboxes.max(1)[0]), dim=1)

            for j in range(chunk_size):
                if data_dict["eval_type"][i][j] != "zt_w_d" and data_dict["eval_type"][i][j] != "zt_wo_d":
                    assert gt_bboxes_bound.shape[0] >= 1
                elif data_dict["eval_type"][i][j] == "mt":
                    assert gt_bboxes_bound.shape[0] >= 2
                gts[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                     data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound": gt_bboxes_bound.cpu().numpy(),
                    "eval_type": data_dict["eval_type"][i][j]
                }
        return gts

    def on_validation_epoch_end(self):
        total_pred_results = {}
        total_gt_results = {}
        for pred_results, gt_results in self.val_test_step_outputs:
            total_pred_results.update(pred_results)
            total_gt_results.update(gt_results)
        self.val_test_step_outputs.clear()
        self.evaluator.set_ground_truths(total_gt_results)
        results = self.evaluator.evaluate(total_pred_results)

        # log
        for metric_name, result in results.items():
            for breakdown, value in result.items():
                self.log(f"val_eval/{metric_name}_{breakdown}", value)
        # if SCANREFER_ENHANCE:
        #     for scene_id in data_dict["scene_id"]:
        #         if scene_id not in self.final_output:
        #             self.final_output[scene_id] = []
        #
        # # scanrefer++ support
        # if SCANREFER_ENHANCE:
        #     for scene_id in data_dict["scene_id"]:
        #         if scene_id not in self.final_output:
        #             self.final_output[scene_id] = []
        #
        # _ = get_eval(
        #     data_dict,
        #     grounding=True,
        #     use_lang_classifier=True,
        #     final_output=self.final_output,  # scanrefer++ support
        #     mem_hash=self.mem_hash,
        #     dont_save=True # scanrefer++ support
        # )
        #
        # log_dict = {
        #     "ref_acc_mean": data_dict["ref_acc_mean"],
        #     "ref_iou_mean": data_dict["ref_iou_mean"],
        #     "best_ious_mean": data_dict["best_ious_mean"],
        #     "ref_iou_rate_0.25": data_dict["ref_iou_rate_0.25"],
        #     "ref_iou_rate_0.5": data_dict["ref_iou_rate_0.5"],
        #     "lang_acc": data_dict["lang_acc"],
        # }
        #
        # # log
        # for key, value in log_dict.items():
        #     ctg = "loss" if "loss" in key else "score"
        #     self.log("val_{}/{}".format(ctg, key), value, on_step=True)


    # def validation_epoch_end(self, outputs):
    #
    #     if SCANREFER_ENHANCE:
    #         # scanrefer+= support
    #         all_preds = {}
    #         all_gts = {}
    #         for key, value in self.final_output.items():
    #             for query in value:
    #                 all_preds[(key, int(query["object_id"]), int(query["ann_id"]))] = query
    #             #os.makedirs(EVAL_SAVE_NAME, exist_ok=True)
    #             # with open(f"{EVAL_SAVE_NAME}/{key}.json", "w") as f:
    #             #     json.dump(value, f)
    #             with open(os.path.join("/home/yza440/Research/D3Net/3dvg_gt", key + ".json"), 'r') as f:
    #                 gt_json = json.load(f)
    #             for query in gt_json:
    #                 all_gts[(key, int(query["object_id"]), int(query["ann_id"]))] = query
    #         iou_25_results, iou_50_results = evaluate_all_scenes(all_preds, all_gts)
    #
    #         self.log("val_score/multi3drefer_0.25", iou_25_results["overall"], prog_bar=False, on_step=False, on_epoch=True)
    #         self.log("val_score/multi3drefer_0.5", iou_50_results["overall"], prog_bar=False, on_step=False,
    #                  on_epoch=True)
    #
    #         self.final_output = {}
    #         self.mem_hash = {}



    def configure_optimizers(self):
        print("=> configure optimizer...")
        optim_class_name = self.cfg.train.optim.classname
        optim = getattr(torch.optim, optim_class_name)
        if optim_class_name == "Adam" or optim_class_name == "AdamW":
            optimizer = optim(filter(lambda p: p.requires_grad, self.parameters()), lr=self.cfg.train.optim.lr, weight_decay=self.cfg.train.optim.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=10,
            gamma=0.8
        )
        return [optimizer], [scheduler]

    # NOTE direct access only during inference
    def forward(self, data_dict):

        if not self.no_detection:
            #######################################
            #                                     #
            #           DETECTION BRANCH          #
            #                                     #
            #######################################

            data_dict = self.detector.feed(data_dict, self.current_epoch)

        if not self.no_grounding:
            ########################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            data_dict = self.listener(data_dict)

        return data_dict
