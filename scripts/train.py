

import os
import sys
import json
import torch
import argparse

import pytorch_lightning as pl

from copy import deepcopy
from omegaconf import OmegaConf
from importlib import import_module

# from pytorch_lightning.plugins import DDPPlugin

from torch.utils.data.dataloader import DataLoader

sys.path.append(".")

def load_conf(args):
    base_cfg = OmegaConf.load("conf/path.yaml")
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    root = os.path.join(cfg.general.output_root, cfg.general.experiment.upper())
    os.makedirs(root, exist_ok=True)

    cfg.general.task = "train"
    cfg.general.root = root

    cfg_backup_path = os.path.join(cfg.general.root, "config.yaml")
    OmegaConf.save(cfg, cfg_backup_path)

    return cfg


def init_listener_data(cfg):
    DATA_MODULE = import_module(cfg.data.module)
    Dataset = getattr(DATA_MODULE, cfg.data.dataset)
    collate_fn = getattr(DATA_MODULE, "sparse_collate_fn")

    # SCAN2CAD = json.load(open(os.path.join(cfg.SCAN2CAD, "scannet_instance_rotations.json")))
    raw_train = json.load(open(cfg["{}_PATH".format(cfg.general.dataset.upper())].train_split))
    raw_val = json.load(open(cfg["{}_PATH".format(cfg.general.dataset.upper())].val_split))

    raw_train_scan_list = sorted(list(set([data["scene_id"] for data in raw_train])))
    raw_val_scan_list = sorted(list(set([data["scene_id"] for data in raw_val])))

    data_train = deepcopy(raw_train)
    data_val = deepcopy(raw_val)
    scan_list_train = deepcopy(raw_train_scan_list)
    scan_list_val = deepcopy(raw_val_scan_list)
    
    print("=> loading train split for listener...")
    dataset_train = Dataset(cfg, cfg.general.dataset, "listener", "train", data_train, scan_list_train)

    print("=> use {} samples for training".format(len(dataset_train)))

    print("=> loading val split for listener...")
    dataset_val = Dataset(cfg, cfg.general.dataset, "listener", "val", data_val, scan_list_val)

    print("=> use {} samples for training".format(len(dataset_val)))

    print("=> loading complete")

    datasets = {
        "train": dataset_train,
        "val": dataset_val
    }

    dataloaders = {
        "train": DataLoader(dataset_train, batch_size=cfg.data.batch_size, \
                shuffle=True, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn),
        "val": DataLoader(dataset_val, batch_size=cfg.data.batch_size, \
                shuffle=False, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn),
    }

    return datasets, dataloaders

def init_data(cfg):
    datasets, dataloaders = {}, {}

    if not cfg.model.no_grounding:
        dataset, dataloader = init_listener_data(cfg)
        datasets["lis"] = dataset
        dataloaders["lis"] = dataloader

    return datasets, dataloaders


def init_monitor(cfg):
    monitor = pl.callbacks.ModelCheckpoint(
        monitor="{}".format(cfg.general.monitor),
        mode="{}".format(cfg.general.monitor_mode),
        # save_weights_only=True,
        dirpath=cfg.general.root,
        filename="model",
        save_last=True
    )

    return monitor

def init_trainer(cfg):
    logger = pl.loggers.WandbLogger(project="D3Net", name="run1", save_dir="new")
    trainer = pl.Trainer(
        # gpus=-1, # use all available GPUs
        # strategy="ddp_find_unused_parameters_false",
        accelerator="gpu", # use multiple GPUs on the same machine
        max_epochs=cfg.train.epochs, 
        num_sanity_val_steps=cfg.train.num_sanity_val_steps, # validate on all val data before training 
        log_every_n_steps=cfg.train.log_every_n_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        callbacks=[monitor], # comment when debug
        logger=logger,
        profiler="simple",
        # resume_from_checkpoint=checkpoint,
        # plugins=DDPPlugin(find_unused_parameters=False)
    )

    return trainer

def init_model(cfg):
    PipelineNet = getattr(import_module("model.pipeline"), "PipelineNet")
    model = PipelineNet(cfg)

    if cfg.model.pretrained_detector and not cfg.model.no_detection and not cfg.model.use_checkpoint:
        device_name = "cuda:{}".format(os.environ.get("LOCAL_RANK", 0))

        print("=> loading pretrained detector to {} ...".format(device_name))
        detector_path = os.path.join(cfg.PRETRAINED_PATH, cfg.model.pretrained_detector)
        detector_weights = torch.load(detector_path, map_location=device_name)
        model.detector.load_state_dict(detector_weights)

    if cfg.model.pretrained_listener and not cfg.model.no_grounding and not cfg.model.use_checkpoint:
        device_name = "cuda:{}".format(os.environ.get("LOCAL_RANK", 0))

        print("=> loading pretrained listener to {} ...".format(device_name))
        listener_path = os.path.join(cfg.PRETRAINED_PATH, cfg.model.pretrained_listener)
        listener_weights = torch.load(listener_path, map_location=device_name)
        model.listener.load_state_dict(listener_weights)

    if cfg.model.freeze_detector:
        print("=> freezing detector...")
        for param in model.detector.parameters():
            param.requires_grad = False

    if hasattr(model, "listener") and cfg.model.freeze_listener:
        print("=> freezing listener...")
        for param in model.listener.parameters():
            param.requires_grad = False

    return model

def start_training(trainer, model, dataloaders):

    if cfg.model.use_checkpoint:
        print("=> configuring trainer with checkpoint from {} ...".format(cfg.model.use_checkpoint))
        checkpoint = os.path.join(cfg.general.output_root, cfg.model.use_checkpoint, "last.ckpt")
    else:
        checkpoint = None

    trainer.fit(
        model=model,
        train_dataloaders=dataloaders["lis"]["train"],
        val_dataloaders=dataloaders["lis"]["val"],
        ckpt_path=checkpoint
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="conf/pointgroup_grounding.yaml", help="path to config file")
    args = parser.parse_args()


    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    datasets, dataloaders = init_data(cfg)

    print("=> initializing model...")
    model = init_model(cfg)

    
    print("=> initializing monitor...")
    monitor = init_monitor(cfg)

    print("=> initializing trainer...")
    trainer = init_trainer(cfg)

    print("=> start training...")
    start_training(trainer, model, dataloaders)
