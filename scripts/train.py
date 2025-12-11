from __future__ import annotations

from nucli_val.data_management.dataset import ImageTranslationPETDataSet, ValImageTranslationPETDataSet, val_collate
from nucli_val.training import Trainer

from nucli_val.models import model_builders

from torch.utils.data import DataLoader
import yaml

import random
import torch

base_seed = 6582 # Half-life of [18F]FDG in seconds :)

def seed_worker(worker_id):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)

def build_model(cfg):
    model = model_builders[cfg["type"]](cfg)
    return model

def build_dataloaders(cfg, train_bs):
    train_cfg, val_cfg = cfg['train'], cfg["val"]
    train_hd, train_ld = train_cfg['hd'], train_cfg['ld']
    train_ds = ImageTranslationPETDataSet(train_ld, train_hd)
    train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=1, worker_init_fn=seed_worker)

    val_loaders = {}

    for dataset, dirs in val_cfg.items():
        ds = ValImageTranslationPETDataSet(dirs["ld"], dirs["hd"])
        val_loaders[dataset] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=val_collate)

    return train_dl, val_loaders

def build_trainer(exp_cfg, run_cfg, model, train_dl, val_loaders):
    return Trainer(model, train_loader=train_dl, val_loaders=val_loaders, use_amp=True, run_name=run_cfg["run_name"], experiment_name=exp_cfg["experiment_name"], val_interval=run_cfg["val_interval"], save_interval=run_cfg["save_interval"])



if __name__=="__main__":
    experiment_config = input("Path to experiment config file:")
    model_config = input("Path to run config file:")

    run_cfg = yaml.safe_load(open(model_config, 'r'))
    exp_cfg = yaml.safe_load(open(experiment_config, 'r'))

    torch.manual_seed(base_seed)



    model = build_model(run_cfg["model"])

    train_loader, val_loaders = build_dataloaders(exp_cfg, run_cfg["batch_size"])

    trainer = build_trainer(exp_cfg, run_cfg, model, train_loader, val_loaders)


    trainer.run(run_cfg["epochs"])

