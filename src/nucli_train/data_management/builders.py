from __future__ import annotations

from nucli_train.utils.registry import Registry

from torch.utils.data import DataLoader

import yaml

import os

import torch

import random

DATASET_REGISTRY = Registry('datasets')

from nucli_train.val.evaluators import EVALUATORS_REGISTRY

base_seed = 6582 # Half-life of [18F]FDG in seconds :)
def seed_worker(worker_id):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def build_data(cfg):
    """
    We expect only one dataset for training and several datasets for validation. 
    The goal is to see if yes or no, the learnt model is generalizable on several datasets then

    For the validation part, we also expect an evaluator that will be used during the validation part of the training. 
    """


    if isinstance(cfg, str):
        cfg = yaml.safe_load(open(cfg, 'r'))
    assert 'train' in cfg, "Configuration must contain 'train' key"
    #assert 'val' in cfg, "Configuration must contain 'val' key"

    train_dataset = DATASET_REGISTRY.get(cfg['train']['type'])(**cfg['train']['params']) # see for example Blosc2Dataset
    train_data = {'dataset': train_dataset, 'batch_size': cfg['train']['batch_size'], 'num_workers': cfg['train']['num_workers']}

    

    val_loaders = {}
    if "val" not in cfg:
        return train_data, val_loaders
    

    val_bs, val_num_workers = cfg["val"]["batch_size"], cfg["val"]["num_workers"]
    global_eval_interval = cfg['val'].get('global_eval_interval', 1)

    for dataset_name, dataset_args in cfg['val']['datasets'].items():

        # Very interesting to be used in the case of different doses (50 pc, 100 pc ...)
        # In the case of self-supervised learning, not useful
        if dataset_args["type"] == "patches_dataset_builder":
            dataset_val_interval = dataset_args.get('interval', global_eval_interval)
            split_dict = yaml.safe_load(open(dataset_args['path'], 'r'))
            base_dir = split_dict['data_dir']
            for center, tracers in split_dict['val'].items():
                for tracer, val_set in tracers.items():
                    for dose in val_set['doses']:
                        ds = DATASET_REGISTRY.get('patches_dataset')(os.path.join(base_dir, center, tracer, dose, 'val.npy'))
                        dl =  DataLoader(ds, batch_size=val_bs, shuffle=False, num_workers=val_num_workers, worker_init_fn=seed_worker)
                        evaluators = [EVALUATORS_REGISTRY.get(evaluator)(f'{dataset_name}/{center}/{tracer}/{dose}')
                            for evaluator in dataset_args.get('evaluators', [])]
                        val_loaders[f'{center}_{tracer}_{dose}'] = {'interval': dataset_val_interval,
                                     'loader': dl,
                                     'evaluators': evaluators}
                        

        else:
            dataset = DATASET_REGISTRY.get(dataset_args['type'])(**dataset_args['params'])
            loader = DataLoader(dataset, 
                                batch_size=val_bs,
                                shuffle=False,
                                num_workers=val_num_workers,
                                worker_init_fn=seed_worker)
            dataset_val_interval = dataset_args.get('interval', global_eval_interval)

            evaluators = [EVALUATORS_REGISTRY.get(evaluator)(dataset_name)
                        for evaluator in dataset_args.get('evaluators', [])]
            val_loaders[dataset_name] = {'interval': dataset_val_interval,
                                        'loader': loader,
                                        'evaluators': evaluators}

    return train_data, val_loaders
