from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

import numpy as np
import os
import time
import mlflow
import shutil
from tqdm import tqdm

import yaml

import matplotlib.pyplot as plt

import random

from os.path import normpath, join, isdir, exists, dirname, abspath, sep

import contextlib

class Seeder:
    def __init__(self, base_seed):
        self.base_seed = base_seed

    def __call__(self, worker_id):
        worker_seed = self.base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer:
    def __init__(
        self,
        model,
        train_data=None,
        val_loaders=None,
        run_name=None,
        experiment_name=None,
        use_amp=False,
        tcompile=False,
        log_freq=2, # batches during epoch
        save_interval=200, resuming={"epoch": 0, "weights_path": None, 'opt' : False}, save_last_opt=True, model_cfg_path=None, data_cfg_path=None
    ):
        """
        Initializes the Trainer.

        Args:
            model: Your torch model class (needs some methods, see docs)
            train_loader, val_loader: Your basic torch dataloader objects 
            use_amp: Whether to use Automatic Mixed Precision.
            log_freq: Interval (in batches) to print losses during train.
            val_interval: Interval (in epochs) to run through val set
            save_interval: Interval (in epochs) to save model weights
        """
        self.model = model
        if train_data is not None:
            self.train_dataset = train_data['dataset']
            self.train_batch_size = train_data['batch_size']
            self.train_workers = train_data['num_workers']
        self.val_loaders = val_loaders
        self.use_amp = use_amp
        self.log_freq = log_freq
        self.save_interval = save_interval
        self.compile = tcompile
        self.save_last_opt = save_last_opt

        self.checkpoint_dir = None


        self.scaler = GradScaler() if self.use_amp else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)


        self.starting_epoch = 1
        if run_name and experiment_name:

            self.optimizers = self.model.get_optimizers() # list, same order as how losses get returned
            self.schedulers = self.model.get_schedulers() # list, can be empty
            
            self.checkpoint_dir = os.path.join("./experiments", experiment_name, run_name)

            assert os.path.exists(model_cfg_path), f"Model config path {model_cfg_path} does not exist."
            assert model_cfg_path.endswith('.yaml'), f"Model config path {model_cfg_path} is not a YAML file."
            
            if os.path.isdir(self.checkpoint_dir):
                if len(os.listdir(self.checkpoint_dir)) == 1 and 'mlflow.yaml' in os.listdir(self.checkpoint_dir):
                    os.remove(join(self.checkpoint_dir, 'mlflow.yaml'))
                assert len(os.listdir(self.checkpoint_dir)) == 0, f"Checkpoint directory {self.checkpoint_dir} is not empty. Please remove it or choose a different run name."

            os.makedirs(self.checkpoint_dir, exist_ok=True)

            mlflow.set_tracking_uri("./experiments")

            if resuming["weights_path"] is not None:
                self.model._load_checkpoint(resuming["weights_path"], resuming["epoch"], resuming["opt"])
                if normpath(resuming["weights_path"]).split(sep)[-2:] == [experiment_name, run_name]:
                    self.starting_epoch = resuming["epoch"] + 1
                mlflow.set_experiment(experiment_name)
                r_id = yaml.safe_load(open(join(os.path.basename(normpath(resuming["weights_path"])), 'mlflow.yaml')))['run_id']
                mlflow.start_run(run_id=r_id)
            else:
                mlflow.set_experiment(experiment_name)
                mlflow.start_run(run_name=run_name)
                r_id = mlflow.active_run().info.run_id
                yaml.dump({'run_id': r_id}, open(join(self.checkpoint_dir, 'mlflow.yaml'), 'w'))


            mlflow.log_artifact(model_cfg_path, artifact_path='configs')
            mlflow.log_artifact(data_cfg_path, artifact_path='configs')

            mlflow.log_param("batch_size", self.train_batch_size)
            for p, v in self.model.get_params().items():
                mlflow.log_param(p, v)






    def wrap_train_step(self, batch):
        # this is such an ugly hack, change at some point
        ctx = autocast(self.device.type, dtype=torch.float16) if self.use_amp else contextlib.nullcontext()

        gen = self.model.train_step(batch)
        
        while True:
            with ctx:
                try:
                    losses = next(gen)
                except StopIteration:
                    break
            yield losses


    def run(self, epochs):
        assert self.train_dataset is not None, "Train dataset must be provided."
        base_seed = 1
        for epoch in range(self.starting_epoch, epochs + 1):
            seed = base_seed + epoch
            seed_everything(seed)
            self.model.train()
            running_losses = {}
            running_intermit_losses = {}

            generator = torch.Generator()
            generator.manual_seed(seed)

            seeder = Seeder(base_seed=seed)
            
            train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, generator=generator, shuffle=True, num_workers=self.train_workers, worker_init_fn=seeder)
            num_batches = len(train_loader)
            loader = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                unit="batch"
            )

            for batch_idx, batch in enumerate(loader, 1):
                if len(self.optimizers) == 1:
                    if self.use_amp:
                        with autocast(self.device.type, dtype=torch.float16):
                            loss_gen = [self.model.train_step(batch)]
                    else:
                        loss_gen = [self.model.train_step(batch)]
                else:
                    loss_gen = self.wrap_train_step(batch)

                for idx, losses in enumerate(loss_gen):

                    optimizer = self.optimizers[idx]
                    loss_value = losses['value']

                    if self.use_amp:
                        self.scaler.scale(loss_value).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss_value.backward()
                        optimizer.step()

                    optimizer.zero_grad()
                    for loss_name, val in losses.items():
                        if loss_name != "info": 
                            if loss_name == 'value':
                                loss_name += str(idx)
                            if loss_name not in running_losses.keys():
                                running_losses[loss_name] = 0.0
                            if loss_name not in running_intermit_losses.keys():
                                running_intermit_losses[loss_name] = 0.0
                            running_losses[loss_name] += val.item()
                            running_intermit_losses[loss_name] += val.item()


                if batch_idx % self.log_freq == 0:
                    avg_losses = {k: v / self.log_freq for k, v in running_intermit_losses.items()}
                    for loss_name, avg_loss in avg_losses.items():
                        mlflow.log_metric(f"loss_per_it/{loss_name}", avg_loss, step=(batch_idx + (epoch-1)*num_batches) * self.train_batch_size)
                    running_intermit_losses = {}
            
            avg_epoch_losses = {k: v / num_batches for k, v in running_losses.items()}
            for loss_name, avg_loss in avg_epoch_losses.items():
                mlflow.log_metric(f"train/{loss_name}", avg_loss, step=epoch)
            loader.close()
            
            self.validate(epoch)

            if epoch%self.save_interval==0:
                models_to_save = self.model.models_to_save()
                for name, submodel in models_to_save.items():
                    if isinstance(submodel, dict):
                        torch.save(submodel, os.path.join(self.checkpoint_dir, name + "_epoch_" + str(epoch) + ".pt"))
                    else:
                        torch.save(submodel.state_dict(), os.path.join(self.checkpoint_dir, name + "_epoch_"  + str(epoch) + ".pt"))
                if self.save_last_opt:
                    self.model.save_opt(self.checkpoint_dir, 'LATEST')

            for scheduler in self.schedulers:
                scheduler.step()
        self.model.save_opt(self.checkpoint_dir, epoch)



    def validate(self, epoch):
        self.model.eval()
        #print(f"Starting eval for epoch {epoch}")
        
        for dataset_name, loader_details in self.val_loaders.items():
            if epoch % loader_details['interval'] != 0:
                continue
            loader = loader_details['loader']
            evaluators = loader_details['evaluators']
            num_batches = len(loader)
            running_val_losses = {}
            print(f"Validating {dataset_name} at epoch {epoch}")
            with torch.no_grad():
                for batch in tqdm(loader):
                    val_output = self.model.validation_step(batch)
                    
                    losses = val_output["losses"]

                    for evaluator in evaluators:
                        evaluator.evaluate_batch(val_output, batch)

                    for loss_name, loss_value in losses.items():
                        if loss_name != "info":
                            if loss_name not in running_val_losses.items():
                                running_val_losses[loss_name] = 0.0
                            running_val_losses[loss_name] += loss_value

            for evaluator in evaluators:
                evaluator.log_epoch(epoch)

                    
            val_losses = {k: v / num_batches for k, v in running_val_losses.items()}
            if self.checkpoint_dir:
                for loss_name, avg_loss in val_losses.items():
                    mlflow.log_metric(f"{dataset_name}/{loss_name}", avg_loss, step=epoch)



            overall_metrics = self.model.compute_metrics()
            if self.checkpoint_dir:
                for metric_name, stats in overall_metrics.items():
                    for stat, value in stats.items():
                        mlflow.log_metric(f"{dataset_name}/{stat}/{metric_name}", value, step=epoch)
            

        self.model.train()
    
