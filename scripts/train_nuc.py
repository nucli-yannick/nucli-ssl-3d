from __future__ import annotations

from nucli_train.data_management.builders import build_dataloaders
from nucli_train.models.builders import build_image_translation_model

from nucli_train.training import Trainer
import os
import mlflow
import yaml
from torch.utils.data import DataLoader

import copy

import random
import yaml
train_loader, _ = build_dataloaders(yaml.safe_load(open('./configs/nuclarity_data.yaml')))

model_cfg = yaml.safe_load(open('./configs/nuclarity_model.yaml'))
epochs = 10000
save_interval = 100
model = build_image_translation_model(model_cfg['model'])
Trainer(model, train_loader=train_loader, val_loaders={}, use_amp=True, run_name='basic-test', experiment_name='nuclarity_data', save_interval=save_interval).run(epochs)