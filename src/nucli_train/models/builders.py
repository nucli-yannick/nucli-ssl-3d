from __future__ import annotations

from .losses import build_losses

from nucli_train.nets import build_network
from nucli_train.utils.registry import Registry

import yaml

MODEL_REGISTRY = Registry('models')
MODEL_BUILDERS_REGISTRY = Registry('model_builders')


def build_model(cfg):
    if isinstance(cfg, str):
        cfg = yaml.safe_load(open(cfg, 'r'))
    if 'model' in cfg.keys():
        cfg = cfg['model']

    if MODEL_BUILDERS_REGISTRY.has(cfg['name']):
        return MODEL_BUILDERS_REGISTRY.get(cfg['name'])(cfg)
    elif MODEL_REGISTRY.has(cfg['name']):
        return MODEL_REGISTRY.get(cfg['name'])(cfg)
    else:
        raise ValueError(f"Model {cfg['name']} not recognized. Available models: ['image_translation']")