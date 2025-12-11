from __future__ import annotations

from nucli_train.utils.registry import Registry

LOSSES_REGISTRY = Registry("losses")

def build_losses(cfg):
    losses = []
    for loss in cfg:
        if LOSSES_REGISTRY.has(loss['name']):
            losses.append({'name':loss['name'], 'f' : LOSSES_REGISTRY.get(loss['name'])(**loss['args']), 'coef' : loss.get('coef', 1)})
        else:
            raise ValueError(f"Loss {loss['name']} not found in losses registry. Available losses: {LOSSES_REGISTRY.list()}")

    return losses