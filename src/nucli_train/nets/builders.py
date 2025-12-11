from __future__ import annotations

from nucli_train.utils.registry import Registry

ARCHITECTURE_BUILDERS_REGISTRY = Registry('architecture_builders')
NETWORK_REGISTRY = Registry('networks')



def build_network(cfg):
    network_name = cfg['name']
    if ARCHITECTURE_BUILDERS_REGISTRY.has(network_name):
        return ARCHITECTURE_BUILDERS_REGISTRY.get(network_name)(cfg['args'])
    elif NETWORK_REGISTRY.has(network_name):
        return NETWORK_REGISTRY.get(network_name)(cfg['args'])
    else:
        raise ValueError(f"Network {network_name} not found in builders or networks registry. Available builders: {ARCHITECTURE_BUILDERS_REGISTRY.list()}, available networks: {NETWORK_REGISTRY.list()}")


