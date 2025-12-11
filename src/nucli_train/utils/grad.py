from __future__ import annotations

import torch.nn as nn


def set_grad(network : nn.Module, requires_grad=True):
    """
    Set the requires_grad attribute for all parameters in the network.
    
    Args:
        network (torch.nn.Module): The neural network model.
        requires_grad (bool): Whether to set requires_grad to True or False.
    """
    for param in network.parameters():
        param.requires_grad = requires_grad