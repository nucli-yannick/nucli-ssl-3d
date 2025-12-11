from __future__ import annotations

import torch.nn as nn
import torch


class ResNet(nn.Module):
    def __init__(self, layers_per_stage=[16, 32, 64, 64],dim='2d', norm='batch'):
        