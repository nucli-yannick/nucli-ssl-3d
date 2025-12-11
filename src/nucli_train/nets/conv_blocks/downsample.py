from __future__ import annotations

import torch.nn as nn
from .builder import DOWNSAMPLERS_REGISTRY

from torchsparse.nn import GlobalAvgPool
import torchsparse

@DOWNSAMPLERS_REGISTRY.register('avgpool')
def avg_pool(dim):
    pool_operation = nn.AvgPool2d if dim == "2d" else nn.AvgPool3d
    class AvgPoolDownSampleBlock(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.downsample = pool_operation(2)
        
        def forward(self, x):
            return self.downsample(x)
        
    return AvgPoolDownSampleBlock


@DOWNSAMPLERS_REGISTRY.register('sparse-global-avgpool')
def sparse_global_avg_pool(dim):
    class SparseGlobalAvgPoolBlock(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.global_pool = GlobalAvgPool()

        def forward(self, x):
            print(f"Type of x before global pool: {type(x)}")
            result = self.global_pool(x)
            print(f"Type of x after global pool: {type(result)}")
            return result

    return SparseGlobalAvgPoolBlock


@DOWNSAMPLERS_REGISTRY.register('sparse-2x2-conv')
def sparse_2x2_conv(dim):
    class Sparse2x2ConvBlock(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.conv = torchsparse.nn.Conv3d(in_channels=args[0], out_channels=args[0], kernel_size=2, stride=2)

        def forward(self, x):
            return self.conv(x)

    return Sparse2x2ConvBlock

@DOWNSAMPLERS_REGISTRY.register('2x2-conv')
def sparse_2x2_conv(dim):
    class Classical2x2ConvBlock(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.conv = nn.Conv3d(in_channels=args[0], out_channels=args[0], kernel_size=2, stride=2)

        def forward(self, x):
            return self.conv(x)

    return Classical2x2ConvBlock