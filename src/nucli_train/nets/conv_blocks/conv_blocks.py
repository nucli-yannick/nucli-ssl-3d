from __future__ import annotations

import copy
from typing import Type

import torchsparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import SparseGRN, SparseDropPath, GRN, SparseDepthWise3D

from timm.models.layers import trunc_normal_, DropPath

from .builder import CONV_BLOCKS_REGISTRY




@CONV_BLOCKS_REGISTRY.register('3-channel-fusion-block')
def input_block_3channel_fusion(conv_layer, dim):
    class NuclarityOutputBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels : int):
            super().__init__()
            assert in_channels == 3, "Input channels must be 3 for 3-channel fusion block"
            mid_ch = 3 * (out_channels // 2)
            self.conv = conv_layer(3, mid_ch, groups=3)

            self.fusion_conv = {'2d' : nn.Conv2d, '3d' : nn.Conv3d}[dim](mid_ch, out_channels, 1)
        def forward(self, x):
            output = self.conv(x)

            output = self.fusion_conv(output)
            return output

    return NuclarityOutputBlock

@CONV_BLOCKS_REGISTRY.register('nuclarity-output-block')
def nuclarity_output_block(_, dim):
    class NuclarityOutputBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels : int):
            super().__init__()
            self.conv = {'2d' : nn.Conv2d, '3d' : nn.Conv3d}[dim](in_channels, out_channels, 1)
        def forward(self, x):
            output = self.conv(x)
            return output


@CONV_BLOCKS_REGISTRY.register('3x3-output-block')
def full_output_block(_, dim):
    class OutputBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels : int):
            super().__init__()
            self.conv = {'2d' : nn.Conv2d, '3d' : nn.Conv3d}[dim](in_channels, out_channels, 3, 1, 1)
        def forward(self, x):
            output = self.conv(x)
            return output

    return OutputBlock

@CONV_BLOCKS_REGISTRY.register('sparse-3x3-output-block')
def sparse_full_output_block(_, dim):
    class SparseOutputBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels : int):
            super().__init__()
            if dim != "3d":
                raise ValueError("SparseOutputBlock only supports 3D convolutions")
            self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        def forward(self, x):
            output = self.conv(x)
            return output
    
    return SparseOutputBlock


@CONV_BLOCKS_REGISTRY.register('denoising-block')
def CustomDenoisingBlockFactory(conv_layer, dim, dense=False, residual=False, convs=1):
    class CustomDenoisingBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(CustomDenoisingBlock, self).__init__()

            amt_dense_layers = convs if dense else 0
            amt_regular_layers = convs if not dense else 0
            self.residual = residual

            assert not (dense and residual), "residual connections and dense layers cannot be fused in this implementation"


            if dense:
                growth_factor = dense
                assert convs * growth_factor == in_channels - out_channels, "Inconsistency in growth factor vs output channel logic"
            if self.residual:
                assert in_channels == out_channels, "residual connection requires input and output channels to be equal"
            self.blocks = nn.ModuleDict({}) # we're gonna put all the submodules in this dict. yay!

            current_channels = in_channels
            for dense_idx in range(amt_dense_layers):
                self.blocks[f'dense_{dense_idx}'] = conv_layer(current_channels, growth_factor)
                current_channels += growth_factor

            for regular_idx in range(amt_regular_layers - 1):
                self.blocks[f'regular_{regular_idx}'] = conv_layer(in_channels, in_channels)

            if amt_regular_layers != 0:
                self.blocks[f'out'] = conv_layer(current_channels, out_channels)

        def forward(self, inputs, dose_emb=False):
            x = inputs
            for key in self.blocks.keys():
                if "regular" in key or "out" in key:
                    x = self.blocks[key](x)
                elif "dense" in key:
                    out = self.blocks[key](x)
                    x = torch.cat([x, out], 1)
                else:
                    raise ModuleNotFoundError(
                        f"We're not supposed to have modules with this key: {key}")  # this is not supposed to trigger

            if self.residual:
                x =  x + inputs

            return x

    return CustomDenoisingBlock


@CONV_BLOCKS_REGISTRY.register('sparse-denoising-block')
def SparseCustomDenoisingBlockFactory(conv_layer, dim, dense=False, residual=False, convs=1):
    """
    This first implementation doesn't support dense and residual
    Adam.
    """
    class SparseCustomDenoisingBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(SparseCustomDenoisingBlock, self).__init__()

            self.blocks = nn.ModuleDict({})  

            current_channels = in_channels

            for regular_idx in range(convs - 1):
                self.blocks[f'regular_{regular_idx}'] = conv_layer(in_channels, in_channels)

            if convs != 0: 
                self.blocks[f'out'] = conv_layer(current_channels, out_channels)

        def forward(self, inputs, dose_emb=False):
            x = inputs
            for key in self.blocks.keys():
                if "regular" in key or "out" in key:
                    x = self.blocks[key](x)
                else:
                    raise ModuleNotFoundError(
                        f"We're not supposed to have modules with this key: {key}")
                
            return x


    return SparseCustomDenoisingBlock


@CONV_BLOCKS_REGISTRY.register('convnextv2-block-3d')
class ConvNeXtV2Block3D(nn.Module):
    """
    Sparse ConvNeXtV2 block implementation.
    Args: 
        dim (int): Number of input channels 
        drop_path (float): Stochastic depth rate.
    Based on: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
    Note: we don't perform pemutations here because we implemented GRN differently.
    Adam.
    """

    def __init__(self, dim, drop_path=0.0):
        super(ConvNeXtV2Block3D, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 7, 1, 3, bias=True, groups=dim)
        self.norm = nn.GroupNorm(1, dim, 1e-6)
        self.pwconv1 = nn.Conv3d(dim, dim * 4, 1, 1, 0)  # Equivalent to a Linear Layer
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Conv3d(dim * 4, dim, 1, 1, 0)  # Equivalent to a Linear Layer
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x): 
        identity = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        x = identity + self.drop_path(x)
        return x



@CONV_BLOCKS_REGISTRY.register('sparse-convnextv2-block-3d')
class SparseConvNeXtV2Block3D(nn.Module):
    """
    Sparse ConvNeXtV2 block implementation.
    Args: 
        dim (int): Number of input channels 
        drop_path (float): Stochastic depth rate.
    Adam.
    """
    def __init__(self, dim, drop_path=0.0): 
        super(SparseConvNeXtV2Block3D, self).__init__()
        self.dwconv = torchsparse.nn.Conv3d(dim, dim, 7, 1, 3, bias=True)
        self.norm = torchsparse.nn.GroupNorm(1, dim, 1e-6) # We use 1 group for GroupNorm to match LayerNorm behavior
        self.pwconv1 = torchsparse.nn.Conv3d(dim, dim * 4, 1, 1, 0) # Equivalent to a Linear Layer
        self.act = torchsparse.nn.ReLU()
        self.grn = SparseGRN(4*dim)
        self.pwconv2 = torchsparse.nn.Conv3d(dim * 4, dim, 1, 1, 0) # Equivalent to a Linear Layer
        self.drop_path = SparseDropPath(drop_path) 

    def forward(self, x: SparseTensor):
        
        identity = x
        
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = identity + self.drop_path(x)
        return x

@CONV_BLOCKS_REGISTRY.register('depthwise-sparse-convnextv2-block-3d')
class DepthWiseSparseConvNeXtV2Block3D(nn.Module):
    """
    Sparse ConvNeXtV2 block implementation with depthwise convolutions.
    Args: 
        dim (int): Number of input channels 
        drop_path (float): Stochastic depth rate.
    Adam.
    """
    def __init__(self, dim, drop_path=0.0): 
        super(DepthWiseSparseConvNeXtV2Block3D, self).__init__()
        self.dwconv = SparseDepthWise3D(dim, kernel_size=7, stride=1, padding=3)
        self.norm = torchsparse.nn.GroupNorm(1, dim, 1e-6) # We use 1 group for GroupNorm to match LayerNorm behavior
        self.pwconv1 = torchsparse.nn.Conv3d(dim, dim * 4, 1, 1, 0) # Equivalent to a Linear Layer
        self.act = torchsparse.nn.ReLU()
        self.grn = SparseGRN(4*dim)
        self.pwconv2 = torchsparse.nn.Conv3d(dim * 4, dim, 1, 1, 0) # Equivalent to a Linear Layer
        self.drop_path = SparseDropPath(drop_path) 

    def forward(self, x: SparseTensor):
        
        identity = x
        
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = identity + self.drop_path(x)
        return x


@CONV_BLOCKS_REGISTRY.register('resblock-3d')
class ResBlock3D(nn.Module): 
    def __init__(self, dim, drop_path=0.0): 
        super(ResBlock3D, self).__init__()
        self.dim = dim
        self.intermediate_dim = dim // 4

        self.conv1 = nn.Conv3d(dim, self.intermediate_dim, 1)
        self.ln1 = nn.GroupNorm(1, self.intermediate_dim)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv3d(self.intermediate_dim, self.intermediate_dim, 3, 1, 1)
        self.ln2 = nn.GroupNorm(1, self.intermediate_dim)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv3d(self.intermediate_dim, self.dim, 1)
        self.ln3 = nn.GroupNorm(1, self.dim)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.ln3(out)

        out = identity + out
        out = self.act(out)
        return out



@CONV_BLOCKS_REGISTRY.register('sparse-resblock-3d')
class SparseResBlock3D(nn.Module):
    def __init__(self, dim, drop_path=0.0): 
        super(SparseResBlock3D, self).__init__()
        self.dim = dim
        self.intermediate_dim = dim // 4

        self.conv1 = torchsparse.nn.Conv3d(dim, self.intermediate_dim, 1)
        self.ln1 = torchsparse.nn.GroupNorm(1, self.intermediate_dim)
        self.act1 = torchsparse.nn.ReLU()

        self.conv2 = torchsparse.nn.Conv3d(self.intermediate_dim, self.intermediate_dim, 3, 1, 1)
        self.ln2 = torchsparse.nn.GroupNorm(1, self.intermediate_dim)
        self.act2 = torchsparse.nn.ReLU()

        self.conv3 = torchsparse.nn.Conv3d(self.intermediate_dim, self.dim, 1)
        self.ln3 = torchsparse.nn.GroupNorm(1, self.dim)
        self.act = torchsparse.nn.ReLU()

    def forward(self, x: SparseTensor):
        identity = x

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.ln3(out)

        out = identity + out
        out = self.act(out)
        return out




