from __future__ import annotations

from nucli_train.utils.registry import Registry

import torchsparse

import torch.nn as nn
CONV_BLOCKS_REGISTRY = Registry('conv_blocks')
DECODER_BLOCKS_REGISTRY = Registry('decoder_blocks')
DOWNSAMPLERS_REGISTRY = Registry('downsamplers')


class SparseNormBuilder: 
    def __init__(self, cfg, dim):
        self.dim = dim
        self.norm_name = cfg['name'].lower()
        self.norm_args = cfg.get('args', {})
    
    def __call__(self, ch):
        if self.norm_name == "batchnorm": 
            return torchsparse.nn.BatchNorm(ch)
        elif self.norm_name == "instancenorm": 
            return torchsparse.nn.InstanceNorm(ch)
        elif self.norm_name == "layernorm": 
            return torchsparse.nn.GroupNorm(1, ch) 
        elif self.norm_name == "groupnorm":
            assert len(self.norm_args), 'GroupNorm should have arguments (either n_groups or group_size)'
            n_groups, group_size = self.norm_args.pop('n_groups', False), self.norm_args.pop('group_size', False)

            assert n_groups ^ group_size

            if n_groups:
                assert (n_groups) > 0 & n_groups <= ch, "n_groups must be > 0 and <= number of channels"
                groups = n_groups
            else:
                groups = ch // group_size
                assert groups != 0

            return torchsparse.nn.GroupNorm(groups, ch)
        
        elif self.norm_name == "none": 
            return torchsparse.nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type: {self.norm_name}")


class NormBuilder:
    def __init__(self, cfg, dim):
        self.dim = dim
        self.norm_name = cfg['name'].lower()
        self.norm_args = cfg.get('args', {})


    def __call__(self, ch):
        if self.norm_name == 'batchnorm':
            return {'2d' : nn.BatchNorm2d, '3d' : nn.BatchNorm3d}[self.dim](ch)
        elif self.norm_name == 'instancenorm':
            return {'2d' : nn.InstanceNorm2d, '3d' : nn.InstanceNorm3d}[self.dim](out_c)
        elif self.norm_name == 'layernorm':
            return nn.GroupNorm(1, ch) 

        elif self.norm_name == 'groupnorm':
            assert len(self.norm_args), 'GroupNorm should have arguments (either n_groups or group_size)'
            n_groups, group_size= self.norm_args.pop('n_groups', False), norm_args.pop('group_size', False)
            
            assert n_groups ^ group_size

            if n_groups:
                assert (n_groups) > 0 & n_groups <= ch, "n_groups must be > 0 and <= number of channels"
                groups = n_groups
            else:
                groups = ch // group_size
                assert groups != 0

            return nn.GroupNorm(groups, ch)

        elif self.norm_name == 'none':
            return nn.Identity()

        else:
            raise ValueError(f"Unknown normalization type: {self.norm_name}")



def ConvLayerBuilder(dim, norm):
    class ConvLayer(nn.Module):
        def __init__(self, in_c, out_c, groups=1):
            super().__init__()
            self.actv = nn.ReLU()
            self.conv = {'2d' : nn.Conv2d, '3d' : nn.Conv3d}[dim](in_c, out_c, 3, 1, 1, groups=groups)

            self.norm = norm(out_c)
            self.layer = nn.Sequential(self.conv, self.norm, self.actv)

        def forward(self, x):
            return self.layer(x)

    return ConvLayer

def SparseConvLayerBuilder(dim, norm): 
    class SparseConvLayer(nn.Module):
        def __init__(self, in_c, out_c, groups=1):
            super().__init__()
            self.actv = torchsparse.nn.ReLU()
            if dim != "3d": 
                raise ValueError("SparseConvLayerBuilder only supports 3D convolutions")
            self.conv = torchsparse.nn.Conv3d(in_c, out_c, 3, 1, 1)

            self.norm = norm(out_c)
            self.layer = nn.Sequential(self.conv, self.norm, self.actv)
        
        def forward(self, x):
            return self.layer(x)
    
    return SparseConvLayer

class ConvBlocksBuilder:
    def __init__(self, cfg):
        assert 'dim' in cfg, "Configuration must include 'dim' key"
        self.dim = cfg['dim']
        assert 'normalization' in cfg, "Configuration must include 'normalization' key"
        norm = cfg['normalization']
        assert 'name' in norm, "Normalization configuration must include 'name' key"
        self.norm = NormBuilder(norm, self.dim)
        self.default_layer = ConvLayerBuilder(self.dim, self.norm)

    def __call__(self, block_name, block_args):
        return CONV_BLOCKS_REGISTRY.get(block_name)(self.default_layer, self.dim, **block_args)

class SparseConvBlocksBuilder: 
    def __init__(self, cfg): 
        assert 'dim' in cfg, "Configuration must include 'dim' key"
        self.dim = cfg['dim']
        assert 'normalization' in cfg, "Configuration must include 'normalization' key"
        norm = cfg['normalization']
        assert 'name' in norm, "Normalization configuration must include 'name' key"
        self.norm = SparseNormBuilder(norm, self.dim)
        self.default_layer = SparseConvLayerBuilder(self.dim, self.norm)
    
    def __call__(self, block_name, block_args):
        print("Building block:", block_name, "with args:", block_args)
        print("Registry contents:", CONV_BLOCKS_REGISTRY._dict.keys())
        return CONV_BLOCKS_REGISTRY.get(block_name)(self.default_layer, self.dim, **block_args)

class DecoderBlocksBuilder:
    def __init__(self, cfg):
        assert 'dim' in cfg, "Configuration must include 'dim' key"
        self.dim = cfg['dim']
        assert 'normalization' in cfg, "Configuration must include 'normalization' key"
        norm = cfg['normalization']
        assert 'name' in norm, "Normalization configuration must include 'name' key"
        self.norm = NormBuilder(norm, self.dim)
        self.default_layer = ConvLayerBuilder(self.dim, self.norm)
    def __call__(self, block_name, block_args):
        return DECODER_BLOCKS_REGISTRY.get(block_name)(self.default_layer, self.dim, **block_args)        