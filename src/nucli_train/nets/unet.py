from __future__ import annotations
from typing import Type

import torchsparse
from .conv_blocks.builder import (
    DecoderBlocksBuilder, 
    ConvBlocksBuilder, 
    SparseConvBlocksBuilder,
    DOWNSAMPLERS_REGISTRY
    )

from .builders import ARCHITECTURE_BUILDERS_REGISTRY

from nucli_train.data_management.processors import PROCESSORS_REGISTRY


import torch.nn as nn
import torch
import os

import math


def to_sparse(x):
    """
    Convert a 5D dense tensor [B, C, D, H, W] into a torchsparse SparseTensor.
    
    Args:
        x (torch.Tensor): Dense tensor of shape [B, C, D, H, W]
    
    Returns:
        torchsparse.SparseTensor: Sparse tensor representation.
    """

    assert x.ndim == 5, "Input tensor must be 5D (B, C, D, H, W)"
    B, C, D, H, W = x.shape

    x_mask = x.sum(dim=1)  
    coords = x_mask.nonzero(as_tuple=False) 

    feats = x[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]]  # [N, C]

    coords = coords.int()

    return torchsparse.SparseTensor(coords=coords, feats=feats)



class UNet(nn.Module):

    def __init__(
        self,
        encoder_block: Type[nn.Module],
        downsample_block: Type[nn.Module],
        decoder_block: Type[nn.Module],
        input_block: Type[nn.Module],
        output_block: Type[nn.Module],
        encoder_features: list | tuple,
        decoder_features: list | tuple,
        bottleneck_features: int,
        input_processor,
        output_processor 
    ):
        """
        Args:
            encoder_block: Basic block used within encoder.
            downsample_block: Used to reduce spatial resolution between encoder stages.
            decoder_block: Takes in previous stage and skip connection to produce next stage.
            input_block, output_block: Used to process input and final feature maps respectively.
            encoder_features: Number of channels per stage in the encoder.
            decoder_features: Number of channels per stage in the decoder.
            encoder_blocks: Number of primary blocks per encoder stage.
            input_processor: Preprocessing module for input (non-trainable).
            output_processor: Postprocessing module for output (non-trainable).
        """
        super().__init__()

        self.input_processor = input_processor
        self.output_processor = output_processor

        self.blocks = nn.ModuleDict()

        # Input block
        self.blocks['input'] = input_block(input_processor.out_channels, encoder_features[0])

        self.blocks[f'encode_{0}'] = encoder_block(encoder_features[0], encoder_features[0])
        
        self.blocks[f'down_{0}'] = downsample_block(encoder_features[0])

        # Encoder
        for i, stage_ch in enumerate(encoder_features[1:]):
            self.blocks[f'encode_{i+1}'] = encoder_block(encoder_features[i], stage_ch)
            
            self.blocks[f'down_{i+1}'] = downsample_block(stage_ch)

        # Bottleneck

        self.blocks['bottleneck_0'] = encoder_block(encoder_features[-1], bottleneck_features)
       
        prev_stage_ch = bottleneck_features


        # Decoder
        for i, stage_ch in enumerate(reversed(decoder_features)):
            enc_ch = encoder_features[-1 - i]
            self.blocks[f'up_{i}'] = decoder_block(prev_stage_ch, enc_ch, stage_ch)

            prev_stage_ch = stage_ch

        # Output block
        self.blocks['output'] = output_block(decoder_features[0], output_processor.in_channels)



    def forward(self, inputs):
 

        x = self.input_processor(inputs)


        print("Input shape at the beginning:", x.shape)
        to_be_catted = []  # use as a stack (push/pop)
        for key in self.blocks.keys(): #nn.module_dict is an ordered dictionary that respects order of insertion
            if "down" in key:
                to_be_catted.append(x)
                x = self.blocks[key](x)
                print(f"After {key}, shape:", x.shape)

            elif "up_" in key:
                x = self.blocks[key](x, to_be_catted.pop())
                print(f"After {key}, shape:", x.shape)
                
            else:
                x = self.blocks[key](x)
                print(f"After {key}, shape:", x.shape)


        return self.output_processor(x, inputs)


    def get_optimizer(self):
        k = 1/16
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=math.sqrt(k)*1e-4
        )
        return optimizer


    def parts_to_save(self):
        """
        Returns a dict with only the trainable parts of the model.
        """
        return {'unet': self.state_dict()}

    def load_checkpoint(self, checkpoint_dir, epoch):
        self.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'unet_epoch_{epoch}.pt')))

@ARCHITECTURE_BUILDERS_REGISTRY.register('unet')
def build_unet_from_cfg(cfg):
    
    conv_block_building = ConvBlocksBuilder(cfg['convolution'])
    

    input_block = conv_block_building(cfg['input_block']['name'], cfg['input_block'].pop('args', {}))
    encoder_block = conv_block_building(cfg['encoder_block']['name'], cfg['encoder_block'].pop('args', {}))
    output_block = conv_block_building(cfg['output_block']['name'], cfg['output_block'].pop('args', {}))

    downsample_block = DOWNSAMPLERS_REGISTRY.get(cfg['downsample_op']['name'])(cfg['convolution']['dim'])

    
    decoder_block = DecoderBlocksBuilder(cfg['convolution'])(
            cfg['decoder_block']['name'], cfg['decoder_block'].pop('args', {})
        )
    

    input_processor = PROCESSORS_REGISTRY.get(cfg['input_processor']['name'])()
    output_processor = PROCESSORS_REGISTRY.get(cfg['output_processor']['name'])()

    return UNet(
        input_block=input_block,
        encoder_block=encoder_block,
        downsample_block=downsample_block,
        decoder_block=decoder_block,
        output_block=output_block,
        encoder_features=cfg['encoder_features'],
        decoder_features=cfg['decoder_features'],
        bottleneck_features=cfg['bottleneck_features'],
        input_processor=input_processor,
        output_processor=output_processor 
    )




class SparseUNet(nn.Module): 

    def __init__(
        self, 
        encoder_block: Type[nn.Module],
        downsample_block: Type[nn.Module],
        decoder_block: Type[nn.Module],
        input_block: Type[nn.Module],
        output_block: Type[nn.Module],
        encoder_features: list | tuple,
        decoder_features: list | tuple,
        bottleneck_features: int,

    ):

        """
        In this first version, we assume that the input processor and the output
        processor are identities. 
        Adam.
        """


        super().__init__()

        self.blocks = nn.ModuleDict()

        # Input block
        self.blocks['input'] = input_block(1, encoder_features[0]) # Only 1 channel for the input, not a RGB image

        self.blocks[f'encode_{0}'] = encoder_block(encoder_features[0], encoder_features[0]) 

        self.blocks[f'down_{0}'] = downsample_block(encoder_features[0])

        # Encoder
        for i, stage_ch in enumerate(encoder_features[1:]):
            self.blocks[f'encode_{i+1}'] = encoder_block(encoder_features[i], stage_ch)
            self.blocks[f'down_{i+1}'] = downsample_block(stage_ch)
        
        # Bottleneck
        self.blocks['bottleneck_0'] = encoder_block(encoder_features[-1], bottleneck_features)

        prev_stage_ch = bottleneck_features

        # Decoder -- note: here the tensors are not sparse anymore
        for i, stage_ch in enumerate(reversed(decoder_features)):
            enc_ch = encoder_features[-1 - i]
            self.blocks[f'up_{i}'] = decoder_block(prev_stage_ch, enc_ch, stage_ch)
            prev_stage_ch = stage_ch
        
        self.blocks['output'] = output_block(decoder_features[0], 1)  # Output is a single channel

        self.bottleneck_features = bottleneck_features

    def forward(self, inputs):
        x = inputs
        original_shape = list(x.shape)
        # Convert to sparse tensor
        x = to_sparse(x)

        to_be_catted = []  # use as a stack (push/pop)
        first_up = False
        for key in self.blocks.keys(): #nn.module_dict is an ordered dictionary that respects order of insertion
            #print(key)
            #print(original_shape)
            if "down" in key:
                to_be_catted.append(torchsparse.utils.to_dense(x.feats, x.coords, spatial_range=(original_shape[0], original_shape[2], original_shape[3], original_shape[4])).permute(0, 4, 1, 2, 3))
                x = self.blocks[key](x)
                
                original_shape[2] = original_shape[2] // 2
                original_shape[3] = original_shape[3] // 2
                original_shape[4] = original_shape[4] // 2

            elif "up_" in key:
                
                if not first_up:
                    first_up = True
                    x = torchsparse.utils.to_dense(x.feats, x.coords, spatial_range=(original_shape[0], original_shape[2], original_shape[3], original_shape[4])).permute(0, 4, 1, 2, 3)
                x = self.blocks[key](x, to_be_catted.pop())
                original_shape = list(x.shape)
                
            else:
                
                x = self.blocks[key](x)
                if "input" in key:
                    original_shape[1] = 16
                elif "output" in key:
                    original_shape[1] = 1
                elif "bottleneck" in key:
                    original_shape[1] = self.bottleneck_features
                elif "encode" in key:
                    if "encode_0" not in key:
                        original_shape[1] = original_shape[1] * 2
            #print(original_shape)
        return x

    def get_optimizer(self):
        k = 1/16
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=math.sqrt(k)*1e-4
        )
        return optimizer

    def parts_to_save(self):
        """
        Returns a dict with only the trainable parts of the model.
        """
        return {'unet': self.state_dict()}

    def load_checkpoint(self, checkpoint_dir, epoch):
        self.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'unet_epoch_{epoch}.pt')))



@ARCHITECTURE_BUILDERS_REGISTRY.register('sparse-unet')
def build_sparse_unet_from_cfg(cfg): 

    

    conv_block_building = SparseConvBlocksBuilder(cfg['convolution'])

    input_block = conv_block_building(cfg['input_block']['name'], cfg['input_block'].pop('args', {}))
    encoder_block = conv_block_building(cfg['encoder_block']['name'], cfg['encoder_block'].pop('args', {}))
    output_block = conv_block_building(cfg['output_block']['name'], cfg['output_block'].pop('args', {}))

    downsample_block = DOWNSAMPLERS_REGISTRY.get(cfg['downsample_op']['name'])(cfg['convolution']['dim'])

    decoder_block = DecoderBlocksBuilder(cfg['convolution'])(
            cfg['decoder_block']['name'], cfg['decoder_block'].pop('args', {})
    )



    return SparseUNet(
        encoder_block=encoder_block,
        downsample_block=downsample_block,
        decoder_block=decoder_block,
        input_block=input_block,
        output_block=output_block,
        encoder_features=cfg['encoder_features'],
        decoder_features=cfg['decoder_features'],
        bottleneck_features=cfg['bottleneck_features'],
    )


class EncoderUNet(nn.Module):
    def __init__(
        self,
        encoder_block: Type[nn.Module],
        downsample_block: Type[nn.Module],
        input_block: Type[nn.Module],
        encoder_features: list | tuple,
        bottleneck_features: int,
        input_processor
    ):
        super().__init__()

        self.input_processor = input_processor
        self.blocks = nn.ModuleDict()

        # Input block
        self.blocks['input'] = input_block(input_processor.out_channels, encoder_features[0])
        self.blocks['encode_0'] = encoder_block(encoder_features[0], encoder_features[0])
        self.blocks['down_0'] = downsample_block(encoder_features[0])

        for i, stage_ch in enumerate(encoder_features[1:]):
            self.blocks[f'encode_{i+1}'] = encoder_block(encoder_features[i], stage_ch)
            self.blocks[f'down_{i+1}'] = downsample_block(stage_ch)

        # Bottleneck
        self.blocks['bottleneck_0'] = encoder_block(encoder_features[-1], bottleneck_features)
       
        prev_stage_ch = bottleneck_features



    def forward(self, inputs):
        x = self.input_processor(inputs)
        for key in self.blocks.keys():
            x = self.blocks[key](x)
        return x

    def get_optimizer(self):
        k = 1/16
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=math.sqrt(k) * 1e-4
        )
        return optimizer
    
    def parts_to_save(self):
            """
            Returns a dict with only the trainable parts of the model.
            """
            return {'encoder_unet': self.state_dict()}


    
    def load_checkpoint(self, checkpoint_dir, epoch):
        self.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'unet_epoch_{epoch}.pt')))


@ARCHITECTURE_BUILDERS_REGISTRY.register('encoder_unet')
def build_encoder_unet_from_cfg(cfg):
    conv_block_building = ConvBlocksBuilder(cfg['convolution'])

    input_block = conv_block_building(cfg['input_block']['name'], cfg['input_block'].pop('args', {}))
    encoder_block = conv_block_building(cfg['encoder_block']['name'], cfg['encoder_block'].pop('args', {}))

    downsample_block = DOWNSAMPLERS_REGISTRY.get(cfg['downsample_op']['name'])(cfg['convolution']['dim'])
    input_processor = PROCESSORS_REGISTRY.get(cfg['input_processor']['name'])()

    return EncoderUNet(
        input_block=input_block,
        encoder_block=encoder_block,
        downsample_block=downsample_block,
        encoder_features=cfg['encoder_features'],
        bottleneck_features=cfg['bottleneck_features'],
        input_processor=input_processor
    )

    