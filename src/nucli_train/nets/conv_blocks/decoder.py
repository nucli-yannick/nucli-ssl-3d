from __future__ import annotations

import torch.nn as nn
import torch
from .builder import DECODER_BLOCKS_REGISTRY

from typing import Type

import torch.nn.functional as F

@DECODER_BLOCKS_REGISTRY.register('simple-decoder-block')
def simple_decoder_block(conv_layer: Type[nn.Module], dim):
    class SimpleDecoderBlock(nn.Module):
        def __init__(self,  prev_stage_c: int, skip_c, out_channels: int):
            super().__init__()
            self.conv_prev = conv_layer(prev_stage_c, prev_stage_c)
            self.main_conv = conv_layer(prev_stage_c + skip_c, out_channels)

            self.type = type


        def forward(self, x, skip_connection):
            x = F.interpolate(x, scale_factor=2, mode="nearest")

            x = self.conv_prev(x)
            x = self.main_conv(torch.cat([x, skip_connection], dim=1))

            return x
    
    return SimpleDecoderBlock

