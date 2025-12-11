from __future__ import annotations

import torch

from nucli_train.utils.registry import Registry


PROCESSORS_REGISTRY = Registry('processors')

@PROCESSORS_REGISTRY.register('perceptual_medical')
class PerceptualMedicalNetProcessor:
    def __init__(self):
        self.in_channels = 1
        self.out_channels = 1
    def __call__(self, pred, target):
        """
        Process the prediction and target tensors for perceptual loss calculation.
        This function normalizes the input tensors by subtracting the mean and dividing by the standard deviation.

        Args:
            pred (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The processed tensor.
        """
        mean = torch.mean(target, dim=(1, 2, 3, 4), keepdim=True)
        std = torch.std(target, dim=(1, 2, 3, 4), keepdim=True) + 1e-8
        return (pred - mean) / std, (target - mean) / std



@PROCESSORS_REGISTRY.register('3-channel')
class MultiChannelProcessor:
    def __init__(self):
        self.in_channels = 1
        self.out_channels = 3

    def __call__(self, x, _=None):
        hdr = x/(x+1)
        log_input = torch.log(x+1)
        scaled_input = x/2.5

        return torch.cat([hdr, log_input, scaled_input], 1)




@PROCESSORS_REGISTRY.register('identity')
class IdentityImageProcessor:
    def __init__(self):

        self.in_channels = 1
        self.out_channels = 1
        
        
    def __call__(self, x, _=None):
        return x

@PROCESSORS_REGISTRY.register('residual')
class ResidualImageProcessor:
    def __init__(self):

        self.in_channels = 1
        self.out_channels = 1
        
        
    def __call__(self, x, inputs):
        return x + inputs


