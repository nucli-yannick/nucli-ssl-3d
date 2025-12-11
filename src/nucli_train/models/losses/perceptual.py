from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from monai.utils import optional_import
from monai.utils.enums import StrEnum


from typing import Type

from .builder import LOSSES_REGISTRY

from torchvision.models.feature_extraction import create_feature_extractor


@LOSSES_REGISTRY.register("PerceptualLossMedicalNet")
class PerceptualLoss3D(nn.Module):
    """
    Base class for perceptual losses in 3D. This class handles averaging the losses, weighting stages, ...
    Assumes the number of channels in ImageProcessor's output is equal to the number of channels the feature extractor expects, OR each channel is processed separately (control through seperate_channels).
    Forward always returns a scalar loss.  TO DO: Implement a way to return channel-wise losses for multi-modal scenarios.

    USER TIP: It is highly recommended that your features are not originating from an activation function, as this introduces signal sparisity + risk of division by zero. 
    """

    def __init__(self, layers : list | tuple, weights : list | tuple, seperate_channels : bool = False, channel_weights : list | tuple = []) -> None:
        super().__init__()

        network = torch.hub.load("warvito/MedicalNet-models", model="medicalnet_resnet10_23datasets")


        self.FE = create_feature_extractor(network, return_nodes=list(layers)).cuda()
        self.weights = {stage: weight for stage, weight in zip(layers, weights)}

        self.FE.eval()
        for param in self.FE.parameters():
            param.requires_grad = False

        self.loss = nn.MSELoss()

        self.sep_ch = seperate_channels
        self.ch_weights = channel_weights if seperate_channels else [1.0]
        


    def _preprocess(self, pred, target):
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

    def forward(self, prediction : torch.Tensor, target : torch.Tensor):

        """
        Compute perceptual loss using the feature extractor. The input and target tensors are inputted in the
        pre-trained network that is used for feature extraction. Then, these extracted features are normalised across
        the channels. Finally, we compute the difference between the input and target features and calculate the mean
        value from the spatial dimensions to obtain the perceptual loss.

        Args:
            input: Input tensor with shape BCDHW.
            target: Target tensor with shape BCDHW.
        """
        processed_pred, processed_target = self._preprocess(prediction, target)

        feats_diff = 0

        its = processed_pred.shape[1] if self.sep_ch else 1

        for i in range(its):

            if self.sep_ch:
                pred_vol = processed_pred[:, i:i+1]
                target_vol = processed_target[:, i:i+1]
            else:
                pred_vol = processed_pred
                target_vol = processed_target


            outs_pred = self.FE(pred_vol)
            outs_target = self.FE(target_vol)

            feats_diff = 0
            for stage, st_weight in self.weights.items():
                feats_pred = outs_pred[stage]
                feats_target = outs_target[stage]

                feats_diff += st_weight * self.ch_weights[i] * torch.sqrt(torch.mean((feats_pred - feats_target)**2))

            return feats_diff




class PerceptualLossSingleView(nn.Module):
    """
    Perceptual losses on 3D data, applying 2D feature extractors to a single view/axis.

    USER TIP: It is highly recommended that your features are not originating from an activation function, as this introduces signal sparisity + risk of division by zero. 
    """
    def __init__(self, network : nn.Module, stage_weights : dict, preprocessor, axis : int, seperate_channels : bool, channel_weights : list | tuple) -> None:
        super().__init__()


        assert (not seperate_channels) ^ len(channel_weights) == preprocessor.out_channels 

        self.processor = preprocessor
        self.FE = create_feature_extractor(network, return_nodes=list(stage_weights.keys()))
        self.weights = stage_weights
        self.axis = axis

        self.perm = [axis, 1] + [2, 3, 4].remove(axis)

        self.FE.eval()
        for param in self.FE.parameters():
            param.requires_grad = False

        self.loss = nn.MSELoss()

        self.sep_ch = seperate_channels
        self.ch_weights = channel_weights if seperate_channels else [1.0]


    def forward(self, prediction : torch.Tensor, target : torch.Tensor):
        """
        Compute perceptual loss using the feature extractor. The input and target tensors are inputted in the
        pre-trained network that is used for feature extraction. Then, these extracted features are normalised across
        the channels. Finally, we compute the difference between the input and target features and calculate the mean
        value from the spatial dimensions to obtain the perceptual loss.

        Args:
            input: Input tensor with shape BCDHW.
            target: Target tensor with shape BCDHW.
        """
        s = prediction.shape
        processed_pred = self.processor.prepare(prediction)
        processed_target = self.processor.prepare(target)

        processed_pred = processed_pred.permute([0] + self.perm).view(-1, *[s[i] for i in self.perm[1:]])
        processed_target = processed_target.permute([0] + self.perm).view(-1, *[s[i] for i in self.perm[1:]])

        feats_diff = 0

        its = processed_pred.shape[1] if self.sep_ch else 1

        for i in range(its):

            if self.sep_ch:
                pred_vol = processed_pred[:, i:i+1]
                target_vol = processed_target[:, i:i+1]
            else:
                pred_vol = processed_pred
                target_vol = processed_target


            outs_pred = self.FE(pred_vol)
            outs_target = self.FE(target_vol)

            feats_diff = 0
            for stage, st_weight in self.weights.items():
                feats_pred = outs_pred[stage]
                feats_target = outs_target[stage]

                feats_diff += st_weight * self.ch_weights[i] * self.loss(feats_pred, feats_target)

            return feats_diff
