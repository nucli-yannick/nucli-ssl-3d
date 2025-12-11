from __future__ import annotations

import torch
import torch.nn as nn

import numpy as np

from os.path import join

from torch.amp import autocast

from nucli_train.models.inference.sliding_window import sliding_window_inferer

from nucli_train.data_management.resampling import resample

from nucli_train.nets import NucUNet

from .builders import MODEL_REGISTRY


@MODEL_REGISTRY.register('nuclarity')
class Nuclarity(nn.Module):
    def __init__(self, low_noise_path, high_noise_path, batch_size : int =8, patch_size : int =64, stride : int =16, target_spacing : tuple = (2.734375, 2.734375, 3.2700)):
        # note: maybe support patch size non uniform at some point
        super().__init__()
        self.low_noise_path = low_noise_path
        self.high_noise_path = high_noise_path

        self.low_noise_model = NucUNet()
        self.low_noise_model.load_checkpoint(low_noise_path, '_')

        self.high_noise_model = NucUNet()
        self.high_noise_model.load_checkpoint(high_noise_path, '_')

        self.target_spacing = target_spacing
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride

    def predict_high_noise(self, x):
        with autocast('cuda', torch.float16), torch.no_grad():
            out = self.high_noise_model(x)

        return out.detach().cpu().squeeze().numpy()

    def predict_low_noise(self, x):
        with autocast('cuda', torch.float16), torch.no_grad():
            out = self.low_noise_model(x)
        
        return out.detach().cpu().squeeze().numpy()

    def validation_step(self, batch):
        inputs, targets = batch["input"].cuda(), batch["target"].cuda()


        outputs = self.predict_low_noise(inputs)
        with autocast('cuda', torch.float16), torch.no_grad():
            outputs = self.low_noise_model(inputs).detach().cpu()

        losses = {}

        outputs = outputs.detach().cpu() # should remove this at some point. Going to cpu only makes sense if we want to save images
        targets = targets.detach().cpu()
        inputs = inputs.detach().cpu()


        metrics = {}

        return {"losses": losses, "metrics": metrics, "predictions": outputs}


    def infer_scan(self, scan : np.array, spacing : tuple | None = None, store_seperate_channels : bool = False, resample_seperate_channels : bool = False):

        # note: if we do not need the entire image for weighting, we can speed this up using blosc2 or some other memmap tech

        if spacing is not None:
            assert self.target_spacing is not None
            scan = resample(scan, spacing, self.target_spacing)

        assert resample_seperate_channels or not store_seperate_channels, "Cannot resample separate channels if not storing separate channels"
        self.cuda()
        self.eval()

        prediction, low_noise_prediction, high_noise_prediction, weight_c0 = sliding_window_inferer(self, scan, self.patch_size, self.stride, batch_size=self.batch_size, store_seperate_channels=store_seperate_channels)

        if spacing is not None:

            resampled_prediction = resample(prediction, self.target_spacing, spacing)
            if resample_seperate_channels:
                resampled_low_noise_prediction = resample(low_noise_prediction, self.target_spacing, spacing)
                resampled_high_noise_prediction = resample(high_noise_prediction, self.target_spacing, spacing)
                return {
                    'resampled_prediction': resampled_prediction,
                    'resampled_low_noise_prediction': resampled_low_noise_prediction,
                    'resampled_high_noise_prediction': resampled_high_noise_prediction,
                    'prediction': prediction,
                    'low_noise_prediction': low_noise_prediction,
                    'high_noise_prediction': high_noise_prediction, 'w_c0' : weight_c0
                }


            return {'prediction' : resampled_prediction, 'prediction_target_resolution' : prediction}

        if store_seperate_channels:
            return {
                'prediction': prediction,
                'low_noise_prediction': low_noise_prediction,
                'high_noise_prediction': high_noise_prediction, 'w_c0' : weight_c0
            }

        return {'prediction': prediction, 'w_c0' : weight_c0}