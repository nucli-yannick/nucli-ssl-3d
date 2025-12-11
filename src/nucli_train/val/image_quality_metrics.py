from __future__ import annotations


from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch


class PSNR_builder:
    """
    Our trainer class needs to call the constructor for torchmetric objects every validation cycle. 
    This functor is made such that we do not have any constructors to which arguments need to be passed.
    """
    def __init__(self, data_range):
        self.range = data_range
    def __call__(self):
        return PeakSignalNoiseRatio(data_range=self.range)

class SSIM_Builder:
    def __init__(self, data_range):
        self.range = data_range
    def __call__(self):
        return StructuralSimilarityIndexMeasure()

class RMSE:
    def __init__(self):
        self.MSE = MeanSquaredError(squared=False)

    def update(self, inputs, targets):
        """
        Assumes inputs, targets of shape B, 1, H, W
        """
        if len(inputs.shape) == 4:
            B, _, H, W = inputs.shape
        elif len(inputs.shape) == 2:
            H, W = inputs.shape
            B = 1

        # torcheval MSE expects 2D input (n_sample, n_output)

        reshaped_input, reshaped_target = inputs.reshape((B, H*W)), targets.reshape((B, H*W))

        self.MSE.update(reshaped_input, reshaped_target)

    def compute(self):
        rmse = self.MSE.compute()
        return rmse
    def forward(self, inputs, targets):
        if len(inputs.shape) == 4:
            B, _, H, W = inputs.shape
        elif len(inputs.shape) == 2:
            H, W = inputs.shape
            B = 1

        # torcheval MSE expects 2D input (n_sample, n_output)

        reshaped_input, reshaped_target = inputs.reshape((B, H*W)), targets.reshape((B, H*W))

        return self.MSE.forward(reshaped_input, reshaped_target)
    
    def reset(self):
        self.MSE.reset()


class SUVMax:
    def __init__(self):
        self.total_error = 0.0
        self.samples = 0
    def forward(self, im1, im2):
        sample_error = abs(im1.max() - im2.max())
        self.total_error += sample_error
        self.samples += 1
        return sample_error
    def compute(self):
        return self.total_error / self.samples

