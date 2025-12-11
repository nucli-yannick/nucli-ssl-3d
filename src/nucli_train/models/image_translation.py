from __future__ import annotations

import torch
import torch.nn as nn

import numpy as np

from os.path import join

from torch.amp import autocast

from nucli_train.models.inference.sliding_window_unet import sliding_window_inferer

from nucli_train.data_management.resampling import resample

from nucli_train.nets import build_network
from nucli_train.models.losses import build_losses
from nucli_train.models.builders import MODEL_BUILDERS_REGISTRY

class ImageTranslationModel(nn.Module):
    def __init__(self, net, loss_functions=None, metrics={}, optimizer=None):
        super().__init__()
        self.network = net
        #self.processor = processor


        if loss_functions:
            self.objectives = {loss_function["name"]: (loss_function["f"], loss_function['coef']) for loss_function in loss_functions}
        
        else:
            self.objectives = {}

        self.opt = {}


        self.metrics = {metric_name: metric_constructor() for metric_name, metric_constructor in metrics.items()}
        self.ld_metrics = {metric_name: metric_constructor() for metric_name, metric_constructor in metrics.items()}

        if optimizer: self.opt = optimizer
        else: self.opt = self.network.get_optimizer()
    
    def cuda(self):
        self.network.cuda()

    def get_params(self):
        return {"paramsM": sum(p.numel() for p in self.network.parameters())/10**6}


    def train_step(self, batch):
        inputs, targets = batch["input"].cuda(), batch["target"].cuda()

        outputs = self.network(inputs)

        losses = self.get_losses(outputs, targets)

        return losses


    def get_losses(self, predicted, target):
        losses = {}
        for name, objective in self.objectives.items():
            val = objective[0](predicted, target)
            losses[name] = val
            if 'value' in losses.keys():
                losses['value'] += objective[1] * val
            else:
                losses['value'] = objective[1] * val

        return losses

    def get_losses_several_inputs(self, loss_inputs_dict): 
        losses = {}
        for name, objective in self.objectives.items():
            all_val = objective[0](**loss_inputs_dict)
            val = all_val["main"]
            losses[name] = val
            if 'value' in losses.keys():
                losses['value'] += objective[1] * val
            else:
                losses['value'] = objective[1] * val
        
        losses["info"] = all_val

        return losses



    def get_metrics(self, predicted, target, inp):
        output = {}
        slices = predicted.shape[0]
        predicted, target, inp = torch.split(predicted, 1), torch.split(target, 1), torch.split(inp, 1)
        for name, metric in self.metrics.items():
            metric_relative = 0.0
            for slice_ind in range(slices):
                denoised_im_metric = float(metric.forward(predicted[slice_ind], target[slice_ind]))
                ld_im_metric = float(self.ld_metrics[name].forward(inp[slice_ind], target[slice_ind]))
                metric_relative += (denoised_im_metric/ld_im_metric - 1)*100
            metric_relative /= slices
            output[name] = metric_relative
        return output
    
    def compute_metrics(self):
        out = {}
        for name, metric in self.metrics.items():
            v = float(metric.compute())
            ld = float(self.ld_metrics[name].compute())
            r = (v/ld - 1)*100
            out[name] = {"model": v, "difference": r}
            metric.reset()
            self.ld_metrics[name].reset()
        return out


    def validation_step(self, batch):
        inputs, targets = batch["input"].cuda(), batch["target"].cuda()



        with autocast('cuda', torch.float16), torch.no_grad():
            outputs = self.network(inputs)

        losses = self.get_losses(outputs, targets)

        outputs = outputs.detach().cpu() # should remove this at some point. Going to cpu only makes sense if we want to save images
        targets = targets.detach().cpu()
        inputs = inputs.detach().cpu()


        metrics = self.get_metrics(outputs, targets, inputs)

        return {"losses": losses, "metrics": metrics, "predictions": outputs}


    def predict(self, images):
        images = images.cuda()
        with autocast('cuda', torch.float16), torch.no_grad():
                outputs = self.network(images)
        return outputs.detach().cpu().squeeze().numpy()

    def save_opt(self, ckpt_dir, epoch):
        torch.save(self.opt.state_dict(), join(ckpt_dir, f'opt_e{epoch}.pt'))


    def _load_checkpoint(self, ckpt_dir, epoch, opt=None):
        self.network.load_checkpoint(ckpt_dir, epoch)
        if opt:
            self.opt.load_state_dict(torch.load(f"{ckpt_dir}/opt_e{epoch}.pth"))

    def reset_metrics(self):
        self.metrics = {k: metric() for k, metric in self.metric_constructors.items()}


    def train(self): # In Python method overloading is not supported, latest definition counts --> we don't need to add mode arg
        self.network.train() # self.FE is in eval and should remain there

    def eval(self):
        self.network.eval() # self.FE is already in eval and we do not want to break things

    def infer_scan(self, scan : np.array, spacing = None, target_spacing=None):
        target_spacing = (2.734375, 2.734375, 3.2700)
        if spacing is not None:
            assert target_spacing is not None
            scan = resample(scan, spacing, target_spacing)
        prediction = sliding_window_inferer(self, scan, 64, 16, batch_size=32)

        if spacing is not None:
            resampled_prediction = resample(prediction, target_spacing, spacing)


            return {'prediction' : resampled_prediction, 'prediction_target_resolution' : prediction}

        return prediction


    def get_schedulers(self):
        return []

    def get_optimizers(self):
        return [self.opt]
    
    def models_to_save(self):
        return self.network.parts_to_save()
    

@MODEL_BUILDERS_REGISTRY.register('image_translation')
def build_image_translation_model(cfg):
    
    network = build_network(cfg['args']['network'])

    losses = build_losses(cfg['args']['losses'])    

    return ImageTranslationModel(network, loss_functions=losses)