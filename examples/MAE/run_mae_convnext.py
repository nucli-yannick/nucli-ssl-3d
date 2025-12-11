from __future__ import annotations



print("Loading MAE preprocessor...")
print("Importing necessary modules...")

from nucli_train.data_management.builders import build_data
from nucli_train.models.builders import build_model, MODEL_REGISTRY, MODEL_BUILDERS_REGISTRY
from nucli_train.training import Trainer
import numpy as np
import yaml
from nucli_train.preprocess.preprocessor import PreprocessorBlosc2

from nucli_train.models.image_translation import ImageTranslationModel

import torch

from einops import rearrange

import mlflow

from nucli_train.nets import build_network
from nucli_train.models.losses import build_losses

import torchsparse


from nucli_train.data_management.builders import build_data
import nucli_train.data_management.transformations 

from einops import rearrange

from nucli_train.data_management.transformations import MAEConvnextTransform


from nucli_train.nets import convnext
print("Importing necessary modules completed.")



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


class MIM(ImageTranslationModel):

    def init_transform(self): 
        self.transform_instance = MAEConvnextTransform(volume_size=128, patch_size=self.network.patch_size, mask_ratio=0.6)
        print(self.transform_instance.patch_size)

    
        
    def train_step(self, batch):
        data = self.transform_instance(batch['data'].cuda())
        mask = data['model_mask']
        input_data = data['data'].cuda()

        output = self.network.forward(input_data, mask)

        losses = self.get_losses_several_inputs({
            "inputs": input_data, 
            "preds": output, 
            "mask": mask
        })

        return losses

    def validation_step(self, batch):
        data = self.transform_instance(batch['data'].cuda())
        mask = data['model_mask']
        input_data = data['data'].cuda()

        with torch.no_grad():
            output = self.network.forward(input_data, mask)
        
        losses = self.get_losses_several_inputs({
            "inputs": input_data, 
            "preds": output, 
            "mask": mask
        })

        preds = output

        if len(preds.shape) == 5: 
                B, C, D, H, W = preds.shape
                preds = preds.reshape(B, C, -1)
                preds = preds.permute(0, 2, 1) 

        output = self.transform_instance.unpatchify(preds)
        print("Output shape after unpatchify: ", output.shape)


        # For some testing ...

        epsilon = 1e-3
        p = int(mask.shape[1] ** (1/3) + epsilon)
        special_size = input_data.shape[2]
        upsampled_mask = self.transform_instance.upsample_mask(mask, int(special_size // p)).unsqueeze(1).type_as(input_data)

        return {
            "losses": losses,
            "output": output,
            "data": input_data,
            "mask": mask, 
            "batch_size": input_data.shape[0], 
            "masked_data": input_data * (1 - upsampled_mask)
        }




@MODEL_BUILDERS_REGISTRY.register('MAE')
def build_MAE(cfg):
    network = build_network(cfg['args']['network'])

    losses = build_losses(cfg['args']['losses'])  

    my_MIM = MIM(network, loss_functions=losses)

    my_MIM.init_transform()

    return my_MIM


print("Build model")
model = build_model('/home/interns_nuclivision_com/nucli-ssl/examples/MAE/MIM_model_convnext_sparse.yaml')

print("Build data")
train_data, val_loaders = build_data("/home/interns_nuclivision_com/aws_backup/adam/autopet_2024/nucli_train/MAE_ConvNeXt_adam_experiment/autopet_2024" + "/main.yaml")


print("Build trainer")
if mlflow.active_run(): mlflow.end_run()
trainer = Trainer(model, train_data=train_data, val_loaders=val_loaders, run_name='final_pres_mmmh_res', experiment_name='MAE-convnext', save_interval=25, model_cfg_path='/home/interns_nuclivision_com/nucli-ssl/examples/MAE/MIM_model_convnext_sparse.yaml', data_cfg_path="/home/interns_nuclivision_com/aws_backup/adam/autopet_2024/nucli_train/MAE_ConvNeXt_adam_experiment/autopet_2024" + "/main.yaml")

print("Run trainer")
trainer.run(5000)