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
        
    def train_step(self, batch):
        masked_data = batch['masked_data'].cuda()
        data = batch['data'].cuda()

        #masked_data_sparse = to_sparse(masked_data)

        output = self.network(masked_data)
        
        losses = self.get_losses(output, data)

        return losses

    def validation_step(self, batch):
        masked_data = batch['masked_data'].cuda()
        mask = batch['mask'].cuda()
        data = batch['data'].cuda()

        with torch.no_grad():
            #masked_data_sparse = to_sparse(masked_data)
            output = self.network(masked_data)
        
        losses = self.get_losses(output, data)

        return {
            "losses": losses,
            "output": output,
            "data": data,
            "mask": mask, 
            "masked_data": masked_data, 
            "batch_size": masked_data.shape[0]
        }



@MODEL_BUILDERS_REGISTRY.register('MAE')
def build_MAE(cfg):
    network = build_network(cfg['args']['network'])

    losses = build_losses(cfg['args']['losses'])    

    return MIM(network, loss_functions=losses)


print("Build model")
model = build_model('/home/interns_nuclivision_com/nucli-ssl/examples/MAE/MIM_model_sparse.yaml')

print("Build data")
train_data, val_loaders = build_data("/home/interns_nuclivision_com/aws_backup/adam/autopet_2024/nucli_train/MAE_adam_experiment/autopet_2024" + "/main.yaml")


print("Build trainer")
if mlflow.active_run(): mlflow.end_run()
trainer = Trainer(model, train_data=train_data, val_loaders=val_loaders, run_name='TEST-sparse-final-Wide', experiment_name='MAE-sparse', save_interval=25, model_cfg_path='/home/interns_nuclivision_com/nucli-ssl/examples/MAE/MIM_model_sparse.yaml', data_cfg_path="/home/interns_nuclivision_com/aws_backup/adam/autopet_2024/nucli_train/MAE_adam_experiment/autopet_2024" + "/main.yaml")

print("Run trainer")
trainer.run(1000)