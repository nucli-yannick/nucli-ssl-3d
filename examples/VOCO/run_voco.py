from __future__ import annotations



print("Loading VOCO preprocessor...")
print("Importing necessary modules...")
from nucli_train.data_management.builders import build_data
from nucli_train.models.builders import build_model, MODEL_REGISTRY, MODEL_BUILDERS_REGISTRY
from nucli_train.training import Trainer
import voco_transform as voto
import numpy as np
import yaml
from nucli_train.preprocess.preprocessor import PreprocessorBlosc2

from nucli_train.models.image_translation import ImageTranslationModel

import torch

from einops import rearrange

import mlflow

from nucli_train.nets import build_network
from nucli_train.models.losses import build_losses


from nucli_train.data_management.builders import build_data
import nucli_train.data_management.transformations 
print("Importing necessary modules completed.")

class MIM(ImageTranslationModel):
        
    def train_step(self, batch):
        data = batch['all_crops'].cuda()
        base_crop_index = batch['base_crop_index'].cuda()
        overlaps = batch["base_target_crop_overlaps"].cuda()

        base_list = []
        target_list = []
        
        for i in range(data.shape[0]):
            base_crop = data[i, :base_crop_index[i], :, :, :]
            target_crops = data[i, base_crop_index[i]:, :, :, :]
            base_list.append(base_crop)
            target_list.append(target_crops)
        
        batch_size = data.shape[0]
        NBASE = base_crop_index[0].item()  # Number of base crops
        nTARGET = data.shape[1] - NBASE  # Number of target crops
        
        
        
        # Concaténer tous les éléments le long de la première dimension
        base_tensor = torch.cat(base_list, dim=0)    
        target_tensor = torch.cat(target_list, dim=0) 


        base_embeddings = self.network(base_tensor[:, None, :, :, :])  # Assuming the network expects a 5D tensor
        target_embeddings = self.network(target_tensor[:, None, :, :, :])
        print("base_embeddings shape:", base_embeddings.shape)

        base_embeddings = rearrange(base_embeddings, "(b NBASE) c 1 1 1 -> b NBASE c", b=batch_size)
        target_embeddings = rearrange(target_embeddings, "(b nTARGET) c 1 1 1 -> b nTARGET c", b=batch_size)

        print("overlaps size:", overlaps.shape)

        losses = self.get_losses_several_inputs({"base_embeddings": base_embeddings, "target_embeddings": target_embeddings, "gt_overlaps": overlaps})

        return losses

    def validation_step(self, batch):
        data = batch['all_crops'].cuda()
        base_crop_index = batch['base_crop_index'].cuda()
        overlaps = batch["base_target_crop_overlaps"].cuda()

        base_list = []
        target_list = []
        
        for i in range(data.shape[0]):
            base_crop = data[i, :base_crop_index[i], :, :, :]
            target_crops = data[i, base_crop_index[i]:, :, :, :]
            base_list.append(base_crop)
            target_list.append(target_crops)
        
        batch_size = data.shape[0]
        NBASE = base_crop_index[0].item()  # Number of base crops
        nTARGET = data.shape[1] - NBASE  # Number of target crops

        num_base = base_crop_index[0].item()
        num_target = data.shape[1] - num_base
        
        
        
        # Concaténer tous les éléments le long de la première dimension
        base_tensor = torch.cat(base_list, dim=0)    
        target_tensor = torch.cat(target_list, dim=0) 

        with torch.no_grad():
            base_embeddings = self.network(base_tensor[:, None, :, :, :])  # Assuming the network expects a 5D tensor
            target_embeddings = self.network(target_tensor[:, None, :, :, :])

            
        base_embeddings = rearrange(base_embeddings, "(b NBASE) c 1 1 1 -> b NBASE c", b=batch_size)
        target_embeddings = rearrange(target_embeddings, "(b nTARGET) c 1 1 1 -> b nTARGET c", b=batch_size)

        all_losses = self.get_losses_several_inputs({"base_embeddings": base_embeddings, "target_embeddings": target_embeddings, "gt_overlaps": overlaps})

        return {
            "losses": all_losses,
            "pred_loss": all_losses["info"]["pred_loss"],
            "reg_loss": all_losses["info"]["reg_loss"],
            "base_embeddings": base_embeddings,
            "target_embeddings": target_embeddings,
            "overlaps": overlaps, 
            "bases": base_tensor,
            "targets": target_tensor,
            "num_base": num_base,
            "num_target": num_target, 
            "batch_size": batch_size, 
            "data": batch["data"]
        }




@MODEL_BUILDERS_REGISTRY.register('VoCo')
def build_VoCo(cfg):
    network = build_network(cfg['args']['network'])

    losses = build_losses(cfg['args']['losses'])    

    return MIM(network, loss_functions=losses)


print("Build model")
model = build_model('/home/interns_nuclivision_com/nucli-ssl/examples/VOCO/MIM_model.yaml')

print("Build data")
train_data, val_loaders = build_data("/home/interns_nuclivision_com/aws_backup/adam/autopet_2024/nucli_train/VOCO_adam_experiment/autopet_2024" + "/main.yaml")


print("Build trainer")
if mlflow.active_run(): mlflow.end_run()
trainer = Trainer(model, train_data=train_data, val_loaders=val_loaders, run_name='TEST-15', experiment_name='VoCo', save_interval=1, model_cfg_path='/home/interns_nuclivision_com/nucli-ssl/examples/VOCO/MIM_model.yaml', data_cfg_path="/home/interns_nuclivision_com/aws_backup/adam/autopet_2024/nucli_train/VOCO_adam_experiment/autopet_2024" + "/main.yaml")

print("Run trainer")
trainer.run(1000)