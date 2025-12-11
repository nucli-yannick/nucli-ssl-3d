from __future__ import annotations

from nucli_train.data_management.dataset import PatchesDataset, Blosc2Dataset
from nucli_train.training import Trainer, AdditiveLosses

from nucli_train.nets.conv_blocks import ConvLayerBuilder, CustomDenoisingBlockFactory, create_simple_decoder_block
from nucli_train.nets.nuclarity import NuclarityOutputBlock, create_nuc_downsample, NuclarityOutputBlockReLU
from nucli_train.nets.unet import UNet

from nucli_train.data_management.processors import IdentityImageProcessor, ResidualImageProcessor, MultiChannelProcessor, AnscombeProcessor, PerceptualMedicalNetProcessor

import torch
import torch.nn as nn

from nucli_train.models import ImageTranslationModel
from torch.utils.data import DataLoader
import yaml
import os
import random

from nucli_train.data_management.splits import create_splits

from nucli_train.training.perceptual import PerceptualLoss3D

processors = {'identity' : IdentityImageProcessor, 'residual' : ResidualImageProcessor, '3-channel' : MultiChannelProcessor, 'Anscombe' : AnscombeProcessor}

downsample_block_factories = {'Nuclarity' : create_nuc_downsample}

output_blocks = {'Nuclarity_3D' : NuclarityOutputBlock, 'Nuclarity_ReLU' : NuclarityOutputBlockReLU}

decoder_blocks = {'Simple' : create_simple_decoder_block}

def build_model(cfg):
    dim = cfg['dim']
    input_processor = processors[cfg['input_processor']['type']](cfg['input_processor'])
    output_processor = processors[cfg['output_processor']['type']](cfg['output_processor'])
    print(cfg['normalization'])
    conv_layer = ConvLayerBuilder(dim, cfg['normalization']['type'], cfg['normalization'].pop('args', None))

    enc_cfg = cfg['encoder']

    encoder_block = CustomDenoisingBlockFactory(conv_layer, dense=enc_cfg['block']["dense"], residual=enc_cfg['block']["residual"], amt_conv_layers=enc_cfg['block']["convs"])


    downsample_block = downsample_block_factories[cfg['downsample']['type']](dim)

    dec_cfg = cfg['decoder']

    decoder_block = decoder_blocks[dec_cfg['type']](conv_layer)
    
    output_block = output_blocks[cfg['output_block']['type']]
    

    network = UNet(encoder_block, downsample_block, decoder_block, nn.Identity, output_block, enc_cfg['features'], dec_cfg['features'], cfg['bottleneck_features'], input_processor, output_processor)

    metrics = {}

    perceptual_net = torch.hub.load("warvito/MedicalNet-models", model="medicalnet_resnet10_23datasets")
    features = {'relu' : 1.0, 'layer1.0.bn2' : 1.0}
    perceptual_loss = PerceptualLoss3D(perceptual_net, features, PerceptualMedicalNetProcessor(None), False, [])
    perceptual_loss.FE.cuda()

    loss_functions = [{'name' : 'L1', 'f' : nn.L1Loss(), 'coef' : 0.2}, {'name' : 'perceptual', 'f' : perceptual_loss, 'coef' : 1.0}]

    #return ImageTranslationModel(network, {"name": "MedicalNetandL1", "f": AdditiveLosses([nn.L1Loss(), perceptual_loss], weights=[1.0, 1.0])}, metrics={})
    return ImageTranslationModel(network, loss_functions, metrics={})




base_seed = 6582 # Half-life of [18F]FDG in seconds :)

def seed_worker(worker_id):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)



def build_dataloaders(cfg, train_bs, val_bs):

    split_dir = f'./experiments/{cfg['experiment_name']}/splits.yaml'
    data_dir = cfg['data_dir']

    os.makedirs(f'./experiments/{cfg['experiment_name']}', exist_ok=True)

    if os.path.exists(split_dir):
        split_dict = yaml.safe_load(open(split_dir, 'r'))
    else:
        split_dict = create_splits(cfg)
        split_file = open(split_dir, 'w+')
        yaml.dump(split_dict, split_file)


    #train_ds = ImageTranslationPETDataSet(split_dict['train'], data_dir, [64, 64, 64])
    #train_ds = PatchesDataset('/home/ec2-user/volume/data/basic_exp/united/fdg/25pc/train.npy')
    train_ds = Blosc2Dataset('/home/ec2-user/volume/data/blosc2/', split='train')
    train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=7, worker_init_fn=seed_worker)

    val_loaders = {}

    for center, tracers in split_dict['val'].items():
        for tracer, val_set in tracers.items():
            for dose in val_set['doses']:
                ds = PatchesDataset(os.path.join('/home/ec2-user/volume/data/basic_exp', center, tracer, dose, 'val.npy'))
                val_loaders[f'{center}_{tracer}_{dose}'] = DataLoader(ds, batch_size=val_bs, shuffle=False, num_workers=4, worker_init_fn=seed_worker)

    return train_dl, val_loaders

def build_trainer(exp_cfg, run_cfg, model, train_dl, val_loaders):
    return Trainer(model, train_loader=train_dl, val_loaders=val_loaders, use_amp=True, run_name=run_cfg["run_name"], experiment_name=exp_cfg["experiment_name"], val_interval=run_cfg["val_interval"], save_interval=run_cfg["save_interval"], resuming={"epoch": 20, "weights_path": '/home/ec2-user/volume/basic-experiments/experiments/chul_train/3c_in_3stage_layernorm/', 'opt' : False})



if __name__=="__main__":
    '''experiment_config = input("Path to experiment config file:")
    model_config = input("Path to run config file:")'''

    experiment_config = '/home/ec2-user/volume/basic-experiments/experiment.yaml'
    model_config = '/home/ec2-user/volume/basic-experiments/layer_3channel_input.yaml'

    run_cfg = yaml.safe_load(open(model_config, 'r'))
    exp_cfg = yaml.safe_load(open(experiment_config, 'r'))

    torch.manual_seed(base_seed)



    model = build_model(run_cfg)

    train_loader, val_loaders = build_dataloaders(exp_cfg, run_cfg["train_bs"], run_cfg['val_bs'])

    trainer = build_trainer(exp_cfg, run_cfg, model, train_loader, val_loaders)

    #torch.autograd.set_detect_anomaly(True)
    trainer.run(run_cfg["epochs"])