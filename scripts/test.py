from __future__ import annotations

from nucli_train.models import model_builders
import yaml
import torch
import os
import numpy as np


def load_model(cfg, pth_path):
    model = model_builders[cfg["type"]](cfg)
    if cfg["type"]=="base_cnn":
        state_dict = torch.load(pth_path)
        if cfg["network"]["name"] == "rdunet":
            torch_state_dict = {}
            for block_name, block in state_dict.items():
                for layer in block.keys():
                    torch_state_dict[block_name + "." + layer] = block[layer]
            state_dict = torch_state_dict
        model.model.load_state_dict(state_dict)
    elif cfg["type"]=="foundation":
        state_dict = torch.load(pth_path)
        model.head.load_state_dict(state_dict)
    return model

if __name__=="__main__":
    '''config_file = input("cfg path:")
    weights_file = input("pth path:")
    ld_dir = input("ld images path:")'''
    #config_file = '/home/ec2-user/volume/foundation-denoising/configs/small_chul_train/rdunet_f0_32.yaml'
    #weights_file = '/home/ec2-user/volume/foundation-denoising/experiments/small_chul_fdg_25pc/rdunet_32f0/RDUNet_encoder_decoder_epoch_20.pt'
    #config_file = '/home/ec2-user/volume/foundation-denoising/configs/small_chul_train/convnext_basic_rd_decoder_64_32.yaml'
    #weights_file = '/home/ec2-user/volume/foundation-denoising/experiments/small_chul_fdg_25pc/convnext_basic_rd_decoder_64_32/head_epoch_20.pt'
    #ld_dir = '/home/ec2-user/volume/foundation-denoising/data/CHUL/small/fdg/val/selection/images/10pc'


    experiment = input('experiment cfg:')
    dataset = input('Dataset to plot results for:')
    run_dir = input('Run cfg:')


    exp_cfg = yaml.safe_load(open(experiment, 'r'))
    run_cfg = yaml.safe_load(open(run_dir, 'r'))

    if dataset in exp_cfg['val']:
        ld_dir = exp_cfg['val'][dataset]['ld']
    else:
        ld_dir = '/home/ec2-user/volume/foundation-denoising/data/CHUL/small/fdg/val/25pc/'

    run_dir = os.path.join('./experiments/', exp_cfg['experiment_name'], run_cfg['run_name'])

    for f in os.listdir(run_dir):
        if '.pt' in f:
            weights_file = os.path.join(run_dir, f)
            break

    model = load_model(run_cfg["model"], weights_file)
    model.cuda()
    save_dir = os.path.join(run_dir, 'denoised_' + dataset)
    os.makedirs(save_dir, exist_ok=True)

    for f in os.listdir(ld_dir):
        images = np.load(os.path.join(ld_dir, f))[:, 92:92 + 256, 3:3 + 256]
        denoised = np.empty_like(images)
        images = torch.tensor(images)
        slices = int(images.shape[0])

        batches = [16]*(slices//16) + [slices % 16]
        i = 0
        for b in batches:
            print(b)
            current_ims = images[i:i+b]
            print(current_ims.shape)
            denoised[i:i+b] = model.predict(torch.unsqueeze(current_ims, 1)).squeeze().numpy()
            i += b
        #denoised = model.predict(torch.unsqueeze(images, 1)).squeeze().numpy()
        np.save(os.path.join(save_dir, f), denoised)
