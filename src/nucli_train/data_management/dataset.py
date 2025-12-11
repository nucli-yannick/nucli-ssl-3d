from __future__ import annotations

import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset
import torch
from nucli_train.utils.registry import Registry


from .builders import DATASET_REGISTRY

import os
import random

import nibabel as nib
import blosc2

from os.path import join

import yaml

TRANS_REGISTRY = Registry('transforms')

@DATASET_REGISTRY.register('patches_dataset')
class PatchesDataset(Dataset):
    def __init__(self, path_to_npy):
        self.samples = open_memmap(path_to_npy)
        self.n_samples = self.samples.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {"input" : torch.tensor(sample[:1]), "target" : torch.tensor(sample[1:])}



@DATASET_REGISTRY.register('patches_dataset_quantification')
class PatchesDatasetQuantification(Dataset):
    def __init__(self, path_to_lc, path_to_sc):

        self.samples_lc = open_memmap(path_to_lc)
        self.samples_sc = open_memmap(path_to_sc)
        self.n_samples = self.samples_lc.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        samples_lc, samples_sc = self.samples_lc[idx:idx+1], self.samples_sc[idx:idx+1]
        return {"input" : torch.tensor(samples_lc), "target" : torch.tensor(samples_sc)}


@DATASET_REGISTRY.register('blosc2_dataset')
class Blosc2Dataset(Dataset):
    def __init__(self, config_path):
        # print(config_path)
        # loads the config file passed as argument

        meta = yaml.safe_load(open(config_path, 'r'))
        
        self.dx, self.dy, self.dz = list((np.array(meta['patch_size']) // 2)) 
        self.valid_centers_folder = 'coords' # folder of valif coordinates for our images

        self.samples = [] 
        self.tracer_scanner_sets = {}

        src_dir = meta['src_dir'] # root folder

        for dataset, centers in meta['data'].items(): # iterate over the datasets 
            for center, tracers in centers.items():   # iterate over the centers where we got data from
                for tracer, tracer_info in tracers.items(): # iterate over the tracers used in eache centers

                    patients = tracer_info['patients']

                    self.tracer_scanner_sets[f'{dataset}/{center}/{tracer}'] = {
                        'pairs' : tracer_info['pairs'], 'src_dir' : join(src_dir, dataset, center, tracer)
                    }  # for each triplet; (dataset, center, tracer) => we identofy the pairs and the source directory

                    if patients == 'all':
                        for pair_name, pair_info in tracer_info['pairs'].items():
                            shared_cases = None
                            for modality in pair_info['inputs'] + pair_info.get('targets', []):
                                if shared_cases is None:
                                    shared_cases = set(os.listdir(join(self.tracer_scanner_sets[f'{dataset}/{center}/{tracer}']['src_dir'], modality)))
                                else:
                                    shared_cases.intersection_update(set(os.listdir(join(self.tracer_scanner_sets[f'{dataset}/{center}/{tracer}']['src_dir'], modality))))
                            for patient in shared_cases:
                                self.samples.append({'id' : patient, 'class' : f'{dataset}/{center}/{tracer}', 'pair_name' : pair_name})

                    else:
                        for patient in patients:
                            for pair_name in tracer_info['pairs'].keys():
                                self.samples.append({'id' : patient, 'class' : f'{dataset}/{center}/{tracer}', 'pair_name' : pair_name})


        self.n_samples = len(self.samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['id'].split('.')[0] # patient001.b2nd

        coords = self.get_coords(join(self.tracer_scanner_sets[sample['class']]['src_dir'], self.valid_centers_folder, sample_id)) # gets center coordinates

        input_arrays, target_arrays = [], [] 

        for mod in self.tracer_scanner_sets[sample['class']]['pairs'][sample['pair_name']]['inputs']:
            input_arrays.append(self.load_patch(join(self.tracer_scanner_sets[sample['class']]['src_dir'], mod, sample_id), coords))

        for mod in self.tracer_scanner_sets[sample['class']]['pairs'][sample['pair_name']]['targets']:
            target_arrays.append(self.load_patch(join(self.tracer_scanner_sets[sample['class']]['src_dir'], mod, sample_id), coords))

        input_tensor = torch.from_numpy(np.stack(input_arrays)).to(torch.float32)
        if len(target_arrays) != 0:
            target_tensor = torch.from_numpy(np.stack(target_arrays)).to(torch.float32)
            return {"input" : input_tensor, "target" : target_tensor}
        else:
            return {"input" : input_tensor}

    def get_coords(self, path):
        coords = np.load(path + '.npy', mmap_mode='r')
        return coords[random.randint(0, coords.shape[0] - 1)].astype(int)

    def load_patch(self, path, coords):
        dparams = {
            'nthreads': 1
        }
        data = blosc2.open(path + '.b2nd', mode='r', dparams=dparams, mmap_mode='r')

        if len(coords) != 0: 
            x_c, y_c, z_c = coords
        else:
            shape_x, shape_y, shape_z = data.shape
            x_c, y_c, z_c = np.random.randint(self.dx, data.shape[0] - 1 - self.dx), np.random.randint(self.dy, data.shape[1] - 1 - self.dy), np.random.randint(self.dz, data.shape[2] - 1 - self.dz)

        
        return data[x_c - self.dx : x_c + self.dx, y_c - self.dy : y_c + self.dy, z_c - self.dz : z_c + self.dz]

@DATASET_REGISTRY.register('blosc2_dataset_transform')
class Blosc2DatasetTransformation(Blosc2Dataset):
    def __init__(self, config_path): 
        super().__init__(config_path)
        meta = yaml.safe_load(open(config_path, 'r'))
        self.transformation_args = meta.get('transformation_args', {})
        self.transformation = TRANS_REGISTRY.get(meta["transformation"])(**self.transformation_args)
        self.on_batch_transformation = meta.get('on_batch_transformation', False)
        self.use_coords = meta.get('use_coords', True)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['id'].split('.')[0] # patient001.b2nd

        if self.use_coords:
            coords = self.get_coords(join(self.tracer_scanner_sets[sample['class']]['src_dir'], self.valid_centers_folder, sample_id))
        else:
            coords = None  # No coordinates used, can be None or a default value


        

        input_arrays, target_arrays = [], [] 

        for mod in self.tracer_scanner_sets[sample['class']]['pairs'][sample['pair_name']]['inputs']:
            input_arrays.append(self.load_patch(join(self.tracer_scanner_sets[sample['class']]['src_dir'], mod, sample_id), coords))

        for mod in self.tracer_scanner_sets[sample['class']]['pairs'][sample['pair_name']]['targets']:
            target_arrays.append(self.load_patch(join(self.tracer_scanner_sets[sample['class']]['src_dir'], mod, sample_id), coords))

        input_tensor = torch.from_numpy(np.stack(input_arrays)).to(torch.float32) # [C, X, Y, Z] with C = 1
        if not self.on_batch_transformation:
            data = self.transformation(input_tensor)
        else: 
            data = input_tensor
        final_data = {"input": input_tensor}
        for key, value in data.items():
            final_data[key] = value
        if len(target_arrays) != 0:
            target_tensor = torch.from_numpy(np.stack(target_arrays)).to(torch.float32)
            final_data["target"] = target_tensor

        return final_data







    

@DATASET_REGISTRY.register('blosc2_regression_test')
class Blosc2DatasetRegressionTest(Dataset):
    def __init__(self, src_dir, meta_path=None):
        if meta_path is not None:
            meta = yaml.safe_load(open(meta_path, 'r'))
        else:
            meta = yaml.safe_load(open(join(src_dir, 'meta.yaml'), 'r'))

        doses = ['10pc', '25pc', '50pc' ]
        
        self.dx, self.dy, self.dz = list((np.array(meta['patch_size']) // 2).astype(int))


        blosc2.set_nthreads(1)

        self.samples = []
        self.tracer_scanner_sets = {}
        src_dir = meta['save_dir']
        for dataset, centers in meta['data'].items():
            for center, tracers in centers.items():
                for tracer, tracer_info in tracers.items():
                    patients = tracer_info['patients']
                    self.tracer_scanner_sets[f'{dataset}/{center}/{tracer}'] = {
                         'src_dir' : join(src_dir, dataset, center, tracer)
                    }
                    for patient in patients:
                        for dose in doses:
                            if os.path.exists(join(self.tracer_scanner_sets[f'{dataset}/{center}/{tracer}']['src_dir'], dose, patient + '.b2nd')):
                                self.samples.append({'id' : patient, 'class' : f'{dataset}/{center}/{tracer}', 'dose' : dose})


        self.n_samples = len(self.samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['id'].split('.')[0]
        dose = sample['dose']

        sc = torch.from_numpy(self.load_patch(join(self.tracer_scanner_sets[sample['class']]['src_dir'], '100pc', sample_id))).unsqueeze(0).to(torch.float32)

        lc = torch.from_numpy(self.load_patch(join(self.tracer_scanner_sets[sample['class']]['src_dir'], dose, sample_id))).unsqueeze(0).to(torch.float32)



        return {'input' : lc, 'target' : sc, "sc" : sc, "lc" : lc, "sample_id" : sample_id, "drf" : dose, "dataset" : sample['class'].split('/')[0], "center" : sample["class"].split('/')[1],"tracer" : sample['class'].split('/')[2]}


    def load_patch(self, path):
        dparams = {
            'nthreads': 1
        }



        data = blosc2.open(path + '.b2nd', mode='r', dparams=dparams, mmap_mode='r')

        center_indices = np.array([s // 2 for s in data.shape])
        start_indices = np.maximum(0, center_indices - np.array([self.dx, self.dy, self.dz]))
        end_indices = np.minimum(data.shape, center_indices + np.array([self.dx, self.dy, self.dz]))

        return data[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]