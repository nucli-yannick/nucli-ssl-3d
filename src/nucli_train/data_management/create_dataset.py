from __future__ import annotations

import numpy as np

import os
from os.path import sep, join, basename

import random

import warnings

import blosc2
from blosc2 import Filter, Codec

import nibabel as nib

from .nnunet_utils import comp_blosc2_params, resample_data_or_seg_to_spacing
from .resampling import resample

import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def save_blosc(path : str, data : np.ndarray, patch_shape : np.ndarray):
    blosc2.set_nthreads(1)
    cparams = {'codec': Codec.ZSTD, 'clevel': 8}
    blocks, chunks = comp_blosc2_params(data.shape, tuple(patch_shape), bytes_per_pixel=data.itemsize)
    blosc2.asarray(np.ascontiguousarray(data), urlpath=path + '.b2nd', chunks=chunks,
                       blocks=blocks, cparams=cparams)

    num_chunks = int(np.prod(chunks))
    return {
        "chunks": num_chunks
    }



def process_patient_blosc(patient, src_dir, save_dir, dataset, center, tracer, modalities, segmentations, target_spacing, patch_shape):
    save_dir = join(save_dir, dataset, center, tracer)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    target_shape = None

    for modality in modalities:
        if os.path.exists(join(save_dir, modality, patient.split('.')[0] + '.b2nd')):
            continue
        if not os.path.exists(join(src_dir, dataset, center, tracer, modality, patient + '.nii.gz')):
            continue
        if not os.path.exists(join(save_dir, modality)):
            os.makedirs(join(save_dir, modality), exist_ok=True)
        volume = nib.load(join(src_dir, dataset, center, tracer, modality, patient + '.nii.gz'))
        if target_spacing:
            mod_arr = resample(volume.get_fdata(), tuple(volume.header['pixdim'][1:4]), tuple(target_spacing))
        else:
            mod_arr = volume.get_fdata().astype(np.float16)
        save_blosc(join(save_dir, modality, patient.split('.')[0]), mod_arr.astype(np.float16), np.array(patch_shape))

    for segmentation in segmentations:
        if os.path.exists(join(save_dir, segmentation, patient.split('.')[0] + '.b2nd')):
            continue
        if not os.path.exists(join(save_dir, segmentation)):
            os.makedirs(join(save_dir, segmentation), exist_ok=True)
        seg_volume = nib.load(join(src_dir, dataset, center, tracer, segmentation, patient + '.nii.gz'))
        if target_spacing:
            seg_arr = resample_data_or_seg_to_spacing(seg_volume.get_fdata().astype(np.uint16), seg_volume.header['pixdim'][1:4], target_spacing, is_seg=True)
        else:
            seg_arr = seg_volume.get_fdata().astype(np.uint16)
        save_blosc(join(save_dir, segmentation, patient.split('.')[0]), seg_arr.astype(np.float16), np.array(patch_shape))
        


def _process_blosc_case(args):
    """
    Worker wrapper: unpack args tuple and call the real processing function.
    """
    (patient, src_dir, save_dir,
     dataset, center, tracer,
     modalities, segmentations,
     tracer_target_spacing, shape) = args

    process_patient_blosc(
        patient,
        src_dir,
        save_dir,
        dataset,
        center,
        tracer,
        modalities, segmentations,
        tracer_target_spacing,
        shape
    )



def save_blosc_data_parallel(meta, num_workers=None):

    shape = meta['patch_size']
    target_spacing = meta.get('target_spacing', None)
    src_dir, save_dir = meta['src_dir'], meta['save_dir']

    if target_spacing:
        assert all(isinstance(x, (int, float)) for x in target_spacing), \
            "Target spacing must be a list of numbers."
        assert all(x > 0 for x in target_spacing), \
            "Target spacing values must be positive."

    # 1) Build task list
    tasks = []
    for split in ['train', 'val']:
        if split not in meta:
            continue

        for dataset, ds_cfg in meta[split].items():
            assert isinstance(ds_cfg, dict)
            for center, ctr_cfg in ds_cfg.items():
                assert isinstance(ctr_cfg, dict)
                for tracer, tracer_cfg in ctr_cfg.items():
                    
                    
                    patients = tracer_cfg.get('patients', [])
                    if not patients:
                        continue
                    modalities = tracer_cfg.get('modalities')

                    segmentations = tracer_cfg.get('segmentations', [])

                    tracer_target_spacing = None
                    if target_spacing or tracer_cfg.get('target_spacing', None):
                        tracer_target_spacing = (
                            target_spacing
                            if target_spacing
                            else tracer_cfg['target_spacing']
                        )

                    if not isinstance(patients, str):
                        for patient in patients:

                            tasks.append((
                                patient, src_dir, save_dir,
                                dataset, center, tracer,
                                modalities, segmentations,
                                tracer_target_spacing, shape
                            ))
                    else:
                        assert patients == 'all'
                        for modality in modalities:
                            for patient in os.listdir(join(src_dir, dataset, center, tracer, modality)):
                                if not patient.endswith('.nii.gz'):
                                    continue
                                tasks.append((
                                    patient, src_dir, save_dir,
                                    dataset, center, tracer,
                                    [modality], [], 
                                    tracer_target_spacing, shape
                                ))
                        for segmentation in segmentations:
                            for patient in os.listdir(join(src_dir, dataset, center, tracer, segmentation)):
                                if not patient.endswith('.nii.gz'):
                                    continue
                                tasks.append((
                                    patient, src_dir, save_dir,
                                    dataset, center, tracer,
                                    [], [segmentation],
                                    tracer_target_spacing, shape
                                ))

    # 2. Execute
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_blosc_case, t) for t in tasks]
        with tqdm(total=total, desc="Processing patients") as pbar:
            for fut in as_completed(futures):
                fut.result()  # propagate exceptions
                pbar.update(1)

    yaml.dump(meta, open(join(save_dir, 'meta.yaml'), 'w'))



def process_tracer(tracer_cfg, tracer_dir, train_patients=None, is_npy=False):
    if 'patients' in tracer_cfg.keys():
        assert isinstance(pairs, dict)
        all_modalities = tracer_cfg.get('modalities', [])
        all_segmentations = tracer_cfg.get('segmentations', [])


        if is_npy:
            all_modalities += 'coords'

        patients = tracer_cfg.get('patients', [])
        target_spacing = tracer_cfg.get('target_spacing', None)
        assert isinstance(target_spacing, (list, type(None)))
        if patients == [] or patients == 0:
            warnings.warn(f"{join(*basename(tracer_dir.split(sep)[-3:]))} has no inputs or patients defined. Skipping.")
            return {'inputs' : [], 'targets' : [], 'patients' : []}
        shared_cases = None
        for modality in all_modalities + all_segmentations:
            if not os.path.isdir(join(tracer_dir, modality)):
                raise FileNotFoundError(f"Modality {modality} not found in {tracer_dir}")

            if shared_cases is None:
                shared_cases = set([file_id.split('.')[0] for file_id in os.listdir(join(tracer_dir, modality))])
                if train_patients: # only passed when creating validation dataset
                    shared_cases = shared_cases - set(train_patients)
            else:
                shared_cases = shared_cases.intersection(set([file_id.split('.')[0] for file_id in os.listdir(join(tracer_dir, modality))]))

        shared_cases = list(shared_cases)

        if isinstance(patients, int):
            assert len(shared_cases) >= patients, f"Not enough cases in {join(*basename(tracer_dir.split(sep)[-3:]))} to create {patients} patients."
            patients = random.sample(shared_cases, patients)
        elif isinstance(patients, list):
            assert (p in shared_cases for p in patients), f"Some patients {patients} not found in {join(*basename(tracer_dir.split(sep)[-3:]))}."
        elif isinstance(patients, str):
            assert patients == 'all', f"Expected 'patients' to be 'all', a list of patient IDs, or an integer, got {patients}."

        if is_npy:
            all_modalities.remove('coords')

        return {'modalities' : all_modalities , 'segmentations' : all_segmentations ,'patients' : patients,  'target_spacing' : target_spacing}
        


        
    else:
        raise NotImplementedError()
        


        

def process_yaml(cfg):
    """
    Process the YAML configuration file to 
                             (i) Ensure all given data is present in src_dir
                             (ii) Create patient splits where necessary
                             (iii) Create dataset metadata and file pairs required by savers
    Returns:
        all_pairs (list): List of tuples containing file paths for each sample.
        meta (dict): Metadata dictionary containing dataset information.
    """
    splits = ['train', 'val']

    meta = {}
    meta['storage-type'] = cfg.get('storage-type')
    assert meta['storage-type'] in ['blosc2', 'npy'], f"{meta['storage-type']} is not implemented"
    meta['src_dir'] = cfg.get('src_dir')
    meta['save_dir'] = cfg.get('save_dir')
    meta['target_spacing'] = cfg.get('target_spacing')
    meta['patch_size'] = cfg.get('patch_size')

    for split in splits:
        if split not in cfg.keys():
            continue
        meta[split] = {}
        assert isinstance(cfg[split], dict)
        for dataset in cfg[split]:
            print(dataset)
            meta[split][dataset] = {}
            assert isinstance(cfg[split][dataset], dict)
            for center in cfg[split][dataset]:
                print(center)
                meta[split][dataset][center] = {}
                assert isinstance(cfg[split][dataset][center], dict)
                for tracer in cfg[split][dataset][center]:
                    print(tracer)
                    assert isinstance(cfg[split][dataset][center][tracer], dict)
                    tracer_data = process_tracer(cfg[split][dataset][center][tracer], join(cfg['src_dir'], dataset, center, tracer), train_patients=None if split == 'train' else meta['train'][dataset][tracer].get('patients', None), is_npy = meta['storage-type']=='npy')
                    meta[split][dataset][center][tracer] = tracer_data
    return meta



def create_dataset(yaml_path : str):
    # 1. ensure correctness of yaml
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    if not yaml_path.endswith('.yaml'):
        raise ValueError(f"Expected a .yaml file, got {yaml_path}")
    
    data_cfg = yaml.safe_load(open(yaml_path, 'r'))

    data_keys = data_cfg.keys()

    if 'val' not in data_keys and 'train' not in data_keys:
        raise ValueError("YAML must contain at least one of 'train' or 'val' keys.")

    assert ('patch_size' in data_keys) and ('src_dir' in data_keys and 'save_dir' in data_keys)


    meta = process_yaml(data_cfg)

    if meta['storage-type'] == 'npy':
        raise NotImplementedError("Numpy storage is not implemented yet. Please use blosc2 storage.")
    elif meta['storage-type'] == 'blosc2':
        save_blosc_data_parallel(meta, 4)
    else:
        raise ValueError(f"Unknown storage type: {meta['storage-type']}")
    



    
