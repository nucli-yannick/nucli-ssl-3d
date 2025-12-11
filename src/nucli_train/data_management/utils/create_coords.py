from __future__ import annotations

import os
from os.path import join

import yaml

import nibabel as nib
import numpy as np

from numpy.lib.format import open_memmap

import blosc2

from tqdm import tqdm

def get_bbox(seg, label):
    """Compute the bounding box of a given label in a 3D array."""
    inds = np.argwhere(seg == label)
    zmin, ymin, xmin = inds.min(axis=0)
    zmax, ymax, xmax = inds.max(axis=0)
    return {
        'z': [int(zmin), int(zmax)],
        'y': [int(ymin), int(ymax)],
        'x': [int(xmin), int(xmax)]
    }

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def extract_patch(image, center, size):
    """Extract a cubic patch from image centered at center with given size."""
    half = size // 2
    zc, yc, xc = center
    Z, Y, X = image.shape
    z1 = clamp(zc - half, 0, Z - size)
    y1 = clamp(yc - half, 0, Y - size)
    x1 = clamp(xc - half, 0, X - size)
    z2 = z1 + size
    y2 = y1 + size
    x2 = x1 + size
    return image[z1:z2, y1:y2, x1:x2]



def open_segmentation(case_base_path, case_name):
    base_name = case_name.split('.')[0]
    if os.path.exists(join(case_base_path, 'annotations', base_name + '.nii.gz')):
        img = nib.load(join(case_base_path, 'annotations', base_name + '.nii.gz')).get_fdata().astype(np.uint16)
    elif os.path.exists(join(case_base_path, 'annotations', base_name + '.nii')):
        img = nib.load(join(case_base_path, 'annotations', base_name + '.nii')).get_fdata().astype(np.uint16)
    elif os.path.exists(join(case_base_path, 'annotations', base_name + '.npy')):
        img = np.load(join(case_base_path, 'annotations', base_name + '.npy'))
    else:
        raise FileNotFoundError(f"Segmentation file for case {case_name} not found in {case_base_path}/annotations")
    return img.squeeze()


def open_image(case_base_path, case_name, dose):
    base_name = case_name.split('.')[0]
    if os.path.exists(join(case_base_path, dose, base_name + '.nii.gz')):
        img = nib.load(join(case_base_path, dose, base_name + '.nii.gz')).dataobj
    elif os.path.exists(join(case_base_path, dose, base_name + '.nii')):
        img = nib.load(join(case_base_path, dose, base_name + '.nii')).dataobj
    elif os.path.exists(join(case_base_path, dose, base_name + '.npy')):
        img = open_memmap(join(case_base_path, dose, base_name + '.npy'))
    elif os.path.exists(join(case_base_path, dose, base_name + '.b2nd')):
        dparams = {
            'nthreads': 1
        }
        img = blosc2.open(join(case_base_path, dose, base_name + '.b2nd'), mode='r', dparams=dparams, mmap_mode='r')
    else:
        return 'Not found'
    return img

def process_cases(cases_base_path):
    # annotations: segmentation 
    # annotations: yaml
    # images: lc
    # images: sc


    # 1.: Load segmentation and yaml's
    # 2: for each lesion: get boxes
    # 3: use these to extract from lc and sc
    # 4: save as npy

    os.makedirs(join(cases_base_path, 'roi_patches'), exist_ok=True)

    for case_name in tqdm(os.listdir(join(cases_base_path, 'annotations'))):
        if not case_name.endswith('.yaml'):
            continue
        

        

        segmentation = open_segmentation(cases_base_path, case_name)
        roi_ids = list(yaml.safe_load(open(join(cases_base_path, 'annotations', case_name.split('.')[0] + '.yaml'), 'r')).keys())

        patches = {'25pc': [], '50pc': [], '100pc': [], 'segmentation': []}

        all_present = {'25pc': True, '50pc': True, '100pc': True}

        for roi_id in sorted(roi_ids):
            bbox = get_bbox(segmentation, int(roi_id))
            patches['segmentation'].append(extract_patch(segmentation,
                                                        center=((bbox['z'][0] + bbox['z'][1]) // 2, 
                                                                (bbox['y'][0] + bbox['y'][1]) // 2, 
                                                                (bbox['x'][0] + bbox['x'][1]) // 2), 
                                                        size=64))
            for dose in ['25pc', '50pc', '100pc']:
                dose_image = open_image(cases_base_path, case_name, dose)

                if dose_image == 'Not found':
                    all_present[dose] = False
                    continue
                
                # Extract patch from the dose image
                patch = extract_patch(dose_image, 
                                      center=((bbox['z'][0] + bbox['z'][1]) // 2, 
                                              (bbox['y'][0] + bbox['y'][1]) // 2, 
                                              (bbox['x'][0] + bbox['x'][1]) // 2), 
                                      size=64)
                print(patch.shape)
                patches[dose].append(patch)
        np.save(join(cases_base_path, 'roi_patches', case_name.split('.')[0] + '_segmentation.npy'), np.stack(patches['segmentation']).astype(np.float16))
        for dose in ['25pc', '50pc', '100pc']:
            if not all_present[dose]:
                print(f"Skipping {dose} for case {case_name} due to missing data.")
                continue
            np.save(join(cases_base_path, 'roi_patches', case_name.split('.')[0] + f'_{dose}.npy'), np.stack(patches[dose]).astype(np.float16))


if __name__ == "__main__":
    base_path = '/home/vicde/blosc2_nuclarity_test/nuclarity_test'
    for center in os.listdir(base_path):
        center_path = join(base_path, center)
        if not os.path.isdir(center_path):
            continue
        for tracer in os.listdir(center_path):
            tracer_path = join(center_path, tracer)
            if not os.path.isdir(tracer_path):
                continue
            print(f"Processing center: {center}, tracer: {tracer}")
            if not os.path.exists(join(tracer_path, 'annotations')):
                print(f"No annotations found for {center}/{tracer}. Skipping.")
                continue
            
            process_cases(tracer_path)