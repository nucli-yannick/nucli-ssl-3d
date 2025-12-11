from __future__ import annotations

from nnunet_utils import resample_data_or_seg_to_spacing

import os
from os.path import join
import shutil

import numpy as np
import nibabel as nib

from tqdm import tqdm

def resample_annotations(base_dir, target_dir, target_spacing):
    for center in os.listdir(base_dir):
        center_path = join(base_dir, center)
        if not os.path.isdir(center_path):
            continue
        
        for tracer in os.listdir(center_path):
            annotations_path = join(center_path, tracer, 'annotations')
            if not os.path.exists(annotations_path):
                continue
            os.makedirs(join(target_dir, center, tracer, 'annotations'), exist_ok=True)

            print(f"Processing center: {center}, tracer: {tracer}")
            
            for file_name in tqdm(os.listdir(annotations_path)):
                if file_name.endswith('.nii.gz'):
                    file_path = join(annotations_path, file_name)
                    
                    # Load the image
                    img = nib.load(file_path)
                    data = np.expand_dims(img.get_fdata(), 0)
                    
                    resampled_segmentation = resample_data_or_seg_to_spacing(data, img.header["pixdim"][1:4], target_spacing).astype(np.uint16)
                    
                    
                    new_file_name = f'{file_name.split(".")[0]}.npy'
                    new_file_path = join(target_dir, center, tracer, 'annotations', new_file_name)

                    np.save(new_file_path, resampled_segmentation)
                    shutil.copy(join(annotations_path, file_name.split('.')[0] + '.yaml'), join(target_dir, center, tracer, 'annotations', file_name.split('.')[0] + '.yaml'))


if __name__ == "__main__":
    base_dir = '/home/vicde/data/nuclarity_test'
    target_dir = '/home/vicde/blosc2_nuclarity_test/nuclarity_test'
    target_spacing = (2.734375, 2.734375, 3.27)

    resample_annotations(base_dir, target_dir, target_spacing)