from __future__ import annotations

from nucli_train.models.builders import build_image_translation_model
import nibabel as nib
import numpy as np

import os
from os.path import join

import yaml

reader_dir = '/home/vicde/data/reader/'

save_dir = '/home/vicde/results_unet/'

import pandas as pd

if __name__ == '__main__':
    model_cfg = yaml.safe_load(open('/home/vicde/nucli-train/configs/nuclarity_model.yaml'))

    model = build_image_translation_model(model_cfg['model'])

    model._load_checkpoint('/home/vicde/nucli-train/experiments/nuclarity_data/basic-test', 2600)
    for center in os.listdir(reader_dir):
        if center not in ['azg', 'vub', 'uzg']:
            continue
        for tracer in os.listdir(join(reader_dir, center)):
            if center == 'vub' and tracer == 'psma':
                continue
            for patient in os.listdir(join(reader_dir, center, tracer, 'annotations')):
                if '.txt' not in patient:
                    continue
                if os.path.exists(join(save_dir, 'reader', center, tracer, 'prediction', patient.split('.')[0] + '.npy')):
                    print(f"Skipping {patient} as it already exists.")
                    continue

                volume = nib.load(join(reader_dir, center, tracer,  '50pc', patient.split('.')[0] + '.nii.gz'))

                print(volume.shape)

                results = model.infer_scan(volume.get_fdata(), volume.header['pixdim'][1:4])
                for k, v in results.items():
                    if k == 'w_c0':
                        row[k] = v
                    else:
                        if not os.path.exists(join(save_dir, 'reader', center, tracer, k)):
                            os.makedirs(join(save_dir, 'reader', center, tracer, k))
                        np.save(join(save_dir, 'reader', center, tracer, k, patient.split('.')[0] + '.npy'), v)

                