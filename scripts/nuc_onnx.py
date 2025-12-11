from __future__ import annotations

import nibabel as nib
import numpy as np

from .src.handler import ONNXModelHandler

import os
from os.path import join

reader_dir = '/home/vicde/data/reader/'

save_dir = '/home/vicde/results_onnx/'

import pandas as pd

if __name__ == '__main__':
    model = ONNXModelHandler('./models/base/low_high.onnx')
    for center in os.listdir(reader_dir):
        if center not in ['azg', 'vub', 'uzg']:
            continue
        for tracer in os.listdir(join(reader_dir, center)):
            rows = []
            if center == 'vub' and tracer == 'psma':
                continue
            for patient in os.listdir(join(reader_dir, center, tracer, 'annotations')):
                if '.txt' not in patient:
                    continue

                row = {}
                volume = nib.load(join(reader_dir, center, tracer,  '50pc', patient.split('.')[0] + '.nii.gz'))

                print(volume.shape)

                results = model.infer_scan(volume.get_fdata(), volume.header['pixdim'][1:4], store_seperate_channels=True, resample_seperate_channels=True)
                for k, v in results.items():
                    if k == 'w_c0':
                        row[k] = v
                    else:
                        if not os.path.exists(join(save_dir, 'reader', center, tracer, k)):
                            os.makedirs(join(save_dir, 'reader', center, tracer, k))
                        np.save(join(save_dir, 'reader', center, tracer, k, patient.split('.')[0] + '.npy'), v)
                row['patient'] = patient.split('.')[0]
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_csv(join(save_dir, 'reader', center, tracer, 'weights.csv'), index=False)
                