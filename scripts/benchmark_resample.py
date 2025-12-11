from __future__ import annotations

import time
import numpy as np
from skimage.transform import resize as sk_resize
import torch
import torch.nn.functional as F
import dask.array as da

import nibabel as nib



def compute_target_shape(vol: np.ndarray, orig_spacing: tuple, target_spacing: tuple) -> tuple:
    """
    Compute target array shape from original volume shape and spacings.
    """
    shape = vol.shape[-len(orig_spacing):]
    return tuple(
        int(round(dim * sp / tsp))
        for dim, sp, tsp in zip(shape, orig_spacing, target_spacing)
    )


def resize_skim(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize using scikit-image (CPU)."""
    return sk_resize(vol, target_shape, order=1, mode='reflect', anti_aliasing=True)


def prepare_tensor(vol: np.ndarray, device: str):
    """
    Ensure volume is torch tensor of shape (N, C, *spatial_dims)."""
    arr = vol
    # Add channel dim if missing
    if arr.ndim in (2, 3):  # (H,W) or (D,H,W)
        arr = arr[np.newaxis, ...]
    # Add batch dim
    if arr.ndim in (3, 4):
        arr = arr[np.newaxis, ...]
    tensor = torch.from_numpy(arr).to(device)
    return tensor



def resize_cucim(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize using cucim (GPU)."""
    import cucim
    from cucim.skimage.transform import resize as cucim_resize
    return cucim_resize(vol, target_shape, order=1, mode='reflect', anti_aliasing=True)


def resize_cucim_blocked(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize using cucim with blocking (GPU)."""
    import cucim
    from cucim.skimage.transform import resize as cucim_resize
    # Block size can be adjusted based on GPU memory
    block_size = (64, 64, 64)
    return cucim_resize(vol, target_shape, order=1, mode='reflect', anti_aliasing=True, block_size=block_size)

def resize_skim_blocked(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize using skimage with blocking (CPU)."""
    # Block size can be adjusted based on CPU memory
    block_size = (64, 64, 64)
    return sk_resize(vol, target_shape, order=1, mode='reflect', anti_aliasing=True, block_size=block_size)

def test_cases():
    # 1. CHUL --> AZG
    # 2. AZG --> CHUL
    # 3. OLVZ --> CHUL
    # 4. OLVZ --> AZG
    # 5. AZG --> OLVZ
    # 6. CHUL --> OLVZ
    # 7. Quadra --> CHUL
    # 8. Quadra --> AZG
    # 9. Quadra --> OLVZ
    # 10. CHUL --> Quadra
    # 11. AZG --> Quadra


    # Methods / Scenarios:
    # Entire volume: One-go
    # Entire volume: Blocking
    # Entire volume: Multi-threaded

def benchmark(volumes_info: list):
    """
    Benchmark resize methods for volumes with individual target spacings.

    volumes_info: list of dicts with keys:
      - 'vol': np.ndarray
      - 'orig_spacing'
      - 'target_spacing'
    """
    methods = {
        'skimage': resize_skimage,
        'torch_gpu': lambda vol, shape: resize_torch(vol, shape, device='cuda'),
        'torch_cpu': lambda vol, shape: resize_torch(vol, shape, device='cpu'),
        'dask': resize_dask
    }
    # Compute shapes
    for info in volumes_info:
        info['target_shape'] = compute_target_shape(
            info['vol'], info['orig_spacing'], info['target_spacing'])

    # Header
    header = 'Method' + ''.join([
        f"\tVol{i+1} ({info['vol'].shape} --> {info['target_shape']})"
        for i, info in enumerate(volumes_info)
    ])
    print(header)

    # Benchmarks
    for name, func in methods.items():
        times = []
        for info in volumes_info:
            t0 = time.perf_counter()
            _ = func(info['vol'], info['target_shape'])
            t1 = time.perf_counter()
            times.append(t1 - t0)
        print(name + ''.join([f"\t{t:.4f}s" for t in times]))



if __name__ == '__main__':
    volumes_info = [
         {
             'vol': nib.load('/home/vicde/data/nuclarity_train/chul/fdg/25pc/chul003.nii.gz').get_fdata(),
             'orig_spacing': nib.load('/home/vicde/data/nuclarity_train/chul/fdg/25pc/chul003.nii.gz').header['pixdim'][1:4],
             'target_spacing':  nib.load('/home/vicde/data/nuclarity_train/azg/fdg/25pc/azg001.nii.gz').header['pixdim'][1:4]
         },
         {
             'vol': nib.load('/home/vicde/data/phantoms/olvz/high/10pc.nii.gz').get_fdata(),
             'orig_spacing': nib.load('/home/vicde/data/nuclarity_train/chul/fdg/25pc/chul003.nii.gz').header['pixdim'][1:4],
             'target_spacing':  nib.load('/home/vicde/data/nuclarity_train/azg/fdg/25pc/azg001.nii.gz').header['pixdim'][1:4]
         }
     ]
    benchmark(volumes_info)
    pass
