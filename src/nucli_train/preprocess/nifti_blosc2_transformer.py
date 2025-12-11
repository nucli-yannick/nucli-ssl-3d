from __future__ import annotations

"""
The goal of this file is to get a .b2nd fil form a nifti file
"""

import numpy as np
import nibabel as nib
import blosc2
import os

from typing import Tuple
import math
from copy import deepcopy



class Blosc2Compressor: 

    def __init__(self): 
        pass

    def compress_nifti(self, input_path: str, output_path: str, patch_size: Tuple[int, int, int]):
        """
        Compress a NIfTI file to .b2nd format
        """
        # Load the NIfTI file
        img = nib.load(input_path)
        volume = img.get_fdata(dtype=np.float32)
        print(f"Location of the output file: {output_path}")

        # Save the compressed file
        self.save_case(volume, patch_size, output_path)

    def save_case(self, volume, patch_size: Tuple[int, int, int], output_path: str, clevel = 8): 

        cparams = {
            "codec": blosc2.Codec.ZSTD,
            "clevel": clevel
        }

        blosc2.set_nthreads(os.cpu_count()) 

        block_size, chunk_size = self.comp_blosc2(volume.shape, patch_size, bytes_per_pixel=volume.itemsize)

         
        
        print(f"Volume shape: {volume.shape}")
        print(f"Volume type: {type(volume)}")
        print(f"Block size: {block_size}, Chunk size: {chunk_size}")
        print(f"Output path: {output_path}")

        blosc2.asarray(
            np.ascontiguousarray(volume),
            urlpath=output_path,
            chunks=chunk_size,
            blocks=block_size,
            cparams=cparams,
            # mmap_mode="w+", # memory-mapped mode for writing
        )
    
   
    def comp_blosc2(self,
        image_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        bytes_per_pixel: int = 8,
        l1_cache_size_per_core_in_bytes: int = 32768,
        l3_cache_size_per_core_in_bytes: int = 1441792,
        safety_factor: float = 0.8
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Compute block and chunk sizes for Blosc2 compression optimized for no-channel 3D data.

        Assumes:
        - image_size is (x, y, z)
        - patch_size is (x, y, z)
        - Each read operation is single-threaded
        - L1-sized blocks, L3-sized chunks
        """
        patch_size = np.array(patch_size)
        block_size = np.array([2 ** max(0, math.ceil(math.log2(p))) for p in patch_size])

        # Shrink block size until it fits in L1 cache
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > l1_cache_size_per_core_in_bytes * safety_factor:
            # Pick the axis with largest ratio between block and patch
            axis_order = np.argsort(block_size / patch_size)[::-1]
            idx = 0
            while idx < 3:
                ax = axis_order[idx]
                if block_size[ax] > 1:
                    block_size[ax] = 2 ** max(0, math.floor(math.log2(block_size[ax] - 1)))
                    block_size[ax] = min(block_size[ax], image_size[ax])
                    break
                idx += 1
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

        block_size = np.minimum(image_size, block_size)

        # Expand to L3-sized chunk while not exceeding image or patch*1.5
        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < l3_cache_size_per_core_in_bytes * safety_factor:
            if np.all(chunk_size == image_size):
                break

            axis_order = np.argsort(chunk_size / block_size)
            idx = 0
            while idx < 3:
                ax = axis_order[idx]
                if chunk_size[ax] < image_size[ax] and patch_size[ax] > 1:
                    proposed = chunk_size[ax] + block_size[ax]
                    if proposed <= image_size[ax]:
                        chunk_size[ax] = proposed
                    break
                idx += 1

            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
            if np.mean(chunk_size / patch_size) > 1.5:
                chunk_size[ax] -= block_size[ax]
                break

        chunk_size = np.minimum(image_size, chunk_size)
        return tuple(block_size), tuple(chunk_size) 