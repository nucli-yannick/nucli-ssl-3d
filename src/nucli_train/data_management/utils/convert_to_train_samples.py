from __future__ import annotations

import os
import argparse
import numpy as np
import nibabel as nib
from numpy.lib.format import open_memmap

from os.path import join
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 3D patches into a memory-mapped .npy when RAM is insufficient."
    )
    parser.add_argument("--main_dir", "-i", type=str, required=True,
                        help="Directory of NIfTI standard-count volumes.")
    parser.add_argument("--target_dir", "-ta", type=str, required=True,
                        help="Directory of .npy center‐coordinate files.")

    parser.add_argument("--dz", type=int, required=True, help="Half‐size along z.")
    parser.add_argument("--dy", type=int, required=True, help="Half‐size along y.")
    parser.add_argument("--dx", type=int, required=True, help="Half‐size along x.")

    parser.add_argument("--split_yaml", "-sy", type=str, required=True,
                        help="Path to yaml with train-val-test splits.")
    parser.add_argument("--split", "-s", type=str, required=True,
                        help="Split to process")
    parser.add_argument("--center", "-ce", type=str, required=True,
                        help="Directory of .npy center‐coordinate files.")
    parser.add_argument("--tracer", "-tr", type=str, required=True,
                        help="Directory of .npy center‐coordinate files.")
    parser.add_argument("--dose", "-do", type=str, required=True,
                        help="Directory of .npy center‐coordinate files.")
    return parser.parse_args()

def strip_nii_extension(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    elif filename.endswith(".nii"):
        return filename[:-4]
    else:
        raise ValueError(f"‘{filename}’ is not a .nii or .nii.gz")

def main():
    args = parse_args()
    main_dir, target_dir = args.main_dir, args.target_dir
    dz, dy, dx = args.dz, args.dy, args.dx
    split_yaml, split = args.split_yaml, args.split
    center, tracer, dose = args.center, args.tracer, args.dose

    centers_dir = join(main_dir, 'coords')
    sc_dir = join(main_dir, '100pc')
    lc_dir = join(main_dir, dose)


    # 1) List and sort all NIfTI files
    nifti_files = yaml.safe_load(open(split_yaml, 'r'))[split][center][tracer][dose]
    nifti_files.sort()
    if not nifti_files:
        raise RuntimeError(f"No NIfTI files in {nifti_dir!r}")

    # 2) FIRST PASS: count total patches
    total_patches = 0
    for nifti_name in nifti_files:
        base = strip_nii_extension(nifti_name)
        centers_path = os.path.join(centers_dir, base + ".npy")
        if not os.path.isfile(centers_path):
            raise FileNotFoundError(f"Missing centers file: {centers_path}")
        centers = open_memmap(centers_path)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError(f"{centers_path} must have shape (N_p, 3)")
        total_patches += centers.shape[0]
    if total_patches == 0:
        raise RuntimeError("No centers → no patches to extract.")
    print(f"Total patches to extract: {total_patches}")

    # 3) Create memmap for output
    patch_shape = (2, 2*dz, 2*dy, 2*dx)
    dtype = np.float32
    os.makedirs(join(target_dir, center, tracer, dose), exist_ok=True)
    all_patches_memmap = open_memmap(
        join(target_dir, center, tracer, dose, split + '.npy'),
        mode="w+",
        dtype=dtype,
        shape=(total_patches, *patch_shape)
    )
    print(f"Created memory-mapped array at '{join(target_dir, center, tracer, dose)}', shape = {all_patches_memmap.shape}")

    # 4) SECOND PASS: extract & write each patch
    write_index = 0
    for nifti_name in nifti_files:
        base = strip_nii_extension(nifti_name)
        lc_path, sc_path= join(lc_dir, nifti_name), join(sc_dir, nifti_name)
        centers_path = os.path.join(centers_dir, base + ".npy")

        volume_sc = nib.load(sc_path).get_fdata().astype(dtype)
        volume_lc = nib.load(lc_path).get_fdata().astype(dtype)
        centers = np.load(centers_path).astype(int)
        Z, Y, X = volume_sc.shape

        print(f"  → Processing '{nifti_name}' with {len(centers)} centers...")
        for (z_c, y_c, x_c) in centers:
            # bounds check
            if (z_c - dz < 0 or z_c + dz > Z or
                y_c - dy < 0 or y_c + dy > Y or
                x_c - dx < 0 or x_c + dx > X):
                raise ValueError(
                    f"Center {(z_c, y_c, x_c)} out of bounds for volume shape {volume.shape}"
                )
            patch_sc = volume_sc[
                z_c - dz : z_c + dz,
                y_c - dy : y_c + dy,
                x_c - dx : x_c + dx
            ]
            patch_lc = volume_lc[
                z_c - dz : z_c + dz,
                y_c - dy : y_c + dy,
                x_c - dx : x_c + dx
            ] 

            patch = np.stack([patch_lc, patch_sc])  # (1, 2*dz, 2*dy, 2*dx)
            all_patches_memmap[write_index, ...] = patch
            write_index += 1

    if write_index != total_patches:
        raise RuntimeError(f"Written {write_index} ≠ expected {total_patches}")
    print("All patches written successfully. Flushing to disk...")
    all_patches_memmap.flush()
    print("Done.")

if __name__ == "__main__":
    main()
