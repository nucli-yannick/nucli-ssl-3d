from __future__ import annotations

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure, binary_dilation, distance_transform_edt

# 3×3×3 connectivity for 3D; for 2D use generate_binary_structure(2, 1)

from scipy import ndimage
from scipy.signal import fftconvolve

from scipy.ndimage import uniform_filter, binary_closing, binary_opening, generate_binary_structure

def compute_valid_centers_3d(volume, suv_thresh, patch_size, min_frac=0.2):
    """
    volume: 3D numpy array of PET SUV values (shape: Z×Y×X)
    suv_thresh: e.g. 0.2  (SUV threshold to define initial mask)
    patch_size: tuple (d, h, w), e.g. (32, 64, 64)
    min_frac: minimum fraction of voxels in mask required inside each patch
    """
    # 1) Build and clean the SUV > suv_thresh mask
    mask0 = (volume > suv_thresh)
    struct = generate_binary_structure(rank=3, connectivity=1)
    mask_clean = binary_closing(mask0, structure=struct, iterations=1)
    mask_clean = binary_opening(mask_clean, structure=struct, iterations=1)

    # 2) (Optional) Dilate slightly to include low-uptake neighborhoods
    mask_dilated = mask_clean.copy()
    if suv_thresh > 0.5:
        # e.g. dilate by 2 voxels if you want to capture faint uptake around mask
        mask_dilated = binary_opening(mask_clean, structure=struct, iterations=1)
        mask_dilated = uniform_filter(mask_dilated.astype(np.float32),
                                      size=(3,3,3), mode='constant', cval=0) > 0.0
        # a single small dilate–erode pass to “soften” boundaries
    
    # 3) Compute local‐mean via uniform_filter
    d, h, w = patch_size
    mask_float = mask_dilated.astype(np.float32)
    local_mean = uniform_filter(mask_float, size=(d, h, w), mode='constant', cval=0.0)

    # 4) Convert local_mean to local_sum
    window_volume = float(d * h * w)
    local_sum = local_mean * window_volume  # still float32

    # 5) Threshold to get valid centers: at least min_frac fraction of patch in mask
    min_voxels = min_frac * window_volume
    valid_centers = (local_sum >= min_voxels)

    # 6) Exclude boundary‐near centers (so full patch stays inside volume)
    dz, dy, dx = d // 2, h // 2, w // 2
    valid_centers[:dz, :, :] = False
    valid_centers[-dz:, :, :] = False
    valid_centers[:, :dy, :] = False
    valid_centers[:, -dy:, :] = False
    valid_centers[:, :, :dx] = False
    valid_centers[:, :, -dx:] = False

    return valid_centers  # boolean 3D array, True = valid crop center


def prune_nonoverlapping_centers(coords, patch_size, overlap_frac=0.0, max_patches=None, seed=0):
    """
    Greedily select centers from `coords` so that their patches overlap by at most `overlap_frac`.
    - coords: (N,3) array of (z,y,x) valid centers
    - patch_size: (d,h,w)
    - overlap_frac: fraction [0..1] of side-length allowed to overlap. 0.0→no overlap; 0.5→50% allowed.
    - max_patches: if not None, stop once len(selected)==max_patches
    - seed: for reproducible shuffling

    Returns:
      selected_centers: list of (z,y,x) centers, pruned for overlap.
    """
    rng = np.random.default_rng(seed)
    coords = coords.copy()
    rng.shuffle(coords)

    d, h, w = patch_size
    # Compute “exclusion radius” in each axis: we want |Δz| ≥ (1 - overlap_frac)*d, etc.
    # Equivalent to saying we exclude any center within r_z = (1-overlap_frac)*d along z (same for y,x).
    r_z = (1 - overlap_frac) * d
    r_y = (1 - overlap_frac) * h
    r_x = (1 - overlap_frac) * w

    selected = []
    still_ok = np.ones(len(coords), dtype=bool)

    for i in range(len(coords)):
        if not still_ok[i]:
            continue
        c_i = coords[i]
        selected.append(tuple(c_i))
        if max_patches is not None and len(selected) >= max_patches:
            break
        z_i, y_i, x_i = c_i
        # Compute absolute deltas to all coords:
        dz = np.abs(coords[:,0] - z_i)
        dy = np.abs(coords[:,1] - y_i)
        dx = np.abs(coords[:,2] - x_i)
        # Mark too-close ones as invalid:
        too_close = (dz < r_z) & (dy < r_y) & (dx < r_x)
        still_ok[too_close] = False

    return np.array(selected, dtype=np.int16)


def plot_patch_mips(volume, coords, chosen_indices, patch_size, case):
    """
    Given a 3D PET volume, coordinates of valid patch centers, and a list of chosen indices,
    extract each patch and plot its axial MIP (Max Intensity Projection).

    Parameters:
    - volume: 3D numpy array (Z, Y, X) of PET SUV values
    - coords: numpy array of shape (num_valid, 3), where each row is (z, y, x) center of a patch
    - chosen_indices: 1D array-like of indices into coords to select which patches to plot
    - patch_size: tuple of (d, h, w) representing the patch dimensions
    """
    d, h, w = patch_size
    dz, dy, dx = d // 2, h // 2, w // 2

    n = len(chosen_indices)
    # Choose a reasonable grid layout (up to 4 columns)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)  # ensure a 2D array even for single row/column

    for idx_plot, center_idx in enumerate(chosen_indices):
        # Compute row, col for subplot
        row = idx_plot // cols
        col = idx_plot % cols
        ax = axes[row, col]

        z, y, x = coords[center_idx]
        # Extract the 3D patch
        patch = volume[
            z - dz : z + dz,
            y - dy : y + dy,
            x - dx : x + dx
        ]

        # Compute axial MIP (max over Z dimension)
        mip_axial = np.max(patch, axis=0)

        ax.imshow(mip_axial, cmap='gray', origin='lower')
        ax.set_title(f'Patch {idx_plot+1} (center={z},{y},{x})\nAxial MIP')
        ax.axis('off')

    # Hide any unused subplots
    total_plots = rows * cols
    for empty_idx in range(n, total_plots):
        row = empty_idx // cols
        col = empty_idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f'./patch_mips/{case}.png')
    plt.close()
import os




# Example usage:
# volume_shape = volume.shape  # e.g., (128, 256, 256)
# selected_centers = np.array([[60, 100, 120], [45, 130, 110], ...])  # shape (n, 3)
# patch_size = (32, 64, 64)
# plot_patch_masks_orthogonal_views(volume_shape, selected_centers, patch_size)
def plot_overlay_patch_masks_orthogonal_views(volume, selected_centers, patch_size, case, alpha=0.3):
    """
    Plot the 3 orthogonal maximum projections of the PET volume with 
    an overlay of the binary mask indicating patch regions for each selected center.

    Parameters:
    - volume: 3D numpy array of the PET SUV values (Z, Y, X)
    - selected_centers: numpy array of shape (n, 3) where each row is (z, y, x)
    - patch_size: tuple (d, h, w) representing the patch dimensions
    - alpha: transparency level for the mask overlay (0.0 to 1.0)
    """
    Z, Y, X = volume.shape
    d, h, w = patch_size
    dz, dy, dx = d // 2, h // 2, w // 2

    # Create an empty 3D mask volume
    mask_3d = np.zeros((Z, Y, X), dtype=bool)

    # Set True for voxels inside any patch
    for (z, y, x) in selected_centers:
        z_min, z_max = max(0, z - dz), min(Z, z + dz)
        y_min, y_max = max(0, y - dy), min(Y, y + dy)
        x_min, x_max = max(0, x - dx), min(X, x + dx)
        mask_3d[z_min:z_max, y_min:y_max, x_min:x_max] = True

    # Compute maximum intensity projections of the volume
    axial_view_vol = np.max(volume, axis=0)      # shape Y×X
    coronal_view_vol = np.max(volume, axis=1)    # shape Z×X
    sagittal_view_vol = np.max(volume, axis=2)   # shape Z×Y

    # Compute maximum projections of the binary mask
    axial_view_mask = np.max(mask_3d, axis=0)    # shape Y×X
    coronal_view_mask = np.max(mask_3d, axis=1)  # shape Z×X
    sagittal_view_mask = np.max(mask_3d, axis=2) # shape Z×Y

    # Plot the three views with overlay
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Axial view (Y vs X)
    axes[0].imshow(axial_view_vol, vmax=2.0, cmap='gray', origin='lower')
    alpha_mask = np.where(axial_view_mask > 0, alpha, 0)
    axes[0].imshow(axial_view_mask, cmap='Reds', alpha=alpha_mask, origin='lower')
    axes[0].set_title('Axial MIP Overlay (Y×X)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # Coronal view (Z vs X)
    axes[1].imshow(coronal_view_vol, vmax=2.0, cmap='gray', origin='lower')
    alpha_mask = np.where(coronal_view_mask > 0, alpha, 0)
    axes[1].imshow(coronal_view_mask, cmap='Reds', alpha=alpha_mask, origin='lower')
    axes[1].set_title('Coronal MIP Overlay (Z×X)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')

    # Sagittal view (Z vs Y)
    # Transpose so that axes are Z (x-axis) vs Y (y-axis)
    axes[2].imshow(sagittal_view_vol.T, vmax=2.0, cmap='gray', origin='lower')
    alpha_mask = np.where(sagittal_view_mask.T > 0, alpha, 0)
    axes[2].imshow(sagittal_view_mask.T, cmap='Reds', alpha=alpha_mask, origin='lower')
    axes[2].set_title('Sagittal MIP Overlay (Z×Y)')
    axes[2].set_xlabel('Z')
    axes[2].set_ylabel('Y')

    plt.tight_layout()
    plt.savefig(f'./patch_masks/{case}.png')
    plt.close()


from concurrent.futures import ProcessPoolExecutor, as_completed

def process_one_case(case_filename: str, input_dir: str, mask_dir: str, coord_dir: str) -> (str, int):
    """
    Load a single NIfTI, compute the mask, save the 2D projection,
    prune centers, generate plots, and save coords. Return (case_filename, num_centers).
    """
    # 1) Ensure matplotlib uses a non‐GUI backend before importing pyplot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 2) Load the volume:
    nifti_path = os.path.join(input_dir, case_filename)
    nifti = nib.load(nifti_path)
    volume = nifti.get_fdata()

    # 3) Compute binary mask of valid centers (your existing function):
    #    compute_valid_centers_3d should return a numpy array of shape volume.shape with 0/1
    mask = compute_valid_centers_3d(volume,
                                    suv_thresh=0.4,
                                    patch_size=(64, 64, 64),
                                    min_frac=0.6)

    # 4) Save a 2D projection (sum over the Y‐axis) to a PNG:
    projection = np.sum(mask, axis=1)  # shape: (D, W) if volume is (D, H, W)
    plt.figure(figsize=(6, 6))
    plt.imshow(projection, cmap='gray')
    plt.axis('off')
    base_name = case_filename.split('.')[0]
    png_path = os.path.join(mask_dir, f'{base_name}.png')
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 5) Extract voxel centers from the mask; prune overlapping:
    coords = np.array(np.where(mask), dtype=np.int16).T  # shape: (N_candidates, 3)
    coords = prune_nonoverlapping_centers(coords,
                                         patch_size=(64, 64, 64),
                                         overlap_frac=0.25,
                                         max_patches=None,
                                         seed=42)
    num_centers = coords.shape[0]

    # 6) Plot orthogonal views and overlays (your existing plotting functions):
    #    These functions should internally call matplotlib (and rely on Agg backend).

    plot_overlay_patch_masks_orthogonal_views(volume,
                                              selected_centers=coords,
                                              patch_size=(64, 64, 64), case=base_name,
                                              alpha=0.3)

    # 7) Save the coords as a .npy file:
    npy_path = os.path.join(coord_dir, f'{base_name}.npy')
    np.save(npy_path, coords)

    # 8) Randomly choose up to 16 indices to generate patch MIPs:
    if num_centers > 0:
        n_to_choose = min(16, num_centers)
        chosen_indices = np.random.choice(num_centers, size=n_to_choose, replace=False)
        plot_patch_mips(volume,
                        coords,
                        chosen_indices,
                        patch_size=(64, 64, 64),
                        case=base_name)

    return case_filename, num_centers
def run_parallel_cases(input_dir: str = './fdg_united',
                       mask_dir: str = './masks',
                       coord_dir: str = './coords'):
    """
    Discover all NIfTI files in `input_dir`, then spawn a pool of worker processes—
    one per logical CPU minus one—to process each case in parallel. Results (PNGs
    and .npy) are saved under `mask_dir` and `coord_dir`, respectively.
    """
    # 1) Create output directories if they don't exist:
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(coord_dir, exist_ok=True)

    # 2) List all filenames in input_dir (assumes only NIfTI files are present or filter as needed):
    all_cases = sorted(os.listdir(input_dir))
    if not all_cases:
        print(f"[INFO] No files found in {input_dir}.")
        return

    # 3) Decide how many worker processes to spawn:
    #    On a CPU with 8 logical cores, use 7 to leave one thread for the OS.
    max_workers = 5

    print(f"[INFO] Found {len(all_cases)} cases. Spawning {max_workers} worker processes...\n")

    # 4) Use a ProcessPoolExecutor to dispatch each case to process_one_case:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {
            executor.submit(process_one_case, case, input_dir, mask_dir, coord_dir): case
            for case in all_cases
        }

        # 5) As each worker completes, print the result or an error:
        for future in as_completed(future_to_case):
            case_name = future_to_case[future]
            try:
                case_name, num_centers = future.result()
                print(f"[DONE] {case_name}: extracted {num_centers} centers")
            except Exception as exc:
                print(f"[ERROR] {case_name}: {exc}")

    print("\n[INFO] All cases processed.")


if __name__ == "__main__":
    # If run as a script, call with default directories:
    run_parallel_cases(input_dir='./fdg_united',
                       mask_dir='./masks',
                       coord_dir='./coords')