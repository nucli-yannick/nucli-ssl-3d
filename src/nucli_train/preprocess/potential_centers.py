from __future__ import annotations

"""
The goal of this script is to compute the potential centers of a .npy 3D volume
"""




import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure, binary_dilation, distance_transform_edt

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
    mask0 = (volume > suv_thresh) # Binary mask same size as volume
    # Only the voxels above the threshold are True.
    struct = generate_binary_structure(rank=3, connectivity=1) # Element structurant pour la connectivité 3D
    mask_clean = binary_closing(mask0, structure=struct, iterations=1) # Applique une fermeture morphologique de fermeture 
    # sur le masque pour combler les trous et combler les espaces
    mask_clean = binary_opening(mask_clean, structure=struct, iterations=1)
    # Applique une ouverture morphologique pour éliminer les petits objets isolés

    # 2) (Optional) Dilate slightly to include low-uptake neighborhoods
    mask_dilated = mask_clean.copy()
    if suv_thresh > 0.5:
        # e.g. dilate by 2 voxels if you want to capture faint uptake around mask
        mask_dilated = binary_opening(mask_clean, structure=struct, iterations=1)
        mask_dilated = uniform_filter(mask_dilated.astype(np.float32),
                                      size=(3,3,3), mode='constant', cval=0) > 0.0
        # a single small dilate–erode pass to "soften" boundaries
        # We take only voxels that are still True after this operation
    
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



# Not used in the case of VoCo
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
    # Compute "exclusion radius" in each axis: we want |Δz| ≥ (1 - overlap_frac)*d, etc.
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

def process_one_case(volume, coord_dir: str, patch_size) -> None:
    """
    Process a single case to compute potential centers and save them to a npy file
    The volume should be 3D
    """

    
    print(f"Shape of the volume: {volume.shape}")

    mask = compute_valid_centers_3d(volume,
                                    suv_thresh=0.4,
                                    patch_size=patch_size,
                                    min_frac=0.6)

    coords = np.array(np.where(mask), dtype=np.int16).T  # shape: (N_candidates, 3)


    print(f"Adam -- -- -- Number of valid centers: {len(coords)}")
    if len(coords) == 0:
        shape = volume.shape
        coords = np.array([[shape[0] // 2, shape[1] // 2, shape[2] // 2]], dtype=np.int16)
        print("No valid centers found, using center of volume as fallback.")
    np.save(coord_dir, coords)






if __name__ == "__main__":

    pass