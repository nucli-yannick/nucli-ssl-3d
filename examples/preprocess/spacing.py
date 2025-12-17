from __future__ import annotations

import nibabel as nib
import os 
import numpy as np
from nucli_train.data_management.resampling import resample 
import matplotlib.pyplot as plt


def save_spacing_distribution(base_dir,exclude_condition, save_dir=None): 
    """
    Computes the distribution of voxel spacings in the dataset.
    """
    spacing_list = []
    for nifti_file in os.listdir(base_dir):
        if not exclude_condition(nifti_file):
            img = nib.load(os.path.join(base_dir, nifti_file))
            spacing_list.append(list(img.header['pixdim'][1:4]))
            print(f"Processed {nifti_file}, spacing: {spacing_list[-1]}")
    spacing_list = np.array(spacing_list)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'spacing_distribution.npy'), spacing_list)

def load_spacing_distribution(file_path):
    """
    Loads the spacing distribution from a file.
    """
    return np.load(file_path, allow_pickle=True)

def plot_distribution(spacing_list):
    """
    Plots the distribution of voxel spacings for all three dimensions.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    spacing_list = np.array(spacing_list)  # Just in case it's not already a NumPy array

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    dim_labels = ['X (Spacing in mm)', 'Y (Spacing in mm)', 'Z (Spacing in mm)']

    for i in range(3):
        sns.histplot(spacing_list[:, i], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution - {dim_labels[i]}')
        axes[i].set_xlabel(dim_labels[i])
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('spacing_distribution_all_dims.png')
    plt.show()


def compare_spacing_image(image_path, target_spacing):
    img = nib.load(image_path)
    img_data = img.get_fdata()
    current_spacing = img.header['pixdim'][1:4]
    target_spacing = np.array([
        4.0, 
        4.0, 
        4.0
    ])
    resampled_data = resample(
        img_data, 
        current_spacing, 
        target_spacing
    )
    return img_data, resampled_data

def plot_comparison_images(original_image, resampled_image, z_slice):

    z_original = original_image.shape[2]
    z_resampled = resampled_image.shape[2]
   

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image[:, :, z_original//z_slice], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(resampled_image[:, :, z_resampled//z_slice], cmap='gray')
    axes[1].set_title('Resampled Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('spacing_comparison.png')






if __name__ == "__main__":

    base_dir = "/home/interns_nuclivision_com/aws_backup/adam/autopet_2024/imagesTr"

    name = "0000"
    images = os.listdir(base_dir)

    while "0000" in name:
        i = np.random.randint(0, 1000)
        name = images[i]
    
    image_path = os.path.join(base_dir, name)

    img_data, resampled_data = compare_spacing_image(image_path, target_spacing=None)

    print(f"Original image shape: {img_data.shape}")
    print(f"Resampled image shape: {resampled_data.shape}")
    
    plot_comparison_images(img_data, resampled_data, z_slice=2)





    

