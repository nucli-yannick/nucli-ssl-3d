from __future__ import annotations

from typing_extensions import Literal

from typing import Tuple
import numpy as np
from einops import rearrange
import torch

from .dataset import TRANS_REGISTRY


@TRANS_REGISTRY.register('voco_transform')
class VocoTransform(): 
    """
    The goal of this script is to write in it all the transformations used in Voco 
    Several transformations are important here: 
        - Creation of the base crops
        - Creation of the target crops 
    """

    def __init__(
        self, 
        voco_base_crop_count: Tuple[int, int, int], 
        voco_crop_size: Tuple[int, int, int], 
        voco_target_crop_count: int = 4
    ): 
        self.voco_base_crop_count = voco_base_crop_count
        self.voco_crop_size = voco_crop_size
        self.voco_target_crop_count = voco_target_crop_count
        self.bounding_boxes = []
        for i in range(self.voco_base_crop_count[0]):
            for j in range(self.voco_base_crop_count[1]):
                for k in range(self.voco_base_crop_count[2]):
                    self.bounding_boxes.append(
                        (
                            i * self.voco_crop_size[0],
                            j * self.voco_crop_size[1],
                            k * self.voco_crop_size[2],
                            (i + 1) * self.voco_crop_size[0],
                            (j + 1) * self.voco_crop_size[1],
                            (k + 1) * self.voco_crop_size[2],
                        )
                    )



    def get_base_crops(self, data):
        """
        Splits the data into base crops.
        Returns all crops.

        :param data: [C, X, Y, Z] data to split into base crops.
        :return: [C, N_subcrops, X_subcrop, Y_subcrop, Z_subcrop] base crops
        """
        base_crops = []
        for i in range(self.voco_base_crop_count[0]):
            for j in range(self.voco_base_crop_count[1]):
                for k in range(self.voco_base_crop_count[2]):
                    crop = data[
                        :,
                        i * self.voco_crop_size[0] : (i + 1) * self.voco_crop_size[0],
                        j * self.voco_crop_size[1] : (j + 1) * self.voco_crop_size[1],
                        k * self.voco_crop_size[2] : (k + 1) * self.voco_crop_size[2],
                    ]
                    base_crops.append(crop)
        return np.stack(base_crops, axis=1)

    
    def get_target_crops(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Defines a random crop that is partially overlapping with some of the base crops.
        The input is a batch of images and a set of potential centers for the crops.
        Args: 
            data (np.ndarray): Input data of shape [C, X, Y, Z].
            potential_centers (np.ndarray): Potential centers for crops. This is a list containing B element of size [N, 3]
        Returns:
        :return: [C, N_subcrops, X_subcrop, Y_subcrop, Z_subcrop], overlaps [B, N_target_crop, N_base_crop]
        """

        image_wise_crop = []
        image_wise_overlaps = []

        crop_size = self.voco_crop_size
        total_volume = crop_size[0] * crop_size[1] * crop_size[2]

        # For each image in batch -- data shape: [B, X, Y, Z]
        for big_crop in data:
            # big_crop shape : [X, Y, Z]
            # local_centers shape: [N, 3] where N is the number of potential centers

            
            target_crops, target_overlaps = [], []
            for _ in range(self.voco_target_crop_count):
            
                x_offset = np.random.randint(0, (big_crop.shape[0] - crop_size[0]) + 1)
                y_offset = np.random.randint(0, (big_crop.shape[1] - crop_size[1]) + 1)
                z_offset = np.random.randint(0, (big_crop.shape[2] - crop_size[2]) + 1)

                crop = big_crop[
                    x_offset : x_offset + crop_size[0],
                    y_offset : y_offset + crop_size[1],
                    z_offset : z_offset + crop_size[2],
                ]
                
                target_crops.append(crop)

                target_base_crop_overlaps = []
                for bbox in self.bounding_boxes:
                    overlap_x = max(0, min(x_offset + crop_size[0], bbox[3]) - max(x_offset, bbox[0]))
                    overlap_y = max(0, min(y_offset + crop_size[1], bbox[4]) - max(y_offset, bbox[1]))
                    overlap_z = max(0, min(z_offset + crop_size[2], bbox[5]) - max(z_offset, bbox[2]))
                    overlap_volume = overlap_x * overlap_y * overlap_z
                    overlap_ratio = overlap_volume / total_volume
                    target_base_crop_overlaps.append(overlap_ratio)
                target_overlaps.append(np.array(target_base_crop_overlaps))
            image_wise_crop.append(np.stack(target_crops, axis=0))
            image_wise_overlaps.append(np.stack(target_overlaps, axis=0))  # [N_target_subcrops, N_base_subcrops]

        return np.stack(image_wise_crop, axis=0), np.stack(image_wise_overlaps, axis=0)
    

    def __call__(self, data) -> dict:
        
   

        if data is None: 
            raise ValueError("Data is None. Please provide valid data.")
        
        base_crops = self.get_base_crops(data)  # [B, N_base_subcrops, X_subcrop, Y_subcrop, Z_subcrop]
        target_crops, gt_overlap = self.get_target_crops(data) # [B, N_target_subcrops, X_subcrop, Y_subcrop, Z_subcrop], [B, N_target_subcrops, N_base_subcrops]


        base_crops_flat = rearrange(base_crops, "b n x y z -> (b n) x y z")
        target_crops_flat = rearrange(target_crops, "b n x y z -> (b n) x y z")
        joint_crops_flat = np.concatenate([base_crops_flat, target_crops_flat], axis=0)
        base_crop_index = base_crops_flat.shape[0]

        return {
            "all_crops": joint_crops_flat, 
            "base_crop_index": base_crop_index,  
            "base_target_crop_overlaps": gt_overlap, 
            "data": data
        }


@TRANS_REGISTRY.register('mae_transform')
class MAETransform():
    def __init__(self, tensor_size, block_size, sparsity_factor=0.75, rng_seed: None | int = None) -> torch.Tensor: 
        self.tensor_size = tensor_size
        self.block_size = block_size
        self.sparsity_factor = sparsity_factor
        self.rng_seed = rng_seed
    
    def create_blocky_mask(self): 
        small_mask_size = tuple(size // self.block_size for size in self.tensor_size)

        flat_mask = torch.ones(np.prod(small_mask_size))
        n_masked = int(self.sparsity_factor * flat_mask.shape[0])
        if self.rng_seed is None: 
            mask_indices = torch.randperm(flat_mask.shape[0])[:n_masked]
        else: 
            gen = torch.Generator.manual_seed(self.rng_seed)
            mask_indices = torch.randperm(flat_mask.shape[0], generator=gen)[:n_masked]
        flat_mask[mask_indices] = 0
        small_mask = torch.reshape(flat_mask, small_mask_size)
        return small_mask

    def mask_creation(
        self, 
        patch_size: tuple[int, int, int]
        ) -> torch.Tensor:
        mask = self.create_blocky_mask()
        mask = mask[None, :, :, :]
        return mask

    def __call__(self, data) -> dict:
        data = data
        patch_size = data.shape[1:]
        mask = self.mask_creation(patch_size)

        rep_D, rep_H, rep_W = (
            data.shape[1] // mask.shape[1],
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
        )
        mask = mask.repeat_interleave(rep_D, dim=1).repeat_interleave(rep_H, dim=2).repeat_interleave(rep_W, dim=3)

        masked_data = data * mask

        return {
            "masked_data": masked_data,
            "mask": mask,
            "data": data
        }


@TRANS_REGISTRY.register('mae_convnext_transform')
class MAEConvnextTransform(): 
    def __init__(
        self, 
        volume_size: int,
        mask_ratio: float, 
        patch_size: float
        ): 

        super().__init__()

        self.volume_size = volume_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patchs = (volume_size // patch_size) ** 3

    def patchify(self, imgs): 
        """
        imgs: (B, 1, D, H, W)
        x: (B, L, patch_size**3 * 1) because 1 is the channel number$
        The goal of this function is to convert he 3D image into a sequence ot patches
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0

        
        batch_size, _, axis_size, H, W = imgs.shape
        d = h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(batch_size, 1, d, p, h, p, w, p))
        x = torch.einsum('bcdefghi->bcdfhegi', x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p**3)) # L = d * h * w
        return x

    def unpatchify(self, x): 
        """
        x: (B, L, patch_size**3 * 1)
        imgs: (B, 1, D, H, W)
        """
        epsilon = 1e-5
        print("Unpatchify input shape: ", x.shape)
        p = self.patch_size
        h = w = d = int(x.shape[1]**(1/3) + epsilon)
        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], 1, d, h, w, p, p, p))
        x = torch.einsum('bcdhwpqs->bcdphqws', x)
        imgs = x.reshape(shape=(x.shape[0], 1, d * p, h * p, w * p))
        return imgs

    def gen_random_mask(self, x, mask_ratio): 
        B = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 3 # Number of patches in the volume
        len_keep = int(L*(1-mask_ratio))

        noise = torch.randn(B, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        epsilon = 1e-3
        p = int(mask.shape[1] ** (1/3) + epsilon)
        return mask.reshape(-1, p, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2).\
                    repeat_interleave(scale, axis=3)

    def __call__(self, data) -> dict: 
       mask = self.gen_random_mask(data, self.mask_ratio)
       return {
            "model_mask": mask, 
            "data": data
       }

















