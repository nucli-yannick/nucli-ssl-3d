from __future__ import annotations

"""
The goal of this script is to write in it all the transformations used in Voco 
Several transformations are important here: 
    - Creation of the base crops
    - Creation of the target crops 
"""

from typing import Literal, Tuple
import numpy as np
from einops import rearrange



class VocoTransform(): 

    def __init__(
        self, 
        voco_base_crop_count: Tuple[int, int, int], 
        voco_crop_size: Tuple[int, int, int], 
        data_key="input", 
        center_key="target_coords",
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

        self.data_key = data_key

    def get_base_crops(self, data):
        """
        Splits the data into base crops.
        Returns all crops.

        :param data: [B, X, Y, Z] data to split into base crops.
        :return: [B, N_subcrops, X_subcrop, Y_subcrop, Z_subcrop] base crops
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
            data (np.ndarray): Input data of shape [B, X, Y, Z].
            potential_centers (np.ndarray): Potential centers for crops. This is a list containing B element of size [N, 3]
        Returns:
        :return: [B, N_subcrops, X_subcrop, Y_subcrop, Z_subcrop], overlaps [B, N_target_crop, N_base_crop]
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
            
                x_offset = np.random.randint(0, (big_crop.shape[1] - crop_size[0]) + 1)
                y_offset = np.random.randint(0, (big_crop.shape[2] - crop_size[1]) + 1)
                z_offset = np.random.randint(0, (big_crop.shape[3] - crop_size[2]) + 1)

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
    

    def __call__(self, data_dict: dict) -> dict:
        """
        Applies the Voco transformation to the input data dictionary.
        Args:
            data_dict (dict): Dictionary containing the data and coordinates.
        Returns:
            dict: Updated dictionary with base crops and target crops and overlaps.
        """
        data = data_dict[self.data_key]

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
            "base_target_crop_overlaps": gt_overlap
        }




