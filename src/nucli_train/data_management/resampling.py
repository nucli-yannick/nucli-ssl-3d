from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import map_coordinates
from skimage.transform import resize
from cucim.skimage.transform import resize as resize_cucim 
from skimage.transform import resize
import cupy as cp 

ANISO_THRESHOLD = 3
SPLINE_ORDER = 3
SPLINE_ORDER_Z = 0

logger = logging.getLogger(__name__)




def get_do_separate_z(
    spacing: tuple,
    anisotropy_threshold=ANISO_THRESHOLD,
) -> bool:
    """
    Determines whether separate processing for the Z-axis is required based on the
    anisotropy of the given spacing.

    Args:
        spacing (tuple): A tuple of floats representing the spacing along each dimension.
        anisotropy_threshold (float, optional): The threshold value to decide if the spacing
            is considered anisotropic. Defaults to ANISO_THRESHOLD.

    Returns:
        bool: True if the maximum spacing divided by the minimum spacing exceeds the
            anisotropy threshold, indicating anisotropy sufficient to warrant separate
            Z-axis processing. False otherwise.
    """
    do_separate_z = (max(spacing) / min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: tuple) -> Union[None, int]:
    """
    Determines the axis with the lowest resolution based on the provided spacings.
    If multiple axes share the lowest resolution and are not unique, returns None.

    Args:
        new_spacing (tuple): A tuple containing spacing values for each dimension.

    Returns:
        Union[None, int]: The index of the axis with the lowest resolution if there is a unique lowest resolution axis.
        Returns None if there is no unique lowest resolution axis or if two axes share the lowest value.

    Raises:
        ValueError: If the spacing values do not conform to expected conditions, such as having more than one low-resolution
        axis or no clear low-resolution axis at all.
    """
    low_res_axes = np.argwhere(new_spacing == np.max(new_spacing)).flatten().tolist()
    if len(low_res_axes) == 2:
        # E.g., this can happen for spacings like (0.24, 1.25, 1.25)
        # In that case we do not want to resample separately in the out of plane axis
        low_res_axis = None
    elif len(low_res_axes) == 1:
        low_res_axis = low_res_axes[0]
    else:
        raise ValueError(
            f"Invalid amount of low-resolution axes, this should not happen, new_spacing: {new_spacing}"
        )

    return low_res_axis


def compute_new_shape(
    old_shape: tuple,
    old_spacing: tuple,
    new_spacing: tuple,
) -> tuple:
    """
    Computes the new shape for an image or volume based on the old and new spacings to
    maintain the same physical size of the data. It adjusts each dimension according to
    the ratio of old to new spacing multiplied by the old dimension size.

    Args:
        old_shape (tuple): A tuple of integers representing the dimensions of the old shape.
        old_spacing (tuple): A tuple of floats indicating the spacing between elements in
                             each dimension of the old shape.
        new_spacing (tuple): A tuple of floats indicating the desired spacing between elements
                             in each dimension of the new shape.

    Returns:
        tuple: A tuple representing the dimensions of the new shape, calculated to maintain
               the physical size of the data.

    Raises:
        ValueError: If the number of dimensions in `old_shape` and `old_spacing` do not match,
                    or if `old_spacing` and `new_spacing` do not have the same number of dimensions.
    """
    if len(old_spacing) != len(old_shape):
        raise ValueError(
            f"Old spacing and old shape should have the same amount of dims, got {old_spacing} and {old_shape} instead."
        )
    if len(old_spacing) != len(new_spacing):
        raise ValueError(
            f"Old spacing and new spacing should have the same amount of dims, got {old_spacing} and {new_spacing} instead."
        )
    new_shape = tuple(
        int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)
    )
    return new_shape


def resample(
    image: np.ndarray,
    current_spacing: tuple,
    new_spacing: tuple,
    new_shape: Optional[tuple] = None,
    order: int = SPLINE_ORDER,
    order_z: int = SPLINE_ORDER_Z,
) -> np.ndarray:
    """
    Resamples an image to a new shape and spacing, optionally applying different resampling
    orders for isotropic and anisotropic dimensions. Handles separate processing along
    the Z-axis if indicated by anisotropy in the spacing.

    Args:
        image (np.ndarray): A 2D numpy array representing the image with shape (D, H, W).
        current_spacing (tuple): A tuple of floats describing the spacing between elements
                                 in each dimension of the current image.
        new_spacing (tuple): A tuple of floats describing the desired spacing between elements
                             in each dimension of the resampled image.
        new_shape (Optional[tuple], optional): A tuple of integers describing the desired shape
                                               of the resampled image. If None, the new shape
                                               will be computed based on the current and new spacings.
        order (int, optional): The order of the spline interpolation for resampling, except for
                               the potentially anisotropic Z-axis. Defaults to SPLINE_ORDER.
        order_z (int, optional): The order of the spline interpolation for the Z-axis when
                                 anisotropy is handled separately. Defaults to SPLINE_ORDER_Z.

    Returns:
        np.ndarray: The resampled image as a 3D numpy array with potentially different spatial dimensions.

    Raises:
        ValueError: If the input image does not have three dimensions or if the new shape does
                    not match the spatial dimensions of the input image.
    """
    low_res_axis = None
    if get_do_separate_z(current_spacing):
        low_res_axis = get_lowres_axis(current_spacing)
    elif get_do_separate_z(new_spacing):
        low_res_axis = get_lowres_axis(new_spacing)

    if image.ndim != 3:
        raise ValueError("Input image must be of shape (D, H, W)")

    if new_shape is None:
        new_shape = compute_new_shape(image.shape, current_spacing, new_spacing)

    if len(new_shape) != image.ndim:
        raise ValueError(
            f"New shape should match spatial dims of image, got {new_shape} vs. {image.shape} instead."
        )

    logger.info(
        f"Resampling input image from shape {image.shape} and spacing {current_spacing} to shape {new_shape} and spacing {new_spacing}"
    )
    image_resampled = _resample(
        image=image,
        new_shape=new_shape,
        low_res_axis=low_res_axis,
        order=order,
        order_z=order_z,
    )
    return image_resampled


def _resample(
    image: np.ndarray,
    new_shape: tuple,
    low_res_axis: Union[None, int] = None,
    order: int = SPLINE_ORDER,
    order_z: int = SPLINE_ORDER_Z,
) -> np.ndarray:
    """
    Performs the resampling of a given image to a new shape, applying isotropic or anisotropic resampling
    techniques based on the presence of a low-resolution axis. This function supports different interpolation
    orders for isotropic and anisotropic resampling to optimize for image quality and computational efficiency.

    Args:
        image (np.ndarray): A 3D numpy array representing the image with shape (D, H, W).
        new_shape (tuple): A tuple of integers indicating the desired shape (D, H, W) to which the image
                           should be resampled.
        low_res_axis (Union[None, int], optional): The axis index identified as having low resolution that
                                                  might require different resampling handling. If None,
                                                  isotropic resampling is performed.
        order (int, optional): The order of the spline interpolation used for isotropic resampling and
                               anisotropic resampling on axes other than the Z-axis. Defaults to SPLINE_ORDER.
        order_z (int, optional): The order of the spline interpolation specifically for the Z-axis in the
                                 case of anisotropic resampling. Defaults to SPLINE_ORDER_Z.

    Returns:
        np.ndarray: The resampled image as a 4D numpy array with dimensions (D, H, W) corresponding
                    to the new shape.

    Raises:
        ValueError: If the final resampled image's shape does not match the desired new shape.

    Notes:
        This function employs scikit-image's `resize` function for resampling, configured not to apply
        anti-aliasing and to use 'edge' mode for handling borders during interpolation.
    """
    resize_kwargs = {"mode": "edge", "anti_aliasing": False}

    if np.all(image.shape == new_shape):
        logger.info("No resampling performed.")
        return image

    if low_res_axis is not None:
        print(f"Performing anisotropic resampling along axis {low_res_axis}")
        resampled_image = _anisotropic_resampling(
            image=image,
            new_shape=new_shape,
            low_res_axis=low_res_axis,
            order=order,
            order_z=order_z,
            resize_kwargs=resize_kwargs,
        )
    else:
        image = cp.asarray(image)
        resampled_image = resize_cucim(image, new_shape, order, **resize_kwargs) 
        #resampled_image = resize(image, new_shape, order, **resize_kwargs)
        resampled_image = cp.asnumpy(resampled_image)

    if resampled_image.shape != new_shape:
        raise ValueError("Resampled image shape does not match the desired new shape.")

    return resampled_image


def _anisotropic_resampling(
    image: np.ndarray,
    new_shape: tuple,
    low_res_axis: int,
    resize_kwargs: dict,
    order: int = SPLINE_ORDER,
    order_z: int = SPLINE_ORDER_Z,
) -> np.ndarray:
    """
    Performs anisotropic resampling of a given image, handling different resolutions along specified axes
    by applying varying resampling techniques based on the axis identified as having low resolution.

    Args:
        image (np.ndarray): A 3D numpy array of the image with shape (D, H, W).
        new_shape (tuple): The target shape for resampling, given as a tuple (D, H, W).
        low_res_axis (int): The axis identified as having a lower resolution, requiring specialized resampling.
        resize_kwargs (dict): A dictionary of keyword arguments to pass to the `resize` function.
        order (int, optional): The order of the spline interpolation for non-low-resolution axes. Defaults to SPLINE_ORDER.
        order_z (int, optional): The order of the spline interpolation for the low-resolution axis. Defaults to SPLINE_ORDER_Z.

    Returns:
        np.ndarray: The anisotropic resampled image as a 3D numpy array with dimensions (D, H, W) corresponding
                    to the new shape.
    Raises:
        ValueError: If the provided `low_res_axis` is not of type int.
    """
    if not isinstance(low_res_axis, int):
        raise ValueError(
            f"Low resolution axis should be of type int, got {low_res_axis}"
        )

    logger.info(f"Performing anisotropic resampling along axis {low_res_axis}")
    if low_res_axis == 0:
        new_shape_2d = new_shape[1:]
    elif low_res_axis == 1:
        new_shape_2d = new_shape[0::2]
    else:
        new_shape_2d = new_shape[:-1]

    shape = image.shape
    resampled_axis = []
    for slice_id in range(shape[low_res_axis]):
        if low_res_axis == 0:
            resampled_axis.append(
                resize(image[slice_id], new_shape_2d, order, **resize_kwargs)
            )
        elif low_res_axis == 1:
            resampled_axis.append(
                resize(image[:, slice_id], new_shape_2d, order, **resize_kwargs)
            )
        else:
            resampled_axis.append(
                resize(image[:, :, slice_id], new_shape_2d, order, **resize_kwargs)
            )
    # stack the slices back along the low-resolution axis
    resampled_axis = np.stack(resampled_axis, low_res_axis)

    if shape[low_res_axis] != new_shape[low_res_axis]:

        # The following few lines are blatantly copied and modified from sklearn's resize()
        rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
        orig_rows, orig_cols, orig_dim = resampled_axis.shape

        row_scale = float(orig_rows) / rows
        col_scale = float(orig_cols) / cols
        dim_scale = float(orig_dim) / dim

        map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
        map_rows = row_scale * (map_rows + 0.5) - 0.5
        map_cols = col_scale * (map_cols + 0.5) - 0.5
        map_dims = dim_scale * (map_dims + 0.5) - 0.5

        coord_map = np.array([map_rows, map_cols, map_dims])
        resampled_image = map_coordinates(
            resampled_axis, coord_map, order=order_z, mode="nearest"
        )
    else:
        resampled_image = resampled_axis
    return resampled_image