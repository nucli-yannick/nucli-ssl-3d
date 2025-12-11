from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Union

import numpy as np
from tqdm import tqdm

import torch

from itertools import product
from typing import Tuple, Union

import numpy as np

CHANNEL_WEIGHTING_Z = 200

def bbox(img: np.ndarray, thresh: float = 0.3) -> tuple[slice]:
    """
    Calculates the bounding box of an image based on a threshold.

    Determines the minimum and maximum coordinates in each dimension of the image
    where the pixel values are greater than the specified threshold, effectively
    creating a bounding box around the significant portion of the image.

    Args:
        img (np.ndarray): The image array for which the bounding box is to be calculated.
        thresh (float, optional): The threshold value used to identify significant
                                  pixels. Defaults to 0.3.

    Returns:
        tuple: A tuple of slice objects representing the bounding box in each dimension.
    """
    cond = np.where(img > thresh)
    _bbox = []
    for i, a in enumerate(cond):
        if a.size == 0:
            # no values are above the threshold, so return full dimension for safety
            print(
                f"WARNING: The input image does not have any values above the SUV threshold of {thresh} in axis {i}"
            )
            _bbox.append(slice(0, img.shape[i]))
        else:
            _bbox.append(slice(np.min(a), np.max(a) + 1))
    return tuple(_bbox)


def pm_metric(x: np.ndarray, kappa=50) -> float:
    """
    Calculates the divergence of Perona-Malik diffusion for an image using
    exponentiated gradients based on intensity differences across the image.

    Args:
        x (np.ndarray): The input image data as a 3D numpy array where each element
            represents pixel intensity.
        kappa (int, optional): The edge-stopping value which controls the diffusion
            rate. A lower value preserves more edges. Defaults to 50.

    Returns:
        float: The mean of the absolute values of the calculated diffusion, representing
        the average diffusion across the image.

    Note:
        Refer to: https://en.wikipedia.org/wiki/Anisotropic_diffusion for more information.
    """
    delta_s = np.zeros_like(x)
    delta_e = delta_s.copy()
    delta_d = delta_s.copy()
    ns = delta_s.copy()
    ew = delta_s.copy()
    ud = delta_s.copy()

    delta_d[:-1, :, :] = x[1:, :, :] - x[:-1, :, :]
    delta_s[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    delta_e[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]

    g_d = np.exp(-((delta_d / kappa) ** 2.0))
    g_s = np.exp(-((delta_s / kappa) ** 2.0))
    g_e = np.exp(-((delta_e / kappa) ** 2.0))

    d = g_d * delta_d
    s = g_s * delta_s
    e = g_e * delta_e

    ud[:] = d
    ns[:] = s
    ew[:] = e
    ud[1:, :, :] -= d[:-1, :, :]
    ns[:, 1:, :] -= s[:, :-1, :]
    ew[:, :, 1:] -= e[:, :, :-1]

    pm = np.abs(ud + ns + ew).mean().item()
    return pm

def get_channel_weighting(
    x: np.ndarray, min_pm: float = 0.1, max_pm: float = 0.2
) -> float:
    """
    Adjust the channel weighting based on the Perona-Malik metric calculated for a
    region of interest (ROI) within the given image data.

    This function computes the Perona-Malik diffusion metric for an ROI in the image,
    then uses this metric to interpolate between two specified weights. The result is
    a two-channel weighting array where the weights sum to 1. The weight distribution
    is dependent on the diffusion metric relative to the provided minimum and maximum
    thresholds.

    Args:
        x (np.ndarray): The input image data as a 3D numpy array.
        min_PM (float, optional): The minimum threshold for the Perona-Malik metric,
            below which the weighting favors the first channel (low-noise) more strongly.
            Defaults to 0.1.
        max_PM (float, optional): The maximum threshold for the Perona-Malik metric,
            above which the weighting favors the second channel (high-noise) more strongly.
            Defaults to 0.2.

    Returns:
        np.ndarray: A two-channel weight array shaped according to the input, with
        weights based on the computed Perona-Malik metric, and extended across the
        original dimensions of the input image.

    """
    roi_image = x[bbox(x)]
    print(f"PM Metric - bbox shape: {roi_image.shape}")
    if roi_image.shape[-1] < CHANNEL_WEIGHTING_Z:
        print(f"Non whole-body PET, using low-noise weight only.")
        pm = 0
    else:
        pm = pm_metric(roi_image)
    print(f"PM Metric - value: {pm}")
    a = (min(max(pm, min_pm), max_pm) - min_pm) / (max_pm - min_pm)
    return np.array([1 - a, a])[..., None, None, None]


def _check_tuple(candidate: Union[tuple, int], dims: int) -> tuple:
    """Check if object is a tuple or integer and return a tuple with predefined length.

    Args:
        candidate (Union[tuple, int]): The object to make a tuple.
        dims (int): The length of the output tuple.

    Raises:
        ValueError: If the length of the candidate dosnt match the dims.
        TypeError: If the candidate's type is not int or tuple.

    Returns:
        tuple: The original object if it was a tuple,
        else a tuple with integer values repeated for dims times.
    """
    if isinstance(candidate, tuple):
        if len(candidate) != dims:
            raise ValueError(f"The candidate has len {len(candidate)}, expected {dims}")
        return candidate
    if isinstance(candidate, int):
        return (candidate,) * dims
    raise TypeError(
        f"Candidate should be of type tuple or int, instead got {type(candidate)}"
    )


class SlidingPatches:
    """Object that generates patches with a predefined shape and stride."""

    def __init__(
        self,
        image: np.ndarray,
        patch_size: Union[tuple, int],
        strides: Union[tuple, int],
    ) -> None:
        """Initializes an SlidingPatches object.

        Args:
            image (np.ndarray): The images on which to take the patches.
            patch_size (Union[tuple, int]): The dimensions of the bounding boxes.
            strides (Union[tuple, int]): The strides to take when acquiring the patches.
        """
        self.image = image
        self.dims = len(self.image.shape)
        self.patch_size = _check_tuple(patch_size, self.dims)
        self.strides = _check_tuple(strides, self.dims)
        if any(
            stride > patch_size
            for stride, patch_size in zip(self.strides, self.patch_size)
        ):
            raise ValueError(
                "The overlap will be zero at some places, make sure the stride is smaller than the patch size."
            )

        self.bbox_corners = None
        self.restart()

    def restart(self) -> None:
        """Restart the iterable containing all the corners of the bounding box."""
        self.bbox_corners = self.get_corner_iterator()

    def get_corner_iterator(self) -> product:
        """Return an iterator over the left upper corners of all patches.

        Returns:
            product: An iterator containing all the upper right corners (tuples)
            of the bounding boxes.
        """
        corner_coordinates = [
            list(np.arange(start=0, stop=d - self.patch_size[i], step=self.strides[i]))
            + [max(d - self.patch_size[i], 0)]
            for i, d in enumerate(self.image.shape)
        ]

        corner_coordinates = product(*corner_coordinates)
        return corner_coordinates

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, tuple]:
        """Yield the next patch from the upper corner iterable.

        Returns:
            Tuple[np.ndarray, tuple]: Tuple containg the image patch and corresponding bounding box.
        """
        corner = next(self.bbox_corners)

        bbox = tuple(
            [slice(start, start + self.patch_size[i]) for i, start in enumerate(corner)]
        )

        patch = self.image[bbox][None, ...]

        return patch, bbox

    def __len__(self) -> int:
        """Return the amount of generated patches.

        Returns:
            int: The amount of patches.
        """
        corner_lenghts = [
            len(np.arange(start=0, stop=d - self.patch_size[i], step=self.strides[i]))
            + 1
            for i, d in enumerate(self.image.shape)
        ]
        return np.prod(np.array(corner_lenghts))


def batch_generator(
    patches: SlidingPatches, batch_size: int
) -> tuple[np.ndarray, list]:
    """Generate batches of sliding window patches.

    Args:
        patches (SlidingPatches): Object that generates the patches with predefined size and stride.
        batch_size (int): The amount of patches to batch in the first dimension.

    Yields:
        Iterator[tuple[np.ndarray, list]]: Tuple of batched patches and corresponding bounding boxes.
    """
    batch = []
    bboxes = []
    for patch, bbox in patches:
        batch.append(patch)
        bboxes.append(bbox)
        if len(batch) == batch_size:
            # yield the batch
            yield np.array(batch), bboxes
            batch = []
            bboxes = []
    # yield the remainder
    if batch:
        yield np.array(batch), bboxes


def append_prediction(
    low_noise: np.ndarray, high_noise: np.ndarray,
    bboxes: np.ndarray,
    predicted_image: np.ndarray,
    overlap: np.ndarray, channel_weights,
    predicted_high_noise: Optional[np.ndarray] = None,
    predicted_low_noise: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Append patch predictions to construct the full image.

    Args:
        yhat (np.ndarray): Batch of predicted patches.
        bboxes (np.ndarray): Batch of bounding boxes.
        predicted_image (np.ndarray): Array that stores the output image.
        overlap (np.ndarray): Array that stores the overlap.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of output image and overlap.
    """
    for i, pred in enumerate(zip(low_noise, high_noise)):
        pred = np.stack(pred, axis=0)
        if pred.shape[0] != channel_weights.shape[0]:
            raise ValueError(
                f"Channel weights and prediction shape do not match at the first dim: {pred.shape} vs. {channel_weights.shape}"
            )
        
        if predicted_high_noise is not None:
            predicted_high_noise[bboxes[i]] += pred[1]
        if predicted_low_noise is not None:
            predicted_low_noise[bboxes[i]] += pred[0]

        predicted_image[bboxes[i]] += np.sum(
            pred * channel_weights, axis=0, keepdims=False
        )
        overlap[bboxes[i]] += 1

    return predicted_image, predicted_high_noise, predicted_low_noise, overlap

def do_inference_no_threads(process_batch, sliding_loader, patches, batch_size, channel_weights, predicted_image, overlap, low_noise_volume=None, high_noise_volume=None):
        
    # Process batches sequentially without ThreadPoolExecutor
    for batch in tqdm(
        sliding_loader, 
        total=int(np.ceil(len(patches) / batch_size)), 
        desc="Slider"
    ):
        low_noise, high_noise, bboxes = process_batch(batch)
        predicted_image, high_noise_volume, low_noise_volume, overlap = append_prediction(
            low_noise=low_noise.astype(np.float32),
            high_noise=high_noise.astype(np.float32),
            bboxes=bboxes, overlap=overlap, channel_weights=channel_weights,
            predicted_image=predicted_image, predicted_high_noise=high_noise_volume, predicted_low_noise=low_noise_volume,
        )
    
    return predicted_image, low_noise_volume, high_noise_volume, overlap


def sliding_window_inferer(
    model,
    input_image: np.ndarray,  # should be of size (D, H, W)
    patch_size: Union[tuple, int],
    strides: Union[tuple, int] = 16,
    batch_size: int = 64, store_seperate_channels: bool = False
):
    """Perform inference on sliding patches of the full image.

    Args:
        ort_session (ort.InferenceSession): ONNX InferenceSession that performs the inference.
        input_image (np.ndarray): Image to perform inference on.
        patch_size (Union[tuple, int]: Patch dimensions.
        strides (Union[tuple, int], optional): Stride sizes. Defaults to 16.
        batch_size (int, optional): Size of the batched patches. Defaults to 64.

    Raises:
        ValueError: If the input image is not 3-dimensional.
        RuntimeError: If the predicted patches don't cover the whole image.

    Returns:
        np.ndarray: Predicted image with the same shape as the input image.
    """



    if len(input_image.shape) != 3:
        raise ValueError(
            f"Expected 3D input shape (D, H, W), instead received {input_image.shape}"
        )

    # get channel weights
    channel_weights = get_channel_weighting(input_image)

    model.cuda()
    model.eval()

    patch_size = _check_tuple(patch_size, 3)

    patches = SlidingPatches(input_image, strides=strides, patch_size=patch_size)
    sliding_loader = batch_generator(patches, batch_size)
    def process_batch(batch_data):
        batch, bboxes = batch_data
        with torch.no_grad():
            low_noise = model.predict_low_noise(torch.tensor(batch.astype(np.float32)).cuda()).cpu()[:, 0].numpy()
            high_noise = model.predict_high_noise(torch.tensor(batch.astype(np.float32)).cuda()).cpu()[:, 0].numpy()
        return low_noise, high_noise, bboxes

    predicted_image = np.zeros_like(input_image).astype(np.float32)
    overlap = np.zeros_like(input_image).astype(np.float32)

    predicted_image, low_noise_volume, high_noise_volume, overlap = do_inference_no_threads(process_batch, sliding_loader, patches, batch_size, channel_weights, predicted_image, overlap, low_noise_volume=np.zeros_like(input_image).astype(np.float32) if store_seperate_channels else None, high_noise_volume=np.zeros_like(input_image).astype(np.float32) if store_seperate_channels else None)


    if np.sum(overlap == 0.0) != 0:
        raise RuntimeError("The patches don't cover the whole image.")

    predicted_image /= overlap
    if store_seperate_channels:
        low_noise_volume /= overlap
        high_noise_volume /= overlap


    return predicted_image, low_noise_volume, high_noise_volume, float(channel_weights[0])


