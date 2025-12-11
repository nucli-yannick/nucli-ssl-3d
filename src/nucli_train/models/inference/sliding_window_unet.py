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
    prediction: np.ndarray,
    bboxes: np.ndarray,
    predicted_image: np.ndarray,
    overlap: np.ndarray, 
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
    for i, pred in enumerate(prediction):        
        predicted_image[bboxes[i]] += pred
        overlap[bboxes[i]] += 1

    return predicted_image, overlap

def do_inference_no_threads(process_batch, sliding_loader, patches, batch_size,  predicted_image, overlap):
        
    # Process batches sequentially without ThreadPoolExecutor
    for batch in tqdm(
        sliding_loader, 
        total=int(np.ceil(len(patches) / batch_size)), 
        desc="Slider"
    ):
        prediction, bboxes = process_batch(batch)
        predicted_image,  overlap = append_prediction(prediction,
            bboxes=bboxes, overlap=overlap,
            predicted_image=predicted_image
        )
    
    return predicted_image, overlap


def sliding_window_inferer(
    model,
    input_image: np.ndarray,  # should be of size (D, H, W)
    patch_size: Union[tuple, int],
    strides: Union[tuple, int] = 16,
    batch_size: int = 64
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

    model.cuda()
    model.eval()

    patch_size = _check_tuple(patch_size, 3)

    patches = SlidingPatches(input_image, strides=strides, patch_size=patch_size)
    sliding_loader = batch_generator(patches, batch_size)
    def process_batch(batch_data):
        batch, bboxes = batch_data
        with torch.no_grad():
            pred = model.predict(torch.tensor(batch.astype(np.float32)).cuda()).cpu()[:, 0].numpy()
        return pred, bboxes

    predicted_image = np.zeros_like(input_image).astype(np.float32)
    overlap = np.zeros_like(input_image).astype(np.float32)

    predicted_image, overlap = do_inference_no_threads(process_batch, sliding_loader, patches, batch_size, predicted_image, overlap)


    if np.sum(overlap == 0.0) != 0:
        raise RuntimeError("The patches don't cover the whole image.")

    predicted_image /= overlap


    return predicted_image


