from typing import Tuple, Union

import cv2
import numpy as np

from scipy.ndimage import median_filter, gaussian_filter
from skimage import exposure

from maxi.data.data_types import MetaData
from maxi.utils.transformations import (
    reverse_normalize,
    rgb2gray,
    intensify_pixels,
)


def transform_res2contours(
    image: np.ndarray,
    original_image: np.ndarray,
    is_image_normalized: bool = False,
    in_range: Tuple[int, int] = None,
    out_range: Tuple[int, int] = (0, 255),
    gaussian_filter_sigma: float = 1.25,
    threshold_factor: float = 1.05,
    color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 1,
    _meta_data: MetaData = {},
    *args,
    **kwargs
) -> np.ndarray:
    """Draws bounding boxes around most significant parts.

    Args:
        image (np.ndarray):
            Perturbed image.
        original_image (np.ndarray):
            Original region.
        is_image_normalized (bool, optional):
            If the original image was normalized. Defaults to False.
            (If true, original copy has to be taken from the meta data dict) 
        in_range (Tuple[int, int], optional):
            Pixel value range of the perturbed image. Defaults to None.
        out_range (Tuple[int, int], optional):
            Pixel value range of the transformed image. Defaults to (0, 255).
        gaussian_filter_sigma (float, optional):
            Standard deviation for gaussian kernel. Defaults to 1.25.
        threshold_factor (float, optional):
            Thresholding factor controlling the thresholding value. Defaults to 1.05.
        color (Tuple[int, int, int], optional):
            BGR color tuple. Defaults to (0, 255, 0).
        line_thickness (int, optional):
            Line thickness for the bounding boxes. Defaults to 1.
        _meta_data (MetaData, optional):
            Dictionary containing image meta data. Defaults to {}.

    Returns:
        Tuple[np.ndarray]: Annotated image with bounding boxes.

    Note:
        Requires the meta data dictionary to reference the original image\
        in order to annotate it with bounding boxes.
    """
    tmp = np.abs(image)

    if in_range and out_range:
        tmp = exposure.rescale_intensity(tmp, in_range=in_range, out_range=out_range)

    tmp = intensify_pixels(tmp, max_value=max(out_range))

    denoised = gaussian_filter(max(out_range) - rgb2gray(tmp.astype(np.uint8)), gaussian_filter_sigma)
    thresholded = cv2.threshold(
        denoised,
        denoised.mean() * threshold_factor,
        maxval=255,
        type=cv2.THRESH_TOZERO,
    )[1]

    conts = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if is_image_normalized:
        assert "original_image" in _meta_data, "Meta data dictionary must contain the key 'original_image'!"
        original_image = _meta_data["original_image"].copy()

    return cv2.drawContours(original_image, conts[0], -1, color=color, thickness=line_thickness)
