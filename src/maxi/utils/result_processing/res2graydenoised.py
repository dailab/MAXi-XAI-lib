from typing import Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure

from maxi.utils.transformations import rgb2gray, intensify_pixels


def transform_res2graydenoised(
    image: np.ndarray,
    in_range: Tuple[int, int] = None,
    out_range: Tuple[int, int] = (0, 255),
    threshold_factor: float = 1.3,
    gaussian_filter_sigma: float = 1.1,
    *args,
    **kwargs,
) -> np.ndarray:
    """[summary]

    Args:
        image (np.ndarray):
            Perturbed image.
        in_range (Tuple[int, int], optional):
            Pixel value range of the perturbed image. Defaults to None.
        out_range (Tuple[int, int], optional):
            Pixel value range of the heatmap. Defaults to (0, 255).
        invert_pixels (bool, optional):
            Invert pixel values by subtracting (255, 255, 255) by\
            the perturbed image.
        gaussian_filter_sigma (float, optional):
            Standard deviation for gaussian kernel. Defaults to 1.25.

    Returns:
        np.ndarray: Image heatmap in BGR.
    """
    if in_range and out_range:
        tmp = exposure.rescale_intensity(image, in_range=in_range, out_range=out_range)

    tmp = intensify_pixels(tmp, max_value=max(out_range))

    if tmp.ndim > 3:
        tmp = np.squeeze(tmp, 0)

    if tmp.ndim > 2 and tmp.shape[2] != 1:  # number of channels
        tmp = rgb2gray(tmp)

    # thresholding
    threshold = tmp.mean() * threshold_factor
    tmp[tmp > threshold] = max(out_range)
    tmp[tmp <= threshold] = min(out_range)

    return gaussian_filter(tmp, sigma=gaussian_filter_sigma)
