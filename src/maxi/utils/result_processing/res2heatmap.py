from typing import Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage import exposure

from maxi.utils.transformations import rgb2gray, intensify_pixels


def get_heatmap(
    mask: np.ndarray,
    *args,
    **kwargs,
) -> np.ndarray:
    """[summary]

    Args:
        mask (np.ndarray): 
            Mask extracted from a perturbed image \
            of shape [w, h].

    Returns:
        np.ndarray: Heatmap in color range 'COLORMAP_JET'.
    """
    map_img = np.uint8(mask)
    return cv2.applyColorMap(map_img, cv2.COLORMAP_JET)


def transform_res2heatmap(
    image: np.ndarray,
    in_range: Tuple[int, int] = None,
    out_range: Tuple[int, int] = (0, 255),
    invert_pixels: bool = True,
    gaussian_filter_sigma: float = 0.75,
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
    assert image.ndim > 2

    tmp = np.abs(image)

    if in_range and out_range:
        tmp = exposure.rescale_intensity(tmp, in_range=in_range, out_range=out_range)

    tmp = intensify_pixels(tmp, max_value=max(out_range))

    if tmp.ndim > 3:
        tmp = np.squeeze(tmp, 0)

    if tmp.shape[2] != 1:  # number of channels
        tmp = rgb2gray(tmp)

    if invert_pixels:
        tmp = max(out_range) - tmp

    tmp = gaussian_filter(tmp, sigma=gaussian_filter_sigma)

    return get_heatmap(tmp)
