import numpy as np
from random import uniform


def generate_rand_image(original_image: np.ndarray) -> np.ndarray:
    """Generates an array where each index is drawn from the range between lowest value in the image\
        and the respective pixel value.

    Args:
        original_image (np.ndarray): Reference image in [width, height, channels]

    Returns:
        np.ndarray: Random image drawn from respective pixel range.
    """
    flattened_image = original_image.flatten()
    low_high = [None] * len(flattened_image)

    for i, entry in enumerate(np.nditer(flattened_image)):
        low_high[i] = {"a": np.min(original_image), "b": entry}

    flatten_res = np.array([uniform(**lh) for lh in low_high], dtype=original_image.dtype)

    if original_image.ndim > 1:
        return np.ascontiguousarray(flatten_res.reshape(original_image.shape))
    return flatten_res
