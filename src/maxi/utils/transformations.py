import numpy as np


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalizes between -1 to 1

    Args:
        image (np.ndarray): Original image

    Returns:
        np.ndarray: Normalized image
    """
    return (image - image.mean()) / image.max()


def reverse_normalize(normalized_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        normalized_image (np.ndarray): [description]
        original_image (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    result = normalized_image * original_image.max() + original_image.mean()
    return result.astype(original_image.dtype)


# from -1 to 1
# def normalize(image: np.ndarray):
#     return 2*(image - np.min(image))/np.ptp(image)-1

# def reverse_normalize(normalized_image: np.ndarray, original_image: np.ndarray):
#     return (normalized_image+1)*np.ptp(original_image) / 2 + np.min(original_image)


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def intensify_pixels(image: np.ndarray, max_value: int = 255) -> np.ndarray:
    reduced_mean = image - image.mean()
    reduced_min = reduced_mean - reduced_mean.min()
    divided_max = reduced_min / reduced_min.max()
    return divided_max * max_value
