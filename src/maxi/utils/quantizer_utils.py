from typing import Any
import numpy as np
import tensorflow as tf
import torch


def sigmoid(z, factor: float = 6):
    return 1 / (1 + np.exp(-z) * factor)


def identity(data: Any) -> Any:
    return data


def calculate_confidence(segmentation_mask: np.ndarray, max_pixel_value: int = 255) -> float:
    if type(segmentation_mask) is np.ndarray:
        return np.mean(segmentation_mask) / max_pixel_value
    elif type(segmentation_mask) is tf.Tensor:
        return tf.reduce_mean(segmentation_mask) / max_pixel_value
    elif type(segmentation_mask) is torch.Tensor:
        return torch.mean(segmentation_mask) / max_pixel_value
    else:
        raise NotImplementedError(
            f"This data type is not supported for confidence calculation! ({type(segmentation_mask)})"
        )
