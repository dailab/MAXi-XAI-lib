from typing import Tuple
import numpy as np


def get_width_height(x: np.ndarray, channels_first: bool) -> Tuple[int, int]:
    if x.ndim == 2:
        return (
            (x.shape[0], x.shape[1]) if x.shape[0] > 1 else (x.shape[1], np.inf)
        )
    elif x.ndim == 3:
        return (
            (x.shape[1], x.shape[2])
            if channels_first
            else (x.shape[0], x.shape[1])
        )
    elif x.ndim == 4:
        return (
            (x.shape[2], x.shape[3])
            if channels_first
            else (x.shape[1], x.shape[2])
        )
    else:
        raise ValueError(
            "Currently, can't extract widths and height dimensions of given matrix!"
        )
