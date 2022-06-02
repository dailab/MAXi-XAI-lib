"""Identity Quantizer Method"""
from typing import Any
from nptyping import NDArray

import numpy as np

from .base_quantizer import BaseQuantizer


class IdentityMethod(BaseQuantizer):
    def __init__(self) -> None:
        """ The IdentityMethod takes an arbitrary prediction and returns it without further processing steps. \
            This method should only be used when the returned prediction from the model is already in a format \
            compatible with the intended explanation method.
        """
        super().__init__()

    def __call__(self, prediction: np.ndarray, *args, **kwargs) -> NDArray[Any]:
        """Returns the resulting annotation as it is.

        Args:
            prediction (np.ndarray): Arbitrary prediction.

        Returns:
            NDArray[Any]: The same arbitrary prediction from the input.
        """
        return super().__call__(prediction)
