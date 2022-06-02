"""Base Explanation Model"""
from abc import ABC, abstractmethod

import numpy as np

from ...data.data_types import InferenceCall, X0_Generator


class BaseExplanationModel(ABC):
    compatible_grad_methods = []

    def __init__(
        self,
        org_img: np.ndarray,
        inference: InferenceCall,
        x0_generator: X0_Generator,
        lower: np.ndarray,
        upper: np.ndarray,
        *args,
        **kwargs
    ) -> None:
        """**Abstract Class**: Base Class for Explanation Methods.

        Description:
            The explanation models have to be implemented as a loss function. This class provides \
            the essential interface for loss functions.

        Args:
            org_img (np.ndarray): Original target image in [width, height, channels].
            inference (InferenceCall): The inference method of an external prediction entity.
            x0_generator (X0_Generator): Method to generate the initial object of optimization.
            lower (np.ndarray): Lower bound for the object of optimization. Has to be of same shape as org_img.
            upper (np.ndarray): Lower bound for the object of optimization. Has to be of same shape as org_img.
        """
        assert org_img.shape == lower.shape and org_img.shape == upper.shape

        self.org_img, self.inference = org_img, inference
        self.inference = inference
        self._x0_generator, self._lower, self._upper = x0_generator, lower, upper

    @abstractmethod
    def get_loss(self, data: np.ndarray, *args, **kwargs) -> float:
        """Computes the loss value for the given data

        Args:
            data (np.ndarray): An image in [width, height, channels].

        Raises:
            NotImplementedError: Method has to be implemented.

        Returns:
            float: The actual loss value.
        """
        raise NotImplementedError
