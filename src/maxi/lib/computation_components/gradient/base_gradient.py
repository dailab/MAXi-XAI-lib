"""Base Gradient Class"""

__all__ = ["BaseGradient"]

from abc import ABC, abstractmethod

from numpy import ndarray

from ...loss.base_explanation_model import BaseExplanationModel


class BaseGradient(ABC):
    def __init__(self, loss: BaseExplanationModel) -> None:
        """**Abstract Class**: Base Class for Gradient Calculation Methods

        Args:
            loss (BaseExplanationModel): Explanation method's specific class instance (loss function).
        """
        self.loss = loss

    @abstractmethod
    def __call__(self, data: ndarray, *args, **kwds) -> ndarray:
        """Gradient Computation Method

        Description:
            Starts one iteration of the gradient computation.

        Note:
            This method needs to be implemented for any gradient method to \
            be used with this library.

        Args:
            data (ndarray): Perturbation matrix in [width, height, channels]

        Raises:
            NotImplementedError: Method has to be implemented by the user.

        Returns:
            ndarray: Matrix containing the computed gradient of shape [width, height, channels]
        """
        raise NotImplementedError
