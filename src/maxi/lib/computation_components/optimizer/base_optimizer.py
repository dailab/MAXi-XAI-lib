"""Base Optimizer Class"""

__all__ = ["BaseOptimizer"]

from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as lina
from scipy.optimize import OptimizeResult

from ..gradient.base_gradient import BaseGradient
from ...loss.base_explanation_model import BaseExplanationModel
from ....utils.general import to_numpy


class BaseOptimizer(ABC):
    def __init__(
        self,
        loss: BaseExplanationModel,
        gradient: BaseGradient,
        org_img: np.ndarray,
        x0: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        algorithm,
    ) -> None:
        """**Abstract Class**: Base Class for Optimizers.

        Description:
            This base class encapsulates the optimization algorithms.

        Args:
            loss (BaseExplanationModel): Subclass instance of ``BaseExplanationModel`` - an explanation methods' \
                loss function.
            gradient (BaseGradient): Subclass instance of ``BaseGradient`` - a particular gradient method.
            org_img (np.ndarray): Original target image in [width, height, channels].
            x0 (np.ndarray): Initial perturbed image to start generation from in [width, height, channels].
            lower (np.ndarray): Lower bound for each entry of the matrix. Has to be of the same shape as the \
             target image.
            upper (np.ndarray): Upper bound for each entry of the matrix. Has to be of the same shape as the \
                target image.
            algorithm: Class containing the optimization algorithm. Has to implement the update method that returns \
                the most recent optimization result (e.g. perturbed image matrix).
        """
        self.loss, self.gradient = loss, gradient
        self.org_image, self.org_shape, self.x0, self.lower, self.upper = (
            org_img,
            org_img.shape,
            x0,
            lower,
            upper,
        )
        self.alg = algorithm
        self.call_count = 0

    @abstractmethod
    def step(self, *args, **kwargs) -> OptimizeResult:
        """Performs one optimization step.

        Description:
            This method has to be implemented by any optimizer class. It should do an update step \
            on the underlying optimization algorithm.

        Raises:
            NotImplementedError: Method has to be implemented by the user.

        Returns:
            OptimizeResult: Represents the optimization result. Furthermore, holds additional information about \
                optimization.
        """
        y = self.alg.update() if self.call_count != 0 else self.x0
        self.call_count = self.call_count + 1
        loss, l1, l2 = (
            to_numpy(self.alg.func(y)),
            lina.norm(y.flatten(), ord=1) * self.l1,
            (lina.norm(y.flatten(), ord=2) ** 2) * self.l2 / 2,
        )
        return OptimizeResult(
            func=(loss + l1 + l2)[0],
            x=y,
            loss=loss[0],
            l1=l1,
            l2=l2,
            nit=self.call_count,
            nfev=self.call_count,
            success=(y is not None),
        )
