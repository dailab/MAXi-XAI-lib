"""Adaptive Exponentiated Gradient Optimizer Module"""

__all__ = ["AdaExpGradOptimizer"]

from typing import Callable

import numpy as np
import numpy.linalg as lina
from scipy.special import lambertw
from scipy.optimize import OptimizeResult

from .base_optimizer import BaseOptimizer
from ..gradient.base_gradient import BaseGradient
from ...loss.base_explanation_model import BaseExplanationModel

from ....utils.general import to_numpy


class AdaExpGrad(object):
    FEV_PER_ITER = 1

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        func_p: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        eta: float = 1.0,
        l1: float = 0,
        l2: float = 0,
    ):
        """Encapsulates the AdaExpGrad optimizer

        Args:
            func (Callable[[np.ndarray], float]): Loss function
            func_p (Callable[[np.ndarray], np.ndarray]): Gradient function
            x0 (np.ndarray): Starting value
            lower (np.ndarray): Lower bound
            upper (np.ndarray): Upper bound
            eta (float, optional): Coefficient to calculate _alpha_. Defaults to 1.0.
            l1 (float, optional): L1 regularization coefficient. Defaults to 0.
            l2 (float, optional): L2 regularization coefficient. Defaults to 0.
        """
        self.func = func
        self.func_p = func_p
        self.x = np.array(x0)
        self.x[:] = x0
        self.d = self.x.size
        self.lower = lower
        self.upper = upper
        self.eta = eta
        self.lam = 1.0
        self.t = 0.0
        self.y = np.array(x0)
        self.y[:] = x0
        self.l1 = l1
        self.l2 = l2

    def update(self) -> np.ndarray:
        self.t = self.t + 1.0
        g = self.func_p(self.x)
        self.step(g)
        self.y[:] = 1.0 / self.t * self.x + (1 - 1.0 / self.t) * self.y
        return self.y

    def evaluate(self, x: np.ndarray) -> float:
        return self.func(x)

    def step(self, g: np.ndarray) -> None:
        self.update_parameters(g)
        self.sgd(g)

    def update_parameters(self, g: np.ndarray) -> None:
        self.lam += lina.norm(g.flatten(), ord=np.inf) ** 2
        # self.lam+=g**2

    def sgd(self, g: np.ndarray) -> None:
        beta = 1.0 / self.d
        alpha = np.sqrt(self.lam) * self.eta
        z = (np.log(np.abs(self.x) / beta + 1.0)) * np.sign(self.x) - g / alpha
        x_sgn = np.sign(z)
        if self.l2 == 0:
            x_val = beta * np.exp(np.abs(z) - self.l1 / alpha) - beta
        else:
            a = beta
            b = self.l2 / alpha
            c = self.l1 / alpha - np.abs(z)
            x_val = lambertw(a * b * np.exp(a * b - c), k=0) / b - a
        y = x_sgn * x_val
        self.x = np.clip(y, self.lower, self.upper)


class AdaExpGradOptimizer(BaseOptimizer):
    def __init__(
        self,
        loss: BaseExplanationModel,
        gradient: BaseGradient,
        org_img: np.ndarray,
        x0: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        l1: float = 0.5,
        l2: float = 0.5,
        eta: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        """Encapsulates the Adaptive Exponentiated Gradient Optimizer

        Args:
            loss (BaseExplanationModel): Subclass instance of ``BaseExplanationModel`` - an explanation methods' \
                loss function.
            gradient (BaseGradient): Subclass instance of ``BaseGradient`` - a particular gradient method.
            org_img (np.ndarray): Original target image.
            x0 (np.ndarray): Initial perturbed image of the optimization.
            lower (np.ndarray): Lower bound for each entry of the matrix. Has to be of the same shape as the \
                target image.
            upper (np.ndarray): Upper bound for each entry of the matrix. Has to be of the same shape as the \
                target image.
            l1 (float, optional): L1 regularization coefficient. Defaults to 0.5.
            l2 (float, optional): L2 regularization coefficient. Defaults to 0.5.
            eta (float, optional): Coefficient to calculate $$\alpha$$. Defaults to 1.0.
                
        Configurable args via keyword arguments in the 'ExplanationGenerator':
            - lower (np.ndarray).
            - upper (np.ndarray).
            - eta (float, optional).
            - l1 (float, optional). Defaults to 0.5.
            - l2 (float, optional). Defaults to 0.5.
        """
        self.eta, self.l1, self.l2 = eta, l1, l2

        alg = AdaExpGrad(
            func=loss,
            func_p=gradient,
            x0=x0,
            lower=lower,
            upper=upper,
            eta=self.eta,
            l1=self.l1,
            l2=self.l2,
        )

        super().__init__(loss, gradient, org_img, x0, lower, upper, alg)

    def step(self, *args, **kwargs) -> OptimizeResult:
        """Performs one optimization step.

        Returns:
            OptimizeResult: Represents the optimization result. Furthermore, holds additional information about \
                optimization.
        """
        return super().step(*args, **kwargs)
