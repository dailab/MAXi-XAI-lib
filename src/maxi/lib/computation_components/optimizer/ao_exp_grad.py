"""Adaptive Optimistic Gradient Optimizer Module"""

__all__ = ["AoExpGradOptimizer"]

from typing import Callable

import numpy as np
import numpy.linalg as lina
from scipy.special import lambertw
from scipy.optimize import OptimizeResult

from .base_optimizer import BaseOptimizer
from ..gradient.base_gradient import BaseGradient
from ...loss.base_explanation_model import BaseExplanationModel
from ....utils.general import to_numpy


class AOExpGrad(object):
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        func_p: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        l1: float = 1.0,
        l2: float = 1.0,
        eta: float = 1.0,
    ):
        self.func = func
        self.func_p = func_p
        self.x = np.zeros(shape=x0.shape)
        self.x[:] = x0
        self.y = np.zeros(shape=x0.shape)
        self.y[:] = x0
        self.d = self.x.size
        self.lower = lower
        self.upper = upper
        self.eta = eta
        self.lam = 0.0
        self.t = 0.0
        self.beta = 0
        self.l1 = l1
        self.l2 = l2
        self.h = np.zeros(shape=self.x.shape)

    def update(self) -> np.ndarray:
        self.t += 1.0
        self.beta += self.t
        g = self.func_p(self.y)
        self.step(g)
        self.h[:] = g
        return self.y

    def step(self, g):
        self.update_parameters(g)
        self.md(g)

    def update_parameters(self, g):
        self.lam += (self.t * lina.norm((g - self.h).flatten(), ord=np.inf)) ** 2

    def md(self, g):
        beta = 1.0 / self.d
        alpha = np.sqrt(self.lam) / np.sqrt(np.log(self.d+1)) * self.eta
        if alpha == 0.0:
            alpha += 1e-6
        z = (np.log(np.abs(self.x) / beta + 1.0)) * np.sign(self.x) - (
            self.t * g - self.t * self.h + (self.t + 1) * g
        ) / alpha
        x_sgn = np.sign(z)
        if self.l2 == 0.0:
            x_val = beta * np.exp(np.maximum(np.abs(z) - self.l1 * (self.t+1) / alpha, 0.0)) - beta
        else:
            a = beta
            b = self.l2 * (self.t + 1) / alpha
            c = np.minimum(self.l1 * (self.t + 1) / alpha - np.abs(z), 0.0)
            abc = -c + np.log(a * b) + a * b
            x_val = (
                np.where(
                    abc >= 15.0,
                    np.log(abc) - np.log(np.log(abc)) + np.log(np.log(abc)) / np.log(abc),
                    lambertw(np.exp(abc), k=0).real,
                )
                / b
                - a
            )
            # x_val = lambertw(a * b * np.exp(a * b - c), k=0).real / b - a
        y = x_sgn * x_val
        self.x = np.clip(y, self.lower, self.upper)
        self.y = (self.t / self.beta) * self.x + ((self.beta - self.t) / self.beta) * self.y


class AoExpGradOptimizer(BaseOptimizer):
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
        **kwargs
    ):
        """Adaptive Optimistic Gradient Optimizer

        Args:
            loss (BaseExplanationModel): Subclass instance of ``BaseExplanationModel`` - an explanation methods' \
                loss function.
            gradient (BaseGradient): Subclass instance of ``BaseGradient`` - a particular gradient method.
            lower (np.ndarray): Lower bound for each entry of the matrix. Has to be of the same shape as the \
                target image.
            upper (np.ndarray): Upper bound for each entry of the matrix. Has to be of the same shape as the \
                target image.
            l1 (float, optional): L1 regularization coefficient. Defaults to 0.5.
            l2 (float, optional): L2 regularization coefficient. Defaults to 0.5.
            eta (float, optional): Coefficient to calculate $$\alpha$$. Defaults to 1.0.
        """
        self.eta, self.l1, self.l2 = eta, l1, l2

        alg = AOExpGrad(
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
