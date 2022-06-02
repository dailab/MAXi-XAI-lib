"""Spectral Adaptive Optimistic Gradient Optimizer Module"""

__all__ = ["SpectralAoExpGradOptimizer"]

from scipy.special import lambertw
from scipy.optimize import OptimizeResult
import numpy as np
import numpy.linalg as lina

import maxi.utils.optimizer_utils as opt_utils
from .base_optimizer import BaseOptimizer
from ..gradient.base_gradient import BaseGradient
from ...loss.base_explanation_model import BaseExplanationModel
from ....utils.general import to_numpy


class SpectralAOExpGrad(object):
    def __init__(self, func, func_p, x0, lower, upper, l1=1.0, l2=1.0, eta=1.0, channels_first: bool = True):
        self.func = func
        self.func_p = func_p
        self.x = np.zeros(shape=x0.shape)
        self.x[:] = x0
        self.y = np.zeros(shape=x0.shape)
        self.y[:] = x0
        width, height = opt_utils.get_width_height(x0, channels_first)
        self.d = np.minimum(width, height)
        self.lower = lower
        self.upper = upper
        self.eta = eta
        self.lam = 0.0
        self.t = 0.0
        self.beta = 0
        self.l1 = l1
        self.l2 = l2
        self.h = np.zeros(shape=self.x.shape)

    def update(self):
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
        s = np.linalg.svd(g - self.h, full_matrices=False, compute_uv=False)
        self.lam += (self.t * s[0]) ** 2

    def md(self, g):
        beta = 1.0 / self.d
        alpha = np.sqrt(self.lam) / np.sqrt(np.log(self.d)) * self.eta
        alpha = np.where(alpha == 0.0, 1e-6, alpha)
        u, _x, v = np.linalg.svd(self.x, full_matrices=False, compute_uv=True)
        z = (
            np.matmul(u * (alpha * np.log(_x / beta + 1.0))[..., None, :], v)
            - self.t * g
            + self.t * self.h
            - (self.t + 1) * g
        )
        u, _z, v = np.linalg.svd(z, full_matrices=False, compute_uv=True)
        _y = beta * np.exp(_z / alpha) - beta
        if self.l2 == 0.0:
            x_val = beta * np.exp(np.maximum(np.log(_y / beta + 1.0) - self.l1 * self.t / alpha, 0.0)) - beta
        else:
            a = beta
            b = self.l2 * self.t / alpha
            c = np.minimum(self.l1 * self.t / alpha - np.log(_y / beta + 1.0), 0.0)
            abc = np.log(a * b) + a * b - c
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
        y = np.matmul(u * x_val[..., None, :], v)
        self.x = np.clip(y, self.lower, self.upper)
        self.y = (self.t / self.beta) * self.x + ((self.beta - self.t) / self.beta) * self.y


class SpectralAoExpGradOptimizer(BaseOptimizer):
    def __init__(
        self,
        loss: BaseExplanationModel,
        gradient: BaseGradient,
        org_img: np.ndarray,
        x0: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        channels_first: bool = True,
        l1: float = 0.5,
        l2: float = 0.5,
        eta: float = 1.0,
        *args,
        **kwargs
    ):
        """Spectral Adaptive Optimistic Gradient Optimizer

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
            channels_first (bool, optional): Whether the number of channels comes before the width
                and the height in the dimensions. E.g. (N, C, W, H) => channels first, (N, W, H, C) => channels last.
                Defaults to True.
            l1 (float, optional): L1 regularization coefficient. Defaults to 0.5.
            l2 (float, optional): L2 regularization coefficient. Defaults to 0.5.
            eta (float, optional): Coefficient to calculate $$\alpha$$. Defaults to 1.0. Defaults to 1.0.
        """
        self.eta, self.l1, self.l2 = eta, l1, l2

        alg = SpectralAOExpGrad(
            func=loss,
            func_p=gradient,
            x0=x0,
            lower=lower,
            upper=upper,
            eta=self.eta,
            l1=self.l1,
            l2=self.l2,
            channels_first=channels_first,
        )

        super().__init__(loss, gradient, org_img, x0, lower, upper, alg)

    def step(self, *args, **kwargs) -> OptimizeResult:
        """Performs one optimization step.

        Returns:
            OptimizeResult: Represents the optimization result. Furthermore, holds additional information about \
                optimization.
        """
        return super().step(*args, **kwargs)
