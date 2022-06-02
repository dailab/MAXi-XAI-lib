"""Gradient Estimation Module"""

__all__ = ["URVGradientEstimator"]

from typing import Callable
from numpy.linalg import norm

import numpy as np
import time

from .base_gradient import BaseGradient
from ....utils.general import to_numpy


class URVGradientEstimator(BaseGradient):
    def __init__(
        self,
        loss: Callable[[np.ndarray], float],
        img_size: int,
        mu: float = None,
        sample_num: int = 100,
        batch_num: int = 1,
        batch_mode: bool = True,
        channels_first: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Uniform Random Vector-based Gradient Estimation

        Args:
            loss (Callable[[np.ndarray], float]): Explanation method's specific class instance (loss function).
            img_size (int): Number of entries in the matrix. In numpy: ``ndarray.size``.
            mu (float, optional): Variance of the normal distribution. Defaults to None.
            sample_num (int, optional): Number of estimation steps. Defaults to 100.
            batch_num (int, optional): Number of batches to split the samples in.
                The batch size is calculated: 'sample_num // batch_num'.
                Parameter is ignored when 'batch_mode' is False. Defaults to 1.
            batch_mode (bool, optional): When enabled, the desired number of samples will be split into
                'batch_num' batches. Might decrease the calculation time significantly.
                Defaults to True.
            channels_first (bool, optional): Whether the number of channels comes before the width
                and the height in the dimensions. E.g. (N, C, W, H) => channels first, (N, W, H, C) => channels last.
                Defaults to False.
        """
        super().__init__(loss=loss)

        self.t, self.d, self.q = sample_num, img_size, sample_num
        self.mu = mu

        self.randomState = np.random.RandomState()

        self._batch_mode = batch_mode
        if batch_mode:
            self._batch_num, self._batch_size = batch_num, sample_num // batch_num
            self.q = self._batch_size
        self._c_dim = 1 if channels_first else -1

    def __call__(self, data: np.ndarray):
        return self.batched_estimate(data) if self._batch_mode else self.non_batched_estimate(data)

    def non_batched_estimate(self, data: np.ndarray) -> np.ndarray:
        """Method for estimating the gradient

        Args:
            data (np.ndarray): Perturbation matrix in [width, height, channels].

        Returns:
            np.ndarray: Matrix containing the computed gradient of shape [width, height, channels].
        """
        mu = 1.0 / np.sqrt(self.t) / np.sqrt(self.d) if self.mu is None else self.mu
        tilde_f_x_r = to_numpy(self.loss.get_loss(data))

        g = np.zeros(data.shape)
        for _ in range(self.q):
            u = self.randomState.normal(size=data.shape)
            u_norm = np.linalg.norm(u)
            u = u / u_norm

            tilde_f_x_l = to_numpy(self.loss.get_loss(data + mu * u))
            g += self.d / mu / self.q * (tilde_f_x_l - tilde_f_x_r) * u
        return g

    def batched_estimate(self, data: np.ndarray) -> np.ndarray:
        """Method for estimating the gradient

        Args:
            data (np.ndarray): Perturbation matrix in [bs, width, height, channels].

        Returns:
            np.ndarray: Matrix containing the computed gradient of shape [bs, width, height, channels].
        """
        mu = 1.0 / self.t / np.sqrt(self.d) if self.mu is None else self.mu
        tilde_f_x_r = to_numpy(self.loss.get_loss(data))

        batched_shape = list(data.shape)
        batched_shape[0] = self.q
        batched_shape = tuple(batched_shape)

        gradients = []
        for _ in range(self._batch_num):
            u = self.randomState.normal(size=batched_shape)

            if data.ndim == 4:
                u_norm = norm(norm(u, axis=self._c_dim), axis=(-2, -1))
                u = np.divide(u, u_norm[:, np.newaxis, np.newaxis, np.newaxis])

                tilde_f_x_l = to_numpy(self.loss.get_loss(data + mu * u))

                factors = self.d / mu / self.q * (tilde_f_x_l - tilde_f_x_r)
                product = np.multiply(factors[:, np.newaxis, np.newaxis, np.newaxis], u)
            elif data.ndim == 2:
                u_norm = norm(u, axis=self._c_dim)
                u = np.divide(u, u_norm[:, np.newaxis])

                tilde_f_x_l = to_numpy(self.loss.get_loss(data + mu * u))

                factors = self.d / mu / self.q * (tilde_f_x_l - tilde_f_x_r)
                product = np.multiply(factors[:, np.newaxis], u)
            elif data.ndim == 3:
                u_norm = norm(norm(u, axis=self._c_dim), axis=-1)
                u = np.divide(u, u_norm[:, np.newaxis, np.newaxis])

                tilde_f_x_l = to_numpy(self.loss.get_loss(data + mu * u))

                factors = self.d / mu / self.q * (tilde_f_x_l - tilde_f_x_r)
                product = np.multiply(factors[:, np.newaxis, np.newaxis], u)
            gradients.append(np.sum(product, axis=0, keepdims=True))

        gradient_sum = np.zeros_like(data, dtype=np.float64)
        for grad in gradients:
            gradient_sum += grad

        return gradient_sum / self._batch_num


class USRVGradientEstimator(BaseGradient):
    def __init__(
        self,
        loss: Callable[[np.ndarray], float],
        img_size: int,
        mu: float = None,
        sample_num: int = 100,
        batch_num: int = 1,
        batch_mode: bool = True,
        channels_first: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Unit Sphere Random Vector-based Gradient Estimation

        Args:
            loss (Callable[[np.ndarray], float]): Explanation method's specific class instance (loss function).
            img_size (int): Number of entries in the matrix. In numpy: ``ndarray.size``.
            mu (float, optional): Variance of the normal distribution. Defaults to None.
            sample_num (int, optional): Number of estimation steps. Defaults to 100.
            batch_num (int, optional): Number of batches to split the samples in.
                The batch size is calculated: 'sample_num // batch_num'.
                Parameter is ignored when 'batch_mode' is False. Defaults to 1.
            batch_mode (bool, optional): When enabled, the desired number of samples will be split into
                'batch_num' batches. Might decrease the calculation time significantly.
                Defaults to True.
            channels_first (bool, optional): Whether the number of channels comes before the width
                and the height in the dimensions. E.g. (N, C, W, H) => channels first, (N, W, H, C) => channels last.
                Defaults to False.
        """
        super().__init__(loss=loss)

        self.t, self.d, self.q = 0, img_size, sample_num
        self.mu = mu

        self.randomState = np.random.RandomState()

        self._batch_mode = batch_mode
        if batch_mode:
            self._batch_num, self._batch_size = batch_num, sample_num // batch_num
            self.q = self._batch_size
        self._c_dim = 1 if channels_first else -1

    def __call__(self, data: np.ndarray):
        return self.batched_estimate(data) if self._batch_mode else self.non_batched_estimate(data)

    def non_batched_estimate(self, data: np.ndarray) -> np.ndarray:
        """Method for estimating the gradient

        Args:
            data (np.ndarray): Perturbation matrix in [width, height, channels].

        Returns:
            np.ndarray: Matrix containing the computed gradient of shape [width, height, channels].
        """
        self.t += 1.0

        mu = 1.0 / np.sqrt(self.t) / np.sqrt(self.d) if self.mu is None else self.mu

        g = np.zeros(data.shape)
        for _ in range(self.q):
            u = self.randomState.normal(size=data.shape)
            u_norm = u / np.linalg.norm(u)

            tilde_f_x_r = to_numpy(self.loss.get_loss(data + mu * u_norm))

            u = self.randomState.normal(size=data.shape)
            u_norm = u / np.linalg.norm(u)

            tilde_f_x_l = to_numpy(self.loss.get_loss(data + mu * u_norm))
            g += self.d / 2 * mu / self.q * (tilde_f_x_l - tilde_f_x_r) * u
        return g

    def batched_estimate(self, data: np.ndarray) -> np.ndarray:
        """Method for estimating the gradient

        Args:
            data (np.ndarray): Perturbation matrix in [bs, width, height, channels].

        Returns:
            np.ndarray: Matrix containing the computed gradient of shape [bs, width, height, channels].
        """
        self.t += 1.0

        mu = 1.0 / self.t / np.sqrt(self.d) if self.mu is None else self.mu

        batched_shape = list(data.shape)
        batched_shape[0] = self._batch_size
        batched_shape = tuple(batched_shape)

        gradients = []
        for _ in range(self._batch_num):
            u = self.randomState.normal(size=batched_shape)

            if data.ndim > 2:
                u_norm = norm(norm(u, axis=self._c_dim), axis=(-2, -1))
                u_norm = np.divide(u, u_norm[:, np.newaxis, np.newaxis, np.newaxis])

                tilde_f_x_r = to_numpy(self.loss.get_loss(data - mu * u_norm))

                u = self.randomState.normal(size=batched_shape)

                u_norm = norm(norm(u, axis=self._c_dim), axis=(-2, -1))
                u_norm = np.divide(u, u_norm[:, np.newaxis, np.newaxis, np.newaxis])

                tilde_f_x_l = to_numpy(self.loss.get_loss(data + mu * u_norm))

                factors = self.d / (2 * mu) / self.q * (tilde_f_x_l - tilde_f_x_r)
                product = np.multiply(factors[:, np.newaxis, np.newaxis, np.newaxis], u)
            else:
                u_norm = norm(u, axis=self._c_dim)
                u_norm = np.divide(u, u_norm[:, np.newaxis])

                tilde_f_x_r = to_numpy(self.loss.get_loss(data + mu * u_norm))

                u = self.randomState.normal(size=batched_shape)

                u_norm = norm(u, axis=self._c_dim)
                u_norm = np.divide(u, u_norm[:, np.newaxis])

                tilde_f_x_l = to_numpy(self.loss.get_loss(data + mu * u_norm))

                factors = self.d / (2 * mu) / self.q * (tilde_f_x_l - tilde_f_x_r)
                product = np.multiply(factors[:, np.newaxis], u)

            gradients.append(np.sum(product, axis=0, keepdims=True))

        gradient_sum = np.zeros_like(data, dtype=np.float64)
        for grad in gradients:
            gradient_sum += grad

        return gradient_sum / self._batch_num
