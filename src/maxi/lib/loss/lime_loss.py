"""Lime Loss Function Module"""

__all__ = ["LimeLoss"]

from typing import Callable, List, Union

import numpy as np

from .base_explanation_model import BaseExplanationModel
from ..computation_components.gradient import (
    LimeGradient,
    URVGradientEstimator,
    USRVGradientEstimator,
)
from ...data.data_types import InferenceCall
from ...utils.general import to_numpy
from ...utils.superpixel_handler import SuperpixelHandler


class LimeLoss(BaseExplanationModel):
    compatible_grad_methods = [
        LimeGradient,
        URVGradientEstimator,
        USRVGradientEstimator,
    ]
    _x0_generator = lambda x: np.full(x.shape, 0)
    # _x0_generator = lambda x: x

    def __init__(
        self,
        org_img: np.ndarray,
        inference: InferenceCall,
        n_samples: int,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
        *args,
        **kwargs,
    ):
        if not hasattr(self, "superpixel_handler"):
            self._init_lower_upper(lower, upper, org_img)
            self._valid_check_lower_upper(org_img)

        super().__init__(
            org_img=org_img,
            inference=inference,
            x0_generator=LimeLoss._x0_generator,
            lower=self._lower,
            upper=self._upper,
        )
        self.n_samples = n_samples
        if not hasattr(self, "superpixel_handler"):
            self._setup_loss_constants()

    def _init_lower_upper(
        self,
        lower: Union[None, np.ndarray],
        upper: Union[None, np.ndarray],
        org_img: np.ndarray,
    ):
        DEFAULT_LB_UB = {
            "lower": np.full(org_img.shape, -5000.0, np.float32),
            # "lower": np.zeros(org_img.shape),
            # "upper": to_numpy(org_img),
            "upper": np.full(org_img.shape, 5000.0, np.float32),
        }

        self._lower = DEFAULT_LB_UB["lower"] if lower is None else lower
        self._upper = DEFAULT_LB_UB["upper"] if upper is None else upper

    def _valid_check_lower_upper(self, org_img: np.ndarray) -> None:
        assert (
            type(self._lower) is np.ndarray
        ), "Invalid lower bound given for optimization"
        assert (
            type(self._upper) is np.ndarray
        ), "Invalid upper bound given for optimization"

        if not hasattr(self, "superpixel_handler"):
            if self._lower.shape != org_img.shape:
                raise ValueError(
                    "Got lower bound matrix of different shape than the"
                    f" input image. ({self._lower.shape} != {org_img.shape})"
                )

            if self._upper.shape != org_img.shape:
                raise ValueError(
                    "Got upper bound matrix of different shape than the"
                    f" input image.({self._upper.shape} != {org_img.shape})"
                )

    @staticmethod
    def _get_indices_of_nonzero_entries(data: np.ndarray) -> np.ndarray:
        return np.where(data > 0)[0]
        # return np.where(data > 0)[0]

    def _setup_locality_measure(self):
        def dist(x: np.ndarray, y: np.ndarray):
            # return np.linalg.norm(np.linalg.norm(z, axis=(1)), axis=(-2, -1))
            return np.linalg.norm(x - y, axis=(-2, -1))

        sigma = max(dist(self._stacked_org_img, self.org_img_plus_Z))

        # exponential kernel
        self.locality_measure = lambda x, y: np.exp(-dist(x, y) ** 2 / sigma ** 2)

    def _setup_loss_constants(self):
        self.non_zero_indices = LimeLoss._get_indices_of_nonzero_entries(
            self.org_img.flatten()
        )

        # set of samples
        self.Z = self._draw_perturbations(self.non_zero_indices, self.n_samples)

        # match org img matrix with Z matrix
        squeezed_org_img = (
            self.org_img if self.org_img.shape[0] > 1 else self.org_img.squeeze(0)
        )
        self._stacked_org_img = np.array(
            [squeezed_org_img for _ in range(self.n_samples)]
        )

        assert self._stacked_org_img.shape == self.Z.shape

        # flatten over 0th axis, from e.g. (2, 3, 28, 28) to (2, 3*28*28)
        self.flatten_Z = self.Z.reshape(self.Z.shape[0], np.prod(self.Z.shape[1:]))
        self.org_img_plus_Z = (self._stacked_org_img + self.Z).clip(
            0, self.org_img.max()
        )

        self._setup_locality_measure()

        self.Pi_Z: np.ndarray = self.locality_measure(
            self._stacked_org_img, self.org_img_plus_Z
        )

        target_idx = to_numpy(self.inference(self.org_img)).argmax()
        self.pred = to_numpy(self.inference(self.org_img_plus_Z))[:, target_idx]

    def _draw_perturbations(
        self, non_zero_indices: np.ndarray, n_samples
    ) -> np.ndarray:
        return np.array(
            [
                self._draw_one_perturbation(non_zero_indices).squeeze(0)
                for _ in range(n_samples)
            ]
        )

    def _draw_one_perturbation(self, non_zero_indices: np.ndarray) -> np.ndarray:
        self.randomState = np.random.RandomState()

        k = self.randomState.randint(low=1, high=len(non_zero_indices))
        indices = np.random.choice(non_zero_indices, k, replace=False)
        indices = np.sort(indices)

        perturbation = np.zeros(self.org_img.flatten().shape, dtype=np.float64)
        values = np.take(self.org_img.flatten(), indices)
        np.put(perturbation, indices, values, mode="raise")
        return perturbation.reshape(self.org_img.shape)

    def get_loss(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        return (
            1
            / self.n_samples
            * (self.Pi_Z.dot(self.pred - (self.flatten_Z.dot(data.flatten()))) ** 2)
        ).reshape((1, 1))


class SuperpixelLimeLoss(LimeLoss):
    def __init__(
        self,
        org_img: np.ndarray,
        inference: InferenceCall,
        n_samples: int,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
        superpixel_handler: SuperpixelHandler = None,
    ):
        self.superpixel_handler = superpixel_handler
        if superpixel_handler:
            self._init_lower_upper(
                lower, upper, self.superpixel_handler.ones_weight_vector
            )
            self._valid_check_lower_upper(org_img)

        super().__init__(
            org_img=org_img,
            inference=inference,
            x0_generator=LimeLoss._x0_generator,
            lower=self._lower,
            upper=self._upper,
            n_samples=n_samples,
        )
        self._setup_loss_constants()

    def _init_lower_upper(
        self,
        lower: Union[None, np.ndarray],
        upper: Union[None, np.ndarray],
        org_img: np.ndarray,
    ):
        DEFAULT_LB_UB = {
            "lower": np.full(
                (self.superpixel_handler.num_superpixels), 0.0, np.float32
            ),
            "upper": np.full(
                (self.superpixel_handler.num_superpixels), 5000.0, np.float32
            ),
        }

        self._lower = DEFAULT_LB_UB["lower"] if lower is None else lower
        self._upper = DEFAULT_LB_UB["upper"] if upper is None else upper

    @staticmethod
    def _get_indices_of_nonzero_entries(sp_label_images: np.ndarray) -> np.ndarray:
        return np.array(
            [
                i
                for i, label_image in enumerate(sp_label_images)
                if label_image.max() > 0
            ]
        )

    def _setup_locality_measure(self):
        def dist(x: np.ndarray, y: np.ndarray):
            # return np.linalg.norm(np.linalg.norm(z, axis=(1)), axis=(-2, -1))
            axis = (-1) if x.ndim == 2 else (-2, -1)
            return np.linalg.norm(x - y, axis=axis)

        sigma = max(dist(self._stacked_org_img, self.org_img_plus_Z))

        # exponential kernel
        self.locality_measure = lambda x, y: np.exp(-dist(x, y) ** 2 / sigma ** 2)

    def _setup_loss_constants(self):
        self.non_zero_indices = SuperpixelLimeLoss._get_indices_of_nonzero_entries(
            self.superpixel_handler.label_images
        )

        # samples of shape (n_superpixels,)
        # self.Z of shape (n_samples, n_superpixels)
        # set of samples
        self.Z = self._draw_perturbations(self.non_zero_indices, self.n_samples)

        # match org img matrix with Z matrix
        self._stacked_org_img = np.array(
            [self.superpixel_handler.ones_weight_vector for _ in range(self.n_samples)]
        )

        assert self._stacked_org_img.shape == self.Z.shape

        # flatten over 0th axis, from e.g. (2, 3, 28, 28) to (2, 3*28*28)
        self.flatten_Z = (
            self.Z.reshape(self.Z.shape[0], np.prod(self.Z.shape[1:]))
            if len(self.Z.shape) > 2
            else self.Z
        )
        self.org_img_plus_Z = (self._stacked_org_img - self.Z).clip(
            0, self.org_img.max()
        )

        # calculate locality measure
        self._setup_locality_measure()
        self.Pi_Z: np.ndarray = self.locality_measure(
            self._stacked_org_img, self.org_img_plus_Z
        )

        target_idx = to_numpy(self.inference(self.org_img)).argmax()
        org_plus_Z_images = self.superpixel_handler.generate_imgs_from_weight_vectors(
            self.org_img_plus_Z
        )
        self.pred = to_numpy(self.inference(org_plus_Z_images))[:, target_idx]

    def _draw_one_perturbation(self, non_zero_indices: np.ndarray) -> np.ndarray:
        self.randomState = np.random.RandomState()

        # draw random indices from non_zero_indices
        # non_zero_indices consist of superpixels that
        # contain at least one non-zero entry
        k = self.randomState.randint(low=1, high=len(non_zero_indices))
        indices = np.random.choice(non_zero_indices, k, replace=False)
        indices = np.sort(indices)

        # draw random weights for non_zero_indices
        weight_vector = np.zeros(
            (self.superpixel_handler.num_superpixels), dtype=np.float64
        )
        random_weights = self.randomState.uniform(low=0.0, high=1.0, size=k)
        np.put(weight_vector, indices, random_weights, mode="raise")
        return np.expand_dims(weight_vector, 0)
        # return self.superpixel_handler.generate_img_from_weight_vector(weight_vector)

    def get_loss(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        return (
            1
            / self.n_samples
            * (self.Pi_Z.dot(self.pred - (self.flatten_Z.dot(align_dims(data)))) ** 2)
        ).reshape((1, 1))


def align_dims(data: np.ndarray) -> np.ndarray:
    return data.reshape(data.shape[-1]) if data.ndim == 2 else data
