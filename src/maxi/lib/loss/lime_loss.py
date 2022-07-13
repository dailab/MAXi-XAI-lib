"""Lime Loss Function Module"""

__all__ = ["LimeLoss"]

from typing import Callable, Union

import numpy as np

from .base_explanation_model import BaseExplanationModel
from ..computation_components.gradient import (
    LimeGradient,
    URVGradientEstimator,
    USRVGradientEstimator,
)
from ...data.data_types import InferenceCall
from ...utils.general import to_numpy


class LimeLoss(BaseExplanationModel):
    compatible_grad_methods = [LimeGradient, URVGradientEstimator, USRVGradientEstimator]
    _x0_generator = lambda x: np.full(x.shape, 0)

    def __init__(
        self,
        org_img: np.ndarray,
        inference: InferenceCall,
        n_samples: int,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
    ):
        self._init_lower_upper(lower, upper, org_img)
        super().__init__(
            org_img=org_img,
            inference=inference,
            x0_generator=LimeLoss._x0_generator,
            lower=self._lower,
            upper=self._upper,
        )
        self.n_samples = n_samples
        self._setup_loss_constants()

    def _init_lower_upper(
        self,
        lower: Union[None, np.ndarray],
        upper: Union[None, np.ndarray],
        org_img: np.ndarray,
    ):
        DEFAULT_LB_UB = {
            "lower": np.full(org_img.shape, -1.0, np.float32),
            # "lower": np.zeros(org_img.shape),
            # "upper": to_numpy(org_img),
            "upper": np.full(org_img.shape, 1.0, np.float32),
        }

        self._lower = DEFAULT_LB_UB["lower"] if lower is None else lower
        self._upper = DEFAULT_LB_UB["upper"] if upper is None else upper

        assert (
            type(self._lower) is np.ndarray
        ), "Invalid lower bound given for optimization"
        assert (
            type(self._upper) is np.ndarray
        ), "Invalid upper bound given for optimization"

        if self._lower.shape != org_img.shape:
            raise ValueError(
                "Got lowwer bound matrix of different shape than the"
                f" input image. ({self._lower.shape} != {org_img.shape})"
            )

        if self._upper.shape != org_img.shape:
            raise ValueError(
                "Got upper bound matrix of different shape than the"
                f" input image.({self._upper.shape} != {org_img.shape})"
            )
    
    @staticmethod
    def _generate_binary_representation(data: np.ndarray) -> np.ndarray:
        return np.where(data > 0, 1, 0)
    
    @staticmethod
    def _get_indices_of_nonzero_entries(data: np.ndarray) -> np.ndarray:
        return np.where(data > 0)[0]
        # return np.where(data > 0)[0]
    
    def _setup_locality_measure(self):
        def dist(x: np.ndarray, y: np.ndarray):
            z = x-y
            # return np.linalg.norm(np.linalg.norm(z, axis=(1)), axis=(-2, -1))
            return np.linalg.norm(x-y, axis=(-2, -1))

        
        sigma = max(dist(self._stacked_org_img, self.org_img_plus_Z))

        # exponential kernel
        self.locality_measure = lambda x, y: np.exp(-dist(x, y) ** 2 / sigma ** 2)
        
    def _setup_loss_constants(self):
        self.binary_org_img = LimeLoss._generate_binary_representation(self.org_img)
        self.non_zero_indices = LimeLoss._get_indices_of_nonzero_entries(self.org_img.flatten())
        
        # set of samples
        self.Z = self._draw_perturbations()

        # match org img matrix with Z matrix
        squeezed_org_img = self.org_img if self.org_img.shape[0] > 1 else self.org_img.squeeze(0)
        self._stacked_org_img = np.array([squeezed_org_img for _ in range(self.n_samples)])

        assert self._stacked_org_img.shape == self.Z.shape

        # flatten over 0th axis, from e.g. (2, 3, 28, 28) to (2, 3*28*28)
        self.flatten_Z = self.Z.reshape(self.Z.shape[0], np.prod(self.Z.shape[1:]))
        self.org_img_plus_Z = (self._stacked_org_img + self.Z).clip(0, self.org_img.max())
        
        self._setup_locality_measure()
        
        self.Pi_Z: np.ndarray = self.locality_measure(
            self._stacked_org_img, self.org_img_plus_Z
        )
        
        target_idx = to_numpy(self.inference(self.org_img)).argmax()
        self.pred = to_numpy(self.inference(self.org_img_plus_Z))[:, target_idx]

    # def _draw_perturbations(
    #     self, shape: tuple, lower: float = -1.0, upper: float = 1.0
    # ) -> np.ndarray:
    #     # first dimension of image should be batch number
    #     batched_shape = list(shape)
    #     batched_shape[0] = self.n_samples
    #     batched_shape = tuple(batched_shape)

    #     self.randomState = np.random.RandomState()
    #     return self.randomState.uniform(size=batched_shape, low=lower, high=upper)
    
    def _draw_perturbations(self,) -> np.ndarray:
        return np.array([self._draw_one_perturbation().squeeze(0) for _ in range(self.n_samples)])
    
    def _draw_one_perturbation(self,) -> np.ndarray:
        self.randomState = np.random.RandomState()
        
        k = self.randomState.randint(low=1, high=len(self.non_zero_indices))
        indices = np.random.choice(self.non_zero_indices, k, replace=False)
        indices = np.sort(indices)
        
        perturbation = np.zeros(self.org_img.flatten().shape, dtype=np.float64)
        values = np.take(self.org_img.flatten(), indices)
        np.put(perturbation, indices, values, mode="raise")
        return perturbation.reshape(self.org_img.shape)
        

    def get_loss(self, data: np.ndarray, *args, **kwargs) -> np.ndarray: 
        return (
            1
            / self.n_samples
            * (self.Pi_Z.dot(self.pred - (self.flatten_Z.dot(data.flatten())))** 2)
        ).reshape((1,1))
