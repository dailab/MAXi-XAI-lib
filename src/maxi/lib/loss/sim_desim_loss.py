from typing import Union

import numpy as np
from maxi.data.data_types import InferenceCall, X0_Generator
from maxi.utils.general import to_numpy
from sklearn.gaussian_process.kernels import RBF

from .base_explanation_model import BaseExplanationModel
from maxi.lib.computation_components import URVGradientEstimator, USRVGradientEstimator


class SimDesimLoss(BaseExplanationModel):
    compatible_grad_methods = [URVGradientEstimator, USRVGradientEstimator]
    # _x0_generator = lambda x: x
    _x0_generator = lambda x: np.full(x.shape, 0)

    def __init__(
        self,
        org_img: np.ndarray,
        inference: InferenceCall,
        target_index: int,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
        *args,
        **kwargs,
    ) -> None:
        """Similiarity Desimilarity Loss

        Args:
            org_img (np.ndarray): Original target image in [width, height, channels].
            inference (InferenceCall): The inference method of an external prediction entity.
            x0_generator (X0_Generator): Method to generate the initial object of optimization.
            lower (np.ndarray): Lower bound for the object of optimization. Has to be of same shape as org_img.
            upper (np.ndarray): Upper bound for the object of optimization. Has to be of same shape as org_img.
            target_index (int): Index of the desired target class.
        """
        super().__init__(
            org_img,
            inference,
            SimDesimLoss._x0_generator,
            lower,
            upper,
            *args,
            **kwargs,
        )
        self._init_lower_upper(lower, upper, org_img)
        self.target_index = target_index
        self.org_prediction = self.inference(self.org_img)
        self.org_prediction_wo_target = np.delete(
            to_numpy(self.org_prediction), target_index
        )
        # Define the RBF kernel with length scale 1.0
        self.kernel = RBF(length_scale=1.0)

    def _init_lower_upper(
        self,
        lower: Union[None, np.ndarray],
        upper: Union[None, np.ndarray],
        org_img: np.ndarray,
    ):
        self._lower = np.full(org_img.shape, 0.0) if lower is None else lower
        self._upper = to_numpy(org_img) if upper is None else upper

        assert (
            type(self._lower) is np.ndarray
        ), "Invalid lower bound given for optimization; self._lower is not a numpy array"
        assert (
            type(self._upper) is np.ndarray
        ), "Invalid upper bound given for optimization; self._upper is not a numpy array"

        if self._lower.shape != org_img.shape:
            raise ValueError(
                f"Got lowwer bound matrix of different shape than the input image. ({self._lower.shape} != {org_img.shape})"
            )

        if self._upper.shape != org_img.shape:
            raise ValueError(
                f"Got upper bound matrix of different shape than the input image.({self._upper.shape} != {org_img.shape})"
            )

    def get_loss(self, data: np.ndarray, *args, **kwargs) -> float:
        """Computes the loss value for the given data

        Args:
            data (np.ndarray): An image in [width, height, channels].

        Returns:
            float: The actual loss value.
        """
        perturbed_pred = self.inference(data)
        perturbed_pred_wo_target = np.delete(perturbed_pred, self.target_index)
        return SimDesimLoss.get_similarity_val(
            self.kernel,
            self.org_prediction[:, self.target_index],
            perturbed_pred[:, self.target_index],
        ) + SimDesimLoss.get_dissimilarity_val(
            self.org_prediction_wo_target, perturbed_pred_wo_target
        )

    @staticmethod
    def get_similarity_val(
        kernel_fnc, first_data: np.ndarray, second_data: np.ndarray
    ) -> float:
        """Computes the similarity value between the original image and the given data

        Args:
            data (np.ndarray): An image in [width, height, channels].

        Returns:
            float: The actual similarity value.
        """
        return -kernel_fnc(first_data, second_data)

    @staticmethod
    def get_dissimilarity_val(first_data: np.ndarray, second_data: np.ndarray) -> float:
        """Computes the similarity value between the original image and the given data

        Args:
            data (np.ndarray): An image in [width, height, channels].

        Returns:
            float: The actual similarity value.
        """
        return np.linalg.norm(first_data - second_data)
