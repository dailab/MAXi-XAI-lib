"""Segmentation Lime Loss Function Module"""

__all__ = ["SuperpixelLimeLoss"]

from typing import Union

import numpy as np

from ..lime_loss import LimeLoss
from ...image_segmentation.base_seg_handler import BaseSegmentationHandler
from ....data.data_types import InferenceCall
from ....utils.general import to_numpy


class SuperpixelLimeLoss(LimeLoss):
    def __init__(
        self,
        org_img: np.ndarray,
        inference: InferenceCall,
        n_samples: int,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
        channels_first: bool = False,
        superpixel_handler: BaseSegmentationHandler = None,
    ):
        """Loss function for Superpixel-LIME (Local Interpretable Model-Agnostic Explanations)

        For the theoretical background refer to:
            https://arxiv.org/abs/1602.04938

        Args:
            org_img (np.ndarray): Original image [width, height, channels] or [channels, width, height].
            inference (InferenceCall): Inference method of an external prediction entity. Has to return an \
                interpretable representation of the underlying prediction, e.g. a binary vector indicating \
                the “presence” or “absence”.
            n_samples (int): Number of samples to draw from the perturbation distribution.
            lower (np.ndarray, optional): Lower bound for the optimization. Has to be of the same shape as the \
                target image. Defaults to None.
            upper (np.ndarray, optional): Upper bound for the optimization. Has to be of the same shape as the \
                target image. Defaults to None.
            superpixel_handler (BaseSegmentationHandler, optional): BaseSegmentationHandler. Defaults to None.
                
        Configurable Parameters:
            n_samples, channels_first
            
        Note:
            The loss function is based on the LIME loss function. The only difference is that the perturbation \
            distribution is drawn from superpixels instead of pixels.
        """
        self.superpixel_handler = superpixel_handler
        if superpixel_handler:
            self._init_lower_upper(
                lower, upper, self.superpixel_handler.ones_weight_vector
            )
            self._valid_check_lower_upper(self._lower, self._upper, org_img)

        super().__init__(
            org_img=org_img,
            inference=inference,
            x0_generator=LimeLoss._x0_generator,
            lower=self._lower,
            upper=self._upper,
            n_samples=n_samples,
            channels_first=channels_first,
        )
        self._setup_loss_constants()

    def _init_lower_upper(
        self,
        lower: Union[None, np.ndarray],
        upper: Union[None, np.ndarray],
        *args,
        **kwargs,
    ):
        """Initializes the lower and upper bound for the optimization.

        Args:
            lower (Union[None, np.ndarray]): Lower bound for the optimization.
            upper (Union[None, np.ndarray]): Upper bound for the optimization.
        """
        DEFAULT_LB_UB = {
            "lower": np.full((self.superpixel_handler.num_segments), -1.0, np.float32),
            "upper": np.full((self.superpixel_handler.num_segments), 1.0, np.float32),
        }

        self._lower = DEFAULT_LB_UB["lower"] if lower is None else lower
        self._upper = DEFAULT_LB_UB["upper"] if upper is None else upper

    @staticmethod
    def _get_indices_of_nonzero_entries(sp_label_images: np.ndarray) -> np.ndarray:
        """Returns the indices of the non-zero entries of a label matrix.

        Args:
            data (np.ndarray): Superpixel Image.

        Returns:
            np.ndarray: Indices of non-zero entries.
        """
        return np.array(
            [
                i
                for i, label_image in enumerate(sp_label_images)
                if label_image.max() > 0
            ]
        )

    def _setup_locality_measure(self):
        """Sets up the locality measure for the perturbation distribution.

        Description:
            The locality measure is used to determine the distance between two images.
        """

        def dist(x: np.ndarray, y: np.ndarray):
            # return np.linalg.norm(np.linalg.norm(z, axis=(1)), axis=(-2, -1))
            axis = (-1) if x.ndim == 2 else (-2, -1)
            return np.linalg.norm(x - y, axis=axis)

        sigma = max(dist(self._stacked_org_img, self.org_img_plus_Z))

        # exponential kernel
        self.locality_measure = lambda x, y: np.exp(-dist(x, y) ** 2 / sigma ** 2)

    def _setup_loss_constants(self):
        """Setup constants for the loss function."""
        # extract non-zero entries from the image
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

        # original image has to be of shape (1, n_superpixels)
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
        # compute the distance between the original image and the perturbed images
        self.Pi_Z: np.ndarray = self.locality_measure(
            self._stacked_org_img, self.org_img_plus_Z
        )

        # get the prediction score of the perturbed images
        target_idx = to_numpy(self.inference(self.org_img)).argmax()
        # get the images from the label weights
        org_plus_Z_images = self.superpixel_handler.generate_imgs_from_weight_vectors(
            self.org_img_plus_Z
        )
        self.pred = to_numpy(self.inference(org_plus_Z_images))[:, target_idx]

    def _draw_one_perturbation(self, non_zero_indices: np.ndarray) -> np.ndarray:
        """Draws one perturbation from the perturbation distribution.

        Description:
            The perturbation is made up of 1 to k random non-zero superpixels of the superpixel image.

        Args:
            non_zero_indices (np.ndarray): Array containing the non-zero indices of the superpixel image.

        Returns:
            np.ndarray: One perturbation superpixel image.
        """
        self.randomState = np.random.RandomState()

        # draw random indices from non_zero_indices
        # non_zero_indices consist of superpixels that
        # contain at least one non-zero entry
        k = self.randomState.randint(low=1, high=len(non_zero_indices))
        indices = np.random.choice(non_zero_indices, k, replace=False)
        indices = np.sort(indices)

        # draw random weights for non_zero_indices
        weight_vector = np.zeros(
            (self.superpixel_handler.num_segments), dtype=np.float64
        )
        random_weights = self.randomState.uniform(low=0.0, high=1.0, size=k)
        np.put(weight_vector, indices, random_weights, mode="raise")
        return np.expand_dims(weight_vector, 0)
        # return self.superpixel_handler.generate_img_from_weight_vector(weight_vector)

    def get_loss(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Gets the loss for the given superpixel image model (importance weights).

        Args:
            data (np.ndarray): Image model.

        Returns:
            np.ndarray: Loss value according to LIME loss.
        """
        return (
            1
            / self.n_samples
            * (self.Pi_Z.dot(self.pred - (self.flatten_Z.dot(align_dims(data)))) ** 2)
        ).reshape((1, 1))


def align_dims(data: np.ndarray) -> np.ndarray:
    return data.reshape(data.shape[-1]) if data.ndim == 2 else data
