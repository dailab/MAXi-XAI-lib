"""CEM Loss Function Module"""

__all__ = ["CEMLoss"]

from typing import Callable, Tuple

import numpy as np
from numpy.linalg import norm

from .base_explanation_model import BaseExplanationModel
from ..computation_components.gradient import (
    URVGradientEstimator,
    USRVGradientEstimator,
)
from ...data.data_types import InferenceCall
from ...utils import loss_utils, x0_generators
from ...utils.general import to_numpy


class CEMLoss(BaseExplanationModel):
    compatible_grad_methods = [URVGradientEstimator, USRVGradientEstimator]
    pp_x0_generator, pn_x0_generator = np.zeros_like, np.zeros_like

    def __init__(
        self,
        mode: str,
        org_img: np.ndarray,
        inference: InferenceCall,
        c: float = 1.0,
        gamma: float = 10,
        K: float = 10,
        AE: Callable[[np.ndarray], np.ndarray] = None,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
        channels_first: bool = False,
    ):
        """ Loss function of the Contrastive-Explanation-Method

        For the theoretical background refer to: \
            https://arxiv.org/pdf/1802.07623.pdf
            https://arxiv.org/pdf/1906.00117.pdf

        Args:
            mode (str): Chose between "PP" for _pertinent positive_ / "PN" for _pertinent negative_.
            org_img (ndarray): Original image [width, height, channels]
            inference (InferenceCall): Inference method of an external prediction entity. Has to return an \
                interpretable representation of the underlying prediction, e.g. a binary vector indicating \
                the “presence” or “absence”.
            c (float, optional): $$f_K$$ regularization coefficient.
            gamma (float, optional): Regularization coefficient for the autoencoder term.
            K (float, optional): Confidence parameter for seperation between probability of target and non-target value.
            AE (Callable[[ndarray], ndarray]): Autoencoder, if None disregard AE error term.
            lower (np.ndarray, optional): Lower bound for the optimization. Has to be of the same shape as the \
                target image.
            upper (np.ndarray, optional): Upper bound for the optimization. Has to be of the same shape as the \
                target image.
            channels_first (bool, optional): Whether the channels dimension comes before the width and height \
                dimensions as in [bs, channels, width, height].
            
        Note:
            The hardcoded lower and upper bounds in ```_init_lower_upper()``` are tailored for images with \
            pixel range of -0.5 to 0.5. For other ranges, it is required to specify it accordingly.\
            Otherwise it might strongly adulterate the results.
        """
        self._setup_mode(mode)
        self._init_lower_upper(lower, upper, org_img)

        super().__init__(
            org_img=org_img,
            inference=inference,
            x0_generator=self._x0_generator,
            lower=self._lower,
            upper=self._upper,
        )

        # loss parameters
        self.c, self.gamma, self.AE, self.K = c, gamma, AE, K
        self.org_img, self._org_img_shape = org_img, org_img.shape
        self.target = self.get_target_idx(org_img)

        # channel dimension
        self._c_dim = 1 if channels_first else -1  # 1 channels first, -1 channels last

    def _init_lower_upper(
        self,
        lower: Tuple[None, np.ndarray],
        upper: Tuple[None, np.ndarray],
        org_img: np.ndarray,
    ):
        DEFAULT_LB_UB = {
            "PP": {
                "lower": np.full(org_img.shape, 0.0),
                # "lower": np.zeros(org_img.shape),
                "upper": to_numpy(org_img),
            },
            "PN": {
                "lower": np.full(org_img.shape, 0.0),
                "upper": 1.0 - to_numpy(org_img),
            },
        }

        self._lower = DEFAULT_LB_UB[self.mode]["lower"] if lower is None else lower
        self._upper = DEFAULT_LB_UB[self.mode]["upper"] if upper is None else upper

        assert type(self._lower) is np.ndarray, "Invalid lower bound given for optimization"
        assert type(self._upper) is np.ndarray, "Invalid upper bound given for optimization"

        if self._lower.shape != org_img.shape:
            raise ValueError(
                f"Got lowwer bound matrix of different shape than the input image. ({self._lower.shape} != {org_img.shape})"
            )

        if self._upper.shape != org_img.shape:
            raise ValueError(
                f"Got upper bound matrix of different shape than the input image.({self._upper.shape} != {org_img.shape})"
            )

    def _setup_mode(self, input: str):
        assert input.upper() in {"PP", "PN"}, "Provided unknown mode for CEM"
        self.mode = input.upper()

        self.get_loss, self._x0_generator = (
            (self.PP, CEMLoss.pp_x0_generator) if self.mode == "PP" else (self.PN, CEMLoss.pn_x0_generator)
        )

    def get_target_idx(self, org_img: np.ndarray) -> int:
        """Retrieves index of the originally classified class in the inference result

        Args:
            org_img (np.ndarray): Original image in [1, width, height, channels] or [1, channels, width, height].

        Returns:
            int: Index of the most likely class.
        """
        res = self.inference(org_img)
        assert res.ndim == 2, "Inference result has to be an one dimensional array"
        assert len(res[0]) >= 2, "Inference result has to represent at least two states"
        assert len(res) == 1, "Loss class currently does not support batched calculations"
        return np.argmax(to_numpy(res))  # index of the original prediction

    def get_loss(self, data: np.ndarray) -> np.ndarray:
        return super().get_loss(data)

    def PN(self, delta: np.ndarray) -> np.ndarray:
        """_Pertinent negative_ loss function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: PN loss value(s), 2D array of shape (bs, 1).
        """
        # if delta.ndim < 2:
        #     delta = np.ascontiguousarray(delta.reshape(self._org_img_shape))
        return self.c * self.f_K_neg(delta) + self.gamma * self.PN_AE_error(delta)

    def PP(self, delta: np.ndarray) -> np.ndarray:
        """Pertinent Positive loss function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: PP loss value(s), 2D array of shape (bs, 1).
        """
        # if delta.ndim < 2:
        #     delta = np.ascontiguousarray(delta.reshape(self._org_img_shape))
        return self.c * self.f_K_pos(delta) + self.gamma * self.PP_AE_error(delta)

    def f_K_neg(self, delta: np.ndarray) -> np.ndarray:
        """f_K term for the pertinent negative

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: negative f_K term loss value, 2D array of shape (bs, 1).
        """
        pred = self.inference(self.org_img + delta)
        return np.maximum(
            loss_utils.np_extract_target_proba(pred, self.target)
            - loss_utils.np_extract_nontarget_proba(pred, self.target),
            -self.K,
        )

    def f_K_pos(self, delta: np.ndarray) -> np.ndarray:
        """f_K term for the pertinent positive

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: positive f_K term loss value, 2D array of shape (bs, 1).
        """
        pred = self.inference(delta)
        return np.maximum(
            loss_utils.np_extract_nontarget_proba(pred, self.target)
            - loss_utils.np_extract_target_proba(pred, self.target),
            -self.K,
        )

    def PN_AE_error(self, delta: np.ndarray) -> np.ndarray:
        """Autoencoder error term for the Pertinent Negative

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: Error value(s), 2D array of shape (bs, 1).
        """
        if not self.AE:
            return 0.0
        adv_img = self.org_img + delta
        return (
            norm(
                norm(adv_img - to_numpy(self.AE(adv_img)), axis=self._c_dim),
                axis=(-2, -1),
            )
            ** 2
        )

    def PP_AE_error(self, delta: np.ndarray) -> np.ndarray:
        """Autoencoder error term for the Pertinent Positive

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: Error value(s), 2D array of shape (bs, 1).
        """
        if not self.AE:
            return 0.0
        return (
            norm(
                norm(delta - to_numpy(self.AE(delta)), axis=self._c_dim),
                axis=(-2, -1),
            )
            ** 2
        )
