"""CEM Loss Function Module"""

__all__ = ["CEMLoss"]

from typing import Callable, Union

import numpy as np
from numpy.linalg import norm
from warnings import warn

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
    # pp_x0_generator, pn_x0_generator = lambda x: x, np.zeros_like
    pp_x0_generator, pn_x0_generator = (
        lambda x: x,
        loss_utils.generate_from_gaussian,
    )

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
        *args,
        **kwargs,
    ):
        """ Loss function of the Contrastive-Explanation-Method

        For the theoretical background refer to: \
            https://arxiv.org/pdf/1802.07623.pdf
            https://arxiv.org/pdf/1906.00117.pdf

        Args:
            mode (str): Chose between "PP" for _pertinent positive_ / "PN" for _pertinent negative_.
            org_img (ndarray): Original image [width, height, channels] or [channels, width, height].
            inference (InferenceCall): Inference method of an external prediction entity. Has to return an \
                interpretable representation of the underlying prediction, e.g. a binary vector indicating \
                the “presence” or “absence”.
            c (float, optional): $$f_K$$ regularization coefficient.
            gamma (float, optional): Regularization coefficient for the autoencoder term.
            K (float, optional): Confidence parameter for seperation between probability of target and non-target value.
            AE (Callable[[ndarray], ndarray]): Autoencoder, if None disregard AE error term.
            lower (np.ndarray, optional): Lower bound for the optimization. Has to be of the same shape as the \
                target image. Defaults to None.
            upper (np.ndarray, optional): Upper bound for the optimization. Has to be of the same shape as the \
                target image. Defaults to None.
            channels_first (bool, optional): Whether the channels dimension comes before the width and height \
                dimensions as in [bs, channels, width, height].
            
        Configurable Parameters:
            c, gamma, K, AE, channels_first
            
        Note:
            The hardcoded lower and upper bounds in ```_init_lower_upper()``` are tailored for images with \
            pixel range of [0, 1] or [0, 255]. For other ranges, it is required to specify it accordingly.\
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

        # read pn target
        self._read_pn_target_from_kwargs(kwargs)

    def _init_lower_upper(
        self,
        lower: Union[None, np.ndarray],
        upper: Union[None, np.ndarray],
        org_img: np.ndarray,
    ):
        DEFAULT_LB_UB = {
            "PP": {
                "lower": np.full(org_img.shape, 0.0),
                # "lower": np.zeros(org_img.shape),
                "upper": to_numpy(org_img),
            },
            "PN": {
                "lower": np.full(org_img.shape, -1.0),
                # "upper": 1.0 - to_numpy(org_img),
                "upper": np.full(org_img.shape, 1.0),
            },
        }
        DEFAULT_LB_UB["PPSMOOTH"] = DEFAULT_LB_UB["PP"]
        DEFAULT_LB_UB["PNSMOOTH"] = DEFAULT_LB_UB["PN"]

        self._lower = DEFAULT_LB_UB[self.mode]["lower"] if lower is None else lower
        self._upper = DEFAULT_LB_UB[self.mode]["upper"] if upper is None else upper

        assert (
            type(self._lower) is np.ndarray
        ), "Invalid lower bound given for optimization"
        assert (
            type(self._upper) is np.ndarray
        ), "Invalid upper bound given for optimization"

        if self._lower.shape != org_img.shape:
            raise ValueError(
                f"Got lowwer bound matrix of different shape than the input image. ({self._lower.shape} != {org_img.shape})"
            )

        if self._upper.shape != org_img.shape:
            raise ValueError(
                f"Got upper bound matrix of different shape than the input image.({self._upper.shape} != {org_img.shape})"
            )

    def _setup_mode(self, input: str):
        assert input.upper() in {
            "PP",
            "PN",
            "PPSMOOTH",
            "PNSMOOTH",
        }, "Provided unknown mode for CEM"
        self.mode = input.upper()

        if self.mode == "PP":
            self.get_loss, self._x0_generator = (self.PP, CEMLoss.pp_x0_generator)
        elif self.mode == "PN":
            self.get_loss, self._x0_generator = (self.PN, CEMLoss.pn_x0_generator)
        elif self.mode == "PPSMOOTH":
            self.get_loss, self._x0_generator = (
                self.PP_smooth,
                CEMLoss.pp_x0_generator,
            )
        else:
            self.get_loss, self._x0_generator = (
                self.PN_smooth,
                CEMLoss.pn_x0_generator,
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
        assert (
            len(res) == 1
        ), "Loss class currently does not support batched calculations"
        return np.argmax(to_numpy(res))  # index of the original prediction

    def _read_pn_target_from_kwargs(self, kwargs: dict) -> None:
        if "pn_target" in kwargs:
            if "PP" in self.mode:
                warn("PN target class is given but not used in PP mode.")
                return
            self.pn_target = np.array([kwargs["pn_target"]])

            if self.target == self.pn_target:
                raise ValueError(
                    f"Target class ({self.target}) and PN target class ({self.pn_target}) are the same."
                )

    def get_loss(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        return super().get_loss(data)

    def PN(self, delta: np.ndarray) -> np.ndarray:
        """_Pertinent Negative_ Loss Function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: PN loss value(s), 2D array of shape (bs, 1).
        """
        # if delta.ndim < 2:
        #     delta = np.ascontiguousarray(delta.reshape(self._org_img_shape))
        return self.c * self.f_K_neg(delta) + self.gamma * self.PN_AE_error(delta)

    def PP(self, delta: np.ndarray) -> np.ndarray:
        """_Pertinent Positive_ Loss Function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: PP loss value(s), 2D array of shape (bs, 1).
        """
        # if delta.ndim < 2:
        #     delta = np.ascontiguousarray(delta.reshape(self._org_img_shape))
        return self.c * self.f_K_pos(delta) + self.gamma * self.PP_AE_error(delta)

    def PP_smooth(self, delta: np.ndarray) -> np.ndarray:
        """_Smooth Pertinent Positive_ loss function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: PP loss value(s), 2D array of shape (bs, 1).
        """
        # if delta.ndim < 2:
        #     delta = np.ascontiguousarray(delta.reshape(self._org_img_shape))
        return self.c * self.f_K_pos_smooth(delta) + self.gamma * self.PP_AE_error(
            delta
        )

    def PN_smooth(self, delta: np.ndarray) -> np.ndarray:
        """_Smooth Pertinent Negative_ Loss Function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: PN loss value(s), 2D array of shape (bs, 1).
        """
        # if delta.ndim < 2:
        #     delta = np.ascontiguousarray(delta.reshape(self._org_img_shape))
        return self.c * self.f_K_neg_smooth(delta) + self.gamma * self.PN_AE_error(
            delta
        )

    def f_K_neg(self, delta: np.ndarray) -> np.ndarray:
        """f_K term for the pertinent negative

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: negative f_K term loss value, 2D array of shape (bs, 1).
        """
        pred = self.inference(self.org_img + delta)

        return (
            np.maximum(
                loss_utils.np_extract_target_proba(pred, self.target)
                - loss_utils.np_extract_target_proba(pred, self.pn_target),
                -self.K,
            )
            if hasattr(self, "pn_target")
            else np.maximum(
                loss_utils.np_extract_target_proba(pred, self.target)
                - loss_utils.np_extract_nontarget_proba(pred, self.target),
                -self.K,
            )
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

    def f_K_neg_smooth(self, delta: np.ndarray) -> np.ndarray:
        """Smooth f_K term for the pertinent negative

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: negative f_K term loss value, 2D array of shape (bs, 1).
        """
        pred = self.inference(self.org_img + delta)

        if not hasattr(self, "pn_target"):
            attack_value = loss_utils.np_extract_target_proba(
                pred, self.target
            ) - loss_utils.np_extract_nontarget_proba(pred, self.target)
        else:
            # difference between value of originally predicted class and value of pn_target class
            attack_value = loss_utils.np_extract_target_proba(
                pred, self.target
            ) - loss_utils.np_extract_target_proba(pred, self.pn_target)

        if attack_value < -10:
            return np.log(1.0 + np.exp(attack_value))
        else:
            return attack_value + np.log(1.0 + np.exp(-attack_value))

    def f_K_pos_smooth(self, delta: np.ndarray) -> np.ndarray:
        """Smooth f_K term for the smooth pertinent positive

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            np.ndarray: positive f_K term loss value, 2D array of shape (bs, 1).
        """
        pred = self.inference(delta)
        attack_value = loss_utils.np_extract_nontarget_proba(
            pred, self.target
        ) - loss_utils.np_extract_target_proba(pred, self.target)

        if attack_value < -10:
            return np.log(1.0 + np.exp(attack_value))
        else:
            return attack_value + np.log(1.0 + np.exp(-attack_value))

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
