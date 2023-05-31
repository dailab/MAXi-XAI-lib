"""[Torch] CEM Loss Function Module"""
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.linalg import norm

from ...data.data_types import InferenceCall
from ...utils import loss_utils
from ..computation_components.gradient import Torch_Gradient
from .cem_loss import CEMLoss


class Torch_CEMLoss(CEMLoss):
    def __init__(
        self,
        mode: str,
        org_img: np.ndarray,
        inference: InferenceCall,
        gamma: float,
        device: str = "cpu",
        K: float = 1.0,
        c: float = 1.0,
        AE: Callable[[np.ndarray], np.ndarray] = None,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
        channels_first: bool = False,
        *args,
        **kwargs
    ) -> None:
        """ PyTorch Loss function of the Contrastive-Explanation-Method

        For the theoretical background refer to: \
            https://arxiv.org/pdf/1802.07623.pdf
            https://arxiv.org/pdf/1906.00117.pdf

        Args:
            mode (str): Chose between "PP" for _pertinent positive_ / "PN" for _pertinent negative_.
            org_img (ndarray): Original image [width, height, channels]
            inference (InferenceCall): Inference method of an external prediction entity. Has to return an \
                interpretable representation of the underlying prediction, e.g. a binary vector indicating \
                the “presence” or “absence”.
            c (float): $$f_K$$ regularization coefficient.
            device (str, optional): Computation device. Defaults to "cpu".
            gamma (float): Regularization coefficient for the autoencoder term.
            K (float): Confidence parameter for seperation between probability of target and non-target value.
            AE (Callable[[ndarray], ndarray]): Autoencoder, if None disregard AE error term.
            lower (np.ndarray, optional): Lower bound for the optimization. Has to be of the same shape as the \
                target image.
            upper (np.ndarray, optional): Upper bound for the optimization. Has to be of the same shape as the \
                target image.
            channels_first (bool, optional): Whether the channels dimension comes before the width and height \
                dimensions as in [bs, channels, width, height].
        
        Configurable Parameters:
            c, gamma, K, AE, channels_first
        
        Note:
            The loss functions are implemented solely using derivable Torch methods. In order to use \
            Torch's automatic differentiation on this class' methods, the model must be implemented in Torch as well.
        """
        self.compatible_grad_methods += [Torch_Gradient]
        self.device = device
        super().__init__(
            mode=mode,
            org_img=torch.tensor(org_img, dtype=torch.float32),
            inference=inference,
            c=torch.tensor(c, dtype=torch.float32),
            gamma=torch.tensor(gamma, dtype=torch.float32),
            K=torch.tensor(K, dtype=torch.float32),
            AE=AE,
            lower=lower,
            upper=upper,
            channels_first=channels_first,
        )
        if hasattr(self, "pn_target") and type(self.pn_target) is np.ndarray:
            self.pn_target = torch.tensor(self.pn_target, dtype=torch.float32)

    def get_target_idx(self, org_img: np.ndarray) -> torch.int64:
        """Retrieves index of the originally classified class in the inference result

        Args:
            org_img (np.ndarray): Original image in [1, width, height, channels] or [1, channels, width, height].

        Returns:
            int: Index of the predicted classification result
        """
        res = self.inference(torch.tensor(org_img, device=self.device))
        assert res.ndim == 2, "Inference result has to be a two dimensional array"
        assert len(res[0]) >= 2, "Inference result has to represent at least two states"
        assert len(res) == 1, "Loss class currently does not support batched calculations"
        return torch.argmax(res)

    def PN(self, delta: np.ndarray) -> torch.Tensor:
        """_Pertinent negative_ Loss Function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            torch.Tensor: PN loss value(s), 2D tensor of shape (bs, 1).
        """
        return super().PN(delta)

    def PP(self, delta: np.ndarray) -> torch.Tensor:
        """_Pertinent Positive_ Loss Function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            torch.Tensor: PP loss value(s), 2D tensor of shape (bs, 1).
        """
        return super().PP(delta)

    def PN_smooth(self, delta: np.ndarray) -> torch.Tensor:
        """_Smooth Pertinent Negative_ Loss Function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            torch.Tensor: PN loss value(s), 2D tensor of shape (bs, 1).
        """
        return super().PN_smooth(delta)

    def PP_smooth(self, delta: np.ndarray) -> torch.Tensor:
        """_Smooth Pertinent Positive_ Loss Function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            torch.Tensor: PP loss value(s), 2D tensor of shape (bs, 1).
        """
        return super().PP_smooth(delta)

    # TODO should get np.ndarray as input
    def f_K_neg(self, delta: torch.Tensor) -> torch.Tensor:
        """f_K term for the pertinent negative

        Args:
            delta (torch.Tensor): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            torch.Tensor: negative f_K term loss value, 2D tensor of shape (bs, 1).
        """
        pred = self.inference(self.org_img + delta)
        return (
            torch.maximum(
                loss_utils.torch_extract_target_proba(pred, self.target)
                - loss_utils.torch_extract_target_proba(pred, self.pn_target),
                -self.K,
            )
            if hasattr(self, "pn_target")
            else torch.maximum(
                loss_utils.torch_extract_target_proba(pred, self.target)
                - loss_utils.torch_extract_nontarget_proba(pred, self.target),
                -self.K,
            )
        )

    # TODO should get np.ndarray as input
    def f_K_pos(self, delta: torch.Tensor) -> torch.Tensor:
        """f_K term for the pertinent positive

        Args:
            delta (torch.Tensor): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            torch.Tensor: positive f_K term loss value, 2D tensor of shape (bs, 1).
        """
        pred = self.inference(delta)
        return torch.maximum(
            loss_utils.torch_extract_nontarget_proba(pred, self.target)
            - loss_utils.torch_extract_target_proba(pred, self.target),
            -self.K,
        )

    def f_K_neg_smooth(self, delta: np.ndarray) -> torch.Tensor:
        """f_K term for the smooth pertinent negative

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            torch.Tensor: negative f_K term loss value, 2D tensor of shape (bs, 1).
        """
        pred = self.inference(self.org_img + torch.tensor(delta, device=self.device))
        if hasattr(self, "pn_target"):
            attack_value = loss_utils.torch_extract_target_proba(
                pred, self.target
            ) - loss_utils.torch_extract_nontarget_proba(pred, self.target)
        else:
            attack_value = loss_utils.torch_extract_target_proba(
                pred, self.target
            ) - loss_utils.torch_extract_nontarget_proba(pred, self.pn_target)

        if attack_value < -10:
            return attack_value + torch.log(1.0 / torch.exp(attack_value) + 1)
        else:
            return attack_value + torch.log(1.0 + torch.exp(-attack_value))

    def f_K_pos_smooth(self, delta: np.ndarray) -> torch.Tensor:
        """f_K term for the pertinent positive

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            torch.Tensor: positive f_K term loss value, 2D tensor of shape (bs, 1).
        """
        pred = self.inference(torch.tensor(delta, device=self.device))
        attack_value = loss_utils.torch_extract_nontarget_proba(
            pred, self.target
        ) - loss_utils.torch_extract_target_proba(pred, self.target)

        if attack_value < -10:
            return attack_value + torch.log(1.0 / torch.exp(attack_value) + 1)
        else:
            return attack_value + torch.log(1.0 + torch.exp(-attack_value))

    # TODO should get np.ndarray as input
    def PN_AE_error(self, delta: torch.Tensor) -> torch.Tensor:
        """Autoencoder error term for the Pertinent Negative

        Args:
            delta (torch.Tensor): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            torch.Tensor: Error value(s), 2D tensor of shape (bs, 1).
        """
        if not self.AE:
            return 0.0
        adv_img = self.org_img + delta
        return norm(norm(adv_img - self.AE(adv_img), axis=self._c_dim), axis=(-2, -1)) ** 2

    # TODO should get np.ndarray as input
    def PP_AE_error(self, delta: torch.Tensor) -> torch.Tensor:
        """Autoencoder error term for the Pertinent Positive

        Args:
            delta (torch.Tensor): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            torch.Tensor: Error value(s), 2D tensor of shape (bs, 1).
        """
        if not self.AE:
            return 0.0
        return norm(norm(delta - self.AE(delta), axis=self._c_dim), axis=(-2, -1)) ** 2
