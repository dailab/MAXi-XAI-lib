"""[TensorFlow] CEM Loss Function Module"""
from typing import Tuple, List, Callable

import numpy as np
import tensorflow as tf

from .cem_loss import CEMLoss
from ..computation_components.gradient import TF_Gradient
from ...data.data_types import InferenceCall
from ...utils import loss_utils


class TF_CEMLoss(CEMLoss):
    def __init__(
        self,
        mode: str,
        org_img: np.ndarray,
        inference: InferenceCall,
        gamma: float,
        K: float,
        c: float = 1.0,
        AE: Callable[[np.ndarray], np.ndarray] = None,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
        channels_first: bool = False,
    ) -> None:
        """ TensoFlow Loss function of the Contrastive-Explanation-Method

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
            gamma (float): Regularization coefficient for the autoencoder term.
            K (float): Confidence parameter for seperation between probability of target and non-target value.
            AE (Callable[[ndarray], ndarray]): Autoencoder, if None disregard AE error term.
            lower (np.ndarray, optional): Lower bound for the optimization. Has to be of the same shape as the \
                target image.
            upper (np.ndarray, optional): Upper bound for the optimization. Has to be of the same shape as the \
                target image.
            channels_first (bool, optional): Whether the channels dimension comes before the width and height \
                dimensions as in [bs, channels, width, height].
            
        Note:
            The loss functions are implemented solely using derivable TensorFlow methods. In order to use \
            TF's automatic differentiation on this class' methods, the model must be implemented in TF as well.
        """
        self.compatible_grad_methods += [TF_Gradient]
        super().__init__(
            mode=mode,
            org_img=tf.convert_to_tensor(org_img, dtype=tf.float32),
            inference=inference,
            c=tf.convert_to_tensor(c, dtype=tf.float32),
            gamma=tf.convert_to_tensor(gamma, dtype=tf.float32),
            K=tf.convert_to_tensor(K, dtype=tf.float32),
            AE=AE,
            lower=lower,
            upper=upper,
            channels_first=channels_first,
        )
        self._org_img_shape = tf.constant(org_img.shape)

    def get_target_idx(self, org_img: tf.Tensor) -> tf.int32:
        """Retrieves index of the originally classified class in the inference result

        Args:
            org_img (tf.Tensor): Original image in [1, width, height, channels] or [1, channels, width, height].

        Returns:
            tf.int32: Index of the most likely class.
        """
        res = self.inference(org_img)
        assert res.ndim == 2, "Inference result has to be a two dimensional array"
        assert len(res[0]) >= 2, "Inference result has to represent at least two states"
        assert len(res) == 1, "Loss class currently does not support batched calculations"
        return tf.cast(tf.argmax(res, axis=-1), tf.int32)

    def PN(self, delta: np.ndarray) -> tf.Tensor:
        """_Pertinent negative_ loss function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            tf.Tensor: PN loss value(s), 2D tensor of shape (bs, 1).
        """
        # if delta.ndim < 2:
        #     delta = tf.reshape(delta, self._org_img_shape)
        if delta.dtype.name != "float32":
            delta = tf.cast(delta, tf.float32)
        return self.c * self.f_K_neg(delta) + self.gamma * self.PN_AE_error(delta)

    def PP(self, delta: np.ndarray) -> tf.Tensor:
        """Pertinent Positive loss function

        Args:
            delta (np.ndarray): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            tf.Tensor: PP loss value(s), 2D tensor of shape (bs, 1).
        """
        # if delta.ndim < 2:
        #     delta = tf.reshape(delta, self._org_img_shape)
        if delta.dtype.name != "float32":
            delta = tf.cast(delta, tf.float32)
        return self.c * self.f_K_pos(delta) + self.gamma * self.PP_AE_error(delta)

    def f_K_neg(self, delta: tf.Tensor) -> tf.Tensor:
        """f_K term for the pertinent negative

        Args:
            delta (tf.Tensor): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            tf.Tensor: negative f_K term loss value, 2D tensor of shape (bs, 1).
        """
        pred = self.inference(self.org_img + delta)
        return tf.maximum(
            loss_utils.tf_extract_target_proba(pred, self.target)
            - loss_utils.tf_extract_nontarget_proba(pred, self.target),
            -self.K,
        )

    def f_K_pos(self, delta: tf.Tensor) -> tf.Tensor:
        """f_K term for the pertinent positive

        Args:
            delta (tf.Tensor): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            tf.Tensor: positive f_K term loss value, 2D tensor of shape (bs, 1).
        """
        pred = self.inference(delta)
        return tf.maximum(
            loss_utils.tf_extract_nontarget_proba(pred, self.target)
            - loss_utils.tf_extract_target_proba(pred, self.target),
            -self.K,
        )

    def PN_AE_error(self, delta: tf.Tensor) -> tf.Tensor:
        """Autoencoder error term for the pertinent negative

        Args:
            delta (tf.Tensor): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            tf.Tensor: Error value(s), 2D tensor of shape (bs, 1).
        """
        if not self.AE:
            return 0.0
        adv_img = self.org_img + delta
        return tf.norm(tf.norm(adv_img - self.AE(adv_img), axis=self._c_dim), axis=[-2, -1]) ** 2

    def PP_AE_error(self, delta: tf.Tensor) -> tf.Tensor:
        """Autoencoder error term for the pertinent positive

        Args:
            delta (tf.Tensor): Perturbation matrix in [bs, width, height, channels] or [bs, channels, width, height].

        Returns:
            tf.Tensor: Error value(s), 2D tensor of shape (bs, 1).
        """
        if not self.AE:
            return 0.0
        return tf.norm(tf.norm(delta - self.AE(delta), axis=self._c_dim), axis=[-2, -1]) ** 2
