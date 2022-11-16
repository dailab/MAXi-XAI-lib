"""Lime Gradient Calculation"""

__all__ = ["Lime_Gradient"]

import numpy as np

from .base_gradient import BaseGradient
from ...loss.base_explanation_model import BaseExplanationModel
from ....utils.superpixel_handler import SuperpixelHandler


class LimeGradient(BaseGradient):
    def __init__(self, loss: BaseExplanationModel, *args, **kwargs):
        from ...loss.lime_loss import LimeLoss, SuperpixelLimeLoss

        assert (
            type(loss) is LimeLoss or type(loss) is SuperpixelLimeLoss
        ), "Invalid loss class given for LimeGradient!"
        super().__init__(loss)

        # assert hasattr(
        #     self.loss, "superpixel_handler"
        # ), "SuperpixelLoss must have a superpixel handler!"
        self._superpixel_mode = hasattr(self.loss, "superpixel_handler")

    def __call__(self, data: np.ndarray, *args, **kwds) -> np.ndarray:
        # if self._superpixel_mode:
        #     data = self.loss.superpixel_handler.generate_img_from_weight_vector(data)

        n, pi_Z, Z, flatten_Z, pred = (
            self.loss.n_samples,
            self.loss.Pi_Z,
            self.loss.Z,
            self.loss.flatten_Z,
            self.loss.pred,
        )

        a = flatten_Z.dot(data.flatten())
        b = a - pred
        c_ = pi_Z * b
        c = (
            c_[:, np.newaxis] * Z
            if self._superpixel_mode
            else c_[:, np.newaxis] * flatten_Z
        )  # ??
        # np.matmul
        d = 1 / n * c.sum(axis=0, keepdims=True)

        if not self._superpixel_mode:
            d = d.reshape(self.loss.org_img.shape)
        return d
