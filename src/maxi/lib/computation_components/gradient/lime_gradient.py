"""Lime Gradient Calculation"""

__all__ = ["LimeGradient"]

import numpy as np

from .base_gradient import BaseGradient
from ...loss.base_explanation_model import BaseExplanationModel
from ...image_segmentation.base_seg_handler import BaseSegmentationHandler


class LimeGradient(BaseGradient):
    def __init__(self, loss: BaseExplanationModel, *args, **kwargs):
        from ...loss.lime_loss import LimeLoss, SuperpixelLimeLoss

        assert (
            type(loss) is LimeLoss or type(loss) is SuperpixelLimeLoss
        ), "Invalid loss class given for LimeGradient!"

        super().__init__(loss)

        self._superpixel_mode = hasattr(self.loss, "superpixel_handler")

    def __call__(self, data: np.ndarray, *args, **kwds) -> np.ndarray:
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
