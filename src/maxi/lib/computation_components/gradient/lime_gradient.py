"""Lime Gradient Calculation"""

__all__ = ["Lime_Gradient"]

import numpy as np

from .base_gradient import BaseGradient
from ...loss.base_explanation_model import BaseExplanationModel


class LimeGradient(BaseGradient):
    def __init__(self, loss: BaseExplanationModel, *args, **kwargs):
        from ...loss.lime_loss import LimeLoss
        assert type(loss) is LimeLoss, "Invalid loss class given for LimeGradient!"
        super().__init__(loss)

    def __call__(self, data: np.ndarray, *args, **kwds) -> np.ndarray:
        n, pi_Z, Z, flatten_Z, pred = ( 
            self.loss.n_samples, 
            self.loss.Pi_Z, 
            self.loss.Z, 
            self.loss.flatten_Z, 
            self.loss.pred 
        )
        
        a = flatten_Z.dot(data.flatten())
        b = a - pred
        c = pi_Z.dot(b) * Z
        d = 1 / n * c.sum(axis=0, keepdims=True)
        
        return d