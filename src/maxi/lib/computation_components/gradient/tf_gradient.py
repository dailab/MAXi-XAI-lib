"""TensorFlow Gradient Calculation"""

__all__ = ["TF_Gradient"]

import numpy as np
import tensorflow as tf

from .base_gradient import BaseGradient
from ...loss.base_explanation_model import BaseExplanationModel


class TF_Gradient(BaseGradient):
    def __init__(self, loss: BaseExplanationModel, *args, **kwargs):
        super().__init__(loss)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(data)
            loss_value = self.loss.get_loss(data)

        return tape.gradient(loss_value, data).numpy()
