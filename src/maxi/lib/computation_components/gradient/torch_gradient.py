"""PyTorch Gradient Calculation"""

__all__ = ["Torch_Gradient"]

import numpy as np
import torch

from .base_gradient import BaseGradient
from ...loss.base_explanation_model import BaseExplanationModel


class Torch_Gradient(BaseGradient):
    def __init__(self, loss: BaseExplanationModel, *args, **kwargs):
        super().__init__(loss)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        torch_data = torch.tensor(data, dtype=torch.float32, requires_grad=True)

        loss = self.loss.get_loss(torch_data)
        loss.backward()

        return torch_data.grad.numpy()
