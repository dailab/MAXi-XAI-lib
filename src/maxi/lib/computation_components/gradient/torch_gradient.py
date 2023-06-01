"""PyTorch Gradient Calculation"""

__all__ = ["Torch_Gradient"]

import numpy as np
import torch

from .base_gradient import BaseGradient
from ...loss.base_explanation_model import BaseExplanationModel


class Torch_Gradient(BaseGradient):
    def __init__(
        self, loss: BaseExplanationModel, device: str = "cpu", *args, **kwargs
    ):
        """Torch Gradient Calculation

        Args:
            loss (BaseExplanationModel): Loss Class. Is parsed by the explanation generator.
            device (str, optional): Computation device. Defaults to "cpu".
        """
        super().__init__(loss)
        self.device = device

    def __call__(self, data: np.ndarray) -> np.ndarray:
        torch_data = torch.tensor(
            data,
            dtype=torch.float32,
            requires_grad=True,
            device=torch.device(self.device),
        )

        loss = self.loss.get_loss(torch_data)
        loss.backward()

        return torch_data.grad.cpu().numpy()
