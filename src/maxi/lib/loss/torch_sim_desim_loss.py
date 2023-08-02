import numpy as np
import torch as th
from maxi.data.data_types import InferenceCall
from maxi.lib.computation_components import Torch_Gradient

from .sim_desim_loss import SimDesimLoss


def rbf_kernel(X, Y, gamma=1):
    """
    Computes the RBF (Gaussian) kernel between two matrices X and Y.
    """
    # Compute pairwise squared Euclidean distances between rows of X and Y
    dist_XY = th.cdist(X, Y, p=2) ** 2

    # Compute the kernel matrix
    return th.exp(-gamma * dist_XY)


class Torch_SimDesimLoss(SimDesimLoss):
    compatible_grad_methods = [Torch_Gradient]

    def __init__(
        self,
        org_img: np.ndarray,
        inference: InferenceCall,
        target_index: int,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
        device: str = "cpu",
        *args,
        **kwargs
    ):
        super().__init__(
            org_img=org_img,
            inference=inference,
            lower=lower,
            upper=upper,
            target_index=target_index,
            *args,
            **kwargs
        )
        self.device = device
        self.org_img = th.tensor(self.org_img, device=device, dtype=th.float32)
        self.target_index = th.tensor(self.target_index, device=device)

        self.org_prediction = th.tensor(self.org_prediction, device=device)
        self.org_prediction_wo_target = th.tensor(
            self.org_prediction_wo_target, device=device
        )

        self.mask = th.ones_like(self.org_prediction, dtype=th.bool, device=device)
        self.mask[:, self.target_index] = False

    def get_loss(self, data: th.Tensor, *args, **kwargs) -> float:
        if not th.is_tensor(data):
            data = th.tensor(data, device=self.device, dtype=th.float32)
        perturbed_pred: th.Tensor = self.inference(self.org_img + data)
        perturbed_pred_wo_target = th.masked_select(perturbed_pred, self.mask)
        return Torch_SimDesimLoss.get_similarity_val(
            rbf_kernel,
            self.org_prediction[:, self.target_index].unsqueeze(1),
            perturbed_pred[:, self.target_index].unsqueeze(1),
        ) + Torch_SimDesimLoss.get_dissimilarity_val(
            self.org_prediction_wo_target, perturbed_pred_wo_target
        )

    @staticmethod
    def get_similarity_val(
        kernel_fnc, first_data: th.Tensor, second_data: th.Tensor
    ) -> th.Tensor:
        return -kernel_fnc(first_data, second_data)

    @staticmethod
    def get_dissimilarity_val(
        first_data: th.Tensor, second_data: th.Tensor
    ) -> th.Tensor:
        return -th.linalg.norm(first_data - second_data)
