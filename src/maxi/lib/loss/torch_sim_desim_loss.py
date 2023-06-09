from .sim_desim_loss import SimDesimLoss

import torch as th


class Torch_SimDesimLoss(SimDesimLoss):
    def __init__(self, device: str = "cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_index = th.tensor(self.target_index, device=device)

        self.org_prediction = th.tensor(self.org_prediction, device=device)
        self.org_prediction_wo_target = th.tensor(
            self.org_prediction_wo_target, device=device
        )

        self.mask = th.ones_like(self.org_prediction, dtype=th.bool, device=device)
        self.mask[self.target_index] = False

    def get_loss(self, data: th.Tensor, *args, **kwargs) -> float:
        perturbed_pred = self.inference(self.org_img + data)
        perturbed_pred_wo_target = th.masked_select(perturbed_pred, self.mask)
        return Torch_SimDesimLoss.get_similarity_val(
            self.org_prediction[self.target_index], perturbed_pred[self.target_index]
        ) + Torch_SimDesimLoss.get_dissimilarity_val(
            self.org_prediction_wo_target, perturbed_pred_wo_target
        )

    @staticmethod
    def get_similarity_val(first_data: th.Tensor, second_data: th.Tensor) -> th.Tensor:
        return th.linalg.norm(first_data - second_data)

    @staticmethod
    def get_dissimilarity_val(
        first_data: th.Tensor, second_data: th.Tensor
    ) -> th.Tensor:
        return -th.linalg.norm(first_data - second_data)
