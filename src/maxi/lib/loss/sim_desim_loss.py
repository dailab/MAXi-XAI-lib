import numpy as np
from maxi.data.data_types import InferenceCall, X0_Generator
from .base_explanation_model import BaseExplanationModel


class SimDesimLoss(BaseExplanationModel):
    def __init__(
        self,
        org_img: np.ndarray,
        inference: InferenceCall,
        x0_generator: X0_Generator,
        lower: np.ndarray,
        upper: np.ndarray,
        target_index: int,
        *args,
        **kwargs
    ) -> None:
        """Similiarity Desimilarity Loss

        Args:
            org_img (np.ndarray): Original target image in [width, height, channels].
            inference (InferenceCall): The inference method of an external prediction entity.
            x0_generator (X0_Generator): Method to generate the initial object of optimization.
            lower (np.ndarray): Lower bound for the object of optimization. Has to be of same shape as org_img.
            upper (np.ndarray): Upper bound for the object of optimization. Has to be of same shape as org_img.
            target_index (int): Index of the desired target class.
        """
        super().__init__(
            org_img, inference, x0_generator, lower, upper, *args, **kwargs
        )
        self.target_index = target_index
        self.org_prediction = self.inference(self.org_img)
        self.org_prediction_wo_target = np.delete(self.org_prediction, target_index)

    def get_loss(self, data: np.ndarray, *args, **kwargs) -> float:
        """Computes the loss value for the given data

        Args:
            data (np.ndarray): An image in [width, height, channels].

        Returns:
            float: The actual loss value.
        """
        perturbed_pred = self.inference(self.org_img + data)
        perturbed_pred_wo_target = np.delete(perturbed_pred, self.target_index)
        return SimDesimLoss.get_similarity_val(
            self.org_prediction[self.target_index],
            perturbed_pred[self.target_index],
        ) + SimDesimLoss.get_dissimilarity_val(
            self.org_prediction_wo_target, perturbed_pred_wo_target
        )

    @staticmethod
    def get_similarity_val(first_data: np.ndarray, second_data: np.ndarray) -> float:
        """Computes the similarity value between the original image and the given data

        Args:
            data (np.ndarray): An image in [width, height, channels].

        Returns:
            float: The actual similarity value.
        """
        return np.linalg.norm(first_data - second_data)

    @staticmethod
    def get_dissimilarity_val(first_data: np.ndarray, second_data: np.ndarray) -> float:
        """Computes the similarity value between the original image and the given data

        Args:
            data (np.ndarray): An image in [width, height, channels].

        Returns:
            float: The actual similarity value.
        """
        return -np.linalg.norm(first_data - second_data)
