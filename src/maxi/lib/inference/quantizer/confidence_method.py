"""Binary Confidence Quantizer Method"""
from typing import Callable

import numpy as np
import torch

from .base_quantizer import BaseQuantizer
from ....data.data_types import Processor
from ....utils import quantizer_utils
from ....utils.general import to_numpy


class BinaryConfidenceMethod(BaseQuantizer):
    def __init__(
        self,
        preprocess: Processor = quantizer_utils.identity,
        confidence_calculator: Callable[[np.ndarray], float] = quantizer_utils.calculate_confidence,
    ) -> None:
        """Binary Confidence Quantizer Method

        Description:
            This quantizer method takes an arbitrary prediction, calculates the confidence score and \
            constructs an array of binary classification format. \
            In order to extract the confidence, the user needs to provide a suitable ``confidence calculator`` \
            method.

        Output format:
            [bs, [-(confidence - 0.5) * 2, (confidence - 0.5) * 2]]

        Args:
            preprocess (Processor, optional): Preprocessing procedure before quantizing. 
                Defaults to identity function.
            confidence_calculator (Callable[[np.ndarray], float], optional): Method to calculate the confidence. 
                Has to reduce the (preprocessed) prediction to a single value (1,). 
                Defaults to calculate_confidence.
        """
        super().__init__(preprocess)
        self.confidence_calculator = confidence_calculator

    def __call__(self, prediction: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Performs the quantization

        Args:
            prediction (np.ndarray): Any type of inference result (e.g. a segmentation mask) in [bs, w, h, c].

        Returns:
            np.ndarray: Prediction of binary classification format ([bs, [-(confidence - 0.5) * 2, (confidence - 0.5) * 2]])
        """
        preprocessed_pred = super().__call__(to_numpy(prediction))

        binary_clsfctn = np.zeros((len(preprocessed_pred), 2))
        array = np.array([-1.0, 1.0])

        for i, pred in enumerate(preprocessed_pred):
            # calculate confidence
            binary_clsfctn[i] = array * (self.confidence_calculator(pred) - 0.5) * 2

        return binary_clsfctn


class TorchBinaryConfidenceMethod(BinaryConfidenceMethod):
    def __init__(
        self,
        preprocess: Processor = quantizer_utils.identity,
        confidence_calculator: Callable[[np.ndarray], float] = quantizer_utils.calculate_confidence,
    ) -> None:
        """Binary Confidence Quantizer Method

        Description:
            This quantizer method takes an arbitrary prediction, calculates the confidence score and \
            constructs an array of binary classification format. \
            In order to extract the confidence, the user needs to provide a suitable ``confidence calculator`` \
            method.

        Output format:
            [-(confidence - 0.5) * 2, (confidence - 0.5) * 2]

        Args:
            preprocess (Processor, optional): Preprocessing procedure before quantizing. 
                Defaults to identity function.
            confidence_calculator (Callable[[np.ndarray], float], optional): Method to calculate the confidence. 
                Has to reduce the (preprocessed) prediction to a single value (1,). 
                Defaults to calculate_confidence.
        """
        super().__init__(preprocess)
        self.confidence_calculator = confidence_calculator
        self._tensor = torch.tensor([[-1.0, 1.0]], requires_grad=True)

    def __call__(self, prediction: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Performs the quantization

        Args:
            prediction (np.ndarray): Any type of inference result (e.g. a segmentation mask).

        Returns:
            np.ndarray: Prediction of binary classification format ([-(confidence - 0.5) * 2, (confidence - 0.5) * 2])
        """
        preprocessed_pred = self.preprocess(prediction)

        # calculate confidence score
        score = (self.confidence_calculator(preprocessed_pred) - 0.5) * 2
        return score * self._tensor
