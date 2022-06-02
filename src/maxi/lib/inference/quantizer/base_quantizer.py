"""Base Quantizer Class"""
from abc import ABC, abstractmethod

import numpy as np

from ....data.data_types import Processor
from ....utils import quantizer_utils


class BaseQuantizer(ABC):
    def __init__(self, preprocess: Processor = quantizer_utils.identity) -> None:
        """**Base Class**: Abstract class for the BaseQuantizer methods
    
        Description:
            The BaseQuantizer translates the produced prediction of an external inference entity into \
            a explanation method compatible format. The quantization function has to be continuous and 
            sensitive to very small changes in the prediction. \
            It has to be parsed to the ``InferenceWrapper``.
        
        Example:
            The _CEM loss function_ requires the prediction to be of _binary/multiclass classification_ format. \
            Now given that your inference model produces _segmentation masks_ of shape [w, h, c], \
            it is required to translate those into a _n_-class (n depends on the underlying use case) classification \
            of shape [n,].
            That is essentially the function of the ``Quantizer``.
            

        Args:
            preprocess (Processor, optional): Additional preprocessing procedure. Defaults to _identity_ function.
        """
        self.preprocess = preprocess

    @abstractmethod
    def __call__(self, prediction: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Performs the quantization

        Args:
            prediction (np.ndarray): Any type of inference result (e.g. a segmentation mask).

        Raises:
            NotImplementedError: Method has to be implemented by the user.

        Returns:
            np.ndarray: Explanation method compatible prediction (e.g. binary classification).
        """
        return self.preprocess(prediction)
