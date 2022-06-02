"""Inference Wrapper Module"""

__all__ = ["InferenceWrapper"]

from typing import Type

from numpy import ndarray

from .quantizer import BaseQuantizer, IdentityMethod
from .processing.identity_processor import identity
from ...data.data_types import InferenceCall, Processor


class InferenceWrapper:
    def __init__(
        self,
        inference_model: InferenceCall,
        quantizer: Type[BaseQuantizer] = None,
        preprocess: Processor = identity,
    ) -> None:
        """This class encapsulates the essential modules regarding the inference \
            as well as the pre- and postprocessing of the data. \
            As a result, it provides an algorithm-conforming interpretation \
            of the model's prediction. \
            Note that the output containing the classification scores has to be a 2D array.

        Args:
            inference_model (InferenceCall): 
                Inference call.
            quantizer (InferenceQuantizer, optional): 
                Instance of a CEMQuantizer subclass, translates the predicted \
                mask into a class distribution. Defaults to None. (None -> IdentityMethod)
            preprocess (Callable[[ndarray], ndarray], optional): Processing procedure prior to inference. Defaults to identity.
            postprocess (Callable[[ndarray], ndarray], optional): Processing procedure post inference. Defaults to identity.
        """
        self.inference_model = inference_model
        self.quantizer = quantizer or IdentityMethod()
        self.preprocess = preprocess

    def __call__(self, img: ndarray, *args, **kwargs) -> ndarray:
        """ Takes an image represented as a ndarray, updates the tile fetcher \
            with the new data, runs the inference and eventually transforms \
            the segmentation masks into a binary classification distribution utilizing an arbitrary \
            quantizer method.

        Args:
            img (NDArray[Int]): Image data in [width, height, channels]

        Returns:
            NDArray[Float]: Array containing a valued class distribution, N entries for N classes. 
        """
        return self.quantizer(self.inference_model(self.preprocess(img)))
