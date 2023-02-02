"""Explanation Generator"""

__all__ = ["ExplanationGenerator"]

import traceback
from collections import OrderedDict
from typing import Callable, Type, Tuple, OrderedDict, Dict, Union

import numpy as np
from scipy.optimize import OptimizeResult

from ..computation_components.gradient.gradient_estimator import URVGradientEstimator
from ..computation_components.gradient.base_gradient import BaseGradient
from ..computation_components.optimizer.base_optimizer import BaseOptimizer
from ..computation_components.optimizer.ada_exp_grad import AdaExpGradOptimizer
from ..loss.base_explanation_model import BaseExplanationModel
from ..loss.cem_loss import CEMLoss
from ..inference.inference_wrapper import InferenceWrapper
from ..image_segmentation import BaseSegmentationHandler
from ...data.data_types import MetaData
from ...utils import logger, general


class ExplanationGenerator:
    def __init__(
        self,
        loss: Type[BaseExplanationModel] = CEMLoss,
        optimizer: Type[BaseOptimizer] = AdaExpGradOptimizer,
        gradient: Type[BaseGradient] = URVGradientEstimator,
        sg_algorithm: Type[BaseSegmentationHandler] = None,
        loss_kwargs: Dict[str, str] = None,
        optimizer_kwargs: Dict[str, str] = None,
        gradient_kwargs: Dict[str, str] = None,
        sg_kwargs: Dict[str, str] = None,
        num_iter: int = 30,
        save_freq: int = np.inf,
        verbose: bool = False,
    ) -> None:
        """XAI Explanation Generator

        Description:
            This class is the central module connecting all the library components with the purpose of generating an \
            explanation for a model's prediction. Those main components consist of the ``ExplanationModel``, \
            ``Optimizer``, ``GradientMethod`` and the ``InferenceWrapper``. \
            The ``Optimizer`` acts as sort of the engine of the explanation procedure, whereas the ``ExplanationModel``\
            poses as the loss function of an explanation method incorporated by the optimizer algorithm.
            After calling the ``run()`` method the optimizer starts producing a perturbed image (x_0) which \
            will get altered and optimized. \
            Optionally, the ``SegmentationHandler`` can be used to segment the image into regions of interest. \
            In order to use the ``SegmentationHandler`` the ``ExplanationModel`` has to be compatible with it. \
            See the ``SegmentationHandler`` documentation for more information. \
            One can also specify the frequency of which savepoints are going to be created.

        Args:
            loss (Type[BaseExplanationModel], optional): Subclass of ``BaseExplanationModel`` - an explanation methods' \
                loss function. Defaults to CEMLoss.
            optimizer (Type[BaseOptimizer], optional): Subclass of ``BaseOptimizer`` - the desired optimization \
                algorithm. Defaults to AdaExpGradOptimizer.
            gradient (Type[BaseGradient], optional): Subclass instance of ``BaseGradient`` - a particular gradient \
                method. Defaults to GradientEstimator.
            sg_algorithm (Type[BaseSegmentationHandler], optional): Subclass of ``BaseSegmentationHandler`` - \
                chosen segmentation algorithm. Defaults to None.
            loss_kwargs (Dict[str, str], optional): Keyword arguments to be parsed to the loss function initilization.
                Defaults to { "mode": "PP", "gamma": 75, "K": 10, "AE": None}.
            optimizer_kwargs (Dict[str, str], optional): Keyword arguments to be parsed to the optimizer initilization.
                Defaults to {"l1": 0.5, "l2": 0.5, "eta": 1.0}.
            gradient_kwargs (Dict[str, str], optional): Keyword arguments to be parsed to the gradient method \
                initilization. Defaults to {"mu": None}.
            sg_kwargs (Dict[str, str], optional): Keyword arguments to be parsed to the segmentation algorithm. 
                Defaults to None. \
            num_iter (int, optional): Number of optimization iterations. Defaults to 30.
            save_freq (int, optional): Frequency of optimizer updates after which the object of optimization is saved. \
                The savepoints will be stored in an OrderedDict and eventually returned. Defaults to np.inf \
                (only result of last iteration is stored).
            verbose (bool, optional): Whether loss is printed. Defaults to False.
        """
        if loss_kwargs is None:
            loss_kwargs = {"mode": "PP", "gamma": 75, "K": 10, "AE": None}
        if optimizer_kwargs is None:
            optimizer_kwargs = {"l1": 0.5, "l2": 0.5, "eta": 1.0}
        if gradient_kwargs is None:
            gradient_kwargs = {"mu": None}

        ExplanationGenerator._check_parsed_args(
            sg_algorithm, loss, optimizer, gradient, sg_kwargs
        )
        self.loss, self.optimizer, self.gradient = loss, optimizer, gradient
        self._loss_kwargs, self._optimizer_kwargs, self._gradient_kwargs = (
            loss_kwargs,
            optimizer_kwargs,
            gradient_kwargs,
        )
        self.iter_count = 0
        self._num_iter = num_iter

        self.log_freq, self.save_freq = 1, min(save_freq, num_iter)

        (
            self._superpixel_mode,
            self._sg_algorithm,
            self._sg_kwargs,
            self.superpixel_handler,
        ) = ("superpixel" in loss.__name__.lower(), sg_algorithm, sg_kwargs, None)

        if self._superpixel_mode and self._sg_kwargs["alg_kwargs"] is None:
            self._sg_kwargs["alg_kwargs"] = {}

        self.logging_cb = logger._callback
        self.verbose = verbose

    @staticmethod
    def _check_parsed_args(
        sg_algorithm: Type[BaseSegmentationHandler],
        loss: Type[BaseExplanationModel],
        optimizer: Type[BaseOptimizer],
        gradient: Type[BaseGradient],
        sg_kwargs: Dict[str, str],
    ) -> None:
        if sg_algorithm and not issubclass(sg_algorithm, BaseSegmentationHandler):
            raise TypeError(
                "Segmentation algorithm must be a subclass of BaseSegmentationHandler"
            )
        if sg_algorithm and "alg_kwargs" not in sg_kwargs and sg_kwargs:
            raise ValueError("'alg_kwargs' key must be specified in sg_kwargs.")
        if not issubclass(loss, BaseExplanationModel):
            raise TypeError("Loss must be a subclass of BaseExplanationModel")
        if not issubclass(optimizer, BaseOptimizer):
            raise TypeError("Optimizer must be a subclass of BaseOptimizer")
        if not issubclass(gradient, BaseGradient):
            raise TypeError("Gradient must be a subclass of BaseGradient")

    def _init_components(
        self,
        image: np.ndarray,
        inference_call: Union[Callable[[np.ndarray], np.ndarray], InferenceWrapper],
    ) -> Tuple[BaseExplanationModel, BaseGradient, BaseOptimizer]:
        """Initializes the components with the given image and inference function.

        Args:
            image (np.ndarray): Image to be explained in [width, height, channels].
            inference_call (Union[Callable[[np.ndarray], np.ndarray], InferenceWrapper]): Inference method returning explanation model \
                compatible predictions. E.g. classification format as in [0.3, 0.2, 3.7].
        Returns:
            Tuple[BaseExplanationModel, BaseGradient, BaseOptimizer]: Initialized components.
        """

        # Segmentation mode
        if self._superpixel_mode:
            self.superpixel_handler = self._sg_algorithm(
                image=image, **self._sg_kwargs["alg_kwargs"]
            )

        # Loss function
        loss_instance: BaseExplanationModel = self.loss(
            inference=inference_call,
            org_img=image,
            superpixel_handler=self.superpixel_handler,
            **self._loss_kwargs,
        )

        # Gradient calculation
        gradient_instance: BaseGradient = self.gradient(
            loss=loss_instance,
            img_size=image.size,
            superpixel_mode=self._superpixel_mode,
            **self._gradient_kwargs,
        )

        if self._superpixel_mode:
            image = self.superpixel_handler.ones_weight_vector

        # Optimization
        optimizer_instance: BaseOptimizer = self.optimizer(
            org_img=image,
            loss=loss_instance.get_loss,
            gradient=gradient_instance,
            num_iter=self._num_iter,
            x0=loss_instance._x0_generator(image),
            lower=loss_instance._lower,
            upper=loss_instance._upper,
            p_cb_epoch=self.save_freq,
            **self._optimizer_kwargs,
        )

        assert type(gradient_instance) in loss_instance.compatible_grad_methods, (
            "Gradient method is not compatible with specified loss class. "
            f"{type(gradient_instance)} <=> {type(loss_instance)} \ņ"
            f"Gradient must be one of the following: {loss_instance.compatible_grad_methods}"
        )

        return loss_instance, gradient_instance, optimizer_instance

    def _explain(self, optimizer: BaseOptimizer) -> OrderedDict[str, np.ndarray]:
        """Starts the explanation procedure. 
        
        Description:
            It will iterate over ``num_iter`` times and apply the \
            ``self.step()`` function which differs between different optimization algorithms. \
            Essentially, ``self.step()`` has to be implemented when adding a new optimization class. \
            The explanations are saved every ``save_freq``'th iteration as savepoints in an OrderedDict.

        Args:
            optimizer (BaseOptimizer): Optimizer instance.

        Returns:
            OrderedDict[str, np.ndarray]: Holds the optimization result of every savepoint. \
                Dictionary keys represent the iteration count when the image was saved. Corresponding value consists \
                of the produced explanation of respective iteration.
        """
        results = OrderedDict()

        while self.iter_count <= self._num_iter:
            opt_result: OptimizeResult = optimizer.step()
            self.iter_count += 1

            #: Every ``save_freq``'th iteration, object of the optimization is saved
            #: e.g. for CEM the perturbed image will be stored
            if general.check_epoch(self.iter_count, self.save_freq, self._num_iter):
                res = opt_result.x.copy()
                results[str(self.iter_count)] = (
                    self.superpixel_handler.generate_img_from_weight_vector(res)
                    if self._superpixel_mode
                    else res
                )

            #: Every ``log_freq``'th iteration, the loss and l1 is logged on the terminal
            if self.verbose and general.check_epoch(
                self.iter_count, self.log_freq, self._num_iter
            ):
                self.logging_cb(opt_result)

        return results

    def run(
        self,
        image: np.ndarray,
        inference_call: Union[Callable[[np.ndarray], np.ndarray], InferenceWrapper],
        meta_data: MetaData = None,
    ) -> Tuple[OrderedDict[str, np.ndarray], MetaData]:
        """Method for starting the explanation procedure

        Args:
            image (np.ndarray): Image to be explained in [width, height, channels].
            inference_call (Union[Callable[[np.ndarray], np.ndarray], InferenceWrapper]): Inference method returning explanation model \
                compatible predictions. The prediction result needs to be a 2D array.
            meta_data (MetaData, optional): Image meta data. Defaults to None.

        Returns:
            Tuple[OrderedDict[str, np.ndarray], OrderedDict[str, np.ndarray], MetaData]:
                OrderedDict containing the explanations, meta data to the explained image.
        """
        assert (
            type(image) is np.ndarray and type(image) is not bool
        ), "Image is of unsupported type"
        assert (
            inference_call and type(inference_call) is not bool
        ), "Inference is of None Type"

        # try:
        loss, gradient, optimizer = self._init_components(image, inference_call)
        return self._explain(optimizer), meta_data
        # except Exception as exc:
        #     print(f"An exception occured: \n {exc}")
        #     traceback.print_exc()
        #     exit()

    def __copy__(self):
        raise NotImplementedError(
            "Copy is not implemented. Asynchronous execution currently not supported."
        )
        return type(self)(
            self.loss,
            self.optimizer,
            self.gradient,
            self._loss_kwargs,
            self._optimizer_kwargs,
            self._gradient_kwargs,
            self._num_iter,
            self.save_freq,
            self.verbose,
        )
