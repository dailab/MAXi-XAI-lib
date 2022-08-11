"""Superpixel Explanation Generator"""

__all__ = ["SuperpixelExplanationGenerator"]

import traceback
from collections import OrderedDict
from typing import Any, Callable, List, Type, Tuple, OrderedDict, Dict, Union

import cv2
import numpy as np
from scipy.optimize import OptimizeResult

from .explanation_generator import ExplanationGenerator
from ..computation_components.gradient.gradient_estimator import URVGradientEstimator
from ..computation_components.gradient.base_gradient import BaseGradient
from ..computation_components.optimizer.base_optimizer import BaseOptimizer
from ..computation_components.optimizer.ada_exp_grad import AdaExpGradOptimizer
from ..loss.base_explanation_model import BaseExplanationModel
from ..loss.cem_loss import CEMLoss
from ..inference.inference_wrapper import InferenceWrapper
from ...data.data_types import MetaData
from ...utils import logger, general, transformations


class SuperpixelExplanationGenerator(ExplanationGenerator):
    def __init__(
        self,
        loss: Type[BaseExplanationModel] = CEMLoss,
        optimizer: Type[BaseOptimizer] = AdaExpGradOptimizer,
        gradient: Type[BaseGradient] = URVGradientEstimator,
        superpixel_mode: bool = False,
        sp_algorithm: str = "SLIC",
        loss_kwargs: Dict[str, str] = None,
        optimizer_kwargs: Dict[str, str] = None,
        gradient_kwargs: Dict[str, str] = None,
        sp_kwargs: Dict[str, str] = None,
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
            One can also specify the frequency of which savepoints are going to be created.

        Args:
            loss (Type[BaseExplanationModel], optional): Subclass of ``BaseExplanationModel`` - an explanation methods' \
                loss function. Defaults to CEMLoss.
            optimizer (Type[BaseOptimizer], optional): Subclass of ``BaseOptimizer`` - the desired optimization \
                algorithm. Defaults to AdaExpGradOptimizer.
            gradient (Type[BaseGradient], optional): Subclass instance of ``BaseGradient`` - a particular gradient \
                method. Defaults to GradientEstimator.
            loss_kwargs (Dict[str, str], optional): Keyword arguments to be parsed to the loss function initilization.
                Defaults to { "mode": "PP", "gamma": 75, "K": 10, "AE": None}.
            optimizer_kwargs (Dict[str, str], optional): Keyword arguments to be parsed to the optimizer initilization.
                Defaults to {"l1": 0.5, "l2": 0.5, "eta": 1.0}.
            gradient_kwargs (Dict[str, str], optional): Keyword arguments to be parsed to the gradient method \
                initilization. Defaults to {"mu": None}.
            num_iter (int, optional): Number of optimization iterations. Defaults to 30.
            save_freq (int, optional): Frequency of optimizer updates after which the object of optimization is saved. \
                The savepoints will be stored in an OrderedDict and eventually returned. Defaults to np.inf \
                (only result of last iteration is stored).
            verbose (bool, optional): Whether loss is printed. Defaults to False.
        """
        super().__init__(
            loss,
            optimizer,
            gradient,
            loss_kwargs,
            optimizer_kwargs,
            gradient_kwargs,
            num_iter,
            save_freq,
            verbose,
        )
        self.sp_algorithm, self.sp_kwargs = (
            SuperpixelExplanationGenerator._retrieve_sp_algorithm(sp_algorithm),
            sp_kwargs,
        )

    @staticmethod
    def _retrieve_sp_algorithm(
        alg_name: str,
    ) -> Union[cv2.ximgproc.SLIC, cv2.ximgproc.SLICO, cv2.ximgproc.MSLIC]:
        alg_name = alg_name.upper()
        if alg_name == "SLIC":
            return cv2.ximgproc.SLIC
        elif alg_name == "SLICO":
            return cv2.ximgproc.SLICO
        elif alg_name == "MSLIC":
            return cv2.ximgproc.MSLIC
        else:
            raise ValueError(f"'{alg_name}' is not a valid superpixel algorithm.")

    def _generate_superpixel_seed(self, img: np.ndarray) -> "Cv2SuperpixelSeed":
        img = np.expand_dims(img.squeeze(axis=0), axis=-1)
        img = transformations.rescale_image_to_0_255(img)

        seeds = cv2.ximgproc.createSuperpixelSLIC(
            img, algorithm=cv2.ximgproc.MSLIC, region_size=2, ruler=200
        )
        seeds.iterate(self.sp_kwargs["num_iterations"])
        self.seed = seeds
        return seeds

    def _build_label_images(self, img: np.ndarray) -> List[np.ndarray]:
        label_map, num_labels = (
            self.seeds.getLabels(),
            self.seeds.getNumberOfSuperpixels(),
        )

        label_maps = [
            np.where(label_map == i, 1, 0).astype(np.uint8) for i in range(num_labels)
        ]

        self.label_images = [
            cv2.bitwise_and(img, img, mask=label_maps[i]) for i in range(num_labels)
        ]
        return self.label_images

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
            Tuple[BaseExplanationModel, BaseGradient, BaseOptimizer]: [description]
        """
        self._generate_superpixel_seed(image)
        self._build_label_images(image)

        # Loss function
        loss_instance: BaseExplanationModel = self.loss(
            inference=inference_call,
            org_img=image,
            **self._loss_kwargs,
        )

        # Gradient calculation
        gradient_instance: BaseGradient = self.gradient(
            loss=loss_instance,
            img_size=image.size,
            **self._gradient_kwargs,
        )

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
                results[str(self.iter_count)] = opt_result.x.copy()

            #: Every ``log_freq``'th iteration, the loss and l1 is logged on the terminal
            if self.verbose and general.check_epoch(
                self.iter_count, self.log_freq, self._num_iter
            ):
                self.logging_cb(opt_result)

        return results

    def __copy__(self):
        raise NotImplementedError("Copy is not yet implemented for this class.")


class SuperpixelHandler:
    def __init__(self, image: np.ndarray, sp_algorithm: str, sp_kwargs: dict) -> None:
        self.image = image
        self.sp_algorithm, self.sp_kwargs = (
            SuperpixelExplanationGenerator._retrieve_sp_algorithm(sp_algorithm),
            sp_kwargs,
        )

    @staticmethod
    def _retrieve_sp_algorithm(
        alg_name: str,
    ) -> Union[cv2.ximgproc.SLIC, cv2.ximgproc.SLICO, cv2.ximgproc.MSLIC]:
        alg_name = alg_name.upper()
        if alg_name == "SLIC":
            return cv2.ximgproc.SLIC
        elif alg_name == "SLICO":
            return cv2.ximgproc.SLICO
        elif alg_name == "MSLIC":
            return cv2.ximgproc.MSLIC
        else:
            raise ValueError(f"'{alg_name}' is not a valid superpixel algorithm.")

    def _generate_superpixel_seed(self, img: np.ndarray) -> "Cv2SuperpixelSeed":
        img = np.expand_dims(img.squeeze(axis=0), axis=-1)
        img = transformations.rescale_image_to_0_255(img)

        seeds = cv2.ximgproc.createSuperpixelSLIC(
            img, algorithm=cv2.ximgproc.MSLIC, region_size=2, ruler=200
        )
        seeds.iterate(self.sp_kwargs["num_iterations"])
        self.seed = seeds
        return seeds

    def _build_label_images(self, img: np.ndarray) -> List[np.ndarray]:
        label_map, num_labels = (
            self.seeds.getLabels(),
            self.seeds.getNumberOfSuperpixels(),
        )

        label_maps = [
            np.where(label_map == i, 1, 0).astype(np.uint8) for i in range(num_labels)
        ]

        self.label_images = [
            cv2.bitwise_and(img, img, mask=label_maps[i]) for i in range(num_labels)
        ]
        return self.label_images
