"""Parameter Seacher Wrapper"""
from copy import copy
from collections import OrderedDict
from typing import Union, Callable, Tuple, OrderedDict
from scipy.optimize import OptimizeResult

import numpy as np
import traceback

from ..explanation_generator import ExplanationGenerator
from ...computation_components.optimizer.base_optimizer import BaseOptimizer
from ...inference.inference_wrapper import InferenceWrapper
from ....data.data_types import MetaData
from ....utils import general


class ParameterSearchWrapper:
    def __init__(
        self,
        explanation_method: ExplanationGenerator,
        initial_l1: float = 1.0,
        initial_l2: float = 1.0,
        max_iter: int = 10,
    ):
        """_summary_

        Args:
            explanation_method (ExplanationGenerator): _description_
            initial_l1 (float, optional): _description_. Defaults to 1.0.
            initial_l2 (float, optional): _description_. Defaults to 1.0.
            max_iter (int, optional): _description_. Defaults to 10.
        """
        assert isinstance(explanation_method, ExplanationGenerator)
        self.explanation_method = explanation_method
        self.initial_l1, self.initial_l2 = initial_l1, initial_l2
        self.max_iter = max_iter

    def _explain_until_negative_loss(
        self,
        explanation_method: ExplanationGenerator,
        image: np.ndarray,
        inference_call: Union[Callable[[np.ndarray], np.ndarray], InferenceWrapper],
        l1: float,
        l2: float,
    ) -> Tuple[OrderedDict[str, np.ndarray], bool]:
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
            bool: Whether the explanation method produced negative (attack) loss values.
        """
        ParameterSearchWrapper._print_start_explanation_params(l1, l2)

        # set new l1 and l2 parameters
        explanation_method._optimizer_kwargs["l1"] = l1
        explanation_method._optimizer_kwargs["l2"] = l2

        _, _, optimizer = explanation_method._init_components(image, inference_call)

        results = OrderedDict()

        while explanation_method.iter_count <= explanation_method._num_iter:
            opt_result: OptimizeResult = optimizer.step()
            explanation_method.iter_count += 1

            if opt_result.loss < 0:
                results[str(explanation_method.iter_count)] = opt_result.x.copy()
                explanation_method.logging_cb(opt_result)
                ParameterSearchWrapper._print_found_negative_loss(l1, l2, explanation_method.iter_count)
                return results, True

            #: Every ``save_freq``'th iteration, object of the optimization is saved
            #: e.g. for CEM the perturbed image will be stored
            if general.check_epoch(
                explanation_method.iter_count, explanation_method.save_freq, explanation_method._num_iter
            ):
                results[str(explanation_method.iter_count)] = opt_result.x.copy()

            #: Every ``log_freq``'th iteration, the loss and l1 is logged on the terminal
            if explanation_method.verbose and general.check_epoch(
                explanation_method.iter_count, explanation_method.log_freq, explanation_method._num_iter
            ):
                explanation_method.logging_cb(opt_result)

        return results, False

    def run(
        self,
        image: np.ndarray,
        inference_call: Union[Callable[[np.ndarray], np.ndarray], InferenceWrapper],
    ) -> Tuple[OrderedDict[str, OrderedDict[str, np.ndarray]], Union[float, None], Union[float, None]]:
        """Method for starting the explanation procedure

        Args:
            image (np.ndarray): Image to be explained in [width, height, channels].
            inference_call (Union[Callable[[np.ndarray], np.ndarray], InferenceWrapper]): Inference method returning \
                explanation model compatible predictions. The prediction result needs to be a 2D array.

        Returns:
            Tuple[OrderedDict[str, OrderedDict[str, np.ndarray]], Union[float, None], Union[float, None]]:
                1. index of the tuple contains OrderedDict of savepoints keyed by the l1 and l2 value of that \
                    particular run. The keys follow this syntax: "l1_[l1 value]_l2_[l2 value]".
                2. index of the tuple contains the l1 value that produced negative loss. It's None, if after \
                    ```max_iter``` iterations of divion by 2 of l1 and l2 the attack loss is still positive.
                3. index of the tuple contains the l2 value that produced negative loss. It's None, if after \
                    ```max_iter``` iterations of divion by 2 of l1 and l2 the attack loss is still positive.
        """
        assert type(image) is np.ndarray and type(image) is not bool, "Image is of unsupported type"
        assert inference_call and type(inference_call) is not bool, "Inference is of None Type"

        # try:
        _i = 0
        l1, l2 = self.initial_l1, self.initial_l2
        found_negative = False
        results = OrderedDict()

        while not found_negative or _i < self.max_iter:
            explanation_method = copy(self.explanation_method)

            result, found_negative = self._explain_until_negative_loss(
                explanation_method, image, inference_call, l1, l2
            )

            results[f"l1_{l1}_l2_{l2}"] = result

            if found_negative:
                return results, l1, l2

            _i += 1
            l1, l2 = l1 / 2, l2 / 2

        ParameterSearchWrapper._print_not_able_to_find_negative_loss(self.max_iter, l1, l2)
        return results, None, None

        # except Exception as exc:
        #     print(f"An exception occured: \n {exc}")
        #     traceback.print_exc()
        #     exit()

    @staticmethod
    def _print_start_explanation_params(l1: float, l2: float, seperator: str = "#", line_length: int = 70) -> None:
        print(seperator * line_length)
        print(f"Start explanation with l1: {l1}, l2: {l2}")
        print(seperator * line_length + "\n")

    @staticmethod
    def _print_found_negative_loss(
        l1: float, l2: float, iter_count: int, seperator: str = "#", line_length: int = 70
    ) -> None:
        print("\n" + seperator * line_length)
        print(f"Found negative loss with l1: {l1}, l2: {l2} within {iter_count} optimization steps.")
        print(seperator * line_length)

    @staticmethod
    def _print_not_able_to_find_negative_loss(
        max_iter: int, l1: float, l2: float, seperator: str = "#", line_length: int = 70
    ) -> None:
        print("\n" + seperator * line_length)
        print(f"Was not able to find negative loss within {max_iter} iterations. \n " f"Last l1: {l1}, last l2: {l2}")
        print(seperator * line_length)
