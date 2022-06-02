"""Asynchronous Explanation Wrapper"""

__all__ = ["AsyncExplanationWrapper"]
from copy import copy
from typing import Dict, List, Tuple, OrderedDict, Union, Callable

import concurrent.futures
import numpy as np

from ...explanation.explanation_generator import ExplanationGenerator
from ...inference.inference_wrapper import InferenceWrapper
from ....data.data_types import MetaData


class AsyncExplanationWrapper:
    def __init__(
        self,
        explanation_method: ExplanationGenerator,
        # save_res_cb: SaveResultCallback = None,
        n_workers: int = 1,
    ) -> None:
        """Asynchronous Explanation Wrapper

        Args:
            explanation_method (ExplanationGenerator): Instance of the ExplanationGenerator class.
            n_workers (int): Number of parallel worker threads.
            
        Note:
            When this class wraps the ```ExplanationGenerator``` and the user has specified a custom lower and \
            upper bound (optimizer keys _lower_ and _upper_), 
        """
        assert isinstance(explanation_method, ExplanationGenerator)
        self._explanation_generators = [copy(explanation_method) for _ in range(n_workers)]
        self.n_workers = n_workers
        # self.save_res_cb = save_res_cb

    def run(
        self,
        original_images: List[np.ndarray],
        inference_calls: Dict[bytes, Union[Callable[[np.ndarray], np.ndarray], InferenceWrapper]],
        _meta_data: Dict[bytes, MetaData] = None,
    ) -> List[Tuple[OrderedDict[str, np.ndarray], MetaData]]:
        """Method to start the asynchronous explanation procedure

        Args:
            original_images (List[np.ndarray]): List containing the target images in [width, height, channels].
            inference_calls (Dict[bytes, InferenceWrapper]): Dictionary, where keys are the image represented in bytes and \
                the belonging value the corresponding inference method.
            _meta_data (Dict[bytes, MetaData], optional):
                Dictionary, where keys have to be the image as bytes and \
                the belonging value the corresponding image meta data. Defaults to None.

        Returns:
            List[Tuple[OrderedDict[str, np.ndarray], MetaData]]: List containing the \
                result tuple for all input images. An entry (tuple) made up of: \
                raw results, meta data.
        """
        assert type(original_images) is list and type(inference_calls) is dict

        results_and_meta: List[
            Tuple[
                OrderedDict[str, np.ndarray],
                MetaData,
            ]
        ] = []
        future_to_bytes: Dict[concurrent.futures.Future, bytes] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for i, image in enumerate(original_images):
                key = image.tobytes()

                try:
                    inference_call = inference_calls[key]
                    meta_data = _meta_data[key] if _meta_data else {}
                except KeyError:
                    raise KeyError(
                        f"Couldn't retrieve the inference method or meta data for the {i}th image in the list. "
                        "Make sure that the dictionary is keyed by the string byte presentation of the array. \n"
                        "[You can get it by calling: *image_matrix*.tobytes() ]"
                    )

                future_to_bytes[
                    executor.submit(
                        self._explanation_generators[i].run,
                        image,
                        inference_call,
                        meta_data,
                    )
                ] = key

            for future in concurrent.futures.as_completed(future_to_bytes):
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"An exception occured: {exc}!")
                else:
                    # if self.save_res_cb:
                    #     self.save_res_cb(
                    #         _meta_data=result[2],
                    #         raw_result=result[1],
                    #         transformed_res=result[0],
                    #     )
                    results_and_meta.append(result)
        return results_and_meta
