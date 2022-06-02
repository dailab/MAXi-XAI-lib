from typing import List, Union

import numpy as np
import yaml

from collections import OrderedDict
from datetime import datetime
from matplotlib.image import imsave
from pathlib import Path
from warnings import warn

from maxi.utils.transformations import reverse_normalize
from maxi.data.data_types import EntityRect, MetaData
from maxi.utils.general import increment_path


class SaveResultCallback:
    def __init__(
        self,
        parent_dir: str = Path("xai") / "run",
        raw_res_range: tuple = (-1, 1),
        transformed_res_range: tuple = (0, 255),
        gray_scale: bool = False,
        rev_norm_on_raw: bool = False,
        ignore_kw: List[str] = ["original_image"],
    ) -> None:
        """Callback to save results and meta information of respective image.

        Args:
            target_directory (str, optional):
                [description]. Defaults to Path("xai")/"run".
        """
        self.parent_dir = parent_dir
        self.target_dir = None

        self.is_gray_scale = gray_scale
        self.rev_norm = rev_norm_on_raw

        self._raw_res_range, self._transformed_res_range = (
            raw_res_range,
            transformed_res_range,
        )

        self._ignore_kw = ignore_kw

    def __call__(
        self,
        raw_result: Union[np.ndarray, OrderedDict],
        transformed_res: Union[np.ndarray, OrderedDict],
        _meta_data: MetaData = None,
        *args,
        **kwargs,
    ) -> None:
        """[summary]

        Optional Meta Data Keys:
            "relative_rect_coords":
                ...
            "original_image":
                ...
            

        Args:
            raw_result (np.ndarray):
                Resulting image from running the explanation method.
            transformed_res (np.ndarray):
                Transformed image after application of the result processor of\
                the explanation method. 
                (raw_result==transformed_res when no result processing is conducted) 
            _meta_data (MetaData):
                Meta information for the current explanation.

        Returns:
            np.ndarray: [description]
        """
        # check if parsed result is a single array or ordereddict
        raw_is_dict, presenter_is_dict = (
            isinstance(raw_result, OrderedDict),
            isinstance(transformed_res, OrderedDict),
        )
        assert raw_is_dict == presenter_is_dict, (
            "Raw results and transformed results have to saved as same "
            f"datatype ({type(raw_result)} != {type(presenter_is_dict)}"
        )

        # automatically increments directory name and creates it
        if self.target_dir is None:
            self.target_dir = increment_path(self.parent_dir)  # new target dir

        # generate subdir name for this series of explanation
        if "relative_rect_coords" in _meta_data:
            entity_rect: EntityRect = _meta_data["relative_rect_coords"]
            dir_name = (
                "x"
                + str(entity_rect["x"])
                + "_y"
                + str(entity_rect["y"])
                + "_w"
                + str(entity_rect["w"])
                + "_h"
                + str(entity_rect["h"])
            )
        else:
            dir_name = str(datetime.now())

        dir_path = self.target_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=False)

        original_img = None
        # try to retrieve reference image
        if "original_image" in _meta_data:
            original_img = _meta_data["original_image"]
            imsave(dir_path / "original_img.png", original_img)

        # save procedure when raw and presenter data stored as dict
        if raw_is_dict and presenter_is_dict:
            # save results from each stage in own subdir
            for raw_res, trans_res in zip(list(raw_result.items()), list(transformed_res.items())):
                savepoint_path = dir_path / f"iter_{raw_res[0]}"
                savepoint_path.mkdir(parents=True, exist_ok=False)

                self.save_result(savepoint_path, raw_res[1], trans_res[1], original_img)
        else:
            self.save_result(dir_path, raw_result, transformed_res, original_img)

        # save meta data
        if _meta_data:
            tmp_meta_data = _meta_data
            tmp_meta_data = del_keys_from_dict(tmp_meta_data, self._ignore_kw)

            with open(dir_path / "meta_data.yml", "w+") as f:
                yaml.dump(tmp_meta_data, f)

    def save_result(
        self,
        path: Path,
        raw_res: np.ndarray,
        transformed_res: np.ndarray,
        original_image: np.ndarray = None,
    ):
        """Saves the generated results in numpy format as well as an image.

        Args:
            raw_res (np.ndarray): [description]
            transformed_res (np.ndarray): [description]
        """
        # optionally reverse normalization on raw results
        if original_image is not None and self.rev_norm:
            raw_res = reverse_normalize(raw_res, original_image)

        if 1 in raw_res.shape:
            raw_res = remove_single_dims(raw_res)

        if 1 in transformed_res.shape:
            transformed_res = remove_single_dims(transformed_res)

        if np.amin(raw_res) < 0:
            warn("Raw result contains negative pixel values! " + "Generated image might not represent the real result.")

        if np.amin(transformed_res) < 0:
            warn(
                "Transformed result contains negative pixel values! "
                + "Generated image might not represent the real result."
            )

        imsave(
            path / "raw_img.png",
            raw_res,
            vmin=min(self._raw_res_range) if self.rev_norm else min(self._transformed_res_range),
            vmax=max(self._raw_res_range) if self.rev_norm else max(self._transformed_res_range),
            cmap="gray" if self.is_gray_scale else None,
        )
        np.save(path / "raw_result.npy", raw_res)

        if not np_equal(raw_res, transformed_res):
            if transformed_res.shape[-1] == 1:  # channels last
                transformed_res = np.squeeze(transformed_res, -1)
            imsave(
                path / "transformed_result.png",
                transformed_res,
                vmin=min(self._transformed_res_range),
                vmax=max(self._transformed_res_range),
                cmap="gray" if self.is_gray_scale else None,
            )
            np.save(path / "transformed_result.npy", transformed_res)


def remove_single_dims(array: np.ndarray) -> np.ndarray:
    while 1 in array.shape:
        idx = array.shape.index(1)
        array = array.squeeze(idx)
    return array


def np_equal(a: np.ndarray, b: np.ndarray) -> bool:
    mask = a == b
    return mask.all()


def del_keys_from_dict(_dict: dict, keys: List[str]) -> None:
    for key in keys:
        _dict.pop(key, None)
    return _dict
