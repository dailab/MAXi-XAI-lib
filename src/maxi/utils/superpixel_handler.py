from typing import List, Union

import cv2
import numpy as np

from . import transformations


class SuperpixelHandler:
    def __init__(self, image: np.ndarray, sp_algorithm: str, sp_kwargs: dict) -> None:
        self.image = image
        self.sp_algorithm, self.sp_kwargs = (
            SuperpixelHandler._retrieve_sp_algorithm(sp_algorithm),
            sp_kwargs,
        )
        self.seed = self._generate_superpixel_seed(image)
        self.label_images = SuperpixelHandler._build_label_images(image, self.seed)

    @property
    def ones_weight_vector(self) -> np.ndarray:
        return np.ones((self.num_superpixels), dtype=np.float32)

    @property
    def zeros_weight_vector(self) -> np.ndarray:
        return np.zeros((self.num_superpixels), dtype=np.float32)

    @property
    def num_superpixels(self) -> int:
        return self.seeds.getLabels()

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

        num_iterations = self.sp_kwargs.get("num_iterations", 25)
        sp_kwargs_copy = self.sp_kwargs.copy()
        if "num_iterations" in sp_kwargs_copy:
            del sp_kwargs_copy["num_iterations"]

        seeds = cv2.ximgproc.createSuperpixelSLIC(
            img, algorithm=cv2.ximgproc.MSLIC, **sp_kwargs_copy
        )
        seeds.iterate(num_iterations)
        return seeds

    @staticmethod
    def _build_label_images(img: np.ndarray, seeds) -> List[np.ndarray]:
        label_map, num_labels = (
            seeds.getLabels(),
            seeds.getNumberOfSuperpixels(),
        )

        label_maps = [
            np.where(label_map == i, 1, 0).astype(np.uint8) for i in range(num_labels)
        ]

        return [
            cv2.bitwise_and(img, img, mask=label_maps[i]) for i in range(num_labels)
        ]

    def generate_img_from_weight_vector(self, weight_vec: np.ndarray) -> np.ndarray:
        assert weight_vec.shape == (self.num_superpixels,), (
            "Weight vector shape mismatch. "
            "Needs to have the same number of entries as the number of superpixels."
        )
        res_img = np.zeros(self.image.shape, dtype=np.uint8)
        for i in range(self.num_superpixels):
            res_img += weight_vec[i] * self.label_images[i]
        return res_img
