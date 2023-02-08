"""CV2 SlicHandler class to handle superpixel generation."""
from typing import List, Union

import cv2
import numpy as np

from maxi.lib.image_segmentation.base_seg_handler import BaseSegmentationHandler


class SlicHandler(BaseSegmentationHandler):
    _channels_first = False

    def __init__(
        self,
        image: np.ndarray,
        sp_algorithm: str = "SLIC",
        region_size: int = 8,
        ruler: int = 200,
        num_iter: int = 10,
    ) -> None:
        """CV2 SlicHandler class to handle superpixel generation.

        References:
            https://docs.opencv.org/3.4/df/d6c/group__ximgproc__superpixel.html
            https://docs.opencv.org/3.4/d3/da9/classcv_1_1ximgproc_1_1SuperpixelSLIC.html

        Args:
            image (np.ndarray): Array containing the image of shape.
            sp_algorithm (str): CV2 superpixel algorithm to use: SLIC, SLICO, MSLIC. Defaults to "SLIC".
            region_size (int): Region size in pixels. Defaults to 8.
            ruler (int): Balances color-space proximity and image-space proximity. Defaults to 200.
            num_iter (int): Number of iterations. Defaults to 10.
        """
        self.sp_algorithm = SlicHandler._retrieve_sp_algorithm(sp_algorithm)
        self.region_size, self.ruler, self.num_iter = (
            region_size,
            ruler,
            num_iter,
        )
        super().__init__(image)

    @staticmethod
    def adjust_image_shape(image: np.ndarray) -> np.ndarray:
        if image.ndim not in [2, 3, 4]:
            raise ValueError("Image has to be 2D, 3D or 4D.")

        # Image has to be channels-last and 3D
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        if image.ndim == 4:
            image = image.squeeze(axis=0)

        if image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)
        return image

    @staticmethod
    def _retrieve_sp_algorithm(
        alg_name: str,
    ) -> "Union[cv2.ximgproc.SLIC, cv2.ximgproc.SLICO, cv2.ximgproc.MSLIC]":
        """Retrieves the CV2 superpixel algorithm class.

        Args:
            alg_name (str): CV2 superpixel algorithm to use: SLIC, SLICO, MSLIC.

        Raises:
            ValueError: Algorithm name not supported.

        Returns:
            Union[cv2.ximgproc.SLIC, cv2.ximgproc.SLICO, cv2.ximgproc.MSLIC]:
                CV2 superpixel algorithm class.
        """
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
        """Generates the superpixel seed.

        Args:
            img (np.ndarray): Array containing image of shape [W, H, C].

        Returns:
            Cv2SuperpixelSeed: Superpixel seed object.
        """
        # img = transformations.rescale_image_to_0_255(img)
        seeds = cv2.ximgproc.createSuperpixelSLIC(
            img,
            algorithm=self.sp_algorithm,
            region_size=self.region_size,
            ruler=self.ruler,
        )
        seeds.iterate(self.num_iter)
        return seeds

    def _build_label_images(self, img: np.ndarray) -> List[np.ndarray]:
        """Builds the label images from the superpixel seed.

        Args:
            img (np.ndarray): Array containing image of shape [W, H, C].
            seeds (Cv2SuperpixelSeed): Superpixel seed object.

        Returns:
            List[np.ndarray]: List containing the label images of shape [W, H, C].

        Note:
            The label images are the seperate superpixel images of the original image.
        """
        self.seed = self._generate_superpixel_seed(img)
        self._num_segments = self.seed.getNumberOfSuperpixels()

        label_map, num_labels = (
            self.seed.getLabels(),
            self.seed.getNumberOfSuperpixels(),
        )

        label_maps = [
            np.where(label_map == i, 1, 0).astype(np.uint8) for i in range(num_labels)
        ]

        return [
            cv2.bitwise_and(img, img, mask=label_maps[i]) for i in range(num_labels)
        ]
