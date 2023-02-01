from typing import List, Union

import cv2
import numpy as np

from maxi.lib.image_segmentation.base_seg_handler import BaseSegmentationHandler


class SuperpixelHandler(BaseSegmentationHandler):
    def __init__(self, image: np.ndarray, sp_algorithm: str, sp_kwargs: dict) -> None:
        """SuperpixelHandler class to handle superpixel generation.

        Description:
            This class is used to generate superpixels from an image and to generate


        Args:
            image (np.ndarray): Array containing the image of shape [W, H, C]
            sp_algorithm (str): CV2 superpixel algorithm to use: SLIC, SLICO, MSLIC.
            sp_kwargs (dict): Superpixel algorithm kwargs.
        """
        self.image = image
        # Currently only MSLIC is supported (hardcoded in _generate_superpixel_seed)
        self.sp_algorithm, self.sp_kwargs = (
            SuperpixelHandler._retrieve_sp_algorithm(sp_algorithm),
            sp_kwargs,
        )
        self.adj_image = SuperpixelHandler.adjust_image_shape(image)
        self.seed = self._generate_superpixel_seed(self.adj_image)
        self._num_segments = self.seed.getNumberOfSuperpixels()
        self._label_images = SuperpixelHandler._build_label_images(
            self.adj_image, self.seed
        )
        self._readjusted_label_images = self.get_readjusted_labelimages()

    @staticmethod
    def adjust_image_shape(image: np.ndarray) -> np.ndarray:
        # Image has to be channels-last and 3D
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        if image.ndim == 4:
            image = image.squeeze(axis=0)

        if image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)
        return image

    def get_readjusted_labelimages(self) -> np.ndarray:
        """Readjusts the label images to the original image shape.

        Example:
            Original image has shape [1, 3, 256, 256]
            Label images have shape [256, 256, 3]
            ---------------------------------------------------------
            Readjusted label images have shape [3, 256, 256]

        Returns:
            np.ndarray: Array containing the readjusted label images.
        """
        if self.label_images[0].shape not in self.image.shape:
            image = [img.reshape(self.image.shape[1:]) for img in self.label_images]
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
        assert self.sp_kwargs, "No superpixel algorithm kwargs given."
        # img = transformations.rescale_image_to_0_255(img)

        num_iterations = self.sp_kwargs.get("num_iterations", 25)
        sp_kwargs_copy = self.sp_kwargs.copy()
        if "num_iterations" in sp_kwargs_copy:
            del sp_kwargs_copy["num_iterations"]

        seeds = cv2.ximgproc.createSuperpixelSLIC(
            img, algorithm=self.sp_algorithm, **sp_kwargs_copy
        )
        seeds.iterate(num_iterations)
        return seeds

    @staticmethod
    def _build_label_images(
        img: np.ndarray, seeds: "Cv2SuperpixelSeed"
    ) -> List[np.ndarray]:
        """Builds the label images from the superpixel seed.

        Args:
            img (np.ndarray): Array containing image of shape [W, H, C].
            seeds (Cv2SuperpixelSeed): Superpixel seed object.

        Returns:
            List[np.ndarray]: List containing the label images of shape [W, H, C].

        Note:
            The label images are the seperate superpixel images of the original image.
        """
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
