"""Base Explanation Model"""
from abc import ABC
from typing import List

import numpy as np


class BaseSegmentationHandler(ABC):
    _channels_first = True

    def __init__(self, image: np.ndarray, *args, **kwargs) -> None:
        """**Abstract Class**: Base Class for Segmentation Handlers.

        Description:
            This class is used as an extra layer to the ``ExplanationGenerator`` to
            apply segmentation algorithms on an image and afterwards generate label images
            (images containing only one superpixel).
            These label images in turn are used to generate the explanations by applying the
            explanation methods directly on the superpixels.
            The ``BaseExplanationModel`` class is used to generate the explanations on solely
            the weight vector of the label images - reducing the complexity of the explanation.

        Args:
            image (np.ndarray): Array containing the image of shape [W, H, C]
            sp_algorithm (str): CV2 superpixel algorithm to use: SLIC, SLICO, MSLIC.
            sp_kwargs (dict): Superpixel algorithm kwargs.
        """
        self.image = image

    @property
    def label_images(self) -> List[np.ndarray]:
        if not hasattr(self, "_label_images"):
            raise NotImplementedError(
                "label_images property has not been initialized in the child class."
            )
        return self._label_images

    @property
    def num_segments(self) -> int:
        if not hasattr(self, "_num_segments"):
            raise NotImplementedError(
                "num_segments property has not been implemented in the child class."
            )
        return self._num_segments

    @property
    def ones_weight_vector(self) -> np.ndarray:
        return np.ones(self.num_segments, dtype=np.float32)

    @property
    def zeros_weight_vector(self) -> np.ndarray:
        return np.zeros(self.num_segments, dtype=np.float32)

    def adjust_image_shape(self, image: np.ndarray) -> np.ndarray:
        if image.ndim not in [2, 3, 4]:
            raise ValueError("Image has to be 2D, 3D or 4D.")

        # Image has to be channels-first and 3D
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        if image.ndim == 4:
            image = image.squeeze(axis=0)

        # channels first
        if self._channels_first:
            if image.shape[2] in [1, 3]:
                image = image.transpose(2, 0, 1)
        else:
            if image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
        return image

    def get_readjusted_labelimages(self) -> np.ndarray:
        """Readjusts the label images to the original image shape if necessary.

        Example:
            Original image has shape [1, 3, 256, 256]
            Label images have shape [256, 256, 3]
            ---------------------------------------------------------
            Readjusted label images have shape [3, 256, 256]

        Returns:
            np.ndarray: Array containing the readjusted label images.
        """
        return [img.reshape(self.image.shape[1:]) for img in self.label_images]

    def _build_label_images(self, img: np.ndarray, *args, **kwargs) -> List[np.ndarray]:
        """Builds the label images with the implemented algorithm.

        Args:
            img (np.ndarray): Array containing image.

        Returns:
            List[np.ndarray]: List containing the label images.

        Note:
            The label images are the seperate superpixel images of the original image.
        """
        raise NotImplementedError(
            "_build_label_images method has not been implemented."
        )

    def generate_img_from_weight_vector(self, weight_vec: np.ndarray) -> np.ndarray:
        """Generates an image from a weight vector.

        Args:
            weight_vec (np.ndarray): Weight vector for the superpixels of shape
                [number_of_superpixels,].

        Returns:
            np.ndarray: Generated image of 3d shape.
        """
        assert weight_vec.shape in [
            (self.num_segments,),
            (1, self.num_segments),
        ], "Weight vector shape mismatch."
        "Needs to have the same number of entries as the number of superpixels. \n"
        f"Got: {weight_vec.shape}, Expected: ({self.num_segments},)"

        if not hasattr(self, "_readjusted_label_images"):
            raise ValueError("Readjusted label images have not been initialized.")

        res_img = np.zeros(self.image.shape, dtype=np.float32)

        for i in range(self.num_segments):
            weight = weight_vec[i] if weight_vec.ndim == 1 else weight_vec[0, i]
            res_img += weight * self._readjusted_label_images[i]
        return res_img

    def generate_imgs_from_weight_vectors(self, weight_vecs: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self.generate_img_from_weight_vector(weight_vec).squeeze(axis=0)
                for weight_vec in weight_vecs
            ]
        )
