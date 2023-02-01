from typing import List
from maxi.lib.image_segmentation.base_seg_handler import BaseSegmentationHandler

# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2


class SegmentationHandler(BaseSegmentationHandler):
    def __init__(
        self, image: np.ndarray, sg_kwargs: dict = None, *args, **kwargs
    ) -> None:
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
        if not sg_kwargs:
            sg_kwargs = {
                "mean_shifting_spatial_radius": 14,
                "mean_shifting_color_radius": 32,
                "peak_local_max_min_dist": 20,
                "invert": True,
            }
        self.segmentation_kwargs = sg_kwargs
        self.adj_image = SegmentationHandler.adjust_image_shape(image)
        self._label_images = self._build_label_images(self.adj_image)
        self._readjusted_label_images = self.get_readjusted_labelimages()

    @staticmethod
    def adjust_image_shape(image: np.ndarray) -> np.ndarray:
        if image.ndim not in [2, 3, 4]:
            raise ValueError("Image has to be 2D, 3D or 4D.")

        # Image has to be channels-first and 3D
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        if image.ndim == 4:
            image = image.squeeze(axis=0)

        # channels first
        if image.shape[2] in [1, 3]:
            image = image.transpose(2, 0, 1)
        return image

    def _build_label_images(self, img: np.ndarray) -> List[np.ndarray]:
        """Builds the label images from the superpixel seed.

        Args:
            img (np.ndarray): Array containing image of shape [W, H, C].
            seeds (Cv2SuperpixelSeed): Superpixel seed object.

        Returns:
            List[np.ndarray]: List containing the label images of shape [W, H, C].

        Note:
            The label images are the seperate superpixel images of the original image.
            During the label image generation, the "_num_segments" attribute
            has to be set.
        """
        # load the image and perform pyramid mean shift filtering
        # to aid the thresholding step
        image = cv2.normalize(
            img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

        norm_image = np.zeros(image.shape)
        norm_image = cv2.normalize(
            image,
            norm_image,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        norm_image = norm_image.astype(np.uint8).transpose(1, 2, 0)

        shifted = cv2.pyrMeanShiftFiltering(
            norm_image,
            self.segmentation_kwargs["mean_shifting_spatial_radius"],
            self.segmentation_kwargs["mean_shifting_color_radius"],
        )

        # convert the mean shift image to grayscale, then apply
        # Otsu's thresholding
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        if self.segmentation_kwargs["invert"]:
            gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(
            D,
            indices=False,
            min_distance=self.segmentation_kwargs["peak_local_max_min_dist"],
            labels=thresh,
        )

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)
        print(f"[INFO] {len(np.unique(labels)) - 1} unique segments found")

        self._num_superpixels = len(np.unique(labels)) - 1

        # loop over the unique labels returned by the Watershed
        # algorithm
        label_images = []
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue
            # otherwise, allocate memory for the label region and draw
            # it on the mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[labels == label] = 255

            w = np.array(mask == 255, dtype=np.uint8)

            # generate image where mask is equal to 255
            cp_img = img.copy().transpose(1, 2, 0)
            res = cv2.bitwise_and(cp_img, cp_img, mask=w)
            label_images.append(res.transpose(2, 0, 1))
        return label_images
