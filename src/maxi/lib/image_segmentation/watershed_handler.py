"""SKImage WatershedHandler class to handle watershed segmentation."""
from typing import List

import cv2
import numpy as np
from scipy import ndimage

# import the necessary packages
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from maxi.lib.image_segmentation.base_seg_handler import BaseSegmentationHandler


class WatershedHandler(BaseSegmentationHandler):
    _channels_first = True

    def __init__(
        self,
        image: np.ndarray,
        ms_spatial_radius: int = 14,
        ms_color_radius: int = 32,
        max_min_dist: int = 20,
        invert: bool = True,
    ) -> None:
        """SKImage WaterhsedHandler class to handle watershed segmentation.

        Description:
            This class is used to generate superpixels from an image and to generate

        Args:
            image (np.ndarray): Array containing the image.
            ms_spatial_radius (int): Mean shifting spatial radius. Defaults to 14.
            ms_color_radius (int): Mean shifting color radius. Defaults to 32.
            max_min_dist (int): Peak local max min dist. Defaults to 20.
            invert (bool): Invert image before applying watershed algorithm. \
                Defaults to True.
        """
        self.ms_spatial_radius, self.ms_color_radius, self.max_min_dist, self.invert = (
            ms_spatial_radius,
            ms_color_radius,
            max_min_dist,
            invert,
        )
        super().__init__(image)

    def _build_label_images(self, img: np.ndarray) -> List[np.ndarray]:
        """Builds the label images by applying the watershed algorithm.

        Args:
            img (np.ndarray): Array containing image of shape [C, W, H].

        Returns:
            List[np.ndarray]: List containing the label images of shape [C, W, H].

        Note:
            The label images are the seperate superpixel images of the original image.
        """
        # load the image and perform pyramid mean shift filtering
        # to aid the thresholding step
        image = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

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
            self.ms_spatial_radius,
            self.ms_color_radius,
        )

        # convert the mean shift image to grayscale, then apply
        # Otsu's thresholding
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        if self.invert:
            gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(
            D,
            indices=False,
            min_distance=self.max_min_dist,
            labels=thresh,
        )

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)

        self._num_segments = len(np.unique(labels)) - 1

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
