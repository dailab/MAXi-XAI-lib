"""Selective Region Processor"""
import numpy as np
import torch

from ....data.data_types import EntityRect


class SelectiveRegionProcessor:
    def __init__(self, orig_image: np.ndarray, entity_region: EntityRect) -> None:
        """Selective Region Processor

        Description:
            Holds a copy of the original image in order to perform inference on \
            model conforming format. The preprocessing procedure includes replacing \
            the region to be altered with the parsed one. The prediction result for the selected \
            region is afterwards extracted in the postprocessing method.
            Thus, this class must hold the coordinates of the region containing the object \
            for explanation.

        Args:
            orig_image (np.ndarray): The original full-sized image.
            entity_region (EntityRect): Coordinates of the image region to be perturbed.
        """
        self.orig_img = orig_image
        self._target = entity_region
        self.check_sizes()

    def check_sizes(self):
        assert (
            self.orig_img.shape[0] >= self.target["w"] and self.orig_img.shape[1] >= self.target["h"]
        ), f"Target region has a dimension which is larger than of the original image; {self.orig_img.shape[0:2]} < {(self.target['w'], self.target['h'])}"

    @property
    def target(self) -> None:
        return self._target

    @target.setter
    def target(self, new_target_region: EntityRect) -> None:
        self._target = new_target_region

    def preprocess(self, new_region: np.ndarray) -> np.ndarray:
        """Replaces the target region in the original image with the parsed one before inference.

        Description:
            Makes a copy of the original image, replaces the target region with \
            the parsed one in 'new_region' and returns the resulting image.

        Args:
            new_region (np.ndarray): The new region to be infered. Shape in [bs, self.w, self.h, c].

        Returns:
            np.ndarray: Batch of full-sized images with 'new_region' at [self.x, self.y] (upper-left).
        """
        if new_region.ndim < 4:
            return self.nonbatched_replacement(new_region)
        else:
            return self.batched_replacement(new_region)

    def batched_replacement(self, new_region: np.ndarray) -> np.ndarray:
        assert (
            new_region.shape[1] == self.target["w"] and new_region.shape[2] == self.target["h"]
        ), f"Parsed region is of different size compared to the target region; {new_region.shape[1:3]} != {(self.target['w'], self.target['h'])}"

        tmp_image = np.stack([self.orig_img.copy() for _ in range(new_region.shape[0])], axis=0)
        tmp_image[
            :,
            self.target["x"] : self.target["x"] + self.target["w"],
            self.target["y"] : self.target["y"] + self.target["h"],
        ] = new_region
        return tmp_image

    def nonbatched_replacement(self, new_region: np.ndarray) -> np.ndarray:
        assert (
            new_region.shape[0] == self.target["w"] and new_region.shape[1] == self.target["h"]
        ), f"Parsed region is of different size compared to the target region; {new_region.shape[0:2]} != {(self.target['w'], self.target['h'])}"

        tmp_image = self.orig_img.copy()
        tmp_image[
            self.target["x"] : self.target["x"] + self.target["w"],
            self.target["y"] : self.target["y"] + self.target["h"],
        ] = new_region
        return tmp_image

    def postprocess(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """Extracts the region of the segmentation mask corresponding to the target region.

        Args:^
            segmentation_mask (np.ndarray): The prediction mask, has to be of the same shape as ``self.orig_img``.

        Returns:
            np.ndarray: Segmentation mask of the target region. Shape in [self.w, self.h].
        """
        assert (
            segmentation_mask.shape[1:] == self.orig_img.shape[:2]
        ), f"Segmentation mask must be of the same shape as the original image; is {segmentation_mask.shape[1:]} != {self.orig_img.shape[:2]}"
        return segmentation_mask[
            :,
            self.target["x"] : self.target["x"] + self.target["w"],
            self.target["y"] : self.target["y"] + self.target["h"],
        ]


class Torch_SelectiveRegionProcessor(SelectiveRegionProcessor):
    def __init__(self, orig_image: np.ndarray, entity_region: EntityRect) -> None:
        """Selective Region Processor

        Description:
            Holds a copy of the original image in order to perform inference on \
            model conforming format. The preprocessing procedure includes replacing \
            the region to be altered with the parsed one. The prediction result for the selected \
            region is afterwards extracted in the postprocessing method.
            Thus, this class must hold the coordinates of the region containing the object \
            for explanation.

        Args:
            orig_image (np.ndarray): The original full-sized image.
            entity_region (EntityRect): Coordinates of the image region to be perturbed.
        """
        import torch

        self.orig_img = orig_image
        self.torch_orig_img = torch.tensor(orig_image, dtype=torch.float32, requires_grad=True)
        self._target = entity_region
        self.check_sizes()

    def preprocess(self, new_region: torch.Tensor) -> torch.Tensor:
        """Replaces the target region in the original image with the parsed one before inference.

        Description:
            Makes a copy of the original image, replaces the target region with \
            the parsed one in 'new_region' and returns the resulting image.

        Args:
            new_region (np.ndarray): The new region to be infered. Shape in [self.w, self.h, c].

        Returns:
            np.ndarray: Full-sized image with 'new_region' at [self.x, self.y] (upper-left).
        """
        assert (
            new_region.shape[0] == self.target["w"] and new_region.shape[1] == self.target["h"]
        ), f"Parsed region is of different size compared to the target region; {new_region.shape[0:2]} != {(self.target['w'], self.target['h'])}"
        tmp_image = torch.cat(
            (
                self.torch_orig_img[: self.target["x"], self.target["y"] : self.target["y"] + self.target["h"], :],
                new_region,
                self.torch_orig_img[
                    self.target["x"] + self.target["w"] :, self.target["y"] : self.target["y"] + self.target["h"], :
                ],
            ),
            dim=0,
        )
        tmp_image = torch.cat(
            (
                self.torch_orig_img[:, : self.target["y"], :],
                tmp_image,
                self.torch_orig_img[:, self.target["y"] + self.target["h"] :, :],
            ),
            dim=1,
        )
        return tmp_image.view((self.orig_img.shape[2], self.orig_img.shape[0], self.orig_img.shape[1]))
