import unittest

import numpy as np

from maxi.lib.inference.processing.selective_region_processor import SelectiveRegionProcessor, EntityRect


class TestSelectiveRegion(unittest.TestCase):
    def setUp(self):
        self.random_image = np.random.randint(0, 255, (25, 25, 3))
        self.target_region = EntityRect(x=0, y=0, w=10, h=10)
        self.region_processor = SelectiveRegionProcessor(orig_image=self.random_image, entity_region=self.target_region)

    def test_preprocess_region(self):
        new_region = np.random.randint(0, 255, (self.target_region["w"], self.target_region["h"], 3))

        image_with_new_region = self.region_processor.preprocess(new_region)

        # check if preprocessing didn't change saved original image
        np.testing.assert_array_equal(self.random_image, self.region_processor.orig_img)

        # check if the target region of returned image contains new region
        np.testing.assert_array_equal(
            image_with_new_region[
                self.target_region["x"] : self.target_region["x"] + self.target_region["w"],
                self.target_region["y"] : self.target_region["y"] + self.target_region["h"],
            ],
            new_region,
        )

        # check if assertionerror raised when parsed image has different shape
        with self.assertRaises(AssertionError):
            insane_region = np.random.randint(0, 255, (self.target_region["w"] + 3, self.target_region["h"] + 3, 3))
            self.region_processor.preprocess(insane_region)

    def test_postprocess_region(self):
        segmentation_mask = np.random.randint(0, 255, self.random_image.shape)

        region_segmentation_mask = self.region_processor.postprocess(segmentation_mask)

        expected_region_mask = segmentation_mask[
            self.target_region["x"] : self.target_region["x"] + self.target_region["w"],
            self.target_region["y"] : self.target_region["y"] + self.target_region["h"],
        ]

        np.testing.assert_array_equal(expected_region_mask, region_segmentation_mask)

        # check if assertionerror raised when parsed image has different shape
        with self.assertRaises(AssertionError):
            insane_region = np.random.randint(0, 255, (1, 1, 3))
            self.region_processor.postprocess(insane_region)


if __name__ == "__main__":
    unittest.main()
