import unittest
from unittest.mock import patch, Mock

import numpy as np

from maxi.lib.inference.quantizer.confidence_method import BinaryConfidenceMethod, calculate_confidence


class TestBinaryConfidenceMethod(unittest.TestCase):
    def setUp(self):
        self.b_conf_method = BinaryConfidenceMethod()

    def test_calculate_confidence(self):
        image_1 = np.full((25, 25, 3), 255)
        image_2 = np.full((25, 25, 3), 127.5)
        image_3 = np.full((25, 25, 3), 0.0)

        images = [image_1, image_2, image_3]
        expected_confidence = [1.0, 0.5, 0.0]

        for image, expected_conf in zip(images, expected_confidence):
            self.assertEqual(calculate_confidence(image, max_pixel_value=255), expected_conf)

    def test_dunder_call(self):
        image_1 = np.full((25, 25, 3), 255)

        mock_preprocessor = Mock(return_value=image_1)
        self.b_conf_method.preprocess = mock_preprocessor

        np.testing.assert_array_equal(self.b_conf_method(image_1), np.array([-1.0, 1.0]))
        np.testing.assert_array_equal(self.b_conf_method.preprocess.call_args[0][0], image_1)


if __name__ == "__main__":
    unittest.main()
