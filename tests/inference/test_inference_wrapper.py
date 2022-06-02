import unittest
from unittest.mock import Mock

import numpy as np

from maxi.lib.inference.inference_wrapper import InferenceWrapper


class TestInferenceWrapper(unittest.TestCase):
    def setUp(self):
        self.mock_inference = Mock(return_value=np.random.randint(0, 255, (25, 25, 3)))
        self.inference_wrapper = InferenceWrapper(inference_model=self.mock_inference)
        self.random_image = np.random.randint(0, 255, (25, 25, 3))

    def test_default_settings(self):
        # test identity preprocessor and quantizer
        np.testing.assert_array_equal(
            self.inference_wrapper(self.random_image),
            self.mock_inference.return_value,
        )

    def test_dunder_call(self):
        mock_preprocessor = Mock(return_value=np.random.randint(0, 125, (25, 25, 3)))
        mock_quantizer = Mock(return_value=np.array([-13.5, 15]))
        self.inference_wrapper.preprocess, self.inference_wrapper.quantizer = mock_preprocessor, mock_quantizer

        # test call order: preprocessing -> inference -> quantizer -> return
        np.testing.assert_array_equal(self.inference_wrapper(self.random_image), mock_quantizer.return_value)
        np.testing.assert_array_equal(self.inference_wrapper.preprocess.call_args[0][0], self.random_image)
        np.testing.assert_array_equal(
            self.inference_wrapper.inference_model.call_args[0][0], mock_preprocessor.return_value
        )
        np.testing.assert_array_equal(
            self.inference_wrapper.quantizer.call_args[0][0], self.mock_inference.return_value
        )


if __name__ == "__main__":
    unittest.main()
