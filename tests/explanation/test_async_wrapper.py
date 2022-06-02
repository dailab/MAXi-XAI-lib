import unittest
from unittest.mock import patch, Mock
from typing import Tuple

import numpy as np
import random
import string
from collections import OrderedDict

from maxi.lib.explanation.explanation_generator import ExplanationGenerator
from maxi.lib.explanation.wrappers.async_explanation import AsyncExplanationWrapper


class MockExplanationGenerator(ExplanationGenerator):
    def __init__(self, *args, **kwargs):
        self.run = Mock(return_value=(OrderedDict(), OrderedDict(), dict()))


class TestExplanationGenerator(unittest.TestCase):
    def setUp(self):
        self.expl_gen = MockExplanationGenerator()
        self.async_expl_gen = AsyncExplanationWrapper(self.expl_gen, n_workers=2)

    def test_run(self):
        random_imgs = [np.random.randint(0, 255, (25, 25, 3)), np.random.randint(0, 255, (25, 25, 3))]
        mock_inferences = {img.tobytes(): Mock() for img in random_imgs}
        random_metadata = {
            img.tobytes(): "".join(random.choices(string.ascii_uppercase + string.digits, k=10)) for img in random_imgs
        }

        output = self.async_expl_gen.run(random_imgs, mock_inferences, random_metadata)

        self.assertEqual(self.expl_gen.run.call_count, len(random_imgs))
        self.assertEqual(len(output), len(random_imgs))

        # check if inference calls and meta data were assigned to the correct image
        args_first_call: Tuple[np.ndarray, Mock, dict] = self.expl_gen.run.call_args_list[0].args
        args_second_call: Tuple[np.ndarray, Mock, dict] = self.expl_gen.run.call_args_list[1].args

        self.assertTrue(
            np.array_equal(args_first_call[0], random_imgs[0]) or np.array_equal(args_first_call[0], random_imgs[1])
        )  # the calling order isn't important

        # determine the order in which images got processed
        i, k = (0, 1) if np.array_equal(args_first_call[0], random_imgs[0]) else (1, 0)

        self.assertEqual(args_first_call[1], mock_inferences[random_imgs[i].tobytes()])
        self.assertEqual(args_first_call[2], random_metadata[random_imgs[i].tobytes()])

        self.assertEqual(args_second_call[1], mock_inferences[random_imgs[k].tobytes()])
        self.assertEqual(args_second_call[2], random_metadata[random_imgs[k].tobytes()])

        # check output type
        self.assertTrue(type(output) is list)
        self.assertTrue(
            type(output[0][0]) is OrderedDict and type(output[0][1]) is OrderedDict and type(output[0][2]) is dict
        )

        # with self.assertRaises(AssertionError):
        #     self.expl_gen.run(None, None)


if __name__ == "__main__":
    unittest.main()
