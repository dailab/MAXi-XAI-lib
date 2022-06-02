import unittest
from unittest.mock import patch, Mock

import numpy as np
from collections import OrderedDict

from maxi.lib.explanation.explanation_generator import ExplanationGenerator
from maxi.lib.computation_components.optimizer.base_optimizer import BaseOptimizer


class MockOptimizer(BaseOptimizer):
    def __init__(self):
        self.init_mock = Mock()
        self.run = Mock(return_value=(OrderedDict(), OrderedDict()))

    def __call__(self, *args, **kwargs):
        self.init_mock(*args, **kwargs)
        return self

    def step(self, *args, **kwargs):
        self.step = Mock()


class TestExplanationGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_loss = Mock()
        self.loss_kwargs = {"test1": "entry"}

        self.mock_gradient = Mock()
        self.gradient_kwargs = {"test2": "entry"}

        self.mock_optimizer = MockOptimizer()
        self.optimizer_kwargs = {"test3": "entry"}

        self.expl_gen = ExplanationGenerator(
            loss=self.mock_loss,
            gradient=self.mock_gradient,
            optimizer=self.mock_optimizer,
            loss_kwargs=self.loss_kwargs,
            gradient_kwargs=self.gradient_kwargs,
            optimizer_kwargs=self.optimizer_kwargs,
        )

    def test_run(self):
        mock_opt_result = Mock()

        random_img = np.random.randint(0, 255, (25, 25, 3))
        mock_inference = Mock()

        output = self.expl_gen.run(random_img, mock_inference)
        """ check component inputs """
        # loss inputs
        loss_kwargs = self.mock_loss.call_args.kwargs
        expected_loss_keys = [
            "inference",
            "org_img",
            "test1",
        ]  # should at least include essential args of respective base class

        self.assertTrue(all(key in loss_kwargs.keys() for key in expected_loss_keys))

        self.assertEqual(loss_kwargs["inference"], mock_inference)
        np.testing.assert_array_equal(loss_kwargs["org_img"], random_img)

        # gradient inputs
        grad_kwargs = self.mock_gradient.call_args.kwargs
        expected_grad_keys = [
            "loss",
            "img_size",
            "test2",
        ]  # should at least include essential args of respective base class

        self.assertTrue(all(key in grad_kwargs.keys() for key in expected_grad_keys))
        self.assertEqual(grad_kwargs["img_size"], random_img.size)

        # optimizer inputs
        optim_kwargs = self.mock_optimizer.init_mock.call_args.kwargs
        expected_optim_keys = [
            "org_img",
            "loss",
            "gradient",
            "x0",
            "lower",
            "upper",
        ]  # should at least include essential args of respective base class

        self.assertTrue(all(key in optim_kwargs.keys() for key in expected_optim_keys))
        self.assertTrue(isinstance(optim_kwargs["loss"], Mock))
        np.testing.assert_array_equal(optim_kwargs["org_img"], random_img)
        self.assertEqual(
            optim_kwargs["x0"].shape, random_img.flatten().shape
        )  # check if image for perturbation was flattened
        self.assertEqual(optim_kwargs["lower"].shape, random_img.flatten().shape)
        self.assertEqual(optim_kwargs["upper"].shape, random_img.flatten().shape)

        # check output type
        self.assertTrue(type(output[0]) is OrderedDict and type(output[1]) is OrderedDict and type(output[2]) is dict)

        with self.assertRaises(AssertionError):
            self.expl_gen.run(None, None)


if __name__ == "__main__":
    unittest.main()
