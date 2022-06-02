import unittest
from unittest.mock import patch, Mock

import numpy as np
from collections import OrderedDict
from scipy.optimize import OptimizeResult

from maxi.lib.computation_components.optimizer.ada_exp_grad import AdaExpGradOptimizer


class TestAdaExpGrad(unittest.TestCase):
    def setUp(self):
        random_img = np.random.randint(0, 255, (25, 25, 3))
        random_presenter_img = np.random.randint(0, 255, random_img.shape)

        mock_loss = Mock(return_value=123)
        mock_gradient = Mock(return_value=np.zeros_like(random_img))
        mock_presenter_cb = Mock(return_value=random_presenter_img)
        mock_presenter_kwargs = {"test": "entry"}

        self.num_iter, cb_epoch = 1, 10

        self.optimizer = AdaExpGradOptimizer(
            loss=mock_loss,
            gradient=mock_gradient,
            num_iter=self.num_iter,
            org_image=random_img,
            x0=np.zeros_like(random_img),
            lower=np.zeros_like(random_img),
            upper=random_img,
            presenter_cb=mock_presenter_cb,
            presenter_kwargs=mock_presenter_kwargs,
            p_cb_epoch=cb_epoch,
            l_cb_epoch=cb_epoch,
        )

        self.p_and_l_epoch = min(self.num_iter, cb_epoch)
        self.assertTrue(
            self.optimizer.p_cb_epoch == self.p_and_l_epoch and self.optimizer.l_cb_epoch == self.p_and_l_epoch
        )

    def test_optimizer_step(self):
        self.assertTrue(type(self.optimizer.step()) is OptimizeResult)

    def test_optimizer_run(self):
        self.optimizer.logging_cb = Mock()

        presenter_res, raw_res = self.optimizer.run()

        self.assertTrue(type(presenter_res) is OrderedDict and type(raw_res) is OrderedDict)
        self.assertEqual(self.optimizer.presenter_cb.call_count, 1)
        self.assertEqual(self.optimizer.logging_cb.call_count, 1)

        p_kwargs_keys = self.optimizer.presenter_cb.call_args.kwargs.keys()
        input_p_kwargs = list(self.optimizer.presenter_kwargs.keys())
        self.assertTrue(
            "image" in p_kwargs_keys
            and "original_image" in p_kwargs_keys
            and "_meta_data" in p_kwargs_keys
            and input_p_kwargs[0] in p_kwargs_keys
        )

        self.optimizer.num_iter = 10
        presenter_res, raw_res = self.optimizer.run()
        self.assertTrue(type(presenter_res) is OrderedDict and type(raw_res) is OrderedDict)
        self.assertEqual(self.optimizer.presenter_cb.call_count, 10)
        self.assertEqual(self.optimizer.logging_cb.call_count, 10)


if __name__ == "__main__":
    unittest.main()
