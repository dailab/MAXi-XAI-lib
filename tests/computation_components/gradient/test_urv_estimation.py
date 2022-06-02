import unittest
from unittest.mock import patch, Mock

import numpy as np

from maxi.lib.computation_components.gradient.gradient_estimator import URVGradientEstimator


class TestURVGradientEstimator(unittest.TestCase):
    def test_estimator_input_output(self):
        mock_loss = Mock(return_value=123)
        iter_count = 10
        gradient_estimator = URVGradientEstimator(loss=mock_loss, img_size=1, iter=iter_count)

        test_input = np.array([1])
        self.assertTrue(type(gradient_estimator(test_input)) is np.ndarray)
        self.assertTrue(gradient_estimator.loss.call_count == iter_count + 1)

    def test_gradient_estimation(self):
        DESIRED_DEC_PREC = 0

        # mse = lambda X, Y: ((X - Y) ** 2).mean()

        grad_iterations = [100]
        sample_size = [100, 1000, 10000]

        mock_functions = [
            lambda x: x ** 2,
        ]  # lambda x: 2 * x ** 3]
        inputs = [np.array([3, -2, 12.5]), np.array([3, 5, -5])]
        expected_gradients = [
            np.array([6, -4, 25]),
        ]  # np.array([54, 150, 150])]

        for input, func, exp_grad in zip(inputs, mock_functions, expected_gradients):
            for grad_iter in grad_iterations:  # different estimation iterations
                gradient_estimator = URVGradientEstimator(loss=func, img_size=input.size, iter=grad_iter)

                res_list = [None] * len(sample_size)
                for i, num_calls in enumerate(sample_size):  # number of test samples
                    res_array = np.zeros((num_calls, len(input)))
                    for j in range(num_calls):  # collect estimated gradients samples
                        res_array[j] = gradient_estimator(input)

                    res_list[i] = res_array

                # losses = list(map(lambda x: mse(x, exp_grad), res_list))
                # self.assertTrue(losses[0] < losses[-1])

                avg_gradients = list(map(lambda x: x.mean(axis=0), res_list))
                for avg_grad in avg_gradients:
                    np.testing.assert_array_almost_equal(avg_grad, exp_grad, decimal=DESIRED_DEC_PREC)


if __name__ == "__main__":
    unittest.main()
