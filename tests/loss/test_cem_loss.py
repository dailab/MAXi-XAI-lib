import unittest
from unittest.mock import PropertyMock, patch, Mock

import numpy as np

from maxi.lib.loss.cem_loss import CEMLoss
from maxi.utils.loss_utils import extract_prob


class TestCemLoss(unittest.TestCase):
    def test_select_mode(self):
        """Tests the mode selection method."""
        sane_modes = ["Pp", "pp", "pP", "Pn", "pn", "pN"]
        insane_modes = ["sf", "12", "*_", "Np", "ppp"]

        with patch.object(CEMLoss, "__init__", lambda a, b, c, d, e, f, h: None):
            loss = CEMLoss(None, None, None, None, None, None)
            loss._lower_upper = {"PP": {"lower": None, "upper": None}, "PN": {"lower": None, "upper": None}}

            for mode in sane_modes:
                loss._setup_mode(mode)

                self.assertEqual(loss.mode, mode.upper())
                if loss.mode == "PP":
                    self.assertEqual(loss.get_loss, loss.PP)
                elif loss.mode == "PN":
                    self.assertEqual(loss.get_loss, loss.PN)

            for mode in insane_modes:
                with self.assertRaises(AssertionError):
                    loss._setup_mode(mode)

    def test_get_target_idx(self):
        mock_inference_results = [
            [-4.5, 13.5],
            [124, -48, 12, 52, 23.536],
        ]
        expected_target_index = [1, 0]
        insane_inference_results = [[[0.5, 2.3, -4]], [[1.4, 5.23], [4.5, 5]], [[[5]]], [-3]]

        mock_inference, insane_mock_inference = Mock(), Mock()
        mock_inference.side_effect, insane_mock_inference.side_effect = (
            [np.array(x) for x in mock_inference_results],
            [np.array(x) for x in insane_inference_results],
        )

        with patch.object(CEMLoss, "__init__", lambda a, b, c, d, e, f, h: None):
            loss = CEMLoss(None, None, None, None, None, None)
            loss.inference = mock_inference

            random_img = np.random.randint(0, 255, (25, 25, 3))

            for expected_idx in expected_target_index:
                self.assertEqual(loss.get_target_idx(random_img), expected_idx)

            loss.inference = insane_mock_inference
            for _ in insane_inference_results:
                with self.assertRaises(AssertionError):
                    loss.get_target_idx(random_img)

    def test_init_loss(self):
        """Tests the initialization method of the CEM loss function."""
        mock_inference = Mock()
        mock_inference.return_value = np.array([-4.5, 13.5])
        expected_target_index = 1

        # test mode for pertinent positive
        loss = CEMLoss(
            mode="PP", org_img=np.random.randint(0, 255, size=(25, 25, 3)), inference=mock_inference, c=1, gamma=0, K=10
        )

        self.assertEqual(loss.get_loss, loss.PP)
        self.assertEqual(loss.target, expected_target_index)

        # test mode for pertinent negative
        loss = CEMLoss(
            mode="PN", org_img=np.random.randint(0, 255, size=(25, 25, 3)), inference=mock_inference, c=1, gamma=0, K=10
        )

        self.assertEqual(loss.get_loss, loss.PN)
        self.assertEqual(loss.target, expected_target_index)

        # test mode different to "PN" & "PP"
        with self.assertRaises(AssertionError):
            loss = CEMLoss(
                mode="23",
                org_img=np.random.randint(0, 255, size=(25, 25, 3)),
                inference=mock_inference,
                c=1,
                gamma=0,
                K=10,
            )

    def test_extract_prob(self):
        mock_inference_results = [
            np.array([-4.5, 13.5]),
            np.array([124, -48, 12, 52, 23.536]),
            np.array([[98.5, -23.2, 19.5, 0.645]]),
        ]
        target_index = [1, 0, 0]
        target_value, non_target_value = [13.5, 124, 98.5], [-4.5, 52, 19.5]

        for inference_res, target_idx, target_val, non_target_val in zip(
            mock_inference_results, target_index, target_value, non_target_value
        ):
            self.assertEqual(extract_prob(inference_res, target_idx, False), target_val)
            self.assertEqual(extract_prob(inference_res, target_idx, True), non_target_val)

    def test_f_K_pos(self):
        mock_inference_results = [[-4.0, 13.0], [5.0, 3.0]]
        mock_inference_results = [np.array(x) for x in mock_inference_results]
        expected_f_K_loss = [-10, -2]

        mock_inference = Mock()
        mock_inference.side_effect = mock_inference_results

        random_delta = np.random.randint(0, 255, (25, 25, 3))
        with patch.object(CEMLoss, "__init__", lambda a, b, c, d, e, f, h: None):
            loss = CEMLoss(None, None, None, None, None, None)
            loss.inference = mock_inference
            loss.K = 10

            for inf_res, f_K_loss in zip(mock_inference_results, expected_f_K_loss):
                loss.target = np.argmax(inf_res)
                self.assertEqual(loss.f_K_pos(random_delta), f_K_loss)

                np.testing.assert_array_equal(
                    loss.inference.call_args[0][0],
                    random_delta,
                    "f_K_pos requires to infer on only the perturbed delta",
                )

    def test_f_K_neg(self):
        mock_inference_results = [[-4.0, 13.0], [5.0, 3.0], [-5, -28]]
        mock_inference_results = [np.array(x) for x in mock_inference_results]
        expected_f_K_loss = [17, 2, 23]

        mock_inference = Mock()
        mock_inference.side_effect = mock_inference_results

        random_delta = np.random.randint(0, 255, (25, 25, 3))
        with patch.object(CEMLoss, "__init__", lambda a, b, c, d, e, f, h: None):
            loss = CEMLoss(None, None, None, None, None, None)
            loss.inference = mock_inference
            loss.org_img = np.random.randint(0, 255, (25, 25, 3))
            loss.K = 10

            for inf_res, f_K_loss in zip(mock_inference_results, expected_f_K_loss):
                loss.target = np.argmax(inf_res)
                self.assertEqual(loss.f_K_neg(random_delta), f_K_loss)

                np.testing.assert_array_equal(
                    loss.inference.call_args[0][0],
                    loss.org_img + random_delta,
                    "f_K_neg requires to infer on the sum of the original image and the perturbed delta",
                )

    def test_PP_loss(self):
        mock_f_K_pos, mock_ae_error = Mock(), Mock()
        mock_f_K_pos.return_value = 2.5
        mock_ae_error.return_value = 3

        delta = np.random.randint(0, 255, (25, 25, 3))
        flattened_delta = delta.flatten()

        with patch.object(CEMLoss, "__init__", lambda a, b, c, d, e, f, h: None):
            loss = CEMLoss(None, None, None, None, None, None)
            loss.get_loss = loss.PP
            loss.c, loss.gamma, loss.target, loss.K, loss._org_img_shape = 2.5, 3.0, 1, 10, delta.shape

            loss.f_K_pos, loss.PP_AE_error = mock_f_K_pos, mock_ae_error

            expected_loss = loss.c * mock_f_K_pos.return_value + loss.gamma * mock_ae_error.return_value

            self.assertEqual(loss.get_loss(flattened_delta), expected_loss)
            np.testing.assert_array_equal(loss.f_K_pos.call_args[0][0], delta)
            np.testing.assert_array_equal(loss.PP_AE_error.call_args[0][0], delta)

    def test_PN_loss(self):
        mock_f_K_pos, mock_ae_error = Mock(), Mock()
        mock_f_K_pos.return_value = 2.5
        mock_ae_error.return_value = 3

        delta = np.random.randint(0, 255, (25, 25, 3))
        flattened_delta = delta.flatten()

        with patch.object(CEMLoss, "__init__", lambda a, b, c, d, e, f, h: None):
            loss = CEMLoss(None, None, None, None, None, None)
            loss.get_loss = loss.PN
            loss.c, loss.gamma, loss.target, loss.K, loss._org_img_shape = 2.5, 3.0, 1, 10, delta.shape

            loss.f_K_neg, loss.PN_AE_error = mock_f_K_pos, mock_ae_error

            expected_loss = loss.c * mock_f_K_pos.return_value + loss.gamma * mock_ae_error.return_value

            self.assertEqual(loss.get_loss(flattened_delta), expected_loss)
            np.testing.assert_array_equal(loss.f_K_neg.call_args[0][0], delta)
            np.testing.assert_array_equal(loss.PN_AE_error.call_args[0][0], delta)


if __name__ == "__main__":
    unittest.main()
