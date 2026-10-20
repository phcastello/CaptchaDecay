"""CTC valid/invalid alignments and gradients through the entire recognizer."""

import unittest
from unittest.mock import patch

import torch

from ml.ctc import ctc_loss, min_ctc_timesteps
from ml.model import CRNN


class CTCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_minimum_alignment_includes_neighboring_repeats(self):
        self.assertEqual(min_ctc_timesteps([]), 0)
        self.assertEqual(min_ctc_timesteps([1, 2]), 2)
        self.assertEqual(min_ctc_timesteps([1, 1]), 3)
        self.assertEqual(min_ctc_timesteps([1, 1, 1, 2, 2]), 8)
        self.assertEqual(min_ctc_timesteps([1, 2, 1]), 3)

    def test_variable_lengths_repeated_targets_finite_loss_and_cnn_gradients(self):
        torch.manual_seed(23)
        model = CRNN(5, hidden_size=8, num_layers=1)
        logits = model(torch.randn(2, 1, 80, 215))
        targets = torch.tensor([1, 1, 2, 3, 2, 4, 2], dtype=torch.long)
        lengths = torch.tensor([3, 4], dtype=torch.long)
        loss = ctc_loss(logits, targets, lengths, ["repeated.png", "longer.png"])
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.cnn[0].weight.grad.abs().sum().item(), 0)

    def test_loss_uses_actual_time_and_matches_torch(self):
        logits = torch.randn(7, 2, 4, requires_grad=True)
        targets = torch.tensor([1, 1, 2, 3, 2], dtype=torch.long)
        lengths = torch.tensor([2, 3], dtype=torch.long)
        actual = ctc_loss(logits, targets, lengths)
        expected = torch.nn.CTCLoss(blank=0, zero_infinity=True)(
            logits.log_softmax(2), targets, torch.tensor([7, 7]), lengths,
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_repeated_label_needs_more_time_than_distinct_label(self):
        logits = torch.randn(2, 1, 3)
        lengths = torch.tensor([2])
        self.assertTrue(torch.isfinite(ctc_loss(logits, torch.tensor([1, 2]), lengths)))
        with self.assertRaisesRegex(ValueError, "repeat.png.*T=2.*minimum timesteps=3"):
            ctc_loss(logits, torch.tensor([1, 1]), lengths, ["repeat.png"])
        with self.assertRaisesRegex(ValueError, "impossible CTC alignment"):
            ctc_loss(logits, torch.tensor([1, 2, 1]), torch.tensor([3]))

    def test_invalid_targets_lengths_and_shapes_fail_before_loss(self):
        logits = torch.randn(5, 1, 3)
        cases = (
            (torch.tensor([0]), torch.tensor([1]), "reserved"),
            (torch.tensor([-1]), torch.tensor([1]), "invalid CTC target ID"),
            (torch.tensor([3]), torch.tensor([1]), "invalid CTC target ID"),
            (torch.tensor([1.0]), torch.tensor([1]), "targets must"),
            (torch.tensor([[1]]), torch.tensor([1]), "targets must"),
            (torch.tensor([1]), torch.tensor([1.0]), "target_lengths must"),
            (torch.tensor([1]), torch.tensor([[1]]), "target_lengths must"),
            (torch.tensor([1]), torch.tensor([1, 1]), "one target length"),
            (torch.tensor([], dtype=torch.long), torch.tensor([0]), "nonempty"),
            (torch.tensor([1]), torch.tensor([-1]), "nonempty"),
            (torch.tensor([1, 2]), torch.tensor([1]), "sum to"),
        )
        for targets, lengths, error in cases:
            with self.subTest(error=error, targets=targets.tolist()), self.assertRaisesRegex(ValueError, error):
                ctc_loss(logits, targets, lengths)
        with self.assertRaisesRegex(ValueError, "one filename"):
            ctc_loss(logits, torch.tensor([1]), torch.tensor([1]), [])
        for shape in ((1, 3), (0, 1, 3), (2, 0, 3), (2, 1, 1)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "CTC logits"):
                ctc_loss(torch.ones(shape), torch.tensor([1]), torch.tensor([1]))
        with self.assertRaisesRegex(ValueError, "floating point"):
            ctc_loss(torch.ones(3, 1, 3, dtype=torch.long), torch.tensor([1]), torch.tensor([1]))

    def test_rejects_nonfinite_logits_log_probabilities_and_loss(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "NaN/Inf.*logits"):
                ctc_loss(torch.full((3, 1, 3), value), torch.tensor([1]), torch.tensor([1]))
        overflowing = torch.tensor([3e38, -3e38, 0]).reshape(1, 1, 3)
        with self.assertRaisesRegex(RuntimeError, "NaN/Inf.*log_softmax"):
            ctc_loss(overflowing, torch.tensor([1]), torch.tensor([1]))
        with patch("ml.ctc.nn.CTCLoss") as criterion:
            criterion.return_value.return_value = torch.tensor(float("nan"))
            with self.assertRaisesRegex(RuntimeError, "NaN/Inf.*loss"):
                ctc_loss(torch.randn(3, 1, 3), torch.tensor([1]), torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
