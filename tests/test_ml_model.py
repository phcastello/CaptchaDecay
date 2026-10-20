"""Shape and architecture checks using random local tensors only."""

import unittest

import torch

from ml.model import CRNN


class CRNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_real_cnn_sequence_lstm_classifier_and_ctc_shapes(self):
        model = CRNN(num_classes=17)
        images = torch.randn(2, 1, 80, 215)
        with torch.no_grad():
            features = model.cnn(images)
            sequence = model.spatial_to_sequence(features)
            recurrent, (hidden, cell) = model.rnn(sequence)
            classified = model.classifier(recurrent)
            logits = model(images)
        self.assertEqual(tuple(features.shape), (2, 256, 5, 53))
        self.assertEqual(tuple(sequence.shape), (2, 53, 256))
        self.assertEqual(tuple(recurrent.shape), (2, 53, 256))
        self.assertEqual(tuple(hidden.shape), (4, 2, 128))
        self.assertEqual(tuple(cell.shape), (4, 2, 128))
        self.assertEqual(tuple(classified.shape), (2, 53, 17))
        self.assertEqual(tuple(logits.shape), (53, 2, 17))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(model.classifier.out_features, 17)
        self.assertTrue(model.rnn.bidirectional)
        self.assertEqual(model.rnn.num_layers, 2)
        self.assertLess(sum(value.numel() for value in model.parameters()), 2_000_000)
        # Raw logits are returned, leaving exactly one log_softmax to the loss.
        self.assertFalse(torch.allclose(logits.logsumexp(dim=2), torch.zeros(53, 2)))

    def test_spatial_sequence_keeps_width_as_time(self):
        features = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        sequence = CRNN.spatial_to_sequence(features)
        self.assertEqual(tuple(sequence.shape), (2, 5, 3))
        for batch in range(2):
            for time in range(5):
                self.assertTrue(torch.equal(sequence[batch, time], features[batch, :, :, time].mean(dim=1)))
        for invalid in (torch.ones(1, 3, 4), torch.ones(0, 3, 4, 5)):
            with self.subTest(shape=tuple(invalid.shape)), self.assertRaisesRegex(ValueError, "CNN features"):
                CRNN.spatial_to_sequence(invalid)

    def test_model_configuration_reconstructs_architecture(self):
        model = CRNN(7, hidden_size=16, num_layers=1)
        restored = CRNN(**model.config)
        restored.load_state_dict(model.state_dict())
        self.assertEqual(model.config, {"num_classes": 7, "hidden_size": 16, "num_layers": 1})
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in restored.parameters()))

    def test_rejects_invalid_constructor_and_input(self):
        for config in ({"num_classes": 1}, {"num_classes": True}, {"num_classes": 2, "hidden_size": 0},
                       {"num_classes": 2, "num_layers": 1.5}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                CRNN(**config)
        model = CRNN(3, hidden_size=8, num_layers=1)
        for shape in ((1, 3, 80, 215), (1, 1, 79, 215), (1, 1, 80, 214), (0, 1, 80, 215), (80, 215)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "shape"):
                model(torch.zeros(shape))
        with self.assertRaisesRegex(ValueError, "float32"):
            model(torch.zeros(1, 1, 80, 215, dtype=torch.float64))
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=bad), self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
                model(torch.full((1, 1, 80, 215), bad))


if __name__ == "__main__":
    unittest.main()
