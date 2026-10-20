"""Short CPU checks of training, validation and the public command line."""

import contextlib
import csv
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import numpy as np
from PIL import Image, ImageDraw

from ml.charset import Charset
from ml.evaluate import main as evaluate_main
from ml.model import CRNN
from ml.train import (
    build_checkpoint,
    load_checkpoint,
    main,
    resolve_device,
    restore_checkpoint,
    save_checkpoint,
    seed_everything,
    train_step,
    validate,
)


def make_dataset(root):
    """Use only generated fixture images, never the repository datasets."""
    root = Path(root)
    rows = []
    for split, labels in (("train", ("aabb", "abba")), ("validation", ("baba",))):
        (root / split).mkdir(parents=True)
        for index, label in enumerate(labels):
            filename = f"{split}_{index}.png"
            image = Image.new("RGB", (215, 80), "white")
            ImageDraw.Draw(image).text((20, 25), label, fill="black")
            image.save(root / split / filename)
            rows.append({"filename": filename, "label": label, "split": split,
                         "width": 215, "height": 80})
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filename", "label", "split", "width", "height"))
        writer.writeheader()
        writer.writerows(rows)
    return root


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        seed_everything(42)
        self.charset = Charset("ab")
        self.model = CRNN(3, hidden_size=8, num_layers=1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        labels = ["aabb", "abba"]
        self.batch = {
            "images": torch.rand(2, 1, 80, 215) * 2 - 1,
            "targets": torch.tensor([value for label in labels for value in self.charset.encode(label)]),
            "target_lengths": torch.tensor([4, 4]),
            "labels": labels,
            "filenames": ["first.png", "second.png"],
        }
        self.device = torch.device("cpu")
        self.config = {"batch_size": 2, "learning_rate": 0.001, "seed": 42,
                       "num_workers": 0, "overfit_samples": None, "threads": 2}
        self.dataset_identity = {"fixture": "synthetic-v1"}
        self.loader_generator = torch.Generator().manual_seed(42)

    def checkpoint(self, epoch=1):
        return build_checkpoint(
            self.model, self.optimizer, epoch, self.charset, self.config,
            self.dataset_identity, {"cer": 1.0, "exact_match": 0.0, "epoch": epoch},
            [{"epoch": epoch, "cer": 1.0}], self.loader_generator,
        )

    def test_cpu_is_explicit_and_cuda_fails_clearly_when_unavailable(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex((ValueError, RuntimeError), "CUDA|cuda"):
                resolve_device("cuda")
        with self.assertRaises((ValueError, RuntimeError)):
            resolve_device("directml")

    def test_seed_controls_random_initialization(self):
        seed_everything(12)
        first = torch.rand(4)
        python_first = random.random()
        numpy_first = np.random.rand(4)
        seed_everything(12)
        self.assertTrue(torch.equal(first, torch.rand(4)))
        self.assertEqual(python_first, random.random())
        self.assertTrue(np.array_equal(numpy_first, np.random.rand(4)))

    def test_train_step_updates_weights_and_has_finite_cnn_gradients(self):
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        self.model.eval()
        loss = train_step(self.model, self.batch, self.optimizer, self.device)
        self.assertTrue(self.model.training)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in self.model.named_parameters()))
        cnn_gradients = [parameter.grad for parameter in self.model.cnn.parameters() if parameter.grad is not None]
        self.assertTrue(cnn_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in cnn_gradients))
        self.assertTrue(any(torch.count_nonzero(gradient) for gradient in cnn_gradients))
        self.assertTrue(self.optimizer.state)

    def test_validation_aggregates_all_batches_without_mutating_model(self):
        self.model.train()
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        grad_enabled = []
        hook = self.model.register_forward_pre_hook(lambda *_: grad_enabled.append(torch.is_grad_enabled()))
        try:
            result = validate(self.model, [self.batch, self.batch], self.device, self.charset)
        finally:
            hook.remove()
        self.assertFalse(self.model.training)
        self.assertEqual(grad_enabled, [False, False])
        self.assertEqual(result["num_images"], 4)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertIn("exact_match", result)
        self.assertIn("cer", result)
        self.assertIn("by_length", result)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_validation_can_limit_smoke_batches(self):
        result = validate(self.model, [self.batch, self.batch], self.device, self.charset, max_batches=1)
        self.assertEqual(result["num_images"], 2)

    def test_validation_metrics_are_weighted_across_unequal_batch_sizes(self):
        charset = self.charset

        def logits_for(texts):
            logits = torch.full((12, len(texts), 3), -20.0)
            logits[:, :, 0] = 20.0
            for batch_index, text in enumerate(texts):
                for timestep, target in enumerate(charset.encode(text)):
                    logits[timestep * 2, batch_index, :] = -20.0
                    logits[timestep * 2, batch_index, target] = 20.0
            return logits

        class ScriptedPredictions(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.outputs = [logits_for(["aabb", "abba"]), logits_for(["a"])]
                self.calls = 0

            def forward(self, _images):
                result = self.outputs[self.calls]
                self.calls += 1
                return result

        last_batch = {
            "images": self.batch["images"][:1],
            "targets": torch.tensor(charset.encode("aaaaa")),
            "target_lengths": torch.tensor([5]),
            "labels": ["aaaaa"],
            "filenames": ["third.png"],
        }
        result = validate(ScriptedPredictions(), [self.batch, last_batch], self.device, charset)
        self.assertEqual(result["num_images"], 3)
        self.assertAlmostEqual(result["exact_match"], 2 / 3)
        self.assertAlmostEqual(result["cer"], 4 / 13)

    def test_nonfinite_loss_stops_before_parameter_update(self):
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        hook = self.model.register_forward_hook(lambda _model, _args, output: output * float("nan"))
        try:
            with self.assertRaisesRegex((ValueError, RuntimeError, FloatingPointError), "finite|NaN|Inf|loss"):
                train_step(self.model, self.batch, self.optimizer, self.device)
        finally:
            hook.remove()
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertFalse(self.optimizer.state)

    def test_infinite_gradient_stops_before_parameter_update(self):
        parameter = next(self.model.parameters())
        before = parameter.detach().clone()
        hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        try:
            with self.assertRaisesRegex((ValueError, RuntimeError, FloatingPointError), "gradient|gradiente|finite|NaN|Inf"):
                train_step(self.model, self.batch, self.optimizer, self.device)
        finally:
            hook.remove()
        self.assertTrue(torch.equal(before, parameter))
        self.assertFalse(self.optimizer.state)

    def test_checkpoint_is_atomic_and_loadable_with_tensor_state(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            payload = self.checkpoint(epoch=3)
            save_checkpoint(checkpoint, payload)
            restored = load_checkpoint(checkpoint)
            self.assertEqual(restored["epoch"], 3)
            for name, value in self.model.state_dict().items():
                self.assertTrue(torch.equal(restored["model_state_dict"][name], value), name)
            self.assertEqual(list(Path(directory).iterdir()), [checkpoint])
            save_checkpoint(checkpoint, self.checkpoint(epoch=4))
            self.assertEqual(load_checkpoint(checkpoint)["epoch"], 4)

    def test_failed_checkpoint_save_preserves_old_file_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, self.checkpoint(epoch=1))
            previous_bytes = path.read_bytes()
            with mock.patch("ml.train.torch.save", side_effect=OSError("simulated write failure")):
                with self.assertRaisesRegex(OSError, "simulated"):
                    save_checkpoint(path, self.checkpoint(epoch=2))
            self.assertEqual(path.read_bytes(), previous_bytes)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_resume_restores_epoch_optimizer_rng_and_matches_continuous_steps(self):
        train_step(self.model, self.batch, self.optimizer, self.device)
        torch.randperm(7, generator=self.loader_generator)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, self.checkpoint(epoch=5))
            expected_python = random.random()
            expected_numpy = np.random.rand(4)
            expected_torch = torch.rand(4)
            expected_order = torch.randperm(7, generator=self.loader_generator)
            train_step(self.model, self.batch, self.optimizer, self.device)
            expected_state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
            restored_model = CRNN(3, hidden_size=8, num_layers=1)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001)
            restored_generator = torch.Generator().manual_seed(999)
            checkpoint = load_checkpoint(path)
            next_epoch = restore_checkpoint(checkpoint, restored_model, restored_optimizer,
                                            self.charset, self.dataset_identity, self.config,
                                            restored_generator)
            self.assertEqual(next_epoch, 6)
            self.assertTrue(restored_optimizer.state)
            self.assertTrue(all(state["step"].item() == 1 for state in restored_optimizer.state.values()))
            self.assertEqual(random.random(), expected_python)
            self.assertTrue(np.array_equal(np.random.rand(4), expected_numpy))
            self.assertTrue(torch.equal(torch.rand(4), expected_torch))
            self.assertTrue(torch.equal(torch.randperm(7, generator=restored_generator), expected_order))
            train_step(restored_model, self.batch, restored_optimizer, self.device)
            for name, value in restored_model.state_dict().items():
                self.assertTrue(torch.equal(expected_state[name], value), name)
            for expected, actual in zip(self.optimizer.state.values(), restored_optimizer.state.values()):
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_resume_rejects_incompatible_charset_dataset_model_and_configuration(self):
        checkpoint = self.checkpoint()
        cases = (
            (self.model, Charset("ba"), self.dataset_identity, self.config, "charset"),
            (self.model, self.charset, {"fixture": "different"}, self.config, "Dataset|dataset"),
            (CRNN(3, hidden_size=9, num_layers=1), self.charset, self.dataset_identity,
             self.config, "model"),
            (self.model, self.charset, self.dataset_identity, {**self.config, "batch_size": 3}, "batch_size"),
        )
        for model, charset, identity, config, message in cases:
            with self.subTest(message=message):
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                with self.assertRaisesRegex(ValueError, message):
                    restore_checkpoint(checkpoint, model, optimizer, charset, identity, config,
                                       torch.Generator())

    def test_cli_dry_run_passes_and_writes_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_dataset(Path(directory) / "fixture")
            run_dir = Path(directory) / "unused-run"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(["--data", str(root), "--device", "cpu", "--batch-size", "2",
                               "--threads", "2", "--run-dir", str(run_dir), "--dry-run"])
            self.assertIn(result, (None, 0))
            self.assertIn("DRY RUN PASSED", output.getvalue())
            self.assertIn("cpu", output.getvalue().lower())
            self.assertEqual(list(Path(directory).rglob("*.pt")), [])

    def test_cli_tiny_fixture_training_resume_preserves_best_and_history(self):
        """Two single-batch fixture epochs; this never opens the actual dataset."""
        with tempfile.TemporaryDirectory() as directory:
            root = make_dataset(Path(directory) / "fixture")
            run_dir = Path(directory) / "run"
            common = ["--data", str(root), "--device", "cpu", "--batch-size", "2",
                      "--threads", "2", "--run-dir", str(run_dir), "--overfit-samples", "1"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(common + ["--epochs", "1"]), 0)
            checkpoint = load_checkpoint(run_dir / "last.pt")
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["config"]["overfit_samples"], 1)
            self.assertTrue(all(state["step"].item() == 1
                                for state in checkpoint["optimizer_state_dict"]["state"].values()))
            self.assertTrue((run_dir / "run_config.json").is_file())
            # A deliberately perfect prior best cannot be beaten, making preservation deterministic.
            checkpoint["best_metrics"] = {"cer": 0.0, "exact_match": 1.0, "epoch": 1}
            save_checkpoint(run_dir / "best.pt", checkpoint)
            save_checkpoint(run_dir / "last.pt", checkpoint)
            best_before = (run_dir / "best.pt").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(common + ["--epochs", "2", "--resume", str(run_dir / "last.pt")]), 0)
            resumed = load_checkpoint(run_dir / "last.pt")
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual([row["epoch"] for row in resumed["history"]], [1, 2])
            self.assertTrue(all(state["step"].item() == 2
                                for state in resumed["optimizer_state_dict"]["state"].values()))
            self.assertEqual((run_dir / "best.pt").read_bytes(), best_before)
            with (run_dir / "history.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual([int(row["epoch"]) for row in csv.DictReader(stream)], [1, 2])
            error = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                self.assertEqual(main(common + ["--epochs", "1"]), 1)
            self.assertIn("not empty", error.getvalue())

    def test_offline_evaluation_restores_checkpoint_charset_and_exports_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_dataset(Path(directory) / "fixture")
            checkpoint = Path(directory) / "fixture.pt"
            save_checkpoint(checkpoint, self.checkpoint(epoch=7))
            checkpoint_bytes = checkpoint.read_bytes()
            destination = Path(directory) / "predictions.csv"
            args = ["--data", str(root), "--split", "validation", "--checkpoint", str(checkpoint),
                    "--device", "cpu", "--threads", "2", "--output-csv", str(destination)]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(evaluate_main(args), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["checkpoint_epoch"], 7)
            self.assertEqual(result["split"], "validation")
            self.assertEqual(result["num_images"], 1)
            self.assertEqual(result["device"], "cpu")
            self.assertIn("cer", result)
            self.assertIn("exact_match", result)
            with destination.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, ["filename", "expected", "predicted", "correct"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["filename"], "validation_0.png")
            self.assertEqual(rows[0]["expected"], "baba")
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(evaluate_main(args), 1)
            self.assertIn("already exists", error.getvalue())


if __name__ == "__main__":
    unittest.main()
