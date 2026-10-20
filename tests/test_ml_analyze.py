"""Offline analysis tests use random, frozen weights and local temporary PNGs.

No optimizer update or backward pass is needed to exercise checkpoint analysis.
"""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ml.charset import Charset
from ml.dataset import SyntheticDataset, SyntheticManifest, collate_ctc
from ml.model import CRNN
from ml.train import build_checkpoint, load_checkpoint, save_checkpoint, validate


def create_fixture(root):
    root = Path(root)
    rows = []
    for split, labels in (("train", ("bbaa",)), ("validation", ("aabb", "ababa"))):
        (root / split).mkdir(parents=True)
        for index, label in enumerate(labels):
            filename = f"{split}_{index}.png"
            image = Image.new("RGB", (215, 80), "white")
            ImageDraw.Draw(image).text((20, 30), label, fill="black")
            image.save(root / split / filename)
            rows.append({"filename": filename, "label": label, "split": split,
                         "width": 215, "height": 80})
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filename", "label", "split", "width", "height"))
        writer.writeheader()
        writer.writerows(rows)
    return root


class AnalyzeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = create_fixture(self.root / "synthetic-fixture")
        self.run = self.root / "run"
        self.run.mkdir()
        self.checkpoint_path = self.run / "best.pt"
        self.output = self.root / "analysis"
        self.charset = Charset("ab")
        torch.manual_seed(42)
        self.model = CRNN(3, hidden_size=8, num_layers=1).eval()
        self.model.requires_grad_(False)
        # Merely instantiate the optimizer required by the checkpoint format.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.manifest = SyntheticManifest(self.data)
        dataset = SyntheticDataset(self.data, "validation", self.charset, manifest=self.manifest)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_ctc)
        self.reference = validate(self.model, loader, torch.device("cpu"), self.charset)
        row = {"epoch": 95, "train_loss": 12.0, "val_loss": self.reference["loss"],
               **{key: value for key, value in self.reference.items() if key != "loss"}}
        self.config = {"seed": 42, "batch_size": 2, "learning_rate": 0.001,
                       "num_workers": 0, "threads": 2, "overfit_samples": None}
        checkpoint = build_checkpoint(
            self.model, self.optimizer, 95, self.charset, self.config, self.manifest.identity,
            {"cer": self.reference["cer"], "exact_match": self.reference["exact_match"], "epoch": 95},
            [row], torch.Generator().manual_seed(42),
        )
        save_checkpoint(self.checkpoint_path, checkpoint)
        self.assertFalse(self.optimizer.state)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def arguments(self, output=None):
        return ["--data", str(self.data), "--split", "validation", "--checkpoint", str(self.checkpoint_path),
                "--device", "cpu", "--batch-size", "2", "--threads", "2", "--seed", "42",
                "--output", str(self.output if output is None else output)]

    def run_analysis(self, arguments=None):
        from ml.analyze import main
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main(self.arguments() if arguments is None else arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_shared_evaluation_loads_epoch_charset_and_validation_only(self):
        from ml.evaluate import evaluate_checkpoint
        visited = []
        original_getitem = SyntheticDataset.__getitem__
        # Analysis must succeed without opening any training PNG.
        (self.data / "train" / "train_0.png").unlink()

        def recorded_getitem(dataset, index):
            visited.append((dataset.split, dataset.records[index].filename))
            return original_getitem(dataset, index)

        with mock.patch.object(SyntheticDataset, "__getitem__", recorded_getitem):
            result = evaluate_checkpoint(self.data, "validation", self.checkpoint_path,
                                         device="cpu", batch_size=2, threads=2)
        self.assertEqual(visited, [("validation", "validation_0.png"), ("validation", "validation_1.png")])
        self.assertEqual(result["charset"].characters, "ab")
        self.assertEqual(result["metrics"]["num_images"], 2)
        self.assertAlmostEqual(result["metrics"]["cer"], self.reference["cer"])
        self.assertAlmostEqual(result["metrics"]["exact_match"], self.reference["exact_match"])
        self.assertEqual(result["model_config"], self.model.config)
        self.assertEqual(result["checkpoint"]["epoch"], 95)
        self.assertEqual(result["parameter_count"], sum(parameter.numel() for parameter in self.model.parameters()))
        self.assertEqual(result["dataset_identity"]["manifest_sha256"], self.manifest.identity["manifest_sha256"])
        self.assertEqual(result["dataset_identity"]["validated_image_splits"], ["validation"])
        self.assertEqual(result["dataset_identity"]["validated_image_count"], 2)
        self.assertTrue(result["preservation_verified"])
        self.assertEqual(len(result["checkpoint_sha256"]), 64)

    def test_analysis_exports_artifacts_without_backward_or_optimizer_updates(self):
        originals = {path: path.read_bytes() for path in self.data.rglob("*") if path.is_file()}
        originals[self.checkpoint_path] = self.checkpoint_path.read_bytes()
        original_weights = {name: tensor.clone() for name, tensor in load_checkpoint(self.checkpoint_path)["model_state_dict"].items()}
        modes = []
        original_forward = CRNN.forward

        def checked_forward(model, images):
            modes.append((model.training, torch.is_grad_enabled(), any(parameter.requires_grad for parameter in model.parameters())))
            return original_forward(model, images)

        with mock.patch.object(CRNN, "forward", checked_forward), \
                mock.patch.object(torch.Tensor, "backward", side_effect=AssertionError("analysis must not backpropagate")), \
                mock.patch.object(torch.optim.AdamW, "__init__", side_effect=AssertionError("analysis needs no optimizer")), \
                mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("analysis must not train")):
            result, stdout, stderr = self.run_analysis()
        self.assertEqual(result, 0, stdout + stderr)
        self.assertTrue(modes)
        self.assertTrue(all(not training and not grad_enabled and not trainable
                            for training, grad_enabled, trainable in modes))
        expected_files = {"predictions.csv", "summary.json", "REPORT.md", "error-gallery.png",
                          "confusion-matrix.png", "by-length.png"}
        self.assertTrue(expected_files.issubset({path.name for path in self.output.iterdir()}))
        for filename in ("error-gallery.png", "confusion-matrix.png", "by-length.png"):
            with Image.open(self.output / filename) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)
                image.verify()
        for path, contents in originals.items():
            self.assertEqual(path.read_bytes(), contents, str(path))
        restored = load_checkpoint(self.checkpoint_path)
        self.assertEqual(restored["epoch"], 95)
        self.assertFalse(restored["optimizer_state_dict"]["state"])
        for name, tensor in restored["model_state_dict"].items():
            self.assertTrue(torch.equal(original_weights[name], tensor), name)

    def test_output_cannot_overwrite_nonempty_directory(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep this report", encoding="utf-8")
        result, stdout, stderr = self.run_analysis()
        self.assertEqual(result, 1, stdout + stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep this report")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_predictions_csv_and_summary_match_all_validation_samples(self):
        from ml.metrics import levenshtein
        result, stdout, stderr = self.run_analysis()
        self.assertEqual(result, 0, stdout + stderr)
        with (self.output / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertTrue({"filename", "expected", "predicted", "correct", "expected_length",
                             "predicted_length", "edit_distance", "sample_cer", "substitutions",
                             "deletions", "insertions"}.issubset(reader.fieldnames))
        self.assertEqual([row["filename"] for row in rows], ["validation_0.png", "validation_1.png"])
        self.assertEqual([row["expected"] for row in rows], ["aabb", "ababa"])
        for row in rows:
            distance = levenshtein(row["expected"], row["predicted"])
            self.assertEqual(int(row["edit_distance"]), distance)
            self.assertEqual(int(row["expected_length"]), len(row["expected"]))
            self.assertEqual(int(row["predicted_length"]), len(row["predicted"]))
            self.assertAlmostEqual(float(row["sample_cer"]), distance / len(row["expected"]))
            self.assertEqual(sum(int(row[field]) for field in ("substitutions", "deletions", "insertions")), distance)
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["global"]["num_images"], 2)
        self.assertEqual(summary["global"]["total_reference_characters"], 9)
        self.assertAlmostEqual(summary["global"]["exact_match"], self.reference["exact_match"])
        self.assertAlmostEqual(summary["global"]["cer"], self.reference["cer"])
        self.assertEqual(set(summary["by_length"]), {"4", "5"})
        self.assertTrue(summary["verification"]["matches_checkpoint"])
        self.assertEqual(summary["verification"]["differences"], [])
        self.assertIn("95", (self.output / "REPORT.md").read_text(encoding="utf-8"))

    def test_stored_metric_divergence_is_reported_and_returns_failure(self):
        checkpoint = load_checkpoint(self.checkpoint_path)
        checkpoint["metrics"]["cer"] = self.reference["cer"] + 0.5
        save_checkpoint(self.checkpoint_path, checkpoint)
        result, stdout, stderr = self.run_analysis()
        self.assertEqual(result, 1, stdout + stderr)
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["verification"]["matches_checkpoint"])
        self.assertTrue(summary["verification"]["differences"])
        self.assertAlmostEqual(summary["global"]["cer"], self.reference["cer"])
        report = (self.output / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("DIVERG", report.upper())

    def test_output_cannot_be_inside_dataset_or_checkpoint_directory(self):
        for output in (self.data / "analysis", self.run / "analysis"):
            with self.subTest(output=output):
                result, stdout, stderr = self.run_analysis(self.arguments(output))
                self.assertEqual(result, 1, stdout + stderr)
                self.assertFalse(output.exists())

    def test_output_cannot_replace_source_directory_or_its_ancestor(self):
        sources = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        for output in (self.data, self.run, self.root):
            with self.subTest(output=output):
                result, stdout, stderr = self.run_analysis(self.arguments(output))
                self.assertEqual(result, 1, stdout + stderr)
                self.assertIn("separate", (stdout + stderr).lower())
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, sources)

    def test_malformed_checkpoint_is_rejected_without_creating_report(self):
        torch.save({"format_version": 1}, self.checkpoint_path)
        result, stdout, stderr = self.run_analysis()
        self.assertEqual(result, 1, stdout + stderr)
        self.assertTrue("checkpoint" in (stdout + stderr).lower())
        self.assertFalse(self.output.exists())

    def rewrite_rows(self, rows):
        with (self.data / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("filename", "label", "split", "width", "height"))
            writer.writeheader()
            writer.writerows(rows)

    def rows(self):
        with (self.data / "labels.csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_shared_evaluation_rejects_empty_validation(self):
        from ml.evaluate import evaluate_checkpoint
        self.rewrite_rows([row for row in self.rows() if row["split"] == "train"])
        with self.assertRaisesRegex(ValueError, "no images.*validation|validation.*empty"):
            evaluate_checkpoint(self.data, "validation", self.checkpoint_path, threads=2)

    def test_shared_evaluation_rejects_unknown_validation_character(self):
        from ml.evaluate import evaluate_checkpoint
        rows = self.rows()
        for row in rows:
            if row["split"] == "validation":
                row["label"] = "aacc"
                break
        self.rewrite_rows(rows)
        with self.assertRaisesRegex(ValueError, "Unknown character.*c.*validation_0"):
            evaluate_checkpoint(self.data, "validation", self.checkpoint_path, threads=2)

    def test_frozen_experiment_rejects_changed_csv_identity(self):
        from ml.evaluate import evaluate_checkpoint
        manifest = self.data / "labels.csv"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "CSV differs|identity"):
            evaluate_checkpoint(self.data, "validation", self.checkpoint_path, threads=2,
                                strict_dataset_identity=True)

    def test_analysis_refuses_train_split(self):
        arguments = self.arguments()
        arguments[arguments.index("validation")] = "train"
        with self.assertRaises(SystemExit) as error:
            self.run_analysis(arguments)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
