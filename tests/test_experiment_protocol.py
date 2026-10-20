"""Check the manual experiment contract without opening data or training."""

import json
import math
from pathlib import Path
import random
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.model import CRNN
from ml.train import parse_args, seed_everything, seed_worker


ROOT = Path(__file__).resolve().parents[1]


class ExperimentProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "experiments/dataset-expansion.json").read_text(encoding="utf-8")
        )

    def arguments(self, experiment):
        # These checks must never dispatch training, even if the fixture changes.
        with mock.patch("ml.train.run", side_effect=AssertionError("No training in protocol checks")):
            return parse_args(self.protocol[experiment]["argv"])

    def test_b_and_c_differ_only_in_scheduler_policy_and_output_directory(self):
        fixed, adaptive = self.arguments("B"), self.arguments("C")
        self.assertEqual(fixed.scheduler, "none")
        self.assertEqual(adaptive.scheduler, "plateau")
        self.assertNotEqual(fixed.run_dir, adaptive.run_dir)
        shared = lambda args: {
            key: value for key, value in vars(args).items() if key not in ("scheduler", "run_dir")
        }
        self.assertEqual(shared(fixed), shared(adaptive))
        for args in (fixed, adaptive):
            self.assertEqual(args.data, Path("dataset/synthetic/v2_9k_train"))
            self.assertEqual(args.device, "cpu")
            self.assertEqual(args.epochs, 30)
            self.assertEqual(args.batch_size, 16)
            self.assertEqual(args.learning_rate, 0.001)
            self.assertEqual(args.seed, 42)
            self.assertEqual(args.num_workers, 0)
            self.assertEqual(args.threads, 6)
            self.assertIsNone(args.resume)
            self.assertIsNone(args.overfit_samples)
            self.assertFalse(args.dry_run)

    def test_b_and_c_have_identical_initial_weights_and_shuffle_protocol(self):
        """Model construction and three tiny loader orders; no forward or update."""
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        states, orders = [], []
        try:
            for experiment in ("B", "C"):
                args = self.arguments(experiment)
                seed_everything(args.seed)
                model = CRNN(31)
                states.append({name: value.detach().clone() for name, value in model.state_dict().items()})
                generator = torch.Generator().manual_seed(args.seed)
                loader = DataLoader(torch.arange(23), shuffle=True, generator=generator,
                                    batch_size=args.batch_size, num_workers=args.num_workers,
                                    worker_init_fn=seed_worker)
                orders.append([[value for batch in loader for value in batch.tolist()]
                               for _ in range(3)])
            for name in states[0]:
                self.assertTrue(torch.equal(states[0][name], states[1][name]), name)
            self.assertEqual(orders[0], orders[1])
            self.assertNotEqual(orders[0][0], orders[0][1])
            for order in orders[0]:
                self.assertEqual(sorted(order), list(range(23)))
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)

    def test_controlled_update_budget_and_same_total_presentations(self):
        baseline = self.protocol["A"]
        self.assertEqual(baseline["train_images"], 2700)
        self.assertEqual(baseline["validation_images"], 300)
        self.assertEqual(baseline["epochs"], 100)
        self.assertEqual(math.ceil(baseline["train_images"] / baseline["batch_size"])
                         * baseline["epochs"], baseline["updates"])
        for experiment in ("B", "C"):
            entry, args = self.protocol[experiment], self.arguments(experiment)
            self.assertEqual((entry["train_images"], entry["validation_images"]), (9000, 300))
            self.assertEqual(math.ceil(entry["train_images"] / args.batch_size) * args.epochs, 16890)
            self.assertEqual(entry["updates"], 16890)
            self.assertEqual(entry["train_images"] * args.epochs,
                             baseline["train_images"] * baseline["epochs"])

    def test_baseline_has_no_training_command_and_is_never_an_output_target(self):
        baseline = self.protocol["A"]
        self.assertEqual(baseline["status"], "completed")
        self.assertNotIn("argv", baseline)
        self.assertEqual(baseline["run_dir"], "outputs/runs/crnn-v2-baseline")
        protected = {Path(path) for path in self.protocol["protected_paths"]}
        self.assertIn(Path("dataset/synthetic/v2_main"), protected)
        self.assertIn(Path(baseline["run_dir"]), protected)
        for experiment in ("B", "C"):
            args = self.arguments(experiment)
            self.assertNotIn(args.run_dir, protected)
            self.assertNotIn(args.data, protected)
            self.assertEqual(self.protocol[experiment]["status"], "prepared_manual_only")
        self.assertFalse(self.protocol["training_executed_during_preparation"])

    def test_copyable_documented_commands_match_the_checked_protocol(self):
        for document in (ROOT / "experiments/README.md", ROOT / "ml/README.md"):
            content = document.read_text(encoding="utf-8")
            for experiment in ("B", "C"):
                command = ".\\.venv-ml\\Scripts\\python.exe -m ml.train " + " ".join(
                    self.protocol[experiment]["argv"]
                )
                self.assertIn(command, content, f"{experiment}: {document.name}")


if __name__ == "__main__":
    unittest.main()
