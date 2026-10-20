"""Short CPU scheduler/checkpoint checks using synthetic fixtures only."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from ml.charset import Charset
from ml.dataset import collate_ctc
from ml.model import CRNN
from ml.train import (build_checkpoint, build_scheduler, load_checkpoint, main, parse_args,
                      restore_checkpoint, save_checkpoint, scheduler_configuration,
                      seed_everything, train_step, update_learning_rate, write_history)


def configuration(**changes):
    return {"batch_size": 2, "learning_rate": 0.001, "seed": 42, "num_workers": 0,
            "overfit_samples": None, "threads": 2, "scheduler": "plateau",
            "plateau_patience": 8, "plateau_factor": 0.5, "plateau_min_delta": 0.002,
            "plateau_cooldown": 2, "plateau_min_lr": 0.00001, "plateau_start_epoch": 10,
            **changes}


def fixture_dataset(root):
    rows = []
    for split, labels in (("train", ("aabb", "abba")), ("validation", ("baba",))):
        (root / split).mkdir(parents=True)
        for index, label in enumerate(labels):
            filename = f"{split}_{index}.png"
            image = Image.new("RGB", (215, 80), "white")
            ImageDraw.Draw(image).text((15 + index, 25), label, fill="black")
            image.save(root / split / filename)
            rows.append({"filename": filename, "label": label, "split": split, "width": 215, "height": 80})
    with (root / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return root


class SchedulerPolicyTests(unittest.TestCase):
    def make_policy(self, **changes):
        self.config = configuration(**changes)
        self.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.001)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        return self.scheduler

    def observe(self, values, start=1):
        return [update_learning_rate(self.optimizer, self.scheduler, value, epoch,
                                     self.config["plateau_start_epoch"])
                for epoch, value in enumerate(values, start=start)]

    def test_defaults_are_optional_and_absolute_cer(self):
        args = parse_args([])
        self.assertEqual(args.scheduler, "none")
        self.assertEqual(scheduler_configuration(vars(args)), {"name": "none"})
        settings = scheduler_configuration(vars(parse_args(["--scheduler", "plateau"])))
        self.assertEqual(settings, {"name": "plateau", "mode": "min", "metric": "validation_cer",
                                    "factor": 0.5, "patience": 8, "threshold": 0.002,
                                    "threshold_mode": "abs", "cooldown": 2,
                                    "min_lr": 0.00001, "start_epoch": 10})

    def test_fixed_lr_is_unchanged(self):
        self.assertIsNone(self.make_policy(scheduler="none"))
        rows = self.observe([1.0] * 30)
        self.assertTrue(all(row == {"lr_used": 0.001, "lr_next": 0.001, "lr_reduced": False} for row in rows))
        self.assertEqual(parse_args(["--learning-rate", "0.000001"]).learning_rate, 0.000001)

    def test_warmup_does_not_observe_or_count_stagnation(self):
        scheduler = self.make_policy()
        before = copy.deepcopy(scheduler.state_dict())
        self.observe([0.2] * 10)
        self.assertEqual(scheduler.state_dict(), before)
        self.observe([0.5], start=11)
        self.assertEqual(scheduler.best, 0.5)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(scheduler.num_bad_epochs, 0)
        self.assertFalse(any(row["lr_reduced"] for row in self.observe([0.5] * 8, start=12)))
        row = self.observe([0.5], start=20)[0]
        self.assertEqual(row, {"lr_used": 0.001, "lr_next": 0.0005, "lr_reduced": True})

    def test_improvement_resets_patience(self):
        scheduler = self.make_policy(plateau_start_epoch=0, plateau_patience=1)
        rows = self.observe([0.5, 0.5, 0.497, 0.497, 0.497])
        self.assertEqual([row["lr_reduced"] for row in rows], [False, False, False, False, True])
        self.assertEqual(scheduler.best, 0.497)

    def test_threshold_is_absolute_not_relative(self):
        scheduler = self.make_policy(plateau_start_epoch=0, plateau_patience=0, plateau_cooldown=0)
        rows = self.observe([0.1, 0.099, 0.0979])
        self.assertTrue(rows[1]["lr_reduced"])
        self.assertFalse(rows[2]["lr_reduced"])
        self.assertEqual(scheduler.best, 0.0979)

    def test_custom_factor_patience_and_cooldown(self):
        self.make_policy(plateau_start_epoch=0, plateau_patience=0,
                         plateau_factor=0.25, plateau_cooldown=2)
        rows = self.observe([1.0] * 5)
        self.assertEqual([row["lr_next"] for row in rows], [0.001, 0.00025, 0.00025, 0.00025, 0.0000625])
        self.assertEqual([row["lr_reduced"] for row in rows], [False, True, False, False, True])

    def test_minimum_lr_is_respected(self):
        self.make_policy(plateau_start_epoch=0, plateau_patience=0, plateau_cooldown=0,
                         plateau_min_lr=0.0001)
        rows = self.observe([1.0] * 12)
        self.assertEqual([row["lr_next"] for row in rows[:5]], [0.001, 0.0005, 0.00025, 0.000125, 0.0001])
        self.assertTrue(all(row["lr_next"] == 0.0001 and not row["lr_reduced"] for row in rows[5:]))

    def test_invalid_cer_is_rejected_even_during_warmup_and_fixed_lr(self):
        for name in ("none", "plateau"):
            for epoch in (1, 11):
                for value in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(name=name, epoch=epoch, value=value):
                        scheduler = self.make_policy(scheduler=name)
                        before = copy.deepcopy(scheduler.state_dict()) if scheduler else None
                        with self.assertRaisesRegex(FloatingPointError, "validation CER.*epoch"):
                            update_learning_rate(self.optimizer, scheduler, value, epoch)
                        self.assertEqual(self.optimizer.param_groups[0]["lr"], 0.001)
                        if scheduler:
                            self.assertEqual(scheduler.state_dict(), before)

    def test_duplicate_or_skipped_scheduler_decisions_are_rejected(self):
        self.make_policy(plateau_start_epoch=0)
        self.observe([0.5])
        for epoch in (1, 3):
            with self.assertRaisesRegex(ValueError, "duplicate/skipped"):
                self.observe([0.5], start=epoch)

    def test_history_preserves_columns_and_records_used_next_and_event(self):
        self.make_policy(plateau_start_epoch=0, plateau_patience=0, plateau_cooldown=0)
        rows = [{"epoch": epoch, "cer": 0.5, "train_loss": 2.0, "val_loss": 3.0, **rates}
                for epoch, rates in enumerate(self.observe([0.5, 0.5, 0.5]), start=1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            write_history(path, rows)
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                result = list(reader)
                self.assertEqual(reader.fieldnames[:10], ["epoch", "train_loss", "val_loss", "exact_match", "cer",
                    "num_images", "exact_match_4", "exact_match_5", "exact_match_6", "epoch_seconds"])
            self.assertEqual([float(row["lr_used"]) for row in result], [0.001, 0.001, 0.0005])
            self.assertEqual([float(row["lr_next"]) for row in result], [0.001, 0.0005, 0.00025])
            self.assertEqual([row["lr_reduced"] for row in result], ["False", "True", "True"])

    def test_invalid_cli_scheduler_settings_are_rejected(self):
        cases = [("--scheduler", "cosine"), ("--plateau-factor", "1"), ("--plateau-factor", "nan"),
                 ("--plateau-factor", "0"), ("--plateau-min-delta", "-0.1"),
                 ("--plateau-min-delta", "inf"), ("--plateau-min-lr", "0.01"),
                 ("--plateau-min-lr", "nan"), ("--plateau-patience", "-1"),
                 ("--plateau-cooldown", "-1"), ("--plateau-start-epoch", "-1")]
        for arguments in cases:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--scheduler", "plateau", *arguments])


class SchedulerCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_training(self, config):
        seed_everything(config["seed"])
        model = CRNN(3, hidden_size=8, num_layers=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        scheduler = build_scheduler(optimizer, config)
        return model, optimizer, scheduler, torch.Generator().manual_seed(config["seed"])

    def assert_state_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_continuous_and_resumed_training_match_weights_optimizer_rng_shuffle_and_lrs(self):
        config = configuration(plateau_start_epoch=1, plateau_patience=1, plateau_cooldown=1)
        charset = Charset("ab")
        identity = {"fixture": "scheduler-determinism"}
        pixels = torch.rand(4, 1, 80, 215, generator=torch.Generator().manual_seed(7)) * 2 - 1
        samples = [{"image": pixels[index], "target": torch.tensor(charset.encode(label)),
                    "label": label, "filename": f"fixture_{index}.png"}
                   for index, label in enumerate(("aabb", "abba", "baba", "baab"))]
        metrics = [0.5, 0.5, 0.5, 0.496, 0.496, 0.496, 0.496, 0.496]

        def advance(training, start, end, history):
            model, optimizer, scheduler, generator = training
            loader = DataLoader(samples, batch_size=2, shuffle=True, generator=generator, collate_fn=collate_ctc)
            orders = []
            for epoch in range(start, end + 1):
                order = []
                for batch in loader:
                    order.extend(batch["filenames"])
                    # Exercise all restored RNG streams during actual updates.
                    noise = (random.random() + float(np.random.random()) + float(torch.rand(()))) * 0.001
                    batch["images"] = batch["images"] + noise
                    train_step(model, batch, optimizer, torch.device("cpu"))
                orders.append(order)
                history.append({"epoch": epoch, "cer": metrics[epoch - 1],
                                **update_learning_rate(optimizer, scheduler, metrics[epoch - 1], epoch, 1)})
            return orders

        continuous = self.make_training(config)
        continuous_history = []
        expected_orders = advance(continuous, 1, 8, continuous_history)
        expected_rng = (random.random(), np.random.random(), torch.rand(3),
                        torch.randperm(9, generator=continuous[3]))
        interrupted = self.make_training(config)
        history = []
        first_orders = advance(interrupted, 1, 5, history)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, build_checkpoint(interrupted[0], interrupted[1], 5, charset, config,
                identity, {"cer": 0.496, "exact_match": 0.0, "epoch": 4}, history, interrupted[3], interrupted[2]))
            checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint["scheduler_state_dict"]["num_bad_epochs"], 1)
            resumed = self.make_training(config)
            next_epoch = restore_checkpoint(checkpoint, resumed[0], resumed[1], charset, identity,
                                            config, resumed[3], resumed[2])
            self.assertEqual(next_epoch, 6)
            self.assertEqual(resumed[2].num_bad_epochs, 1)
            remaining_orders = advance(resumed, next_epoch, 8, checkpoint["history"])
        self.assertEqual(expected_orders, first_orders + remaining_orders)
        self.assertEqual(continuous_history, checkpoint["history"])
        self.assertTrue(any(row["lr_reduced"] for row in continuous_history))
        self.assert_state_equal(continuous[0].state_dict(), resumed[0].state_dict())
        self.assert_state_equal(continuous[1].state_dict(), resumed[1].state_dict())
        self.assert_state_equal(continuous[2].state_dict(), resumed[2].state_dict())
        self.assertEqual(random.random(), expected_rng[0])
        self.assertEqual(np.random.random(), expected_rng[1])
        self.assertTrue(torch.equal(torch.rand(3), expected_rng[2]))
        self.assertTrue(torch.equal(torch.randperm(9, generator=resumed[3]), expected_rng[3]))

    def test_legacy_fixed_checkpoint_loads_and_cannot_enable_scheduler(self):
        config = configuration(scheduler="none")
        training = self.make_training(config)
        charset = Charset("ab")
        checkpoint = build_checkpoint(training[0], training[1], 1, charset, config, {}, {}, [], training[3])
        del checkpoint["scheduler_state_dict"]
        del checkpoint["scheduler_config"]
        for key in list(checkpoint["config"]):
            if key == "scheduler" or key.startswith("plateau_"):
                del checkpoint["config"][key]
        restored = self.make_training(configuration(scheduler="none"))
        self.assertEqual(restore_checkpoint(checkpoint, restored[0], restored[1], charset, {},
            configuration(scheduler="none"), restored[3]), 2)
        plateau = self.make_training(configuration())
        with self.assertRaisesRegex(ValueError, "legacy.*cannot enable"):
            restore_checkpoint(checkpoint, plateau[0], plateau[1], charset, {},
                               configuration(), plateau[3], plateau[2])

    def test_incompatible_scheduler_resume_and_state_are_rejected(self):
        config = configuration(plateau_start_epoch=0)
        training = self.make_training(config)
        update_learning_rate(training[1], training[2], 0.5, 1, 0)
        checkpoint = build_checkpoint(training[0], training[1], 1, Charset("ab"), config, {}, {}, [], training[3], training[2])
        for key, value in (("scheduler", "none"), ("plateau_patience", 7), ("plateau_factor", 0.25),
                           ("plateau_min_delta", 0.003), ("plateau_cooldown", 1),
                           ("plateau_min_lr", 0.000001), ("plateau_start_epoch", 1)):
            with self.subTest(key=key):
                changed = {**config, key: value}
                restored = self.make_training(changed)
                with self.assertRaisesRegex(ValueError, "scheduler configuration mismatch"):
                    restore_checkpoint(checkpoint, restored[0], restored[1], Charset("ab"), {}, changed,
                                       restored[3], restored[2])
        for key, value in (("last_epoch", 0), ("_last_lr", [0.1]), ("threshold_mode", "rel")):
            altered = copy.deepcopy(checkpoint)
            altered["scheduler_state_dict"][key] = value
            restored = self.make_training(config)
            with self.subTest(state=key), self.assertRaisesRegex(ValueError, "scheduler"):
                restore_checkpoint(altered, restored[0], restored[1], Charset("ab"), {}, config,
                                   restored[3], restored[2])

    def test_checkpoint_requires_current_scheduler_decision(self):
        config = configuration(plateau_start_epoch=0)
        model, optimizer, scheduler, generator = self.make_training(config)
        with self.assertRaisesRegex(ValueError, "current epoch"):
            build_checkpoint(model, optimizer, 1, Charset("ab"), config, {}, {}, [], generator, scheduler)

    def test_cli_fixture_reduces_logs_restores_current_lr_and_preserves_cer_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture_dataset(Path(directory) / "fixture")
            run_dir = Path(directory) / "run"
            common = ["--data", str(root), "--run-dir", str(run_dir), "--device", "cpu", "--threads", "2",
                      "--batch-size", "2", "--scheduler", "plateau", "--plateau-start-epoch", "0",
                      "--plateau-patience", "0", "--plateau-cooldown", "0"]
            observations = [
                {"cer": 0.5, "exact_match": 0.0, "loss": 9.0, "num_images": 1, "by_length": {}},
                {"cer": 0.5, "exact_match": 1.0, "loss": 10.0, "num_images": 1, "by_length": {}},
                {"cer": 0.5, "exact_match": 0.0, "loss": 1.0, "num_images": 1, "by_length": {}},
                {"cer": 0.5, "exact_match": 0.0, "loss": 0.5, "num_images": 1, "by_length": {}},
            ]
            output = io.StringIO()
            with mock.patch("ml.train.CRNN", side_effect=lambda n: CRNN(n, hidden_size=8, num_layers=1)), \
                    mock.patch("ml.train.validate", side_effect=observations), contextlib.redirect_stdout(output):
                self.assertEqual(main(common + ["--epochs", "3"]), 0)
                before = load_checkpoint(run_dir / "last.pt")
                self.assertEqual(before["scheduler_state_dict"]["last_epoch"], 3)
                self.assertEqual(before["optimizer_state_dict"]["param_groups"][0]["lr"], 0.00025)
                best_bytes = (run_dir / "best.pt").read_bytes()
                self.assertEqual(main(common + ["--epochs", "4", "--resume", str(run_dir / "last.pt")]), 0)
            checkpoint = load_checkpoint(run_dir / "last.pt")
            self.assertEqual([row["lr_used"] for row in checkpoint["history"]], [0.001, 0.001, 0.0005, 0.00025])
            self.assertEqual([row["lr_next"] for row in checkpoint["history"]], [0.001, 0.0005, 0.00025, 0.000125])
            self.assertEqual(checkpoint["scheduler_state_dict"]["last_epoch"], 4)
            self.assertEqual(checkpoint["best_metrics"]["epoch"], 2)
            self.assertEqual((run_dir / "best.pt").read_bytes(), best_bytes)
            self.assertIn("ReduceLROnPlateau: validation CER stagnated", output.getvalue())
            self.assertIn("LR used:", output.getvalue())
            self.assertIn("LR next:", output.getvalue())
            self.assertTrue(all(state["step"].item() == 4 for state in checkpoint["optimizer_state_dict"]["state"].values()))

    def test_cli_dry_runs_both_policies_forward_backward_without_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture_dataset(Path(directory) / "fixture")
            for policy in ("none", "plateau"):
                output = io.StringIO()
                run_dir = Path(directory) / policy
                with self.subTest(policy=policy), contextlib.redirect_stdout(output):
                    self.assertEqual(main(["--data", str(root), "--device", "cpu", "--threads", "2",
                        "--batch-size", "2", "--run-dir", str(run_dir), "--scheduler", policy, "--dry-run"]), 0)
                report = json.loads(output.getvalue().split("DRY RUN PASSED")[0])
                self.assertEqual(report["scheduler"], policy)
                self.assertTrue(report["parameters_changed"])
                self.assertEqual(report["scheduler_observations"], 1 if policy == "plateau" else 0)
                self.assertEqual(report["lr_used"], 0.001)
                self.assertEqual(report["lr_next"], 0.001)
                self.assertFalse(run_dir.exists())
            self.assertFalse(list(Path(directory).rglob("*.pt")))


if __name__ == "__main__":
    unittest.main()
