"""Small CPU policy/checkpoint checks; generated fixtures only, no full training."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageDraw
import torch

from ml.charset import Charset
from ml.model import CRNN
from ml.train import (EarlyStopping, build_checkpoint, build_early_stopping, build_scheduler,
                      early_stopping_configuration, load_checkpoint, main, parse_args,
                      restore_checkpoint, run, save_checkpoint, seed_everything, update_learning_rate, versions)


def configuration(**changes):
    return {"batch_size": 2, "learning_rate": 0.001, "seed": 42, "num_workers": 0,
            "overfit_samples": None, "threads": 2, "scheduler": "plateau",
            "plateau_patience": 8, "plateau_factor": 0.5, "plateau_min_delta": 0.002,
            "plateau_cooldown": 2, "plateau_min_lr": 0.00001, "plateau_start_epoch": 10,
            "early_stopping_patience": 25, "early_stopping_min_delta": 0.002,
            "early_stopping_start_epoch": 11, **changes}


class EarlyStoppingPolicyTests(unittest.TestCase):
    def test_default_remains_disabled(self):
        args = parse_args([])
        self.assertEqual(args.early_stopping_patience, 0)
        self.assertIsNone(build_early_stopping(vars(args)))
        self.assertEqual(early_stopping_configuration({}), {"enabled": False})
        configured = parse_args(["--early-stopping-patience", "25"])
        self.assertEqual(early_stopping_configuration(vars(configured)),
                         {"enabled": True, "mode": "min", "metric": "validation_cer",
                          "patience": 25, "min_delta": 0.002, "start_epoch": 11})

    def test_warmup_is_ignored_and_first_monitored_epoch_sets_baseline(self):
        stopping = EarlyStopping(configuration())
        initial = stopping.state_dict()
        for epoch in range(1, 11):
            self.assertFalse(stopping.step(0.01, epoch))
        self.assertEqual(stopping.state_dict(), initial)
        self.assertFalse(stopping.step(0.5, 11))
        self.assertEqual(stopping.best_reference_cer, 0.5)
        self.assertEqual(stopping.bad_epochs, 0)
        self.assertEqual(stopping.observations, 1)

    def test_stagnation_stops_after_exact_patience(self):
        stopping = EarlyStopping(configuration(early_stopping_start_epoch=1, early_stopping_patience=2))
        self.assertEqual([stopping.step(value, epoch) for epoch, value in enumerate((.5, .5, .6), 1)],
                         [False, False, True])
        self.assertTrue(stopping.stopped)
        self.assertEqual(stopping.bad_epochs, 2)
        with self.assertRaisesRegex(ValueError, "already exhausted"):
            stopping.step(.1, 4)

    def test_sufficient_improvement_resets_patience(self):
        stopping = EarlyStopping(configuration(early_stopping_start_epoch=1, early_stopping_patience=2))
        values = (.5, .5, .497, .497, .497)
        self.assertEqual([stopping.step(value, epoch) for epoch, value in enumerate(values, 1)],
                         [False, False, False, False, True])
        self.assertEqual(stopping.best_reference_cer, .497)

    def test_absolute_min_delta_must_be_strictly_exceeded(self):
        stopping = EarlyStopping(configuration(early_stopping_start_epoch=1,
                                               early_stopping_min_delta=.125))
        stopping.step(1.0, 1)
        stopping.step(.875, 2)  # Exactly the threshold is insufficient.
        self.assertEqual(stopping.best_reference_cer, 1.0)
        self.assertEqual(stopping.bad_epochs, 1)
        stopping.step(.874, 3)
        self.assertEqual(stopping.best_reference_cer, .874)
        self.assertEqual(stopping.bad_epochs, 0)

    def test_small_improvements_accumulate_against_significant_reference(self):
        stopping = EarlyStopping(configuration(early_stopping_start_epoch=1))
        for epoch, value in enumerate((.5, .499, .4979), 1):
            stopping.step(value, epoch)
        self.assertEqual(stopping.best_reference_cer, .4979)
        self.assertEqual(stopping.bad_epochs, 0)

    def test_scheduler_reduces_and_observes_effects_before_default_early_stop(self):
        config = configuration()
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.001)
        scheduler, stopping = build_scheduler(optimizer, config), EarlyStopping(config)
        reductions = []
        for epoch in range(1, 101):
            rates = update_learning_rate(optimizer, scheduler, .5, epoch, 10)
            if rates["lr_reduced"]:
                reductions.append(epoch)
            if stopping.step(.5, epoch):
                break
        self.assertEqual(reductions, [20, 31])
        self.assertEqual(epoch, 36)
        self.assertEqual(optimizer.param_groups[0]["lr"], .00025)

    def test_duplicate_skipped_decisions_and_nonfinite_cer_are_rejected(self):
        stopping = EarlyStopping(configuration())
        stopping.step(.5, 11)
        for epoch in (11, 13):
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "duplicate/skipped"):
                stopping.step(.5, epoch)
        before = stopping.state_dict()
        for value in (float("inf"), float("nan")):
            with self.assertRaises(FloatingPointError):
                stopping.step(value, 12)
        self.assertEqual(stopping.state_dict(), before)

    def test_invalid_cli_parameters_are_rejected(self):
        for arguments in (("--early-stopping-patience", "-1"),
                          ("--early-stopping-min-delta", "-0.1"),
                          ("--early-stopping-min-delta", "nan"),
                          ("--early-stopping-min-delta", "inf"),
                          ("--early-stopping-start-epoch", "0")):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(list(arguments))

    def test_cuda_run_disables_tf32_and_cpu_run_preserves_backend_flags(self):
        original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            for device in ("cpu", "cuda"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                with mock.patch("ml.train.resolve_device", return_value=torch.device(device)), \
                        mock.patch("ml.train.seed_everything"), mock.patch("ml.train.torch.set_num_threads"), \
                        mock.patch("ml.train.SyntheticManifest", side_effect=ValueError("stop before dataset/model")):
                    with self.assertRaisesRegex(ValueError, "stop before dataset"):
                        run(parse_args(["--device", device]))
                expected = device == "cpu"
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, expected)
                self.assertEqual(torch.backends.cudnn.allow_tf32, expected)
                self.assertEqual(versions()["cuda_matmul_allow_tf32"], expected)
                self.assertEqual(versions()["cudnn_allow_tf32"], expected)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


class EarlyStoppingCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def training(self, config):
        seed_everything(42)
        model = CRNN(3, hidden_size=8, num_layers=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        return (model, optimizer, build_scheduler(optimizer, config), build_early_stopping(config),
                torch.Generator().manual_seed(42))

    def checkpoint(self, config, training, epoch, history=None):
        model, optimizer, scheduler, stopping, generator = training
        return build_checkpoint(model, optimizer, epoch, Charset("ab"), config, {"fixture": "early-stopping"},
                                {"cer": .5, "exact_match": 0., "epoch": 11}, history or [], generator,
                                scheduler=scheduler, early_stopping=stopping)

    def restore(self, checkpoint, config):
        training = self.training(config)
        model, optimizer, scheduler, stopping, generator = training
        epoch = restore_checkpoint(checkpoint, model, optimizer, Charset("ab"),
                                   {"fixture": "early-stopping"}, config, generator,
                                   scheduler=scheduler, early_stopping=stopping)
        return epoch, training

    def advance(self, training, start, end):
        _, optimizer, scheduler, stopping, _ = training
        history = []
        for epoch in range(start, end + 1):
            rates = update_learning_rate(optimizer, scheduler, .5, epoch, 10)
            stopped = stopping.step(.5, epoch)
            history.append({"epoch": epoch, **rates, "early_stopping": stopping.state_dict()})
            if stopped:
                break
        return history

    def test_interrupted_resume_preserves_counters_scheduler_and_final_stop_epoch(self):
        config = configuration()
        continuous = self.training(config)
        expected = self.advance(continuous, 1, 100)
        interrupted = self.training(config)
        first = self.advance(interrupted, 1, 22)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, self.checkpoint(config, interrupted, 22, first))
            checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint["early_stopping_state_dict"]["bad_epochs"], 11)
            epoch, resumed = self.restore(checkpoint, config)
        self.assertEqual(epoch, 23)
        self.assertEqual(resumed[3].bad_epochs, 11)
        remaining = self.advance(resumed, epoch, 100)
        self.assertEqual(expected, first + remaining)
        self.assertEqual(resumed[3].state_dict(), continuous[3].state_dict())
        self.assertEqual(resumed[2].state_dict(), continuous[2].state_dict())
        self.assertEqual(remaining[-1]["epoch"], 36)

    def test_legacy_disabled_resume_remains_supported_but_cannot_enable_stopping(self):
        disabled = configuration(scheduler="none", early_stopping_patience=0)
        training = self.training(disabled)
        checkpoint = self.checkpoint(disabled, training, 1)
        del checkpoint["early_stopping_config"]
        del checkpoint["early_stopping_state_dict"]
        for key in list(checkpoint["config"]):
            if key.startswith("early_stopping_"):
                del checkpoint["config"][key]
        self.assertEqual(self.restore(checkpoint, disabled)[0], 2)
        with self.assertRaisesRegex(ValueError, "legacy checkpoint cannot enable"):
            self.restore(checkpoint, configuration(scheduler="none"))

    def test_resume_rejects_configuration_changes_and_disabling_active_state(self):
        config = configuration()
        training = self.training(config)
        self.advance(training, 1, 12)
        checkpoint = self.checkpoint(config, training, 12)
        for changes in ({"early_stopping_patience": 24}, {"early_stopping_min_delta": .003},
                        {"early_stopping_start_epoch": 12}, {"early_stopping_patience": 0}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "early stopping configuration mismatch"):
                self.restore(checkpoint, {**config, **changes})

    def test_missing_state_or_corrupt_counters_fail_instead_of_resetting(self):
        config = configuration()
        training = self.training(config)
        self.advance(training, 1, 12)
        checkpoint = self.checkpoint(config, training, 12)
        missing = copy.deepcopy(checkpoint)
        del missing["early_stopping_state_dict"]
        with self.assertRaisesRegex(ValueError, "lacks saved early stopping"):
            self.restore(missing, config)
        for key, value in (("bad_epochs", -1), ("observations", 0),
                           ("best_reference_cer", float("nan")), ("stopped", True)):
            altered = copy.deepcopy(checkpoint)
            altered["early_stopping_state_dict"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "early stopping"):
                self.restore(altered, config)

    def test_checkpoint_requires_current_validation_decision(self):
        config = configuration(scheduler="none", early_stopping_start_epoch=1)
        with self.assertRaisesRegex(ValueError, "completed epoch"):
            self.checkpoint(config, self.training(config), 1)

    def test_stopped_checkpoint_preserves_state_and_cannot_restart(self):
        config = configuration(early_stopping_patience=1)
        training = self.training(config)
        self.advance(training, 1, 12)
        checkpoint = self.checkpoint(config, training, 12)
        self.assertTrue(checkpoint["early_stopping_state_dict"]["stopped"])
        clone = EarlyStopping(config)
        clone.load_state_dict(checkpoint["early_stopping_state_dict"], completed_epoch=12)
        self.assertTrue(clone.stopped)
        with self.assertRaisesRegex(ValueError, "already stopped"):
            self.restore(checkpoint, config)


class EarlyStoppingIntegrationTests(unittest.TestCase):
    def dataset(self, root):
        rows = []
        for split, labels in (("train", ("aabb", "abba")), ("validation", ("baba",))):
            (root / split).mkdir(parents=True)
            for index, label in enumerate(labels):
                name = f"{split}_{index}.png"
                image = Image.new("RGB", (215, 80), "white")
                ImageDraw.Draw(image).text((20 + index, 25), label, fill="black")
                image.save(root / split / name)
                rows.append({"filename": name, "label": label, "split": split, "width": 215, "height": 80})
        with (root / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def validation(self, cer):
        return {"cer": cer, "exact_match": 0., "loss": 1., "num_images": 1, "by_length": {}}

    def test_dry_run_round_trips_checkpoint_with_active_policies_and_no_artifacts(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            self.dataset(root)
            run_dir = Path(directory) / "unused-run"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(["--data", str(root), "--device", "cpu", "--batch-size", "2", "--threads", "2",
                               "--run-dir", str(run_dir), "--scheduler", "plateau", "--plateau-start-epoch", "0",
                               "--early-stopping-patience", "2", "--early-stopping-start-epoch", "1", "--dry-run"])
            self.assertEqual(result, 0)
            report = json.loads(output.getvalue().split("\nDRY RUN PASSED")[0])
            self.assertTrue(report["checkpoint_restored"])
            self.assertTrue(report["checkpoint_optimizer_restored"])
            self.assertTrue(report["checkpoint_rng_restored"])
            self.assertTrue(report["checkpoint_policies_restored"])
            self.assertEqual(report["checkpoint_map_location"], "cpu")
            self.assertEqual(report["scheduler_observations"], 1)
            self.assertFalse(run_dir.exists())
            self.assertEqual(list(Path(directory).rglob("*.pt")), [])

    def test_tiny_cpu_resume_stops_and_preserves_original_best_checkpoint(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            self.dataset(root)
            run_dir = Path(directory) / "run"
            arguments = ["--data", str(root), "--device", "cpu", "--batch-size", "2", "--threads", "2",
                         "--run-dir", str(run_dir), "--scheduler", "plateau", "--plateau-start-epoch", "0",
                         "--plateau-patience", "0", "--plateau-cooldown", "0",
                         "--early-stopping-patience", "2", "--early-stopping-start-epoch", "1"]
            small_model = lambda classes: CRNN(classes, hidden_size=8, num_layers=1)
            with mock.patch("ml.train.CRNN", side_effect=small_model), \
                    mock.patch("ml.train.validate", return_value=self.validation(.5)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments + ["--epochs", "1"]), 0)
            output = io.StringIO()
            with mock.patch("ml.train.CRNN", side_effect=small_model), \
                    mock.patch("ml.train.validate", return_value=self.validation(.6)), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(main(arguments + ["--epochs", "10", "--resume", str(run_dir / "last.pt")]), 0)
            best, last = load_checkpoint(run_dir / "best.pt"), load_checkpoint(run_dir / "last.pt")
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(last["epoch"], 3)
            self.assertEqual(last["best_metrics"], best["best_metrics"])
            self.assertTrue(last["early_stopping_state_dict"]["stopped"])
            self.assertEqual(last["early_stopping_state_dict"]["bad_epochs"], 2)
            self.assertEqual([row["epoch"] for row in last["history"]], [1, 2, 3])
            self.assertIn("EARLY STOPPING", output.getvalue())
            self.assertIn("ReduceLROnPlateau", output.getvalue())

    def test_small_new_best_is_saved_even_when_it_does_not_reset_patience(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            self.dataset(root)
            run_dir = Path(directory) / "run"
            arguments = ["--data", str(root), "--device", "cpu", "--batch-size", "2", "--threads", "2",
                         "--run-dir", str(run_dir), "--epochs", "10", "--early-stopping-patience", "2",
                         "--early-stopping-start-epoch", "1"]
            metrics = [self.validation(cer) for cer in (.5, .499, .4985)]
            with mock.patch("ml.train.CRNN", side_effect=lambda classes: CRNN(classes, hidden_size=8, num_layers=1)), \
                    mock.patch("ml.train.validate", side_effect=metrics), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            best, last = load_checkpoint(run_dir / "best.pt"), load_checkpoint(run_dir / "last.pt")
            self.assertEqual(best["epoch"], 3)
            self.assertEqual(best["best_metrics"]["cer"], .4985)
            self.assertEqual(last["early_stopping_state_dict"]["best_reference_cer"], .5)
            self.assertTrue(last["early_stopping_state_dict"]["stopped"])

    def test_terminal_resume_repairs_interrupted_best_and_history_without_training(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            self.dataset(root)
            run_dir = Path(directory) / "run"
            arguments = ["--data", str(root), "--device", "cpu", "--batch-size", "2", "--threads", "2",
                         "--run-dir", str(run_dir), "--epochs", "10", "--early-stopping-patience", "2",
                         "--early-stopping-start-epoch", "1"]
            small_model = lambda classes: CRNN(classes, hidden_size=8, num_layers=1)
            def interrupted_save(path, payload):
                if Path(path).name == "best.pt" and payload["epoch"] == 3:
                    raise OSError("simulated interruption after last.pt before best.pt")
                save_checkpoint(path, payload)
            metrics = [self.validation(cer) for cer in (.5, .499, .4985)]
            with mock.patch("ml.train.CRNN", side_effect=small_model), \
                    mock.patch("ml.train.validate", side_effect=metrics), \
                    mock.patch("ml.train.save_checkpoint", side_effect=interrupted_save), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(arguments), 1)
            last = load_checkpoint(run_dir / "last.pt")
            self.assertEqual(last["epoch"], 3)
            self.assertTrue(last["early_stopping_state_dict"]["stopped"])
            self.assertEqual(load_checkpoint(run_dir / "best.pt")["epoch"], 2)
            last_bytes = (run_dir / "last.pt").read_bytes()
            config_bytes = (run_dir / "run_config.json").read_bytes()
            with (run_dir / "history.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual([int(row["epoch"]) for row in csv.DictReader(stream)], [1, 2])
            output = io.StringIO()
            with mock.patch("ml.train.CRNN", side_effect=small_model), \
                    mock.patch("ml.train.train_step") as train, \
                    mock.patch.object(torch.optim.AdamW, "step") as optimizer_step, \
                    mock.patch("ml.train.validate") as validation, contextlib.redirect_stdout(output):
                self.assertEqual(main(arguments + ["--resume", str(run_dir / "last.pt")]), 0)
            train.assert_not_called()
            optimizer_step.assert_not_called()
            validation.assert_not_called()
            self.assertIn("EARLY STOPPING ALREADY COMPLETE", output.getvalue())
            repaired = load_checkpoint(run_dir / "best.pt")
            self.assertEqual(repaired["epoch"], 3)
            self.assertEqual(repaired["best_metrics"], last["best_metrics"])
            for name, value in last["model_state_dict"].items():
                self.assertTrue(torch.equal(value, repaired["model_state_dict"][name]), name)
            with (run_dir / "history.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual([int(row["epoch"]) for row in csv.DictReader(stream)], [1, 2, 3])
            self.assertEqual((run_dir / "last.pt").read_bytes(), last_bytes)
            self.assertEqual((run_dir / "run_config.json").read_bytes(), config_bytes)


if __name__ == "__main__":
    unittest.main()
