"""CPU-only checks; torch-dependent tests skip if PyTorch is not installed."""

from argparse import Namespace
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

from benchmarks.backend_probe import (ATOL, build_case, compare_snapshots, ctc_min_timesteps,
                                      ensure_finite, native_guard, parse_args, probe, summarize)


class ProbeHelperTests(unittest.TestCase):
    def test_ctc_alignment_accounts_for_repeats(self):
        self.assertEqual(ctc_min_timesteps([1, 1, 1, 2, 2]), 8)
        self.assertEqual(ctc_min_timesteps([1, 2, 3]), 3)
        self.assertEqual(ctc_min_timesteps([]), 0)

    def test_timing_units_and_throughput(self):
        result = summarize([0.01, 0.03, 0.02], 8)
        self.assertAlmostEqual(result["mean_ms"], 20)
        self.assertAlmostEqual(result["median_ms"], 20)
        self.assertAlmostEqual(result["stdev_ms"], 10)
        self.assertAlmostEqual(result["images_per_second"], 400)
        for samples in ([], [0], [-1]):
            with self.assertRaises(ValueError):
                summarize(samples, 8)

    def test_cli_rejects_invalid_sizes(self):
        for option in ("--batch-size", "--iterations", "--repeats", "--threads"):
            with self.subTest(option=option), patch("sys.stderr"), self.assertRaises(SystemExit):
                parse_args([option, "0"])
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--warmup", "-1"])
        self.assertEqual(parse_args(["--warmup", "0"]).warmup, 0)
        self.assertEqual(parse_args(["--output", "somewhere.json"]).output, Path("somewhere.json"))

    def test_fallback_is_an_error_and_warning_policy_is_restored(self):
        filters = warnings.filters[:]
        with self.assertRaises(UserWarning), native_guard():
            warnings.warn("The operator is not currently supported and will fall back to run on the CPU.")
        self.assertEqual(warnings.filters, filters)
        with native_guard() as caught:
            warnings.warn("ordinary diagnostic warning")
        self.assertEqual(len(caught), 1)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional; use .venv-directml for CPU probe tests")
class ProbeCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)

    def test_reject_nan_and_zero_gradients(self):
        torch = self.torch
        with self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
            ensure_finite(torch, torch.tensor([float("nan")]), "gradient")
        with self.assertRaisesRegex(RuntimeError, "all zero"):
            ensure_finite(torch, torch.zeros(3), "gradient", nonzero=True)

    def test_comparisons_report_missing_stages_and_mismatches(self):
        torch = self.torch
        a = {"output": torch.ones(2), "grad:input": torch.ones(2)}
        b = {"output": torch.ones(2) + ATOL / 2}
        result = compare_snapshots(torch, a, b)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["tensors"]["output"]["status"], "passed")
        self.assertEqual(result["tensors"]["grad:input"]["status"], "not_comparable")
        self.assertEqual(compare_snapshots(torch, {}, {})["status"], "not_comparable")
        self.assertEqual(compare_snapshots(torch, a, {})["status"], "not_comparable")
        self.assertEqual(compare_snapshots(torch, a, {**a, "output": torch.zeros(2)})["status"], "mismatch")

    def test_same_seed_recreates_inputs_and_parameters(self):
        torch = self.torch
        first = build_case(torch, "integrated", 1, 42, torch.device("cpu"))
        second = build_case(torch, "integrated", 1, 42, torch.device("cpu"))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first[0].parameters(), second[0].parameters())))
        self.assertEqual(first[1].dtype, torch.float32)
        output = first[3]()
        self.assertEqual(list(output.shape), [53, 1, 17])
        self.assertTrue(torch.isfinite(first[4](output)).item())

    def test_integrated_cpu_training_changes_parameters(self):
        result, snapshots = probe(self.torch, "integrated", 1, Namespace(seed=42, iterations=3),
                                  self.torch.device("cpu"))
        self.assertEqual(result["status"], "passed", result.get("error"))
        self.assertEqual(result["iterations_completed"], 3)
        self.assertTrue(result["changed_parameters"])
        self.assertIn("grad:input", snapshots)

    def test_noop_optimizer_cannot_pass(self):
        torch = self.torch
        original = build_case

        def with_noop_optimizer(*args):
            values = original(*args)
            values[2].step = lambda: None
            return values

        with patch("benchmarks.backend_probe.build_case", side_effect=with_noop_optimizer):
            result, _ = probe(torch, "linear", 1, Namespace(seed=42, iterations=3), torch.device("cpu"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_stage"], "parameters")
        self.assertIn("no parameter changed", result["error"])


if __name__ == "__main__":
    unittest.main()
