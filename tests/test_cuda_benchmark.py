"""CPU-safe safety checks for the bounded CUDA diagnostic command."""

from argparse import Namespace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.benchmark_cuda import main, select_device, validate_sample_sizes


class CUDABenchmarkSafetyTests(unittest.TestCase):
    def test_refuses_to_consume_a_full_training_or_validation_epoch(self):
        args = Namespace(warmup=3, batches=8, validation_batches=3)
        validate_sample_sizes(args, 1250, 19)
        with self.assertRaisesRegex(ValueError, "training epoch"):
            validate_sample_sizes(args, 11, 19)
        with self.assertRaisesRegex(ValueError, "validation epoch"):
            validate_sample_sizes(args, 1250, 4)
        args.batches = 30
        with self.assertRaisesRegex(ValueError, "limited"):
            validate_sample_sizes(args, 1250, 19)

    def test_refuses_cpu_fallback_and_unindexed_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "cannot fall back"):
                select_device()
        with patch("torch.cuda.is_available", return_value=True):
            for device in ("cpu", "cuda"):
                with self.subTest(device=device), self.assertRaisesRegex(ValueError, "explicit CUDA"):
                    select_device(device)

    def test_existing_report_is_preserved_without_running_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text("previous report", encoding="utf-8")
            with patch("scripts.benchmark_cuda.benchmark") as benchmark, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    main(["--output", str(path)])
            benchmark.assert_not_called()
            self.assertEqual(path.read_text(encoding="utf-8"), "previous report")


if __name__ == "__main__":
    unittest.main()
