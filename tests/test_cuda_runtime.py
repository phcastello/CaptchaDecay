"""GPU primitive tests are optional; no historical CRNN implementation is present."""

import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from scripts.check_cuda import check_runtime, select_a2000


@unittest.skipIf(torch is None, "PyTorch não instalado; testes opcionais de CUDA.")
class DeviceSelectionTests(unittest.TestCase):
    def test_rejects_unavailable_cuda_without_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA indisponível"):
                select_a2000()

    def test_rejects_wrong_gpu(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_name", return_value="Different GPU"):
            with self.assertRaisesRegex(RuntimeError, "exatamente uma A2000"):
                select_a2000()


@unittest.skipUnless(torch is not None and torch.cuda.is_available()
                     and any("A2000" in torch.cuda.get_device_name(i)
                             for i in range(torch.cuda.device_count())),
                     "GPU: requer NVIDIA A2000 e PyTorch CUDA.")
class CudaRuntimeTests(unittest.TestCase):
    def test_gpu_ctc_backward_update_and_primitive_checkpoint(self):
        report = check_runtime()
        self.assertTrue(report["cuda_available"])
        self.assertTrue(report["isolated"])
        self.assertIn("A2000", report["gpu"])
        self.assertEqual(report["ctc_backward"], "passed")


if __name__ == "__main__":
    unittest.main()
