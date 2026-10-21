"""Optional GPU checks of the original CRNN, using temporary synthetic PNGs only."""

import csv
import math
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
from ml.ctc import ctc_loss
from ml.dataset import SyntheticDataset, SyntheticManifest, collate_ctc
from ml.decode import greedy_decode
from ml.model import CRNN
from ml.train import (build_checkpoint, build_early_stopping, build_scheduler,
                      load_checkpoint, resolve_device, restore_checkpoint,
                      save_checkpoint, seed_everything, train_step,
                      update_learning_rate, validate)


CHARACTERS = "23456789abcdefghiklmnprstuvwyz"


class CudaDevicePolicyTests(unittest.TestCase):
    def test_explicit_cuda_request_rejects_unavailable_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "explicitly requested.*unavailable"):
                resolve_device("cuda")

    def test_cpu_selection_remains_available_without_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("cpu"), torch.device("cpu"))


@unittest.skipUnless(torch.cuda.is_available(), "GPU: requer PyTorch com CUDA disponível.")
class CudaCrnnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        cls.previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cls.previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        torch.backends.cuda.matmul.allow_tf32 = cls.previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = cls.previous_cudnn_tf32

    def setUp(self):
        seed_everything(42)
        self.device = resolve_device("cuda")
        self.charset = Charset(CHARACTERS)
        self.temporary = tempfile.TemporaryDirectory(prefix="captchadecay-crnn-cuda-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rows = []
        number = 0
        for split, labels in (("train", ("aabb", "79mh9u")),
                              ("validation", ("szdglp", "84ku"))):
            (self.root / split).mkdir()
            for label in labels:
                number += 1
                filename = f"{number:08d}.png"
                image = Image.new("RGB", (215, 80), "white")
                ImageDraw.Draw(image).text((10 + number, 20), label, fill="black")
                image.save(self.root / split / filename)
                rows.append({"filename": filename, "label": label, "split": split,
                             "width": 215, "height": 80, "length": len(label)})
        with (self.root / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        self.manifest = SyntheticManifest(self.root)
        self.loader_generator = torch.Generator().manual_seed(42)
        options = {"batch_size": 2, "collate_fn": collate_ctc, "num_workers": 0}
        self.train_loader = DataLoader(
            SyntheticDataset(self.root, "train", self.charset, manifest=self.manifest),
            shuffle=True, generator=self.loader_generator, **options)
        self.validation_loader = DataLoader(
            SyntheticDataset(self.root, "validation", self.charset, manifest=self.manifest),
            **options)
        self.batch = next(iter(self.train_loader))

    def model_and_optimizer(self, device=None):
        model = CRNN(self.charset.num_classes).to(device or self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        return model, optimizer

    def test_gpu_ctc_accepts_cpu_metadata_and_backpropagates(self):
        logits = torch.randn(53, 2, self.charset.num_classes,
                             device=self.device, requires_grad=True)
        cpu_loss = ctc_loss(logits, self.batch["targets"], self.batch["target_lengths"],
                            self.batch["filenames"])
        gpu_loss = ctc_loss(logits, self.batch["targets"].to(self.device),
                            self.batch["target_lengths"].to(self.device))
        self.assertTrue(cpu_loss.is_cuda)
        torch.testing.assert_close(cpu_loss, gpu_loss)
        cpu_loss.backward()
        self.assertTrue(logits.grad.is_cuda)
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue(torch.any(logits.grad != 0).item())

    def test_gpu_full_crnn_adamw_clipping_validation_and_greedy(self):
        model, optimizer = self.model_and_optimizer()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 1186815)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        observed = set()
        handles = []

        def observe(module, _inputs, outputs):
            output = outputs[0] if isinstance(outputs, tuple) else outputs
            self.assertTrue(output.is_cuda)
            self.assertEqual(output.dtype, torch.float32)
            observed.add(type(module))

        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d,
                                   torch.nn.LSTM, torch.nn.Linear)):
                handles.append(module.register_forward_hook(observe))
        original_ctc = torch.nn.CTCLoss.forward
        ctc_calls = []

        def capture_ctc(module, probabilities, targets, input_lengths, target_lengths):
            self.assertTrue(probabilities.is_cuda)
            self.assertTrue(targets.is_cuda)
            self.assertEqual(targets.dtype, torch.long)
            self.assertEqual(input_lengths.device.type, "cpu")
            self.assertEqual(target_lengths.device.type, "cpu")
            self.assertTrue(torch.isfinite(probabilities).all().item())
            ctc_calls.append(tuple(probabilities.shape))
            return original_ctc(module, probabilities, targets, input_lengths, target_lengths)

        try:
            with mock.patch.object(torch.nn.CTCLoss, "forward", capture_ctc):
                loss = train_step(model, self.batch, optimizer, self.device)
                result = validate(model, self.validation_loader, self.device,
                                  self.charset, return_predictions=True)
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(math.isfinite(loss) and loss > 0)
        self.assertEqual(observed, {torch.nn.Conv2d, torch.nn.BatchNorm2d,
                                    torch.nn.LSTM, torch.nn.Linear})
        self.assertEqual(ctc_calls, [(53, 2, 31), (53, 2, 31)])
        for name, parameter in model.named_parameters():
            self.assertTrue(parameter.is_cuda, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(parameter.grad.is_cuda, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        gradient_norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(parameter.grad) for parameter in model.parameters()]))
        self.assertLessEqual(gradient_norm.item(), 5.0001)
        for name in ("cnn.0.weight", "rnn.weight_ih_l0", "classifier.weight"):
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)
        self.assertEqual(result["num_images"], 2)
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertEqual(len(result["predictions"]), 2)
        with torch.no_grad():
            logits = model(self.batch["images"].to(self.device))
        self.assertEqual(tuple(logits.shape), (53, 2, 31))
        decoded = greedy_decode(logits, self.charset)
        self.assertEqual(len(decoded), 2)
        self.assertTrue(all(set(label) <= set(CHARACTERS) for label in decoded))

    def test_gpu_project_checkpoint_resume_optimizer_rng_scheduler_and_cpu_map(self):
        config = {"batch_size": 2, "learning_rate": 0.001, "seed": 42,
                  "num_workers": 0, "overfit_samples": None, "threads": 2,
                  "scheduler": "plateau", "plateau_start_epoch": 0,
                  "plateau_patience": 8, "plateau_factor": 0.5,
                  "plateau_min_delta": 0.002, "plateau_cooldown": 2,
                  "plateau_min_lr": 0.00001, "early_stopping_patience": 25,
                  "early_stopping_min_delta": 0.002, "early_stopping_start_epoch": 1}
        model, optimizer = self.model_and_optimizer()
        scheduler = build_scheduler(optimizer, config)
        stopping = build_early_stopping(config)
        train_step(model, self.batch, optimizer, self.device)
        result = validate(model, self.validation_loader, self.device, self.charset)
        rates = update_learning_rate(optimizer, scheduler, result["cer"], 1, 0)
        stopping.step(result["cer"], 1)
        history = [{"epoch": 1, "cer": result["cer"], **rates}]
        best = {"epoch": 1, "cer": result["cer"], "exact_match": result["exact_match"]}
        payload = build_checkpoint(model, optimizer, 1, self.charset, config,
                                   self.manifest.identity, best, history, self.loader_generator,
                                   scheduler=scheduler, early_stopping=stopping)
        path = self.root / "cuda-checkpoint.pt"
        save_checkpoint(path, payload)
        checkpoint = load_checkpoint(path)
        self.assertTrue(all(value.device.type == "cpu"
                            for value in checkpoint["model_state_dict"].values()))
        cuda_checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.assertTrue(all(value.is_cuda for value in cuda_checkpoint["model_state_dict"].values()))
        del cuda_checkpoint

        def rng_sample(generator):
            return (random.random(), float(np.random.random()), torch.rand(3),
                    torch.rand(3, device=self.device), torch.randperm(8, generator=generator))

        expected_random = rng_sample(self.loader_generator)
        resumed_model, resumed_optimizer = self.model_and_optimizer()
        resumed_scheduler = build_scheduler(resumed_optimizer, config)
        resumed_stopping = build_early_stopping(config)
        resumed_generator = torch.Generator().manual_seed(999)
        seed_everything(999)
        next_epoch = restore_checkpoint(
            checkpoint, resumed_model, resumed_optimizer, self.charset, self.manifest.identity,
            config, resumed_generator, scheduler=resumed_scheduler, early_stopping=resumed_stopping)
        self.assertEqual(next_epoch, 2)
        for expected, actual in zip(expected_random, rng_sample(resumed_generator)):
            if isinstance(expected, torch.Tensor):
                self.assertTrue(torch.equal(expected, actual))
            else:
                self.assertEqual(expected, actual)
        self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
        self.assertEqual(resumed_stopping.state_dict(), stopping.state_dict())
        for name, expected in model.state_dict().items():
            self.assertTrue(torch.equal(expected, resumed_model.state_dict()[name]), name)
        for state in resumed_optimizer.state.values():
            for name in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(state[name].is_cuda, name)
        before = resumed_model.classifier.weight.detach().clone()
        self.assertTrue(math.isfinite(train_step(resumed_model, self.batch, resumed_optimizer, self.device)))
        self.assertFalse(torch.equal(before, resumed_model.classifier.weight))

        # The actual CUDA checkpoint also loads through the project's CPU path.
        cpu_model, cpu_optimizer = self.model_and_optimizer(torch.device("cpu"))
        cpu_scheduler = build_scheduler(cpu_optimizer, config)
        cpu_stopping = build_early_stopping(config)
        self.assertEqual(restore_checkpoint(
            checkpoint, cpu_model, cpu_optimizer, self.charset, self.manifest.identity, config,
            torch.Generator(), scheduler=cpu_scheduler, early_stopping=cpu_stopping), 2)
        self.assertTrue(math.isfinite(train_step(cpu_model, self.batch, cpu_optimizer, torch.device("cpu"))))
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in cpu_model.parameters()))


if __name__ == "__main__":
    unittest.main()
