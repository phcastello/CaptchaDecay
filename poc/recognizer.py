"""Unlabelled PNG inference using the published, frozen experiment D model."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import time

import torch

from ml.charset import Charset
from ml.dataset import read_image_tensor
from ml.decode import greedy_decode
from ml.evaluate import file_sha256
from ml.model import CRNN
from ml.train import load_checkpoint, resolve_device

DEFAULT_CHECKPOINT = Path("outputs/runs/crnn-v2-20k-cuda-plateau/best.pt")
D_SHA256 = "da2c608b9776f7ea54e15996e8bcb1c8bcfa54f2c4fed0f8c536ff7f093bfddf"
D_EPOCH = 46


class Recognizer:
    """Load once before navigation; recognize(bytes) has no label/target input."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cpu", threads=6):
        started = time.perf_counter()
        self.path = Path(checkpoint)
        if not self.path.is_file():
            raise FileNotFoundError(f"Checkpoint D ausente: {self.path}")
        digest = file_sha256(self.path)
        if digest != D_SHA256:
            raise ValueError("SHA-256 incompatível: esta PoC exige o checkpoint D publicado.")
        state = load_checkpoint(self.path)
        if state["epoch"] != D_EPOCH:
            raise ValueError("Época incompatível: esperado D, época 46.")
        self.device = resolve_device(device)
        if type(threads) is not int or threads < 1:
            raise ValueError("threads deve ser inteiro positivo.")
        torch.set_num_threads(threads)
        self.charset = Charset.from_dict(state["charset"])
        if state["model_config"].get("num_classes") != self.charset.num_classes:
            raise ValueError("Classes do checkpoint incompatíveis com o charset.")
        self.model = CRNN(**state["model_config"]).to(self.device)
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self._before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        if file_sha256(self.path) != digest:
            raise RuntimeError("Checkpoint alterado durante o carregamento.")
        self._synchronize()
        self.metadata = {
            "checkpoint": str(self.path), "checkpoint_sha256": digest,
            "checkpoint_epoch": state["epoch"], "device": str(self.device),
            "model_config": self.model.config, "charset": self.charset.to_dict(),
            "model_load_seconds": time.perf_counter() - started,
        }

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def recognize(self, png: bytes) -> dict:
        """Original 215×80 PNG -> exact greedy text, hashes and timing."""
        self._synchronize()
        started = time.perf_counter()
        image = read_image_tensor(io.BytesIO(png)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(image)
            text = greedy_decode(logits, self.charset)[0]
        self._synchronize()
        seconds = time.perf_counter() - started
        if not text:
            raise ValueError("Previsão vazia; nenhuma resposta será submetida.")
        return {**self.metadata, "text": text, "image_sha256": hashlib.sha256(png).hexdigest(),
                "inference_seconds": seconds}

    def verify_preservation(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Modelo deixou o estado congelado.")
        if any(not torch.equal(self._before[k], v) for k, v in self.model.state_dict().items()):
            raise RuntimeError("Pesos/buffers de BatchNorm alterados durante inferência.")
        if file_sha256(self.path) != self.metadata["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint alterado durante inferência.")
