"""Validate server CUDA primitives; use ml.train --dry-run for the full CRNN."""

import json
import sys
import tempfile
from pathlib import Path

import torch


def select_a2000():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível; execução em CPU não é permitida neste teste.")
    matches = [index for index in range(torch.cuda.device_count())
               if "A2000" in torch.cuda.get_device_name(index)]
    if len(matches) != 1:
        raise RuntimeError(f"Esperada exatamente uma A2000 visível; encontradas {matches}.")
    device = torch.device("cuda", matches[0])
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < 11 * 1024 ** 3:
        raise RuntimeError("A A2000 detectada não possui os 12 GB esperados.")
    torch.cuda.set_device(device)
    return device


def check_runtime():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Execute este script com .venv-ml/bin/python.")
    device = select_a2000()
    torch.manual_seed(42)
    # Matrix multiplication and simple backward explicitly on the selected GPU.
    matrix = torch.randn(64, 64, device=device, requires_grad=True)
    matrix.square().matmul(matrix).mean().backward()
    assert matrix.grad.is_cuda and torch.isfinite(matrix.grad).all()

    # Native CUDA CTC with CPU length metadata and CUDA int64 targets.
    # This intentionally exercises primitives, without inventing a CRNN model.
    layer = torch.nn.Linear(8, 29).to(device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)
    inputs = torch.randn(16, 2, 8, device=device)
    targets = torch.tensor([1, 2, 3, 4, 2, 3, 4, 5], device=device, dtype=torch.long)
    input_lengths = torch.tensor([16, 16], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)
    before = layer.weight.detach().clone()
    log_probs = layer(inputs).log_softmax(2)
    loss = torch.nn.CTCLoss(blank=0)(log_probs, targets, input_lengths, target_lengths)
    assert loss.is_cuda and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and p.grad.is_cuda and torch.isfinite(p.grad).all()
               for p in layer.parameters())
    torch.nn.utils.clip_grad_norm_(layer.parameters(), 5, error_if_nonfinite=True)
    optimizer.step()
    assert not torch.equal(before, layer.weight)

    with tempfile.TemporaryDirectory(prefix="captchadecay-cuda-") as directory:
        checkpoint_path = Path(directory) / "primitives.pt"
        torch.save({"model": layer.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)
        restored = torch.nn.Linear(8, 29).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        restored.load_state_dict(checkpoint["model"])
        restored_optimizer = torch.optim.AdamW(restored.parameters())
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        assert torch.equal(restored.weight, layer.weight)
        cpu_copy = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert all(not value.is_cuda for value in cpu_copy["model"].values())
        restored_optimizer.zero_grad(set_to_none=True)
        restored(inputs).square().mean().backward()
        restored_optimizer.step()
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "scope": "CUDA primitives only; CRNN/scheduler/early-stopping/resume not validated",
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "isolated": sys.prefix != sys.base_prefix,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "vram_bytes": torch.cuda.get_device_properties(device).total_memory,
        "cuda_allocatable_bytes": total,
        "free_vram_bytes": free, "ctc_loss": loss.item(),
        "matmul_backward": "passed", "ctc_backward": "passed",
        "adamw_gradient_clipping_weight_update": "passed",
        "primitive_checkpoint_restore_cuda_and_cpu": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(check_runtime(), indent=2))
