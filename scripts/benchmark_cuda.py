"""Bounded float32 CUDA benchmark; never saves weights or runs a full epoch.

Run from the repository root with ``python -m scripts.benchmark_cuda``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader

from ml.dataset import SyntheticDataset, SyntheticManifest, collate_ctc
from ml.model import CRNN
from ml.train import positive, seed_everything, seed_worker, train_step, validate, versions


def select_device(name="cuda:0"):
    """Require an explicitly indexed CUDA device and the intended A2000 GPU."""
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; the benchmark cannot fall back to CPU.")
    device = torch.device(name)
    if device.type != "cuda" or device.index is None:
        raise ValueError("Select an explicit CUDA device, for example --device cuda:0.")
    if device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device.index} is unavailable.")
    properties = torch.cuda.get_device_properties(device)
    if "A2000" not in properties.name or properties.total_memory < 10 * 2**30:
        raise ValueError(f"Expected NVIDIA A2000 12 GB; found {properties.name!r}, "
                         f"{properties.total_memory / 2**30:.2f} GiB.")
    torch.cuda.set_device(device)
    return device, properties


def validate_sample_sizes(args, train_batches, validation_batches):
    """Keep diagnostics bounded, including when invoked on a small dataset."""
    if args.warmup + args.batches >= train_batches:
        raise ValueError("Benchmark must consume fewer batches than a training epoch.")
    if 1 + args.validation_batches >= validation_batches:
        raise ValueError("Benchmark must consume fewer batches than a validation epoch.")
    if args.warmup + args.batches > 32 or 1 + args.validation_batches > 8:
        raise ValueError("Short benchmark is limited to 32 training and 8 validation batches, including warmup.")


def nvidia_inventory():
    """Physical GPU indices complement CUDA's possibly remapped device indices."""
    fields = ("index", "name", "uuid", "memory.total", "memory.free", "driver_version", "pci.bus_id")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error": str(error)}
    return {"available": True, "memory_unit": "MiB",
            "physical_devices": [dict(zip(fields, (value.strip() for value in row)))
                                 for row in csv.reader(io.StringIO(result.stdout)) if row]}


def benchmark(args):
    started = time.perf_counter()
    device, properties = select_device(args.device)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    # Match experiment D's strict float32 training policy on Ampere GPUs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free_before, total_memory = torch.cuda.mem_get_info(device)
    manifest = SyntheticManifest(args.data)
    train = SyntheticDataset(args.data, "train", manifest=manifest)
    validation = SyntheticDataset(args.data, "validation", charset=train.charset, manifest=manifest)
    generator = torch.Generator().manual_seed(args.seed)
    # Match ml.train's production defaults: no pinned memory, no loader workers.
    options = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False,
                   collate_fn=collate_ctc, worker_init_fn=seed_worker)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(validation, shuffle=False, **options)
    validate_sample_sizes(args, len(train_loader), len(val_loader))
    model = CRNN(train.charset.num_classes).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    torch.cuda.synchronize(device)
    startup_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats(device)
    training_times, validation_times = [], []
    training_losses = []
    iterator = iter(train_loader)
    with torch.autocast(device_type="cuda", enabled=False):
        for index in range(args.warmup + args.batches):
            torch.cuda.synchronize(device)
            batch_started = time.perf_counter()
            batch = next(iterator)
            loss = train_step(model, batch, optimizer, device)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - batch_started
            if index >= args.warmup:
                training_times.append(seconds)
                training_losses.append(loss)
        # Independently warm up validation; timing includes PNG loading,
        # normalization, transfer, CTC, greedy decoding and metric calculation.
        iterator = iter(val_loader)
        for index in range(1 + args.validation_batches):
            torch.cuda.synchronize(device)
            batch_started = time.perf_counter()
            batch = next(iterator)
            validate(model, [batch], device, train.charset)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - batch_started
            if index:
                validation_times.append(seconds)
    train_mean = statistics.mean(training_times)
    val_mean = statistics.mean(validation_times)
    epoch_seconds = train_mean * len(train_loader) + val_mean * len(val_loader)
    free_after, _ = torch.cuda.mem_get_info(device)
    return {
        "device": str(device), "gpu": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": nvidia_inventory(), "precision": "float32", "amp": False,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "seed": args.seed, "batch_size": args.batch_size,
        "num_workers": 0, "pin_memory": False,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dataset": manifest.identity, "versions": versions(),
        "warmup_training_batches": args.warmup,
        "measured_training_batches": args.batches,
        "warmup_validation_batches": 1,
        "measured_validation_batches": args.validation_batches,
        "training_seconds_per_batch": train_mean,
        "training_batch_stdev_seconds": statistics.stdev(training_times) if len(training_times) > 1 else 0,
        "validation_seconds_per_batch": val_mean,
        "training_batch_seconds": training_times,
        "validation_batch_seconds": validation_times,
        "measured_training_losses": training_losses,
        "training_batches_per_epoch": len(train_loader),
        "validation_batches_per_epoch": len(val_loader),
        "estimated_training_epoch_seconds": train_mean * len(train_loader),
        "estimated_validation_epoch_seconds": val_mean * len(val_loader),
        "estimated_epoch_seconds": epoch_seconds,
        "estimated_30_epochs_seconds": epoch_seconds * 30,
        "estimated_100_epochs_seconds": epoch_seconds * 100,
        "startup_seconds": startup_seconds,
        "memory": {
            "total_bytes": total_memory,
            "free_before_model_bytes": free_before,
            "free_after_benchmark_bytes": free_after,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "training_updates_in_memory": args.warmup + args.batches,
        "checkpoints_written": False,
        "limitations": (
            "Short sample; estimate includes loading/preprocessing, host-to-device transfer, "
            "original train_step finite checks, clipping, AdamW and validation/decoding. "
            "Excludes checkpoint/history I/O, scheduler decisions, startup and later-epoch "
            "variation. Validation estimate uses full batches although its last batch may "
            "be smaller. Dataset manifest verification warms filesystem caches. No full "
            "epoch is executed; the scratch model and optimizer are discarded."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/synthetic/v2_20k_train"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=positive, default=16)
    parser.add_argument("--threads", type=positive, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=positive, default=3)
    parser.add_argument("--batches", type=positive, default=8)
    parser.add_argument("--validation-batches", type=positive, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("--output already exists; choose another path")
    try:
        report = benchmark(args)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(json.dumps(report, indent=2))
    except (ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
