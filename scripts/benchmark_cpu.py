"""Short in-memory CPU benchmark; never writes model weights or runs an epoch."""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

from ml.dataset import SyntheticDataset, SyntheticManifest, collate_ctc
from ml.model import CRNN
from ml.train import positive, seed_everything, seed_worker, train_step, validate, versions


def process_memory():
    """Current and peak resident process memory, when the OS exposes it."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {"resident_mib": counters.WorkingSetSize / 2**20,
                    "peak_resident_mib": counters.PeakWorkingSetSize / 2**20}
    return {"resident_mib": None, "peak_resident_mib": None}


def benchmark(args):
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    startup = time.perf_counter()
    manifest = SyntheticManifest(args.data)
    train = SyntheticDataset(args.data, "train", manifest=manifest)
    validation = SyntheticDataset(args.data, "validation", charset=train.charset, manifest=manifest)
    generator = torch.Generator().manual_seed(args.seed)
    options = dict(batch_size=args.batch_size, num_workers=0, collate_fn=collate_ctc,
                   worker_init_fn=seed_worker)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(validation, shuffle=False, **options)
    if args.warmup + args.batches >= len(train_loader):
        raise ValueError("Benchmark must consume fewer batches than a training epoch.")
    if 1 + args.validation_batches >= len(val_loader):
        raise ValueError("Benchmark must consume fewer batches than a validation epoch.")
    model = CRNN(train.charset.num_classes).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    startup_seconds = time.perf_counter() - startup
    training_times = []
    validation_times = []
    iterator = iter(train_loader)
    for index in range(args.warmup + args.batches):
        started = time.perf_counter()
        batch = next(iterator)
        train_step(model, batch, optimizer, torch.device("cpu"))
        if index >= args.warmup:
            training_times.append(time.perf_counter() - started)
    # Warm up the inference path separately. Loading/preprocessing is timed too.
    iterator = iter(val_loader)
    for index in range(1 + args.validation_batches):
        started = time.perf_counter()
        batch = next(iterator)
        validate(model, [batch], torch.device("cpu"), train.charset)
        if index:
            validation_times.append(time.perf_counter() - started)
    train_mean = statistics.mean(training_times)
    val_mean = statistics.mean(validation_times)
    epoch_seconds = train_mean * len(train_loader) + val_mean * len(val_loader)
    return {"device": "cpu", "seed": args.seed, "batch_size": args.batch_size,
            "dataset": manifest.identity, "versions": versions(),
            "warmup_training_batches": args.warmup, "measured_training_batches": args.batches,
            "measured_validation_batches": args.validation_batches,
            "training_seconds_per_batch": train_mean, "validation_seconds_per_batch": val_mean,
            "training_batch_seconds": training_times, "validation_batch_seconds": validation_times,
            "training_batches_per_epoch": len(train_loader), "validation_batches_per_epoch": len(val_loader),
            "estimated_epoch_seconds": epoch_seconds, "estimated_30_epochs_seconds": epoch_seconds * 30,
            "startup_seconds": startup_seconds, "memory": process_memory(),
            "training_updates_in_memory": args.warmup + args.batches,
            "checkpoints_written": False,
            "limitations": "Short sample; excludes checkpoint I/O and later-epoch variation. Separate processes may differ."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/synthetic/v2_9k_train"))
    parser.add_argument("--batch-size", type=positive, default=16)
    parser.add_argument("--threads", type=positive, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=positive, default=2)
    parser.add_argument("--batches", type=positive, default=6)
    parser.add_argument("--validation-batches", type=positive, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("--output already exists; choose another path")
    report = benchmark(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
