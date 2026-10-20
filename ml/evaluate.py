"""Offline evaluation on synthetic files using the checkpoint's exact charset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from .charset import Charset
from .dataset import SyntheticDataset, SyntheticManifest, collate_ctc
from .model import CRNN
from .train import atomic_write, load_checkpoint, positive, nonnegative, resolve_device, validate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a CRNN on local synthetic data only.")
    parser.add_argument("--data", type=Path, default=Path("dataset/synthetic/v2_main"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=positive, default=16)
    parser.add_argument("--num-workers", type=nonnegative, default=0)
    parser.add_argument("--threads", type=positive, default=6)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args(argv)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_csv(path, fields, rows, *, lowercase_booleans=False):
    """Atomic CSV export preserving exact text and explicit boolean values."""
    def writer(stream):
        output = csv.DictWriter(stream, fieldnames=fields)
        output.writeheader()
        for row in rows:
            output.writerow({key: (str(row[key]).lower() if lowercase_booleans and isinstance(row[key], bool)
                                   else row[key]) for key in fields})
    atomic_write(path, writer)


def frozen_inference(checkpoint, dataset, device="cpu", batch_size=16, num_workers=0):
    """Shared forward/CTC pipeline; caller audits and preserves its local data."""
    charset = Charset.from_dict(checkpoint["charset"])
    if checkpoint["model_config"].get("num_classes") != charset.num_classes:
        raise ValueError("Checkpoint model classes are incompatible with its charset.")
    if dataset.charset.to_dict() != charset.to_dict():
        raise ValueError("Dataset charset differs from checkpoint.")
    model = CRNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=collate_ctc, shuffle=False)
    with torch.inference_mode():
        metrics = validate(model, loader, device, charset, return_predictions=True)
    if any(not torch.equal(before[key], value) for key, value in model.state_dict().items()):
        raise RuntimeError("Frozen model state changed during evaluation.")
    return {"metrics": metrics, "model_config": model.config,
            "parameter_count": sum(p.numel() for p in model.parameters())}


def evaluate_checkpoint(data, split, checkpoint_path, device="cpu", batch_size=16,
                        num_workers=0, threads=6, strict_dataset_identity=False):
    """Shared frozen inference, reading PNGs from the requested synthetic split only.

    The complete CSV is parsed for consistency, but other splits' PNGs are not
    opened. Optimizer state is loaded as checkpoint data and never instantiated
    or restored into an optimizer. No training function is called.
    """
    device = resolve_device(str(device))
    torch.set_num_threads(threads)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    charset = Charset.from_dict(checkpoint["charset"])
    if checkpoint["model_config"].get("num_classes") != charset.num_classes:
        raise ValueError("Checkpoint model classes are incompatible with its charset.")
    manifest = SyntheticManifest(data, image_splits=(split,))
    recorded_manifest_hash = checkpoint["dataset_identity"].get("manifest_sha256")
    if strict_dataset_identity and recorded_manifest_hash != manifest.identity["manifest_sha256"]:
        raise ValueError("Dataset CSV differs from the checkpoint; frozen-experiment analysis refused.")
    dataset = SyntheticDataset(data, split, charset=charset, manifest=manifest)
    inference = frozen_inference(checkpoint, dataset, device, batch_size, num_workers)
    if checkpoint_hash != file_sha256(checkpoint_path):
        raise RuntimeError("Checkpoint bytes changed during evaluation.")
    if file_sha256(dataset.root / "labels.csv") != manifest.identity["manifest_sha256"]:
        raise RuntimeError("Dataset labels.csv changed during evaluation.")
    for record in dataset.records:
        if file_sha256(record.path) != manifest.image_hashes[(split, record.filename)]:
            raise RuntimeError(f"PNG changed during evaluation: {record.filename}")
    return {**inference, "checkpoint": checkpoint, "dataset": dataset,
            "dataset_identity": manifest.identity, "charset": charset,
            "checkpoint_sha256": checkpoint_hash, "device": str(device),
            "preservation_verified": True}


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.output_csv and args.output_csv.exists():
            raise ValueError(f"Prediction output already exists: {args.output_csv}")
        evaluation = evaluate_checkpoint(args.data, args.split, args.checkpoint, args.device,
                                         args.batch_size, args.num_workers, args.threads)
        result = evaluation["metrics"]
        predictions = result.pop("predictions", [])
        result.update(device=evaluation["device"], split=args.split,
                      checkpoint_epoch=evaluation["checkpoint"]["epoch"],
                      dataset_fingerprint=evaluation["dataset_identity"]["fingerprint"])
        if args.output_csv:
            export_csv(args.output_csv, ("filename", "expected", "predicted", "correct"), predictions)
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    except (ValueError, RuntimeError, OSError, FloatingPointError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
