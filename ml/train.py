"""CPU-first CRNN training. Nothing runs when this module is imported."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from pathlib import Path
import random
import sys
import tempfile
import time

import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .charset import Charset
from .ctc import ctc_loss
from .dataset import SyntheticDataset, SyntheticManifest, collate_ctc
from .decode import greedy_decode
from .metrics import MetricAccumulator
from .model import CRNN


def resolve_device(name):
    if name not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda. DirectML is not supported.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was explicitly requested but is unavailable in this PyTorch installation.")
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def seed_worker(_worker_id):
    random.seed(torch.initial_seed() % (2**32))
    np.random.seed(torch.initial_seed() % (2**32))


def versions():
    return {"python": sys.version, "torch": str(torch.__version__), "pillow": PIL.__version__, "numpy": np.__version__,
            "cuda_build": torch.version.cuda, "threads": torch.get_num_threads()}


def ensure_finite(value, context):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"NaN/Inf in {context}; training stopped.")


def scheduler_configuration(config):
    """The only supported scheduling policy; missing keys preserve legacy fixed LR."""
    name = config.get("scheduler", "none")
    if name == "none":
        return {"name": "none"}
    if name != "plateau":
        raise ValueError(f"Unsupported scheduler: {name!r}.")
    return {"name": "plateau", "mode": "min", "metric": "validation_cer",
            "factor": config.get("plateau_factor", 0.5),
            "patience": config.get("plateau_patience", 8),
            "threshold": config.get("plateau_min_delta", 0.002), "threshold_mode": "abs",
            "cooldown": config.get("plateau_cooldown", 2),
            "min_lr": config.get("plateau_min_lr", 0.00001),
            "start_epoch": config.get("plateau_start_epoch", 10)}


def build_scheduler(optimizer, config):
    settings = scheduler_configuration(config)
    if settings["name"] == "none":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **{key: settings[key] for key in
                      ("mode", "factor", "patience", "threshold", "threshold_mode", "cooldown", "min_lr")})


def finite_validation_cer(value, epoch):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid validation CER at epoch {epoch}: {value!r}.") from error
    if not math.isfinite(result):
        raise FloatingPointError(f"NaN/Inf in validation CER at epoch {epoch}; training stopped before scheduler/checkpoint.")
    return result


def update_learning_rate(optimizer, scheduler, validation_cer, epoch, start_epoch=10):
    """Decide the next epoch's LR once, after the current validation.

    Epochs <= start_epoch never reach ReduceLROnPlateau, including its initial
    best-metric/plateau counters. Its last_epoch therefore counts observations,
    rather than all training epochs. Checkpoints are saved after this decision.
    """
    cer = finite_validation_cer(validation_cer, epoch)
    lr_used = optimizer.param_groups[0]["lr"]
    if scheduler is not None:
        expected_previous = max(0, epoch - start_epoch - 1)
        if scheduler.last_epoch != expected_previous:
            raise ValueError(f"Scheduler epoch mismatch: epoch {epoch} expects {expected_previous} "
                             f"previous CER observations, found {scheduler.last_epoch}; refusing a duplicate/skipped decision.")
        if epoch > start_epoch:
            scheduler.step(cer)
    lr_next = optimizer.param_groups[0]["lr"]
    return {"lr_used": lr_used, "lr_next": lr_next, "lr_reduced": lr_next < lr_used}


def train_step(model, batch, optimizer, device, max_norm=5.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["images"].to(device))
    loss = ctc_loss(logits, batch["targets"], batch["target_lengths"], batch["filenames"])
    ensure_finite(loss, "training loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise RuntimeError(f"Missing gradient for {name}.")
            ensure_finite(parameter.grad, f"gradient {name}")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
    if norm.item() == 0:
        raise FloatingPointError("All gradients are zero; training stopped.")
    optimizer.step()
    for name, parameter in model.named_parameters():
        ensure_finite(parameter, f"parameter {name} after optimizer step")
    return float(loss.detach().item())


@torch.no_grad()
def validate(model, loader, device, charset, max_batches=None, return_predictions=False):
    model.eval()
    metrics = MetricAccumulator()
    total_loss = 0.0
    predictions = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        logits = model(batch["images"].to(device))
        loss = ctc_loss(logits, batch["targets"], batch["target_lengths"], batch["filenames"])
        ensure_finite(loss, "validation loss")
        decoded = greedy_decode(logits, charset)
        metrics.update(batch["labels"], decoded)
        total_loss += loss.item() * len(decoded)
        if return_predictions:
            predictions.extend({"filename": filename, "expected": expected,
                                "predicted": predicted, "correct": expected == predicted}
                               for filename, expected, predicted in
                               zip(batch["filenames"], batch["labels"], decoded))
    result = metrics.compute()
    if not result["num_images"]:
        raise ValueError("Validation loader is empty.")
    result["loss"] = total_loss / result["num_images"]
    if return_predictions:
        result["predictions"] = predictions
    return result


def atomic_write(path, writer, *, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb" if binary else "w", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False,
                                         **({} if binary else {"encoding": "utf-8", "newline": ""})) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(path, payload):
    atomic_write(path, lambda stream: torch.save(payload, stream), binary=True)


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise ValueError(f"Cannot read CRNN checkpoint {path}: {error}") from error
    required = {"model_state_dict", "optimizer_state_dict", "epoch", "model_config", "charset",
                "config", "dataset_identity", "rng_state", "best_metrics", "history"}
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1 or not required.issubset(checkpoint):
        raise ValueError("Unsupported or incomplete CRNN checkpoint.")
    if (type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < 1
            or not isinstance(checkpoint["history"], list)
            or any(not isinstance(checkpoint[key], dict) for key in required - {"epoch", "history"})):
        raise ValueError("Invalid checkpoint metadata types or epoch.")
    Charset.from_dict(checkpoint["charset"])
    return checkpoint


def build_checkpoint(model, optimizer, epoch, charset, config, dataset_identity,
                     best_metrics, history, loader_generator, scheduler=None):
    rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
           "loader": loader_generator.get_state()}
    numpy_state = np.random.get_state()
    rng["numpy"] = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
    if next(model.parameters()).device.type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state_all()
    settings = scheduler_configuration(config)
    if (settings["name"] == "plateau") != (scheduler is not None):
        raise ValueError("Scheduler object does not match checkpoint configuration.")
    if scheduler is not None and scheduler.last_epoch != max(0, epoch - settings["start_epoch"]):
        raise ValueError("Cannot checkpoint scheduler before the current epoch's validation decision.")
    return {"format_version": 1, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
            "model_config": model.config, "charset": charset.to_dict(), "config": config,
            "seed": config["seed"], "dataset_identity": dataset_identity,
            "metrics": history[-1] if history else {}, "best_metrics": best_metrics,
            "history": history, "rng_state": rng, "versions": versions(),
            "scheduler_config": settings,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None}


def restore_checkpoint(checkpoint, model, optimizer, charset, dataset_identity, config, loader_generator,
                       scheduler=None):
    if checkpoint["charset"] != charset.to_dict():
        raise ValueError("Checkpoint charset is incompatible with the training charset.")
    if checkpoint["dataset_identity"] != dataset_identity:
        raise ValueError("Dataset identity differs from checkpoint (CSV or PNG contents changed).")
    if checkpoint["model_config"] != model.config:
        raise ValueError("Checkpoint model configuration differs.")
    for key in ("batch_size", "learning_rate", "seed", "num_workers", "overfit_samples", "threads"):
        if config.get(key) != checkpoint["config"].get(key):
            raise ValueError(f"Resume configuration mismatch for {key}.")
    settings = scheduler_configuration(config)
    recorded_settings = checkpoint.get("scheduler_config", scheduler_configuration(checkpoint["config"]))
    state = checkpoint.get("scheduler_state_dict")
    if settings["name"] == "plateau" and ("scheduler_config" not in checkpoint or not isinstance(state, dict)):
        raise ValueError("Checkpoint is incompatible with --scheduler plateau: saved scheduler state/configuration is required; "
                         "a legacy/fixed-LR checkpoint cannot enable a new scheduler on resume.")
    if settings != recorded_settings:
        raise ValueError("Resume scheduler configuration mismatch; use the original --scheduler and plateau parameters.")
    if settings["name"] == "plateau":
        if scheduler is None or not isinstance(state, dict) or "scheduler_config" not in checkpoint:
            raise ValueError("Checkpoint is incompatible with --scheduler plateau: saved scheduler state/configuration is required; "
                             "a legacy checkpoint cannot enable a new scheduler on resume.")
        if state.get("last_epoch") != max(0, checkpoint["epoch"] - settings["start_epoch"]):
            raise ValueError("Checkpoint scheduler epoch is inconsistent with its completed epoch.")
        expected_state = scheduler.state_dict()
        for key in ("mode", "factor", "patience", "threshold", "threshold_mode", "cooldown", "min_lrs", "eps"):
            if state.get(key) != expected_state[key]:
                raise ValueError(f"Checkpoint scheduler state is incompatible for {key}.")
        optimizer_lrs = [group["lr"] for group in checkpoint["optimizer_state_dict"]["param_groups"]]
        if state.get("_last_lr") != optimizer_lrs:
            raise ValueError("Checkpoint scheduler LR differs from the optimizer LR.")
    elif scheduler is not None or state is not None:
        raise ValueError("Checkpoint scheduler is incompatible with --scheduler none.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(state)
    random.setstate(checkpoint["rng_state"]["python"])
    numpy_state = checkpoint["rng_state"]["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(checkpoint["rng_state"]["torch"])
    loader_generator.set_state(checkpoint["rng_state"]["loader"])
    if next(model.parameters()).device.type == "cuda" and "cuda" in checkpoint["rng_state"]:
        torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])
    return checkpoint["epoch"] + 1


def write_history(path, history):
    fields = ["epoch", "train_loss", "val_loss", "exact_match", "cer", "num_images",
              "exact_match_4", "exact_match_5", "exact_match_6", "epoch_seconds",
              "lr_used", "lr_next", "lr_reduced"]
    def writer(stream):
        output = csv.DictWriter(stream, fieldnames=fields)
        output.writeheader()
        output.writerows({key: row.get(key, "") for key in fields} for row in history)
    atomic_write(path, writer)


def dry_run(model, train_loader, val_loader, optimizer, device, charset, started=None,
            scheduler=None, plateau_start_epoch=10):
    model_started = time.perf_counter()
    if started is None:
        started = model_started
    batch = next(iter(train_loader))
    before = next(model.cnn.parameters()).detach().clone()
    shapes = {}
    hook = model.register_forward_hook(lambda _module, _input, output: shapes.update(temporal=list(output.shape)))
    try:
        loss = train_step(model, batch, optimizer, device)
    finally:
        hook.remove()
    first = next(model.cnn.parameters())
    if torch.equal(before, first.detach()):
        raise RuntimeError("Dry-run failed: CNN weights did not change.")
    if first.grad is None or not torch.any(first.grad != 0).item():
        raise RuntimeError("Dry-run failed: no nonzero gradient reached the CNN.")
    result = validate(model, val_loader, device, charset, max_batches=1, return_predictions=True)
    # A diagnostic observation exercises the active scheduler path without
    # running its warmup epochs or creating training history/checkpoints.
    diagnostic_epoch = plateau_start_epoch + 1 if scheduler is not None else 1
    rates = update_learning_rate(optimizer, scheduler, result["cer"], diagnostic_epoch, plateau_start_epoch)
    report = {"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
              "input_shape": list(batch["images"].shape), "temporal_shape": shapes["temporal"],
              "loss": loss, "validation": result, "parameters_changed": True,
              **rates, "scheduler": "plateau" if scheduler is not None else "none",
              "scheduler_observations": scheduler.last_epoch if scheduler is not None else 0,
              "scheduler_diagnostic_epoch": diagnostic_epoch,
              "seconds": time.perf_counter() - started,
              "model_seconds": time.perf_counter() - model_started}
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print("DRY RUN PASSED")
    return report


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a local synthetic CRNN. CPU is the default; DirectML unsupported.")
    parser.add_argument("--data", type=Path, default=Path("dataset/synthetic/v2_main"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=positive, default=30, help="total desired epochs, including completed epochs on resume")
    parser.add_argument("--batch-size", type=positive, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--scheduler", choices=("none", "plateau"), default="none")
    parser.add_argument("--plateau-patience", type=nonnegative, default=8)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-min-delta", type=float, default=0.002,
                        help="minimum absolute CER improvement (threshold_mode=abs)")
    parser.add_argument("--plateau-cooldown", type=nonnegative, default=2)
    parser.add_argument("--plateau-min-lr", type=float, default=0.00001)
    parser.add_argument("--plateau-start-epoch", type=nonnegative, default=10,
                        help="skip CER observations through this epoch; default: monitor starting at epoch 11")
    parser.add_argument("--seed", type=nonnegative, default=42)
    parser.add_argument("--num-workers", type=nonnegative, default=0)
    parser.add_argument("--threads", type=positive, default=6, help="PyTorch CPU threads (default: 6)")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/runs/crnn-v2-baseline"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overfit-samples", type=positive, help="deliberately restrict training to a seeded synthetic subset")
    args = parser.parse_args(argv)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.scheduler == "plateau":
        if not math.isfinite(args.plateau_factor) or not 0 < args.plateau_factor < 1:
            parser.error("--plateau-factor must be finite and between 0 and 1 (exclusive)")
        if not math.isfinite(args.plateau_min_delta) or args.plateau_min_delta < 0:
            parser.error("--plateau-min-delta must be finite and nonnegative")
        if not math.isfinite(args.plateau_min_lr) or not 0 <= args.plateau_min_lr <= args.learning_rate:
            parser.error("--plateau-min-lr must be finite, nonnegative and no greater than --learning-rate")
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume cannot be combined")
    return args


def run(args):
    started = time.perf_counter()
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    manifest = SyntheticManifest(args.data)
    train_data = SyntheticDataset(args.data, "train", manifest=manifest)
    charset = train_data.charset
    val_data = SyntheticDataset(args.data, "validation", charset=charset, manifest=manifest)
    if not len(train_data) or not len(val_data):
        raise ValueError("Both synthetic train and validation must contain images.")
    if args.overfit_samples:
        if args.overfit_samples > len(train_data):
            raise ValueError("--overfit-samples exceeds the training dataset size.")
        indices = sorted(random.Random(args.seed).sample(range(len(train_data)), args.overfit_samples))
        train_data = Subset(train_data, indices)
        print(f"OVERFIT SANITY MODE: {len(indices)} training images; validation remains separate.")
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                      "collate_fn": collate_ctc, "worker_init_fn": seed_worker}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    model = CRNN(charset.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scheduler = build_scheduler(optimizer, vars(args))
    if args.dry_run:
        dry_run(model, train_loader, val_loader, optimizer, device, charset, started=started,
                scheduler=scheduler, plateau_start_epoch=args.plateau_start_epoch)
        return 0
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(weight_decay=0.0001, gradient_max_norm=5.0, best_criterion="lowest CER, then highest exact match")
    run_dir = args.run_dir.resolve()
    history = []
    best = {"cer": float("inf"), "exact_match": -1.0, "epoch": 0}
    start_epoch = 1
    if args.resume:
        if args.resume.resolve() != run_dir / "last.pt":
            raise ValueError("Resume requires --resume <run-dir>/last.pt and the original --run-dir.")
        checkpoint = load_checkpoint(args.resume)
        start_epoch = restore_checkpoint(checkpoint, model, optimizer, charset, manifest.identity, config, generator,
                                         scheduler=scheduler)
        history, best = checkpoint["history"], checkpoint["best_metrics"]
        if best["epoch"] == checkpoint["epoch"]:
            # Repair interruption between atomic writes of last.pt and best.pt.
            save_checkpoint(run_dir / "best.pt", checkpoint)
        else:
            previous_best = load_checkpoint(run_dir / "best.pt")
            if (previous_best["best_metrics"] != best
                    or previous_best["dataset_identity"] != manifest.identity
                    or previous_best["charset"] != charset.to_dict()
                    or previous_best["model_config"] != model.config
                    or previous_best["epoch"] != best["epoch"]):
                raise ValueError("best.pt is incompatible with last.pt; preserve and inspect this run directory.")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Run directory is not empty: {run_dir}. Choose another --run-dir or explicitly --resume its last.pt.")
    if start_epoch > args.epochs:
        raise ValueError(f"Checkpoint already completed {start_epoch - 1} epochs; --epochs is the TOTAL desired epochs.")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(run_dir / "run_config.json", lambda stream: json.dump(
        {"config": config, "model": model.config, "charset": charset.to_dict(),
         "dataset": manifest.identity, "versions": versions()}, stream, indent=2))
    write_history(run_dir / "history.csv", history)
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"train: {len(train_data)}; validation: {len(val_data)}; charset: {charset.characters}")
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            train_total += train_step(model, batch, optimizer, device) * len(batch["labels"])
            train_count += len(batch["labels"])
        validation = validate(model, val_loader, device, charset)
        finite_validation_cer(validation["cer"], epoch)
        improved = (validation["cer"], -validation["exact_match"]) < (best["cer"], -best["exact_match"])
        if improved:
            best = {"cer": validation["cer"], "exact_match": validation["exact_match"], "epoch": epoch}
        row = {"epoch": epoch, "train_loss": train_total / train_count, "val_loss": validation["loss"],
               "exact_match": validation["exact_match"], "cer": validation["cer"],
               "num_images": validation["num_images"], "by_length": validation["by_length"],
               "epoch_seconds": time.perf_counter() - started}
        for length in (4, 5, 6):
            row[f"exact_match_{length}"] = validation["by_length"].get(str(length), {}).get("exact_match", "")
        history.append(row)
        print(f"Train loss: {row['train_loss']:.6f}; val loss: {row['val_loss']:.6f}; "
              f"exact match: {row['exact_match']:.4%}; CER: {row['cer']:.6f}; "
              f"duration: {row['epoch_seconds']:.2f}s; best checkpoint: epoch {best['epoch']}", flush=True)
        row.update(update_learning_rate(optimizer, scheduler, row["cer"], epoch, args.plateau_start_epoch))
        print(f"LR used: {row['lr_used']:.6f}; LR next: {row['lr_next']:.6f}", flush=True)
        if row["lr_reduced"]:
            print("ReduceLROnPlateau: validation CER stagnated; "
                  f"learning rate: {row['lr_used']:.6g} -> {row['lr_next']:.6g}", flush=True)
        checkpoint = build_checkpoint(model, optimizer, epoch, charset, config, manifest.identity, best, history, generator,
                                      scheduler=scheduler)
        save_checkpoint(run_dir / "last.pt", checkpoint)
        if improved:
            save_checkpoint(run_dir / "best.pt", checkpoint)
        write_history(run_dir / "history.csv", history)
        print("Validation exact match by length: " + json.dumps(validation["by_length"], sort_keys=True), flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, FloatingPointError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
