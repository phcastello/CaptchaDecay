"""Synthetic float32 training probe. No project imports, images, labels or HTTP.

Fallback warnings are errors, so an unsupported native operation cannot pass by
running on CPU. Explicit CPU copies below are observation/synchronization only.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import warnings


CASES = ("conv2d", "batchnorm2d", "relu", "maxpool2d", "cnn", "bilstm", "linear", "ctc")
FALLBACK_PATTERN = r"(?is).*(fall.?back|falling back|not currently supported|not supported.*CPU).*"
RTOL, ATOL = 1e-3, 1e-5


class NativeOperationUnavailable(RuntimeError):
    """Reject an operation before the DirectML CPU fallback can execute."""


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "directml", "both"), default="both")
    parser.add_argument("--batch-size", type=positive, default=4)
    parser.add_argument("--iterations", type=positive, default=5)
    parser.add_argument("--warmup", type=nonnegative, default=2)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--seed", type=nonnegative, default=42)
    parser.add_argument("--threads", type=positive, default=6)
    parser.add_argument("--device-index", type=nonnegative, default=0)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/directml-results.json"))
    return parser.parse_args(argv)


def ctc_min_timesteps(target):
    """CTC needs one extra blank between each pair of adjacent repeats."""
    return len(target) + sum(a == b for a, b in zip(target, target[1:]))


def summarize(samples, batch):
    if not samples or any(x <= 0 for x in samples):
        raise ValueError("timings must be nonempty and positive")
    mean = statistics.mean(samples)
    return {"mean_ms": mean * 1000, "median_ms": statistics.median(samples) * 1000,
            "stdev_ms": statistics.stdev(samples) * 1000 if len(samples) > 1 else 0,
            "min_ms": min(samples) * 1000, "max_ms": max(samples) * 1000,
            "images_per_second": batch / mean}


@contextmanager
def native_guard():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("error", message=FALLBACK_PATTERN)
        yield caught


@contextmanager
def native_dispatch_guard(torch, device):
    """Audit actual dispatched DML operators, including LSTM's decomposition.

    Warning promotion alone is insufficient: some nested C++ calls print a
    warning and continue. Inspect registrations before dispatching each operator.
    Composite kernels may decompose; their actual suboperations are audited too.
    The private dispatcher API is intentionally confined to this diagnostic.
    """
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    audit = {"operators": {}, "blocked_operator": None}
    if device.type != "privateuseone":
        yield audit
        return

    class NativeOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            values, _ = tree_flatten((args, kwargs))
            uses_dml = any(isinstance(v, torch.Tensor) and v.device.type == "privateuseone"
                           for v in values)
            uses_dml = uses_dml or any(isinstance(v, torch.device) and v.type == "privateuseone"
                                       for v in values)
            if uses_dml:
                schema = func._schema
                name = schema.name + ("." + schema.overload_name if schema.overload_name else "")
                keys = [key for key in ("PrivateUse1", "CompositeImplicitAutograd",
                                        "CompositeExplicitAutograd", "CompositeExplicitAutogradNonFunctional")
                        if torch._C._dispatch_has_kernel_for_dispatch_key(name, key)]
                audit["operators"][name] = keys
                if not keys:
                    audit["blocked_operator"] = name
                    raise NativeOperationUnavailable(
                        f"{name}: no PrivateUse1 or composite kernel; CPU fallback blocked before execution")
            return func(*args, **kwargs)

    with NativeOnly():
        yield audit


def environment(torch):
    packages = {}
    for name in ("torch", "torch-directml", "torchvision", "numpy", "pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "executable": sys.executable,
              "prefix": sys.prefix, "base_prefix": sys.base_prefix,
              "platform": platform.platform(), "packages": packages,
              "torch_runtime": torch.__version__, "cuda_build": torch.version.cuda,
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()}
    if sys.platform == "win32":
        command = """[PSCustomObject]@{
          OS=(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber);
          CPU=(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);
          RAM=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory;
          GPU=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate)
        } | ConvertTo-Json -Depth 5"""
        try:
            proc = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                  capture_output=True, text=True, timeout=30, check=True)
            result["windows"] = json.loads(proc.stdout)
        except (subprocess.SubprocessError, ValueError, OSError) as error:
            result["windows_error"] = str(error)
    return result


def build_case(torch, name, batch, seed, device):
    nn = torch.nn
    # Generate on CPU with exactly the same order/seed on both backends.
    torch.manual_seed(seed)

    def cnn():
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8),
                             nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16),
                             nn.ReLU(), nn.MaxPool2d(2))

    class TinyCRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = cnn()
            self.projection = nn.Linear(16 * 20, 128)
            self.rnn = nn.LSTM(128, 128, num_layers=2, bidirectional=True)
            self.head = nn.Linear(256, 17)

        def forward(self, x):
            features = self.cnn(x)  # B,16,20,53
            sequence = features.permute(3, 0, 1, 2).flatten(2)  # T,B,320
            sequence, _ = self.rnn(self.projection(sequence))
            return self.head(sequence)

    if name == "conv2d":
        model, shape = nn.Conv2d(1, 8, 3, padding=1), (batch, 1, 80, 215)
    elif name == "batchnorm2d":
        model, shape = nn.BatchNorm2d(8), (batch, 8, 40, 107)
    elif name == "relu":
        model, shape = nn.ReLU(), (batch, 8, 40, 107)
    elif name == "maxpool2d":
        model, shape = nn.MaxPool2d(2), (batch, 8, 40, 106)
    elif name == "cnn":
        model, shape = cnn(), (batch, 1, 80, 215)
    elif name == "bilstm":
        model, shape = nn.LSTM(128, 128, num_layers=2, bidirectional=True), (24, batch, 128)
    elif name == "linear":
        model, shape = nn.Linear(256, 17), (24, batch, 256)
    elif name == "ctc":
        model, shape = nn.Identity(), (24, batch, 17)
    elif name == "integrated":
        model, shape = TinyCRNN(), (batch, 1, 80, 215)
    else:
        raise ValueError(name)
    source = torch.randn(shape, dtype=torch.float32)
    model = model.to(device).train()
    x = source.to(device).detach().requires_grad_(True)
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3, foreach=False) if parameters else None
    # Deliberate repeated symbols exercise the true alignment constraint.
    targets = [[1 + (i % 15), 1 + (i % 15), 16, 2, 3] for i in range(batch)]
    temporal = 53 if name == "integrated" else 24
    if any(ctc_min_timesteps(t) > temporal for t in targets):
        raise ValueError("invalid CTC alignment")
    target_tensor = torch.tensor([v for row in targets for v in row], dtype=torch.long).to(device)
    input_lengths = torch.full((batch,), temporal, dtype=torch.long).to(device)
    target_lengths = torch.full((batch,), 5, dtype=torch.long).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    # A zero target makes the auxiliary objective deterministic and nonnegative.
    def forward():
        output = model(x)
        return output[0] if isinstance(output, tuple) else output

    def loss_for(output):
        if name in ("ctc", "integrated"):
            return criterion(output.log_softmax(2), target_tensor, input_lengths, target_lengths)
        return output.square().mean()

    return model, x, optimizer, forward, loss_for


def cpu_copy(tensor):
    return tensor.detach().to("cpu", non_blocking=False).clone()


def ensure_finite(torch, tensor, label, nonzero=False):
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{label}: NaN/Inf")
    if nonzero and not torch.count_nonzero(tensor).item():
        raise RuntimeError(f"{label}: all zero")


def probe(torch, name, batch, args, device):
    result = {"status": "failed", "device": str(device), "batch": batch,
              "stages": {}, "iterations_completed": 0, "losses": [],
              "operation_forward_stage": "loss" if name == "ctc" else "forward"}
    snapshots = {}
    stage = "setup"
    caught = []
    audit = {}
    try:
        with native_guard() as caught, native_dispatch_guard(torch, device) as audit:
            model, x, optimizer, forward, loss_for = build_case(
                torch, name, batch, args.seed, device)
            result["stages"][stage] = "passed"
            result["parameter_devices"] = sorted({str(p.device) for p in model.parameters()})
            result["input_device"] = str(x.device)
            for iteration in range(max(3, args.iterations)):
                if optimizer:
                    optimizer.zero_grad(set_to_none=True)
                x.grad = None
                before = {key: cpu_copy(p) for key, p in model.named_parameters()}
                stage = "forward"
                output = forward()
                if output.device != device:
                    raise RuntimeError(f"unexpected output device: {output.device}")
                observed = cpu_copy(output)
                ensure_finite(torch, observed, "output")
                result["output_shape"] = list(output.shape)
                result["output_device"] = str(output.device)
                result["stages"][stage] = "passed"
                if iteration == 0:
                    snapshots["input_logits" if name == "ctc" else "output"] = observed
                stage = "loss"
                loss = loss_for(output)
                observed_loss = cpu_copy(loss)
                ensure_finite(torch, observed_loss, "loss", nonzero=True)
                result["losses"].append(observed_loss.item())
                result["stages"][stage] = "passed"
                if iteration == 0:
                    snapshots["loss"] = observed_loss
                stage = "backward"
                loss.backward()
                result["stages"][stage] = "passed"
                stage = "gradients"
                grads = {"input": x.grad, **{key: p.grad for key, p in model.named_parameters()}}
                grad_info = {}
                for key, grad in grads.items():
                    if grad is None:
                        raise RuntimeError(f"missing gradient: {key}")
                    if grad.device != device:
                        raise RuntimeError(f"gradient {key} on {grad.device}")
                    observed_grad = cpu_copy(grad)
                    # Individual biases may have a zero gradient by symmetry. Require
                    # a nonzero input gradient and nonzero trainable gradients overall.
                    ensure_finite(torch, observed_grad, key, nonzero=(key == "input"))
                    grad_info[key] = {"finite": True, "nonzero": bool(torch.count_nonzero(observed_grad)),
                                      "max_abs": observed_grad.abs().max().item()}
                    if iteration == 0:
                        snapshots[f"grad:{key}"] = observed_grad
                if before and not any(v["nonzero"] for k, v in grad_info.items() if k != "input"):
                    raise RuntimeError("all trainable gradients are zero")
                result["gradients"] = grad_info
                result["stages"][stage] = "passed"
                if optimizer:
                    stage = "optimizer"
                    optimizer.step()
                    result["stages"][stage] = "passed"
                    stage = "parameters"
                    changed = []
                    for key, param in model.named_parameters():
                        observed_param = cpu_copy(param)
                        ensure_finite(torch, observed_param, key)
                        if not torch.equal(before[key], observed_param):
                            changed.append(key)
                        if iteration == 0:
                            snapshots[f"update:{key}"] = observed_param - before[key]
                    if not changed:
                        raise RuntimeError("no parameter changed after Adam.step")
                    result["changed_parameters"] = changed
                    result["stages"][stage] = "passed"
                else:
                    result["stages"]["optimizer"] = "not_applicable"
                result["iterations_completed"] += 1
            result["status"] = "passed"
    except Exception as error:
        result["failed_stage"] = stage
        result["stages"][stage] = "failed"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        result["fallback_blocked"] = isinstance(error, (Warning, NativeOperationUnavailable))
    result["warnings"] = [str(item.message) for item in caught]
    result["dispatch_audit"] = audit
    return result, snapshots


def compare_snapshots(torch, left, right):
    rows = {}
    for key in sorted(left.keys() | right.keys()):
        if key not in left or key not in right:
            rows[key] = {"status": "not_comparable", "reason": "stage unavailable on one backend"}
            continue
        a, b = left[key], right[key]
        if a.shape != b.shape:
            rows[key] = {"status": "mismatch", "reason": "different shapes"}
            continue
        absolute = (a - b).abs()
        passed = torch.allclose(a, b, rtol=RTOL, atol=ATOL)
        rows[key] = {"status": "passed" if passed else "mismatch",
                     "max_abs": absolute.max().item(),
                     "max_relative": (absolute / a.abs().clamp_min(ATOL)).max().item()}
    compared = sum(r["status"] != "not_comparable" for r in rows.values())
    return {"rtol": RTOL, "atol": ATOL, "tensors": rows, "tensors_compared": compared,
            "status": ("mismatch" if any(r["status"] == "mismatch" for r in rows.values())
                       else "not_comparable" if not compared
                       else "partial" if any(r["status"] == "not_comparable" for r in rows.values())
                       else "passed")}


def benchmark(torch, name, batch, args, device):
    result = {"status": "failed", "device": str(device), "repeats": []}
    caught = []
    try:
        # The caller first requires the same case/shapes to pass the audited
        # training probe. Keep Python dispatcher auditing outside timed sections;
        # any fallback warning during timing still invalidates the measurement.
        with native_guard() as caught:
            for repeat in range(args.repeats):
                model, x, optimizer, forward, loss_for = build_case(
                    torch, name, batch, args.seed, device)
                samples = {"forward": [], "backward_optimizer": [], "total": []}
                for iteration in range(args.warmup + args.iterations):
                    start = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    x.grad = None
                    output = forward()
                    loss = loss_for(output)
                    # Blocking CPU read completes forward/loss; its cost is included.
                    loss_cpu = cpu_copy(loss)
                    split = time.perf_counter()
                    loss.backward()
                    optimizer.step()
                    # Observe EVERY parameter to complete all optimizer branches,
                    # not merely a scalar produced before the optimizer ran.
                    params_cpu = [cpu_copy(p) for p in model.parameters()]
                    end = time.perf_counter()
                    ensure_finite(torch, loss_cpu, "timed loss", nonzero=True)
                    for value in params_cpu:
                        ensure_finite(torch, value, "timed parameter")
                    if iteration >= args.warmup:
                        samples["forward"].append(split - start)
                        samples["backward_optimizer"].append(end - split)
                        samples["total"].append(end - start)
                result["repeats"].append({"index": repeat + 1, "seconds": samples,
                                          **{k: summarize(v, batch) for k, v in samples.items()}})
            for key in ("forward", "backward_optimizer", "total"):
                all_samples = [v for run in result["repeats"] for v in run["seconds"][key]]
                result[key] = summarize(all_samples, batch)
            result["repeat_mean_stdev_ms"] = statistics.stdev(
                r["total"]["mean_ms"] for r in result["repeats"]) if args.repeats > 1 else 0
            result["status"] = "passed"
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    result["warnings"] = [str(item.message) for item in caught]
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        import torch
    except ImportError as error:
        raise SystemExit(f"Use the isolated benchmark environment: {error}") from error
    torch.set_num_threads(args.threads)
    torch.set_warn_always(True)
    result = {"schema_version": 1, "config": {**vars(args), "output": str(args.output)},
              "environment": environment(torch), "backends": {}, "comparisons": {},
              "synchronization": "blocking loss CPU copy after forward; blocking copies of every "
                                 "parameter after Adam.step; copies included in timing; no CUDA API",
              "fallback_policy": "native-only: pre-dispatch kernel audit plus fallback warnings "
                                 "promoted to errors; no hybrid run",
              "timing_scope": "preloaded synthetic batch; forward includes loss/zero_grad; "
                              "backward includes Adam and synchronization; validation excluded"}
    devices = {}
    names = ("cpu", "directml") if args.device == "both" else (args.device,)
    for backend in names:
        try:
            if backend == "cpu":
                device = torch.device("cpu")
                label = "CPU"
            else:
                import torch_directml
                result["environment"]["directml_devices"] = [
                    torch_directml.device_name(i).rstrip("\x00")
                    for i in range(torch_directml.device_count())]
                device = torch_directml.device(args.device_index)
                label = torch_directml.device_name(args.device_index).rstrip("\x00")
                if "AMD Radeon RX 6700 XT" not in label:
                    print(f"NOTICE: selected adapter is {label}", flush=True)
            devices[backend] = device
            result["backends"][backend] = {"device": str(device), "name": label, "probes": {}, "timings": {}}
        except Exception as error:
            result["backends"][backend] = {"status": "unavailable", "error": str(error)}
    snapshots = {}
    for backend, device in devices.items():
        data = result["backends"][backend]
        for name in CASES:
            print(f"[{backend}] {name} batch={args.batch_size}", flush=True)
            data["probes"][name], snapshots[backend, name] = probe(
                torch, name, args.batch_size, args, device)
            row = data["probes"][name]
            message = row.get("error", "").splitlines()
            print(f"  {row['status']} {row.get('failed_stage', '')} {message[0] if message else ''}", flush=True)
        required = ("cnn", "bilstm", "linear", "ctc")
        if all(data["probes"][n]["status"] == "passed" for n in required):
            print(f"[{backend}] integrated", flush=True)
            data["probes"]["integrated"], snapshots[backend, "integrated"] = probe(
                torch, "integrated", args.batch_size, args, device)
        else:
            data["probes"]["integrated"] = {"status": "skipped",
                "reason": "critical components failed native compatibility", "failed_components": [
                    n for n in required if data["probes"][n]["status"] != "passed"]}
    if len(devices) == 2:
        for name in (*CASES, "integrated"):
            result["comparisons"][name] = compare_snapshots(
                torch, snapshots.get(("cpu", name), {}), snapshots.get(("directml", name), {}))
    # Performance is gated on successful full forward/backward/update diagnostics.
    for backend, device in devices.items():
        data = result["backends"][backend]
        for name in ("cnn", "integrated"):
            if data["probes"][name]["status"] == "passed":
                print(f"[{backend}] timing {name}: {args.repeats} repeats", flush=True)
                data["timings"][name] = benchmark(torch, name, args.batch_size, args, device)
            else:
                data["timings"][name] = {"status": "skipped", "reason": "compatibility failed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)
    # Diagnostic failures are expected results, but visible to automation.
    return 0 if all(b.get("probes", {}).get("integrated", {}).get("status") == "passed"
                    for b in result["backends"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
