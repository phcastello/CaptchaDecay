"""CTC loss with explicit validation of the alignment and numeric inputs."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def min_ctc_timesteps(target: Sequence[int]) -> int:
    """Repeated neighbors need a separating blank, in addition to their labels."""
    return len(target) + sum(left == right for left, right in zip(target, target[1:]))


def ctc_loss(
    logits: Tensor,
    targets: Tensor,
    target_lengths: Tensor,
    filenames: Sequence[str] | None = None,
) -> Tensor:
    """Calculate mean CTC loss using the actual time dimension of ``logits``.

    Targets are concatenated, nonblank character IDs. Input/target lengths are
    kept on CPU, which works for the CPU and optional CUDA backends. Impossible
    alignments fail before ``zero_infinity=True`` can hide them.
    """
    if logits.ndim != 3 or any(size <= 0 for size in logits.shape) or logits.shape[2] < 2:
        raise ValueError(f"CTC logits must have nonempty [T,B,C] shape and C >= 2; got {tuple(logits.shape)}")
    if not logits.is_floating_point():
        raise ValueError("CTC logits must be floating point")
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("NaN/Inf in CTC logits")
    timesteps, batch_size, num_classes = logits.shape
    if targets.ndim != 1 or targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("CTC targets must be a flat int32/int64 tensor")
    if target_lengths.ndim != 1 or target_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("CTC target_lengths must be a flat int32/int64 tensor")
    if target_lengths.numel() != batch_size:
        raise ValueError(f"CTC needs one target length per image ({batch_size}); got {target_lengths.numel()}")
    if filenames is not None and len(filenames) != batch_size:
        raise ValueError(f"CTC needs one filename per image ({batch_size}); got {len(filenames)}")
    lengths = target_lengths.detach().cpu().tolist()
    if any(length <= 0 for length in lengths):
        raise ValueError("CTC labels must be nonempty: every target length must be > 0")
    if sum(lengths) != targets.numel():
        raise ValueError(f"CTC target lengths sum to {sum(lengths)}, but there are {targets.numel()} target IDs")
    encoded = targets.detach().cpu().tolist()
    offset = 0
    for index, length in enumerate(lengths):
        label = encoded[offset:offset + length]
        context = str(filenames[index]) if filenames is not None else f"batch image {index}"
        for value in label:
            if value < 1 or value >= num_classes:
                raise ValueError(
                    f"{context}: invalid CTC target ID {value}; expected 1..{num_classes - 1} (blank 0 is reserved)"
                )
        minimum = min_ctc_timesteps(label)
        if minimum > timesteps:
            raise ValueError(
                f"{context}: impossible CTC alignment: T={timesteps}, label length={length}, "
                f"minimum timesteps={minimum} including consecutive repeats"
            )
        offset += length
    input_lengths = torch.full((batch_size,), timesteps, dtype=torch.long)
    log_probabilities = logits.log_softmax(dim=2)
    if not torch.isfinite(log_probabilities).all().item():
        raise RuntimeError("NaN/Inf in CTC log_softmax probabilities; inspect the magnitude of the logits")
    loss = nn.CTCLoss(blank=0, zero_infinity=True)(
        log_probabilities,
        targets.to(device=logits.device, dtype=torch.long),
        input_lengths,
        target_lengths.to(device="cpu", dtype=torch.long),
    )
    if not torch.isfinite(loss).item():
        raise RuntimeError("NaN/Inf in CTC loss after log_softmax; inspect logits and target alignment")
    return loss
