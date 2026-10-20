"""Local greedy CTC decoding; no beam search or language model."""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .charset import Charset


def greedy_decode(logits: Tensor, charset: "Charset") -> list[str]:
    """Collapse neighboring IDs before removing blank (reserved ID zero)."""
    if logits.ndim != 3 or any(size <= 0 for size in logits.shape):
        raise ValueError(f"Decoder expects nonempty [T,B,C] logits; got {tuple(logits.shape)}")
    if logits.shape[2] != charset.num_classes:
        raise ValueError(f"Decoder has {logits.shape[2]} classes, but charset requires {charset.num_classes}")
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("NaN/Inf in greedy decoder logits")
    paths = logits.argmax(dim=2).transpose(0, 1).detach().cpu().tolist()
    results = []
    for path in paths:
        collapsed = []
        previous = None
        for index in path:
            if index != previous and index != 0:
                collapsed.append(index)
            previous = index
        results.append(charset.decode(collapsed))
    return results
