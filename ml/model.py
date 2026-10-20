"""Compact CNN/BiLSTM recognizer for the synthetic 80 by 215 images."""

import torch
from torch import Tensor, nn


class CRNN(nn.Module):
    """Turn an image batch into raw CTC logits with shape ``[T, B, C]``.

    The first two pools halve both spatial axes. The final two only halve
    height, retaining 53 temporal positions for an input width of 215.
    Height is averaged before the LSTM; width always remains the time axis.
    """

    def __init__(self, num_classes: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        for name, value, minimum in (
            ("num_classes", num_classes, 2),
            ("hidden_size", hidden_size, 1),
            ("num_layers", num_layers, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
        self.config = {
            "num_classes": num_classes,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        }
        blocks = []
        in_channels = 1
        for out_channels, pool in zip((32, 64, 128, 256), ((2, 2), (2, 2), (2, 1), (2, 1))):
            blocks.extend((
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            ))
            in_channels = out_channels
        self.cnn = nn.Sequential(*blocks)
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    @staticmethod
    def spatial_to_sequence(features: Tensor) -> Tensor:
        """Average only the height axis: ``[B,C,H,W] -> [B,W,C]``."""
        if features.ndim != 4 or any(size <= 0 for size in features.shape):
            raise ValueError(f"CNN features must have nonempty [B,C,H,W] shape; got {tuple(features.shape)}")
        return features.mean(dim=2).transpose(1, 2).contiguous()

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 80, 215) or images.shape[0] == 0:
            raise ValueError(f"CRNN input must have shape [B,1,80,215] with B > 0; got {tuple(images.shape)}")
        if images.dtype != torch.float32:
            raise ValueError(f"CRNN input must be float32; got {images.dtype}")
        if not torch.isfinite(images).all().item():
            raise RuntimeError("NaN/Inf in CRNN input images")
        features = self.cnn(images)
        sequence = self.spatial_to_sequence(features)
        sequence, _ = self.rnn(sequence)
        return self.classifier(sequence).transpose(0, 1).contiguous()
