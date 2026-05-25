"""Model architectures — **agent-modifiable**.

The agent is free to redesign these nets to improve the official score. The only
hard requirement is the signature of :func:`build_model`: it takes the resolved
config and the number of classes and returns an ``nn.Module`` mapping a
``(N, C, H, W)`` float tensor in [0, 1] to ``(N, num_classes)`` logits.

Torch is imported lazily so that ``import harness`` stays cheap and the
non-training parts of the repo work without the optional ``[torch]`` extra.
"""

from __future__ import annotations

from typing import Any


def _torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for training. Install with: "
            "uv pip install -e '.[torch]'"
        ) from exc
    import torch
    import torch.nn as nn

    return torch, nn


def build_model(config: dict, num_classes: int):
    """Build the model named by ``config['model']['name']``."""
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "tiny_cnn")
    if name == "tiny_cnn":
        return TinyCNN(
            num_classes=num_classes,
            width=int(model_cfg.get("width", 16)),
            depth=int(model_cfg.get("depth", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            in_channels=3,
        )
    raise ValueError(f"unknown model name: {name!r}")


class TinyCNN:  # thin factory wrapper so the class is importable without torch
    """A small configurable CNN. Instantiating returns an ``nn.Module``."""

    def __new__(cls, num_classes: int, width: int = 16, depth: int = 2,
                dropout: float = 0.0, in_channels: int = 3):
        torch, nn = _torch()

        class _TinyCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                blocks = []
                c_in = in_channels
                c_out = width
                for _ in range(depth):
                    blocks += [
                        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                        nn.BatchNorm2d(c_out),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                    ]
                    c_in = c_out
                    c_out = c_out * 2
                self.features = nn.Sequential(*blocks)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(c_in, num_classes)

            def forward(self, x):  # x: (N, C, H, W)
                z = self.features(x)
                z = self.pool(z).flatten(1)
                z = self.drop(z)
                return self.head(z)

        return _TinyCNN()
