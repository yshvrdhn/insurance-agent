"""Training surface — **agent-modifiable**.

This is the main file the harness edits to improve the official score. It owns:

* the optimizer, learning-rate schedule and regularization,
* the compute budget (training stops at the first of max-epochs / max-seconds),
* optional per-sample weights, so :mod:`harness.mining` can upweight the rare,
  hard foggy frames -- the single biggest lever on the worst-group score,
* augmentation (``brightness_jitter`` / ``hflip`` / ``contrast_jitter``) for
  generic robustness.

It must keep one stable contract with the immutable evaluator: :func:`train`
returns an object exposing ``predict(X) -> np.ndarray[int]`` over float images
of shape ``(N, H, W, C)`` in [0, 1]. Everything else is fair game.

Training reads only the ``train`` and ``val`` splits. Touching ``test`` here is
a contract violation (``program.md``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from harness.data import IMG_C, IMG_H, IMG_W, class_names, load_split
from harness.model import build_model


def set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainedModel:
    """The artifact handed to the evaluator. Duck-typed on ``predict``."""

    config: dict
    num_classes: int
    state_dict: Any
    norm: dict
    metrics: dict = field(default_factory=dict)
    _model: Any = None

    def _ensure_model(self):
        if self._model is None:
            import torch

            m = build_model(self.config, self.num_classes)
            m.load_state_dict(self.state_dict)
            m.eval()
            self._model = m
        return self._model

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        model = self._ensure_model()
        xt = _to_tensor(_apply_eval_norm(x.astype(np.float32), self.norm))
        with torch.no_grad():
            logits = model(xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def save(self, path: str | Path) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "num_classes": self.num_classes,
                "state_dict": self.state_dict,
                "norm": self.norm,
                "metrics": self.metrics,
            },
            str(path),
        )
        return path


def _to_tensor(x_nhwc: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(
        x_nhwc.transpose(0, 3, 1, 2))).float()


def _apply_eval_norm(x: np.ndarray, norm: dict) -> np.ndarray:
    if norm.get("per_image"):
        m = x.mean(axis=(1, 2, 3), keepdims=True)
        s = x.std(axis=(1, 2, 3), keepdims=True) + 1e-5
        return (x - m) / s
    return x


def _augment(x: np.ndarray, aug: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply training-time augmentation to a float batch ``(N, H, W, C)``."""
    out = x
    if aug.get("hflip"):
        flip = rng.random(len(out)) < 0.5
        out = out.copy()
        out[flip] = out[flip, :, ::-1, :]
    bj = float(aug.get("brightness_jitter", 0.0))
    if bj > 0:  # multiplicative brightness -> robustness to fog haze
        factor = np.exp(rng.normal(0.0, bj, size=(len(out), 1, 1, 1)))
        out = out * factor.astype(np.float32)
    cj = float(aug.get("contrast_jitter", 0.0))
    if cj > 0:
        m = out.mean(axis=(1, 2, 3), keepdims=True)
        scale = np.exp(rng.normal(0.0, cj, size=(len(out), 1, 1, 1)))
        out = m + (out - m) * scale.astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def train(config: dict, dataset_path: str | Path,
          sample_weights: np.ndarray | None = None) -> TrainedModel:
    """Train under the configured budget; return a :class:`TrainedModel`."""
    import torch
    import torch.nn.functional as F

    seed = int(config.get("seed", 0))
    set_seed(seed)
    rng = np.random.default_rng(seed)

    train_cfg = config.get("train", {})
    aug = config.get("augment", {})
    norm = {"per_image": bool(config.get("augment", {}).get("per_image_norm", False))}
    budget = config.get("budget", {})

    xtr, ytr, _ = load_split(dataset_path, "train")
    xva, yva, _ = load_split(dataset_path, "val")
    num_classes = len(class_names())

    model = build_model(config, num_classes)
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    batch_size = int(train_cfg.get("batch_size", 64))
    max_epochs = int(budget.get("max_epochs", 12))
    max_seconds = float(budget.get("max_seconds", 60.0))

    if sample_weights is None:
        sample_weights = np.ones(len(xtr), dtype=np.float64)
    sample_weights = sample_weights / sample_weights.sum()

    sched = None
    if train_cfg.get("cosine_lr"):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    start = time.time()
    steps = 0
    n = len(xtr)
    steps_per_epoch = max(1, n // batch_size)
    yva_t = torch.from_numpy(yva).long()
    best_val = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(max_epochs):
        model.train()
        # weighted sampling lets mining upweight hard frames
        order = rng.choice(n, size=n, replace=True, p=sample_weights)
        for bi in range(steps_per_epoch):
            idx = order[bi * batch_size:(bi + 1) * batch_size]
            if len(idx) == 0:
                continue
            xb = _augment(xtr[idx], aug, rng)
            xb = _apply_eval_norm(xb, norm)
            xt = _to_tensor(xb)
            yt = torch.from_numpy(ytr[idx]).long()
            opt.zero_grad()
            loss = F.cross_entropy(model(xt), yt,
                                   label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
            loss.backward()
            opt.step()
            steps += 1
        if sched is not None:
            sched.step()

        model.eval()
        with torch.no_grad():
            xv = _apply_eval_norm(xva.astype(np.float32), norm)
            val_pred = model(_to_tensor(xv)).argmax(1)
            val_acc = (val_pred == yva_t).float().mean().item()
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if time.time() - start > max_seconds:
            break

    elapsed = time.time() - start
    model.load_state_dict(best_state)
    metrics = {
        "val_acc": round(best_val, 4),
        "epochs_run": epoch + 1,
        "steps": steps,
        "seconds": round(elapsed, 2),
    }
    return TrainedModel(
        config=config,
        num_classes=num_classes,
        state_dict={k: v.detach().clone() for k, v in best_state.items()},
        norm=norm,
        metrics=metrics,
    )
