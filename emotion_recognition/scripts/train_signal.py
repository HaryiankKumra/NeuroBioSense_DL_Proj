"""Stage 2 training script: Signal pretraining on WESAD or scratch fallback.

If WESAD is unavailable, this script can be skipped and Stage 3 will train the
signal branch with stronger regularization from scratch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from emotion_recognition.models.signal_module import SignalModule
from emotion_recognition.utils.metrics import evaluate_classification


class WESADWindowDataset(Dataset):
    """Windowed WESAD dataset adaptor.

    Expected npz keys:
    - signals: shape (N, T, 6)
    - labels: shape (N,) mapped to 3 classes (stress/amusement/neutral)
    """

    def __init__(self, npz_path: str | Path) -> None:
        super().__init__()
        data = np.load(npz_path)
        self.signals = data["signals"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)

        if self.signals.ndim != 3 or self.signals.shape[-1] != 6:
            raise ValueError("signals must have shape (N, T, 6)")

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.signals[index])  # (T, 6)
        y = torch.tensor(self.labels[index], dtype=torch.long)  # ()
        return x, y


class SignalStage2Model(nn.Module):
    """Signal encoder + 3-way Stage 2 classifier."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.signal_module = SignalModule(channels=6)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sig_emb, _, _ = self.signal_module(x)  # (B, T, 6) -> (B, 256)
        log_probs = self.classifier(sig_emb)  # (B, 256) -> (B, 3)
        return log_probs


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)  # (B, T, 6)
        labels = labels.to(device, non_blocking=True)  # (B,)

        if train:
            optimizer.zero_grad(set_to_none=True)

        log_probs = model(signals)  # (B, T, 6) -> (B, 3)
        loss = criterion(log_probs, labels)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(float(loss.item()))
        preds = log_probs.argmax(dim=1)  # (B, 3) -> (B,)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = evaluate_classification(np.asarray(y_true), np.asarray(y_pred), num_classes=3)
    return float(np.mean(losses) if losses else 0.0), metrics


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 Signal pretraining on WESAD")
    parser.add_argument("--wesad-train-npz", type=str, default="")
    parser.add_argument("--wesad-val-npz", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="signal_stage2.pth")
    parser.add_argument("--device", type=str, default="mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.wesad_train_npz or not Path(args.wesad_train_npz).exists():
        print("WESAD train data unavailable. Skipping Stage 2 (train from scratch in Stage 3).")
        return

    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    train_ds = WESADWindowDataset(args.wesad_train_npz)
    val_ds = WESADWindowDataset(args.wesad_val_npz) if args.wesad_val_npz and Path(args.wesad_val_npz).exists() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if val_ds is not None
        else None
    )

    model = SignalStage2Model(num_classes=3).to(device)
    class_weights = compute_class_weights(train_ds.labels, num_classes=3).to(device)
    criterion = nn.NLLLoss(weight=class_weights)

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)

        if val_loader is not None:
            val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            val_f1 = float(val_metrics["macro_f1"])
            print(
                f"[Stage2][Epoch {epoch:03d}] train_loss={train_loss:.4f} train_f1={train_metrics['macro_f1']:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
            )
        else:
            val_f1 = float(train_metrics["macro_f1"])
            print(
                f"[Stage2][Epoch {epoch:03d}] train_loss={train_loss:.4f} train_f1={train_metrics['macro_f1']:.4f}"
            )

        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                "signal_module": model.signal_module.state_dict(),
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_state is None:
        best_state = {"signal_module": model.signal_module.state_dict()}

    torch.save(best_state, output_path)
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump({"best_val_macro_f1": best_f1}, f, indent=2)

    print(f"Saved Stage 2 weights to: {output_path}")


if __name__ == "__main__":
    main()
