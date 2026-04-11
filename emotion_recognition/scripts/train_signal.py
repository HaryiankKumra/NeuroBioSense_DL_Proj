"""Stage 2 training script: Signal pretraining on WESAD or scratch fallback.

If WESAD is unavailable, this script can be skipped and Stage 3 will train the
signal branch with stronger regularization from scratch.
"""

from __future__ import annotations

import argparse
import json
import pickle
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


# Raw WESAD labels: 1=baseline, 2=stress, 3=amusement.
# Stage-2 class order requested: stress, amusement, neutral.
WESAD_TO_STAGE2 = {
    2: 0,  # stress
    3: 1,  # amusement
    1: 2,  # neutral/baseline
}


def _resample_vector(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == target_len:
        return x
    src_idx = np.linspace(0, len(x) - 1, num=len(x), dtype=np.float32)
    dst_idx = np.linspace(0, len(x) - 1, num=target_len, dtype=np.float32)
    y = np.interp(dst_idx, src_idx, x).astype(np.float32)
    return y


def _resample_labels_nearest(labels: np.ndarray, target_len: int) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if len(labels) == target_len:
        return labels.astype(np.int64)
    src_idx = np.arange(len(labels), dtype=np.float32)
    dst_idx = np.linspace(0, len(labels) - 1, num=target_len, dtype=np.float32)
    nearest = np.clip(np.rint(dst_idx).astype(np.int64), 0, len(labels) - 1)
    return labels[nearest].astype(np.int64)


def _load_subject_signal_32hz(subject_pkl: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load WESAD subject and resample wrist channels to 32 Hz timeline.

    Returns:
        signal_32hz: (T, 6) with [BVP, EDA, TEMP, ACC_X, ACC_Y, ACC_Z]
        labels_32hz: (T,) raw WESAD labels
    """
    with open(subject_pkl, "rb") as f:
        payload = pickle.load(f, encoding="latin1")

    wrist = payload["signal"]["wrist"]
    acc = np.asarray(wrist["ACC"], dtype=np.float32)  # (T32, 3)
    n32 = acc.shape[0]

    bvp = _resample_vector(np.asarray(wrist["BVP"], dtype=np.float32).reshape(-1), target_len=n32)  # (T64,) -> (T32,)
    eda = _resample_vector(np.asarray(wrist["EDA"], dtype=np.float32).reshape(-1), target_len=n32)  # (T4,) -> (T32,)
    temp = _resample_vector(np.asarray(wrist["TEMP"], dtype=np.float32).reshape(-1), target_len=n32)  # (T4,) -> (T32,)

    label_raw = np.asarray(payload["label"], dtype=np.int64).reshape(-1)  # (~700Hz,)
    labels_32hz = _resample_labels_nearest(label_raw, target_len=n32)  # (~700Hz,) -> (T32,)

    signal = np.column_stack([bvp, eda, temp, acc[:, 0], acc[:, 1], acc[:, 2]]).astype(np.float32)  # -> (T32, 6)
    return signal, labels_32hz


def _window_subject(
    signal: np.ndarray,
    labels: np.ndarray,
    window_size: int = 160,
    stride: int = 80,
    min_majority_ratio: float = 0.8,
) -> Tuple[List[np.ndarray], List[int]]:
    """Create 5s windows and map labels to Stage-2 classes."""
    x_list: List[np.ndarray] = []
    y_list: List[int] = []

    n = signal.shape[0]
    for start in range(0, max(1, n - window_size + 1), stride):
        end = start + window_size
        if end > n:
            break

        x_win = signal[start:end]  # (160, 6)
        y_win_raw = labels[start:end]  # (160,)

        valid_mask = np.isin(y_win_raw, list(WESAD_TO_STAGE2.keys()))
        if valid_mask.mean() < min_majority_ratio:
            continue

        valid_labels = y_win_raw[valid_mask]
        raw_majority = int(np.bincount(valid_labels).argmax())
        mapped = WESAD_TO_STAGE2[raw_majority]

        x_list.append(x_win.astype(np.float32))
        y_list.append(mapped)

    return x_list, y_list


def prepare_wesad_npz(
    wesad_root: str | Path,
    train_npz: str | Path,
    val_npz: str | Path,
    seed: int = 42,
) -> Tuple[int, int]:
    """Prepare windowed Stage-2 training/validation NPZ files from raw WESAD."""
    rng = np.random.default_rng(seed)
    root = Path(wesad_root)
    subject_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.upper().startswith("S")])
    if not subject_dirs:
        raise RuntimeError(f"No WESAD subject folders found under: {root}")

    subject_ids = np.array([p.name for p in subject_dirs])
    rng.shuffle(subject_ids)

    split = max(1, int(round(0.2 * len(subject_ids))))
    val_subjects = set(subject_ids[:split].tolist())
    train_subjects = set(subject_ids[split:].tolist())

    x_train: List[np.ndarray] = []
    y_train: List[int] = []
    x_val: List[np.ndarray] = []
    y_val: List[int] = []

    for subject_dir in subject_dirs:
        pkl_path = subject_dir / f"{subject_dir.name}.pkl"
        if not pkl_path.exists():
            continue

        signal, labels = _load_subject_signal_32hz(pkl_path)
        x_list, y_list = _window_subject(signal, labels, window_size=160, stride=80)

        if subject_dir.name in val_subjects:
            x_val.extend(x_list)
            y_val.extend(y_list)
        else:
            x_train.extend(x_list)
            y_train.extend(y_list)

    if not x_train:
        raise RuntimeError("No train windows generated from WESAD.")
    if not x_val:
        raise RuntimeError("No validation windows generated from WESAD.")

    train_npz = Path(train_npz)
    val_npz = Path(val_npz)
    train_npz.parent.mkdir(parents=True, exist_ok=True)
    val_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        train_npz,
        signals=np.stack(x_train, axis=0).astype(np.float32),  # (N, 160, 6)
        labels=np.asarray(y_train, dtype=np.int64),  # (N,)
    )
    np.savez_compressed(
        val_npz,
        signals=np.stack(x_val, axis=0).astype(np.float32),  # (N, 160, 6)
        labels=np.asarray(y_val, dtype=np.int64),  # (N,)
    )

    return len(x_train), len(x_val)


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
    parser.add_argument("--dataset-root", type=str, default="Dataset")
    parser.add_argument("--wesad-root", type=str, default="")
    parser.add_argument("--wesad-train-npz", type=str, default="")
    parser.add_argument("--wesad-val-npz", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="signal_stage2.pth")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    wesad_root = Path(args.wesad_root) if args.wesad_root else dataset_root / "WESAD"

    train_npz_path = Path(args.wesad_train_npz) if args.wesad_train_npz else Path("artifacts/wesad_train.npz")
    val_npz_path = Path(args.wesad_val_npz) if args.wesad_val_npz else Path("artifacts/wesad_val.npz")

    if not train_npz_path.exists() or not val_npz_path.exists():
        if not wesad_root.exists():
            print("WESAD root unavailable and NPZ files missing. Skipping Stage 2.")
            return

        print("Preparing WESAD NPZ windows from raw subject files...")
        n_train, n_val = prepare_wesad_npz(
            wesad_root=wesad_root,
            train_npz=train_npz_path,
            val_npz=val_npz_path,
            seed=args.seed,
        )
        print(f"Prepared WESAD windows: train={n_train}, val={n_val}")

    if args.prepare_only:
        print("Preparation complete (--prepare-only).")
        return

    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    train_ds = WESADWindowDataset(train_npz_path)
    val_ds = WESADWindowDataset(val_npz_path) if val_npz_path.exists() else None

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
