"""Stage 1 training script: Face pretraining on FER2013 + CK+.

Backbone: InceptionResnetV1 (facenet-pytorch, vggface2)
Policy: freeze all except last inception block
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets

from emotion_recognition.models.facenet_backbone import FaceNetBackbone
from emotion_recognition.models.projection_head import ProjectionHead
from emotion_recognition.utils.metrics import evaluate_classification
from emotion_recognition.utils.preprocessing import build_video_transform


# Canonical Stage 1 class order follows FER folder labels:
# angry, disgust, fear, happy, neutral, sad, surprise
CKPLUS_TO_STAGE1 = {
    0: 0,  # anger
    1: 1,  # disgust
    2: 2,  # fear
    3: 3,  # happiness
    4: 5,  # sadness
    5: 6,  # surprise
    6: 4,  # neutral
}


class FaceStage1Model(nn.Module):
    """Backbone + projection + stage-1 classifier."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.backbone = FaceNetBackbone(pretrained="vggface2")
        self.backbone.set_stage1_policy()

        self.projection = ProjectionHead(input_dim=512, hidden_dim=256, output_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_512 = self.backbone(x)  # (B, 3, 160, 160) -> (B, 512)
        proj_128 = self.projection(feat_512)  # (B, 512) -> (B, 128)
        log_probs = self.classifier(proj_128)  # (B, 128) -> (B, 7)
        return log_probs


class CombinedImageFolder(Dataset):
    """Concatenate multiple ImageFolder datasets with shared transforms."""

    def __init__(self, roots: List[Path], transform) -> None:
        super().__init__()
        self.datasets: List[datasets.ImageFolder] = []
        self.index_map: List[Tuple[int, int]] = []
        for idx, root in enumerate(roots):
            ds = datasets.ImageFolder(str(root), transform=transform)
            self.datasets.append(ds)
            for local_i in range(len(ds)):
                self.index_map.append((idx, local_i))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int):
        ds_idx, local_idx = self.index_map[index]
        image, label = self.datasets[ds_idx][local_idx]
        return image, label


class CKPlusCSVDataset(Dataset):
    """CK+ CSV loader (emotion/pixels/Usage) with 7-class mapping.

    The source CSV may contain contempt (label=7), which is excluded.
    """

    def __init__(self, csv_path: str | Path, transform, split: str) -> None:
        super().__init__()
        df = pd.read_csv(csv_path)

        required = {"emotion", "pixels"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"CK CSV must include columns {required}")

        if "Usage" in df.columns:
            usage = df["Usage"].astype(str).str.strip().str.lower()
            if split == "train":
                df = df[usage == "training"]
            else:
                df = df[usage != "training"]

        df = df[df["emotion"].isin(list(CKPLUS_TO_STAGE1.keys()))].reset_index(drop=True)
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        src_label = int(row["emotion"])
        label = CKPLUS_TO_STAGE1[src_label]

        pixels = np.fromstring(str(row["pixels"]), dtype=np.uint8, sep=" ")
        if pixels.size != 48 * 48:
            raise ValueError(f"Invalid CK+ pixel length at index={index}: {pixels.size}")

        gray = pixels.reshape(48, 48)
        rgb = np.stack([gray, gray, gray], axis=-1)  # (48, 48) -> (48, 48, 3)
        image = self.transform(rgb)  # (48,48,3) -> (3,160,160)
        return image, label


def compute_class_weights(dataset: Dataset, num_classes: int = 7) -> torch.Tensor:
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y))
    labels_np = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)  # (N,) -> (7,)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)  # (B, 3, 160, 160)
        labels = labels.to(device, non_blocking=True)  # (B,)

        if train:
            optimizer.zero_grad(set_to_none=True)

        log_probs = model(images)  # (B, 3, 160, 160) -> (B, 7)
        loss = criterion(log_probs, labels)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(float(loss.item()))
        pred = log_probs.argmax(dim=1)  # (B, 7) -> (B,)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    metrics = evaluate_classification(np.asarray(y_true), np.asarray(y_pred), num_classes=7)
    return float(np.mean(losses) if losses else 0.0), metrics


def save_stage1_weights(model: FaceStage1Model, output_path: Path) -> None:
    """Save only backbone + projection head for Stage 3 transfer."""
    state = {
        "backbone": model.backbone.state_dict(),
        "projection_head": model.projection.state_dict(),
    }
    torch.save(state, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 Face pretraining on FER2013 + CK+")
    parser.add_argument("--dataset-root", type=str, default="Dataset")
    parser.add_argument("--fer-root", type=str, default="", help="Root folder for FER ImageFolder format")
    parser.add_argument("--ck-root", type=str, default="", help="Optional CK+ ImageFolder root")
    parser.add_argument("--ck-csv", type=str, default="", help="Optional CK+ CSV path (emotion/pixels/Usage)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="facenet_stage1.pth")
    parser.add_argument("--device", type=str, default="mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    transform_train = build_video_transform(train=True, stage=1)
    transform_eval = build_video_transform(train=False, stage=1)

    dataset_root = Path(args.dataset_root)
    fer_root = Path(args.fer_root) if args.fer_root else dataset_root / "FER"
    ck_csv_path = Path(args.ck_csv) if args.ck_csv else dataset_root / "CKPLUS" / "ckextended.csv"
    ck_root = Path(args.ck_root) if args.ck_root else None

    train_parts: List[Dataset] = []
    val_parts: List[Dataset] = []

    # FER folder mode: prefer explicit train/test subfolders.
    if fer_root.exists():
        fer_train_dir = fer_root / "train"
        fer_test_dir = fer_root / "test"

        if fer_train_dir.is_dir() and fer_test_dir.is_dir():
            train_parts.append(datasets.ImageFolder(str(fer_train_dir), transform=transform_train))
            val_parts.append(datasets.ImageFolder(str(fer_test_dir), transform=transform_eval))
        else:
            full_train_fer = CombinedImageFolder([fer_root], transform=transform_train)
            full_eval_fer = CombinedImageFolder([fer_root], transform=transform_eval)

            total = len(full_train_fer)
            val_count = int(round(total * float(args.val_split)))
            val_count = max(1, min(val_count, total - 1))

            rng = np.random.default_rng(seed=args.seed)
            indices = np.arange(total)
            rng.shuffle(indices)
            val_idx = indices[:val_count].tolist()
            train_idx = indices[val_count:].tolist()

            train_parts.append(Subset(full_train_fer, train_idx))
            val_parts.append(Subset(full_eval_fer, val_idx))

    # CK+ CSV mode (preferred for your downloaded dataset).
    if ck_csv_path.exists():
        train_parts.append(CKPlusCSVDataset(ck_csv_path, transform=transform_train, split="train"))
        val_parts.append(CKPlusCSVDataset(ck_csv_path, transform=transform_eval, split="eval"))
    elif ck_root is not None and ck_root.exists():
        full_train_ck = CombinedImageFolder([ck_root], transform=transform_train)
        full_eval_ck = CombinedImageFolder([ck_root], transform=transform_eval)

        total = len(full_train_ck)
        val_count = int(round(total * float(args.val_split)))
        val_count = max(1, min(val_count, total - 1))

        rng = np.random.default_rng(seed=args.seed + 1)
        indices = np.arange(total)
        rng.shuffle(indices)
        val_idx = indices[:val_count].tolist()
        train_idx = indices[val_count:].tolist()

        train_parts.append(Subset(full_train_ck, train_idx))
        val_parts.append(Subset(full_eval_ck, val_idx))

    if not train_parts or not val_parts:
        raise RuntimeError(
            "No Stage 1 datasets found. Provide FER root and/or CK CSV/root. "
            f"Checked FER={fer_root}, CK_CSV={ck_csv_path}."
        )

    train_dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_dataset = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = FaceStage1Model(num_classes=7).to(device)
    class_weights = compute_class_weights(train_dataset, num_classes=7).to(device)

    criterion = nn.NLLLoss(weight=class_weights)

    head_params = list(model.projection.parameters()) + list(model.classifier.parameters())
    backbone_params = model.backbone.trainable_parameters()

    optimizer = Adam(
        [
            {"params": head_params, "lr": 1e-3},
            {"params": backbone_params, "lr": 1e-5},
        ]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(
            f"[Stage1][Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            best_state = {
                "backbone": model.backbone.state_dict(),
                "projection_head": model.projection.state_dict(),
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        torch.save(best_state, output_path)
    else:
        save_stage1_weights(model, output_path)

    metrics_path = output_path.with_suffix(".json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"best_val_macro_f1": best_f1}, f, indent=2)

    print(f"Saved Stage 1 weights to: {output_path}")


if __name__ == "__main__":
    main()
