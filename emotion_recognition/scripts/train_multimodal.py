"""Stage 3 training script: Multimodal fine-tuning on NeuroBioSense.

Key features:
- Participant-level split (no clip leakage)
- Stage-specific freeze policy
- Weighted NLL with label smoothing
- Cosine annealing, grad clipping, and early stopping on macro-F1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from emotion_recognition.models.full_model import MultimodalEmotionModel
from emotion_recognition.utils.dataset import NeuroBioSenseDataset, build_neurobiosense_datasets
from emotion_recognition.utils.metrics import evaluate_classification

EMOTION7_CLASS_NAMES = ["J", "SA", "A", "D", "SU", "N", "F"]
VALENCE3_CLASS_NAMES = ["negative", "neutral", "positive"]
VALENCE2_CLASS_NAMES = ["negative", "positive"]

# Emotion ids from dataset.py: J=0, SA=1, A=2, D=3, SU=4, N=5, F=6
VALENCE3_MAP: Dict[int, int] = {
    0: 2,
    4: 2,
    5: 1,
    1: 0,
    2: 0,
    3: 0,
    6: 0,
}
VALENCE2_MAP: Dict[int, int] = {
    0: 1,
    4: 1,
    1: 0,
    2: 0,
    3: 0,
    6: 0,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RepeatDataset(Dataset):
    """Virtual dataset that repeats a base dataset for stronger on-the-fly augmentation."""

    def __init__(self, base: NeuroBioSenseDataset, repeats: int) -> None:
        self.base = base
        self.repeats = max(1, int(repeats))

    def __len__(self) -> int:
        return len(self.base) * self.repeats

    def __getitem__(self, index: int):
        return self.base[index % len(self.base)]


class LabelMappedDataset(Dataset):
    """Dataset view that remaps labels and optionally drops unmapped classes."""

    def __init__(self, base: Dataset, label_map: Dict[int, int]) -> None:
        self.base = base
        self.label_map = {int(k): int(v) for k, v in label_map.items()}
        self.indices: List[int] = []
        self.labels: List[int] = []

        raw_labels = extract_indexed_labels(base)
        for idx, lab in enumerate(raw_labels):
            if lab in self.label_map:
                self.indices.append(idx)
                self.labels.append(self.label_map[lab])

        if len(self.indices) == 0:
            raise RuntimeError("LabelMappedDataset produced 0 samples after mapping.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        base_idx = self.indices[index]
        video, signal, _ = self.base[base_idx]
        new_label = torch.tensor(self.labels[index], dtype=torch.long)
        return video, signal, new_label


def extract_indexed_labels(dataset: Dataset) -> List[int]:
    """Get indexed labels without decoding media tensors."""
    if hasattr(dataset, "labels"):
        return [int(x) for x in dataset.labels]

    if hasattr(dataset, "samples"):
        return [int(sample.label_id) for sample in dataset.samples]

    if hasattr(dataset, "base"):
        return extract_indexed_labels(dataset.base)

    raise TypeError(f"Unsupported dataset type for indexed labels: {type(dataset)!r}")


def build_task_datasets(
    train_ds: NeuroBioSenseDataset,
    val_ds: NeuroBioSenseDataset,
    test_ds: NeuroBioSenseDataset,
    task: str,
) -> Tuple[Dataset, Dataset, Dataset, int, List[str]]:
    """Create task-specific label mapping views."""
    if task == "valence3":
        mapped_train = LabelMappedDataset(train_ds, label_map=VALENCE3_MAP)
        mapped_val = LabelMappedDataset(val_ds, label_map=VALENCE3_MAP)
        mapped_test = LabelMappedDataset(test_ds, label_map=VALENCE3_MAP)
        return mapped_train, mapped_val, mapped_test, 3, VALENCE3_CLASS_NAMES

    if task == "valence2":
        mapped_train = LabelMappedDataset(train_ds, label_map=VALENCE2_MAP)
        mapped_val = LabelMappedDataset(val_ds, label_map=VALENCE2_MAP)
        mapped_test = LabelMappedDataset(test_ds, label_map=VALENCE2_MAP)
        return mapped_train, mapped_val, mapped_test, 2, VALENCE2_CLASS_NAMES

    return train_ds, val_ds, test_ds, 7, EMOTION7_CLASS_NAMES


class LabelSmoothingNLLLoss(nn.Module):
    """NLL-style loss with label smoothing over log-probabilities."""

    def __init__(self, class_weights: torch.Tensor | None = None, smoothing: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.smoothing = smoothing

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted smoothed NLL.

        Args:
            log_probs: Tensor of shape (B, C), already LogSoftmax'ed.
            target: Tensor of shape (B,).
        """
        bsz, num_classes = log_probs.shape
        smooth = self.smoothing

        with torch.no_grad():
            true_dist = torch.full_like(log_probs, fill_value=smooth / (num_classes - 1))  # (B, C)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smooth)  # (B, C)

        per_sample_loss = -(true_dist * log_probs).sum(dim=1)  # (B, C) -> (B,)

        if self.class_weights is not None:
            sample_weights = self.class_weights[target]  # (B,) -> (B,)
            per_sample_loss = per_sample_loss * sample_weights

        return per_sample_loss.mean()


class FocalNLLLoss(nn.Module):
    """Focal loss over log-probabilities with optional label smoothing."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
        smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = log_probs.size(1)

        # Base NLL term.
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (B, C) -> (B,)
        nll = -log_pt

        # Optional smoothing term in log-probability space.
        if self.smoothing > 0:
            smooth_loss = -log_probs.mean(dim=1)  # (B, C) -> (B,)
            nll = (1.0 - self.smoothing) * nll + self.smoothing * smooth_loss

        pt = torch.exp(log_pt).clamp_min(1e-9)  # (B,) -> (B,)
        focal_factor = torch.pow(1.0 - pt, self.gamma)
        per_sample_loss = focal_factor * nll

        if self.class_weights is not None:
            per_sample_loss = per_sample_loss * self.class_weights[target]

        return per_sample_loss.mean()


def compute_class_weights_from_dataset(dataset: Dataset, num_classes: int = 7) -> torch.Tensor:
    """Compute class weights from indexed labels without decoding media."""
    labels = extract_indexed_labels(dataset)
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def summarize_parameter_groups(model: MultimodalEmotionModel) -> Dict[str, int]:
    """Print and return frozen/trainable parameter summary by module."""
    groups = {
        "face_backbone": model.face_module.backbone,
        "projection_head": model.face_module.projection_head,
        "face_bilstm": model.face_module.temporal_bilstm,
        "face_attention": model.face_module.temporal_attention,
        "signal_channel_attention": model.signal_module.channel_attention,
        "signal_cnn": model.signal_module.cnn_blocks,
        "signal_bilstm": model.signal_module.bilstm,
        "signal_attention": model.signal_module.temporal_attention,
        "cross_modal_attention": model.cross_modal_attention,
        "fusion": model.fusion,
        "classifier": model.classifier,
    }

    summary: Dict[str, int] = {}
    print("\nParameter Group Summary (trainable/frozen):")
    print("-" * 72)
    for name, module in groups.items():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = total - trainable
        summary[f"{name}_trainable"] = int(trainable)
        summary[f"{name}_frozen"] = int(frozen)
        print(f"{name:28s} trainable={trainable:10d} frozen={frozen:10d}")
    print("-" * 72)

    total_all = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TOTAL                        trainable={total_trainable:10d} frozen={total_all - total_trainable:10d}\n")

    return summary


def build_optimizer(model: MultimodalEmotionModel) -> Adam:
    """Build optimizer with required LR policy per component."""
    groups = []

    # Optionally train a subset of FaceNet with very small LR.
    groups.append({"params": [p for p in model.face_module.backbone.parameters() if p.requires_grad], "lr": 1e-5})

    # Projection head trainable with lr=1e-3.
    groups.append({"params": [p for p in model.face_module.projection_head.parameters() if p.requires_grad], "lr": 1e-3})

    # Signal stack uses conservative LR when trainable.
    groups.append({"params": [p for p in model.signal_module.cnn_blocks.parameters() if p.requires_grad], "lr": 1e-4})
    groups.append({"params": [p for p in model.signal_module.bilstm.parameters() if p.requires_grad], "lr": 1e-4})
    groups.append({"params": [p for p in model.signal_module.temporal_attention.parameters() if p.requires_grad], "lr": 1e-4})

    # Optional trainable signal channel-attention also with conservative lr.
    groups.append({"params": [p for p in model.signal_module.channel_attention.parameters() if p.requires_grad], "lr": 1e-4})

    # Fusion and classifier stack lr=1e-3.
    groups.append({"params": [p for p in model.cross_modal_attention.parameters() if p.requires_grad], "lr": 1e-3})
    groups.append({"params": [p for p in model.fusion.parameters() if p.requires_grad], "lr": 1e-3})
    groups.append({"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": 1e-3})

    # Face temporal dynamics beyond projection also use lr=1e-3.
    groups.append({"params": [p for p in model.face_module.temporal_bilstm.parameters() if p.requires_grad], "lr": 1e-3})
    groups.append({"params": [p for p in model.face_module.temporal_attention.parameters() if p.requires_grad], "lr": 1e-3})

    filtered_groups = [g for g in groups if len(g["params"]) > 0]
    optimizer = Adam(filtered_groups)
    return optimizer


def run_epoch(
    model: MultimodalEmotionModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
    train: bool,
    num_classes: int,
    aggregation_mode: str = "mean",
    disable_face: bool = False,
    disable_signal: bool = False,
) -> Tuple[float, Dict[str, np.ndarray | float]]:
    model.train(mode=train)
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for video, signal, labels in loader:
        if train:
            video = video.to(device, non_blocking=True)  # (B, T_v, 3, 160, 160)
            signal = signal.to(device, non_blocking=True)  # (B, T_s, 6)
            if disable_signal:
                signal = torch.zeros_like(signal)
            labels = labels.to(device, non_blocking=True)  # (B,)

            optimizer.zero_grad(set_to_none=True)
            log_probs, _ = model(
                video,
                signal,
                use_face=not disable_face,
                use_signal=not disable_signal,
            )  # ((B,T_v,3,160,160),(B,T_s,6)) -> (B,7)
            loss = criterion(log_probs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            preds = log_probs.argmax(dim=1)  # (B, 7) -> (B,)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
        else:
            # Expected eval shapes with batch_size=1:
            # video: (1, N_w, T_v, 3, 160, 160), signal: (1, T_s, 6), labels: (1,)
            clip_windows = video.squeeze(0).to(device, non_blocking=True)  # (1,N_w,T_v,3,160,160) -> (N_w,T_v,3,160,160)
            clip_signal = signal.squeeze(0).to(device, non_blocking=True)  # (1,T_s,6) -> (T_s,6)
            clip_label = labels.squeeze(0).to(device, non_blocking=True)  # (1,) -> ()

            if disable_face:
                clip_windows = clip_windows[:1]

            n_w = clip_windows.size(0)
            signal_rep = clip_signal.unsqueeze(0).repeat(n_w, 1, 1)  # (T_s,6) -> (N_w,T_s,6)
            if disable_signal:
                signal_rep = torch.zeros_like(signal_rep)

            window_log_probs, _ = model(
                clip_windows,
                signal_rep,
                use_face=not disable_face,
                use_signal=not disable_signal,
            )  # ((N_w,T_v,3,160,160),(N_w,T_s,6)) -> (N_w,7)

            if aggregation_mode == "majority":
                votes = window_log_probs.argmax(dim=1)  # (N_w,7) -> (N_w,)
                counts = torch.bincount(votes, minlength=num_classes).float()  # (N_w,) -> (C,)
                agg_probs = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-9)  # (7,) -> (7,)
                agg_log_probs = torch.log(agg_probs).unsqueeze(0)  # (7,) -> (1,7)
            else:
                agg_probs = torch.exp(window_log_probs).mean(dim=0).clamp_min(1e-9)  # (N_w,7) -> (7,)
                agg_log_probs = torch.log(agg_probs).unsqueeze(0)  # (7,) -> (1,7)

            label_batch = clip_label.view(1)  # () -> (1,)
            loss = criterion(agg_log_probs, label_batch)
            losses.append(float(loss.item()))

            pred = agg_log_probs.argmax(dim=1)  # (1,7) -> (1,)
            y_true.append(int(label_batch.item()))
            y_pred.append(int(pred.item()))

    metrics = evaluate_classification(np.asarray(y_true), np.asarray(y_pred), num_classes=num_classes)
    mean_loss = float(np.mean(losses) if losses else 0.0)
    return mean_loss, metrics


def maybe_load_pretrained_weights(
    model: MultimodalEmotionModel,
    facenet_stage1: str | None,
    signal_stage2: str | None,
) -> None:
    if facenet_stage1 and Path(facenet_stage1).exists():
        payload = torch.load(facenet_stage1, map_location="cpu")
        if "backbone" in payload:
            model.face_module.backbone.load_state_dict(payload["backbone"], strict=False)
        if "projection_head" in payload:
            model.face_module.projection_head.load_state_dict(payload["projection_head"], strict=False)
        print(f"Loaded Stage 1 weights from: {facenet_stage1}")
    else:
        print("Stage 1 weights not found. Training with default FaceNet initialization.")

    if signal_stage2 and Path(signal_stage2).exists():
        payload = torch.load(signal_stage2, map_location="cpu")
        if "signal_module" in payload:
            model.signal_module.load_state_dict(payload["signal_module"], strict=False)
            print(f"Loaded Stage 2 weights from: {signal_stage2}")
    else:
        print("Stage 2 weights unavailable. Signal module will train from scratch/frozen policy as configured.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 multimodal fine-tuning on NeuroBioSense")
    parser.add_argument("--dataset-root", type=str, default="Dataset")
    parser.add_argument("--video-root", type=str, default="")
    parser.add_argument("--signal-csv", type=str, default="")
    parser.add_argument("--demographics-csv", type=str, default="")
    parser.add_argument("--facenet-stage1", type=str, default="facenet_stage1.pth")
    parser.add_argument("--signal-stage2", type=str, default="signal_stage2.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="multimodal_stage3.pth")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval-aggregation", type=str, choices=["mean", "majority"], default="mean")
    parser.add_argument(
        "--neuro-only",
        action="store_true",
        help="Train Stage 3 only on NeuroBioSense with stronger synthetic epoch expansion.",
    )
    parser.add_argument(
        "--augment-repeats",
        type=int,
        default=1,
        help="Repeat train dataset N times to increase augmented views per epoch.",
    )
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use WeightedRandomSampler to oversample minority classes.",
    )
    parser.add_argument(
        "--unfreeze-signal-cnn",
        action="store_true",
        help="Unfreeze signal CNN blocks (useful for Neuro-only training).",
    )
    parser.add_argument(
        "--unfreeze-face-last-block",
        action="store_true",
        help="Unfreeze last FaceNet inception blocks for domain adaptation.",
    )
    parser.add_argument(
        "--freeze-face-all",
        action="store_true",
        help="Freeze full face branch during Stage 3.",
    )
    parser.add_argument(
        "--disable-face",
        action="store_true",
        help="Bypass face branch and use zero face embeddings (signal-only ablation).",
    )
    parser.add_argument(
        "--freeze-signal-all",
        action="store_true",
        help="Freeze full signal branch during Stage 3.",
    )
    parser.add_argument(
        "--disable-signal",
        action="store_true",
        help="Set signal inputs to zeros during train/eval (face-priority ablation).",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["label_smoothing", "focal"],
        default="label_smoothing",
        help="Training loss type.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for supported losses.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss.",
    )
    parser.add_argument(
        "--sampler-max-ratio",
        type=float,
        default=4.0,
        help="Cap minority oversampling ratio when balanced sampler is enabled.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["emotion7", "valence3", "valence2"],
        default="emotion7",
        help="Target task: original 7-class emotion, 3-class valence, or binary valence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.neuro_only:
        args.facenet_stage1 = ""
        args.signal_stage2 = ""
        args.balanced_sampler = True
        args.unfreeze_signal_cnn = True
        args.unfreeze_face_last_block = True
        if args.augment_repeats < 4:
            args.augment_repeats = 4

    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset_root = Path(args.dataset_root)
    default_video_root = dataset_root / "NeuroBioSense Dataset" / "NeuroBioSense" / "Advertisement Categories"
    default_signal_csv = dataset_root / "NeuroBioSense Dataset" / "NeuroBioSense" / "Biosignal Files" / "Pre-Processed" / "32-Hertz.csv"
    default_demo_xlsx = dataset_root / "NeuroBioSense Dataset" / "NeuroBioSense" / "Participant Data" / "Participant_demographic_information.xlsx"

    video_root = Path(args.video_root) if args.video_root else default_video_root
    signal_csv_path = Path(args.signal_csv) if args.signal_csv else default_signal_csv

    if args.demographics_csv:
        demographics_path = Path(args.demographics_csv)
    else:
        demographics_path = default_demo_xlsx if default_demo_xlsx.exists() else None

    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")
    if not signal_csv_path.exists():
        raise FileNotFoundError(f"Signal CSV not found: {signal_csv_path}")

    train_ds, val_ds, test_ds, stats = build_neurobiosense_datasets(
        video_root=video_root,
        signal_csv_path=signal_csv_path,
        demographics_csv_path=demographics_path,
        stage=3,
        t_v=10,
        t_s=128,
        seed=args.seed,
    )

    train_ds, val_ds, test_ds, num_classes, class_names = build_task_datasets(train_ds, val_ds, test_ds, task=args.task)

    print(f"Task: {args.task} | num_classes={num_classes} | classes={class_names}")

    train_data: Dataset = train_ds
    if args.augment_repeats > 1:
        train_data = RepeatDataset(train_ds, repeats=args.augment_repeats)

    train_sampler = None
    train_shuffle = True
    if args.balanced_sampler:
        base_labels = np.asarray(extract_indexed_labels(train_ds), dtype=np.int64)
        class_counts = np.bincount(base_labels, minlength=num_classes).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        ratio = class_counts.max() / class_counts
        ratio = np.clip(ratio, 1.0, float(args.sampler_max_ratio))

        repeats = max(1, len(train_data) // len(train_ds))
        repeated_labels = np.tile(base_labels, repeats)
        if len(repeated_labels) < len(train_data):
            extra = base_labels[: len(train_data) - len(repeated_labels)]
            repeated_labels = np.concatenate([repeated_labels, extra], axis=0)

        sample_weights = ratio[repeated_labels]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_data),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MultimodalEmotionModel(num_classes=num_classes).to(device)
    maybe_load_pretrained_weights(model, args.facenet_stage1, args.signal_stage2)

    # Stage-specific freezing policy.
    if args.disable_face or args.freeze_face_all:
        for p in model.face_module.parameters():
            p.requires_grad = False
    elif args.neuro_only:
        # Start from fully trainable then selectively freeze for stability.
        for p in model.parameters():
            p.requires_grad = True

        # FaceNet remains mostly frozen; optionally adapt the top representation block.
        model.face_module.backbone.freeze_all()
    else:
        model.apply_stage3_freezing()

    if args.unfreeze_face_last_block and not (args.disable_face or args.freeze_face_all):
        model.face_module.backbone.unfreeze_last_inception_block()

    if args.freeze_signal_all or args.disable_signal:
        for p in model.signal_module.parameters():
            p.requires_grad = False
    elif args.unfreeze_signal_cnn:
        for p in model.signal_module.cnn_blocks.parameters():
            p.requires_grad = True
    else:
        model.signal_module.freeze_cnn_blocks()

    summary = summarize_parameter_groups(model)

    class_weights = compute_class_weights_from_dataset(train_ds, num_classes=num_classes).to(device)
    # Avoid double re-weighting when oversampling is already active.
    criterion_weights = None if args.balanced_sampler else class_weights
    if args.loss_type == "focal":
        criterion = FocalNLLLoss(
            class_weights=criterion_weights,
            gamma=args.focal_gamma,
            smoothing=args.label_smoothing,
        )
    else:
        criterion = LabelSmoothingNLLLoss(class_weights=criterion_weights, smoothing=args.label_smoothing)

    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            num_classes=num_classes,
            aggregation_mode=args.eval_aggregation,
            disable_face=args.disable_face,
            disable_signal=args.disable_signal,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            num_classes=num_classes,
            aggregation_mode=args.eval_aggregation,
            disable_face=args.disable_face,
            disable_signal=args.disable_signal,
        )
        scheduler.step()

        print(
            f"[Stage3][Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        val_f1 = float(val_metrics["macro_f1"])
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_without_improve = 0
            best_state = {
                "model": model.state_dict(),
                "normalization_stats": stats.to_dict(),
                "param_summary": summary,
            }
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={args.patience}).")
            break

    if best_state is None:
        best_state = {
            "model": model.state_dict(),
            "normalization_stats": stats.to_dict(),
            "param_summary": summary,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)

    # Evaluate best checkpoint on test split.
    model.load_state_dict(best_state["model"])
    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer,
        device,
        train=False,
        num_classes=num_classes,
        aggregation_mode=args.eval_aggregation,
        disable_face=args.disable_face,
        disable_signal=args.disable_signal,
    )

    report = {
        "task": args.task,
        "class_names": class_names,
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_overall_acc": float(test_metrics["overall_acc"]),
        "test_per_class_acc": test_metrics["per_class_acc"].tolist(),
        "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved Stage 3 model to: {output_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
