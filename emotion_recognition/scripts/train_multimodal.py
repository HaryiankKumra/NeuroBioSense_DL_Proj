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
from torch.utils.data import DataLoader

from emotion_recognition.models.full_model import MultimodalEmotionModel
from emotion_recognition.utils.dataset import build_neurobiosense_datasets
from emotion_recognition.utils.metrics import evaluate_classification


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def compute_class_weights_from_loader(loader: DataLoader, num_classes: int = 7) -> torch.Tensor:
    labels: List[int] = []
    for _, _, y in loader:
        labels.extend(y.numpy().astype(np.int64).tolist())
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

    # Projection head trainable with lr=1e-3.
    groups.append({"params": [p for p in model.face_module.projection_head.parameters() if p.requires_grad], "lr": 1e-3})

    # Signal BiLSTM + temporal attention lr=1e-4.
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
    aggregation_mode: str = "mean",
) -> Tuple[float, Dict[str, np.ndarray | float]]:
    model.train(mode=train)
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for video, signal, labels in loader:
        if train:
            video = video.to(device, non_blocking=True)  # (B, T_v, 3, 160, 160)
            signal = signal.to(device, non_blocking=True)  # (B, T_s, 6)
            labels = labels.to(device, non_blocking=True)  # (B,)

            optimizer.zero_grad(set_to_none=True)
            log_probs, _ = model(video, signal)  # ((B,T_v,3,160,160),(B,T_s,6)) -> (B,7)
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

            n_w = clip_windows.size(0)
            signal_rep = clip_signal.unsqueeze(0).repeat(n_w, 1, 1)  # (T_s,6) -> (N_w,T_s,6)

            window_log_probs, _ = model(clip_windows, signal_rep)  # ((N_w,T_v,3,160,160),(N_w,T_s,6)) -> (N_w,7)

            if aggregation_mode == "majority":
                votes = window_log_probs.argmax(dim=1)  # (N_w,7) -> (N_w,)
                counts = torch.bincount(votes, minlength=7).float()  # (N_w,) -> (7,)
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

    metrics = evaluate_classification(np.asarray(y_true), np.asarray(y_pred), num_classes=7)
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
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--signal-csv", type=str, required=True)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    train_ds, val_ds, test_ds, stats = build_neurobiosense_datasets(
        video_root=args.video_root,
        signal_csv_path=args.signal_csv,
        demographics_csv_path=args.demographics_csv if args.demographics_csv else None,
        stage=3,
        t_v=10,
        t_s=128,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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

    model = MultimodalEmotionModel(num_classes=7).to(device)
    maybe_load_pretrained_weights(model, args.facenet_stage1, args.signal_stage2)

    # Apply mandatory Stage 3 freezing policy.
    model.apply_stage3_freezing()
    summary = summarize_parameter_groups(model)

    class_weights = compute_class_weights_from_loader(train_loader, num_classes=7).to(device)
    criterion = LabelSmoothingNLLLoss(class_weights=class_weights, smoothing=0.1)

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
            aggregation_mode=args.eval_aggregation,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            aggregation_mode=args.eval_aggregation,
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
        aggregation_mode=args.eval_aggregation,
    )

    report = {
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
