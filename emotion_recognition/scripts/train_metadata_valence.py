"""Metadata-assisted binary valence training.

This script trains a lightweight classifier using ad-level metadata features:
- ad_code
- advertisement category

It is designed as a practical fallback when strict physiological alignment is weak.

Data augmentation used here:
- minority-class random oversampling on the training split

Outputs:
- artifacts/final_valence_metadata.json (metrics)
- artifacts/final_valence_metadata.pkl (sklearn pipeline)
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from emotion_recognition.scripts.train_multimodal import VALENCE2_MAP
from emotion_recognition.utils.dataset import ClipSample, scan_video_samples, split_participants
from emotion_recognition.utils.metrics import evaluate_classification


@dataclass
class MetadataRow:
    participant_id: str
    ad_code: str
    category: str
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train metadata-assisted binary valence baseline")
    parser.add_argument("--dataset-root", type=str, default="Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oversample-minority", action="store_true", help="Apply random oversampling to minority class")
    parser.add_argument("--skip-val-tuning", action="store_true", help="Use provided hyperparameters directly")
    parser.add_argument("--c", type=float, default=1.0, help="LogisticRegression C used when --skip-val-tuning")
    parser.add_argument(
        "--class-weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Class weighting used when --skip-val-tuning",
    )
    parser.add_argument("--output-json", type=str, default="artifacts/final_valence_metadata.json")
    parser.add_argument("--output-model", type=str, default="artifacts/final_valence_metadata.pkl")
    return parser.parse_args()


def _rows_from_samples(samples: Sequence[ClipSample]) -> List[MetadataRow]:
    rows: List[MetadataRow] = []
    for s in samples:
        if s.label_id not in VALENCE2_MAP:
            continue
        rows.append(
            MetadataRow(
                participant_id=s.participant_id,
                ad_code=s.ad_code,
                category=s.category,
                label=int(VALENCE2_MAP[s.label_id]),
            )
        )
    return rows


def _split_rows(
    rows: Sequence[MetadataRow],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> Tuple[List[MetadataRow], List[MetadataRow], List[MetadataRow]]:
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    train_rows = [r for r in rows if r.participant_id in train_set]
    val_rows = [r for r in rows if r.participant_id in val_set]
    test_rows = [r for r in rows if r.participant_id in test_set]
    return train_rows, val_rows, test_rows


def _to_xy(rows: Sequence[MetadataRow]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[r.ad_code, r.category] for r in rows], dtype=object)
    y = np.asarray([r.label for r in rows], dtype=np.int64)
    return x, y


def _oversample_minority(x: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return x, y

    max_count = int(counts.max())
    idx_all = np.arange(len(y), dtype=np.int64)

    aug_idx: List[np.ndarray] = []
    for cls in classes:
        cls_idx = idx_all[y == cls]
        if len(cls_idx) == 0:
            continue
        if len(cls_idx) < max_count:
            sampled = rng.choice(cls_idx, size=max_count - len(cls_idx), replace=True)
            cls_idx = np.concatenate([cls_idx, sampled], axis=0)
        aug_idx.append(cls_idx)

    full_idx = np.concatenate(aug_idx, axis=0)
    rng.shuffle(full_idx)
    return x[full_idx], y[full_idx]


def _build_pipeline(c: float, class_weight: str | None) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), [0, 1]),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=1000, C=float(c), class_weight=class_weight)
    return Pipeline([("pre", pre), ("lr", clf)])


def _fit_with_val_selection(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[float, str | None]:
    candidates_c = [0.1, 0.3, 1.0, 3.0, 10.0]
    candidates_w: List[str | None] = [None, "balanced"]

    best = None
    for c in candidates_c:
        for cw in candidates_w:
            model = _build_pipeline(c=c, class_weight=cw)
            model.fit(x_train, y_train)
            pred_val = model.predict(x_val)
            acc = float((pred_val == y_val).mean())
            if best is None or acc > best[0]:
                best = (acc, c, cw)

    assert best is not None
    _, best_c, best_cw = best
    return float(best_c), best_cw


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    video_root = dataset_root / "NeuroBioSense Dataset" / "NeuroBioSense" / "Advertisement Categories"

    all_samples = scan_video_samples(video_root)
    train_ids, val_ids, test_ids = split_participants(all_samples, test_size=0.15, val_size=0.15, seed=args.seed)

    rows = _rows_from_samples(all_samples)
    train_rows, val_rows, test_rows = _split_rows(rows, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    x_train, y_train = _to_xy(train_rows)
    x_val, y_val = _to_xy(val_rows)
    x_test, y_test = _to_xy(test_rows)

    if args.oversample_minority:
        x_train_aug, y_train_aug = _oversample_minority(x_train, y_train, seed=args.seed)
    else:
        x_train_aug, y_train_aug = x_train, y_train

    if args.skip_val_tuning:
        best_c = float(args.c)
        best_cw: str | None = None if args.class_weight == "none" else "balanced"
    else:
        best_c, best_cw = _fit_with_val_selection(
            x_train=x_train_aug,
            y_train=y_train_aug,
            x_val=x_val,
            y_val=y_val,
        )

    # Final fit on train+val using selected hyperparameters.
    x_trainval = np.concatenate([x_train, x_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    if args.oversample_minority:
        x_trainval, y_trainval = _oversample_minority(x_trainval, y_trainval, seed=args.seed + 1)

    model = _build_pipeline(c=best_c, class_weight=best_cw)
    model.fit(x_trainval, y_trainval)

    pred_test = model.predict(x_test).astype(np.int64)
    metrics = evaluate_classification(y_test.astype(np.int64), pred_test.astype(np.int64), num_classes=2)

    report = {
        "task": "valence2",
        "model": "metadata_logistic",
        "features": ["ad_code", "category"],
        "augmentation": {
            "oversample_minority": bool(args.oversample_minority),
        },
        "seed": int(args.seed),
        "selected_hyperparams": {
            "C": float(best_c),
            "class_weight": best_cw,
        },
        "test_overall_acc": float(metrics["overall_acc"]),
        "test_macro_f1": float(metrics["macro_f1"]),
        "test_per_class_acc": metrics["per_class_acc"].tolist(),
        "test_confusion_matrix": metrics["confusion_matrix"].tolist(),
    }

    output_json = Path(args.output_json)
    output_model = Path(args.output_model)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with output_model.open("wb") as f:
        pickle.dump(model, f)

    print(json.dumps(report, indent=2))
    print(f"Saved report to {output_json}")
    print(f"Saved model to {output_model}")


if __name__ == "__main__":
    main()
