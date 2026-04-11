"""Evaluation metrics for imbalanced 7-class emotion classification."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> np.ndarray:
    """Compute confusion matrix using NumPy.

    Args:
        y_true: Ground-truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        num_classes: Number of classes.

    Returns:
        Confusion matrix, shape (num_classes, num_classes).
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_accuracy_np(cm: np.ndarray) -> np.ndarray:
    """Per-class accuracy from confusion matrix.

    Args:
        cm: Confusion matrix, shape (C, C).

    Returns:
        Accuracy vector, shape (C,).
    """
    tp = np.diag(cm).astype(np.float64)  # (C, C) -> (C,)
    denom = cm.sum(axis=1).astype(np.float64)  # (C, C) -> (C,)
    acc = np.divide(tp, np.maximum(denom, 1.0))  # (C,) / (C,) -> (C,)
    return acc


def macro_f1_score_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> float:
    """Macro F1-score without external dependencies."""
    cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)
    tp = np.diag(cm).astype(np.float64)  # (C, C) -> (C,)
    fp = cm.sum(axis=0).astype(np.float64) - tp  # (C,) -> (C,)
    fn = cm.sum(axis=1).astype(np.float64) - tp  # (C,) -> (C,)

    precision = np.divide(tp, np.maximum(tp + fp, 1.0))  # (C,) -> (C,)
    recall = np.divide(tp, np.maximum(tp + fn, 1.0))  # (C,) -> (C,)
    f1 = np.divide(2.0 * precision * recall, np.maximum(precision + recall, 1e-12))  # (C,) -> (C,)
    return float(np.mean(f1))


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> Dict[str, np.ndarray | float]:
    """Return standard evaluation bundle."""
    cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)
    per_class_acc = per_class_accuracy_np(cm)
    macro_f1 = macro_f1_score_np(y_true, y_pred, num_classes=num_classes)
    overall_acc = float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0

    return {
        "macro_f1": macro_f1,
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
    }
