"""Utility package for data preprocessing, dataset building, and metrics."""

from .dataset import EMOTION_TO_ID, NeuroBioSenseDataset, build_neurobiosense_datasets
from .metrics import confusion_matrix_np, evaluate_classification, macro_f1_score_np, per_class_accuracy_np

__all__ = [
    "EMOTION_TO_ID",
    "NeuroBioSenseDataset",
    "build_neurobiosense_datasets",
    "macro_f1_score_np",
    "per_class_accuracy_np",
    "confusion_matrix_np",
    "evaluate_classification",
]
