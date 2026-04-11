"""PyTorch dataset for NeuroBioSense multimodal emotion recognition.

This dataset scans the required video structure:
{participant_id}/{ad_code}/{emotion_label}/{clip_id}.MP4
and aligns each clip with physiological rows from 32-Hertz.csv.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .preprocessing import (
    load_video_tensor,
    make_sliding_windows,
    sample_training_window,
)
from .signal_processing import (
    SIGNAL_COLUMNS,
    SignalNormalizationStats,
    augment_signal,
    extract_signal_segment,
    fit_signal_normalizer,
    infer_id_columns,
    load_32hz_csv,
    load_duration_mapping,
    lookup_duration,
    normalize_signal_np,
    resample_signal_to_fixed_length,
)

EMOTION_TO_ID: Dict[str, int] = {
    "J": 0,
    "SA": 1,
    "A": 2,
    "D": 3,
    "SU": 4,
    "N": 5,
    "F": 6,
}

ID_TO_EMOTION: Dict[int, str] = {v: k for k, v in EMOTION_TO_ID.items()}


@dataclass
class ClipSample:
    """Index entry for a multimodal sample."""

    video_path: Path
    participant_id: str
    ad_code: str
    emotion_code: str
    label_id: int


class NeuroBioSenseDataset(Dataset):
    """Dataset returning aligned multimodal tensors.

    Train split:
    - video tensor shape (T_v, 3, 160, 160) from one random window

    Val/test split:
    - video tensor shape (N_w, T_v, 3, 160, 160) with all windows
    """

    def __init__(
        self,
        samples: Sequence[ClipSample],
        signal_df: pd.DataFrame,
        signal_id_columns: Dict[str, str],
        signal_stats: SignalNormalizationStats,
        duration_df: Optional[pd.DataFrame],
        split: str,
        stage: int = 3,
        t_v: int = 10,
        t_s: int = 128,
        train: bool = False,
        every_n: int = 4,
        stride: int = 5,
        temporal_jitter: bool = True,
    ) -> None:
        super().__init__()
        self.samples = list(samples)
        self.signal_df = signal_df
        self.signal_id_columns = signal_id_columns
        self.signal_stats = signal_stats
        self.duration_df = duration_df
        self.split = split
        self.stage = stage
        self.t_v = t_v
        self.t_s = t_s
        self.train = train
        self.every_n = every_n
        self.stride = stride
        self.temporal_jitter = temporal_jitter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        # Load sampled video frames.
        frames_tensor, raw_duration_sec = load_video_tensor(
            sample.video_path,
            every_n=self.every_n,
            train=self.train,
            stage=self.stage,
            temporal_jitter=self.train and self.temporal_jitter,
        )
        # (T, 3, 160, 160)

        if self.train:
            video_window = sample_training_window(frames_tensor, window_size=self.t_v, stride=self.stride)
            # (T, 3, 160, 160) -> (T_v, 3, 160, 160)
        else:
            windows = make_sliding_windows(frames_tensor, window_size=self.t_v, stride=self.stride)
            video_window = windows
            # (N_w, T_v, 3, 160, 160) -> (N_w, T_v, 3, 160, 160)

        duration_sec = lookup_duration(
            self.duration_df,
            participant_id=sample.participant_id,
            ad_code=sample.ad_code,
            fallback_duration_sec=raw_duration_sec,
        )

        sig_segment = extract_signal_segment(
            self.signal_df,
            participant_id=sample.participant_id,
            ad_code=sample.ad_code,
            duration_sec=duration_sec,
            id_columns=self.signal_id_columns,
        )
        # (?, 6) aligned to clip duration

        sig_norm = normalize_signal_np(sig_segment, self.signal_stats)  # (?, 6) -> (?, 6)
        sig_resampled = resample_signal_to_fixed_length(sig_norm, target_length=self.t_s)  # (?,6) -> (T_s,6)

        if self.train and self.stage == 3:
            sig_resampled = augment_signal(sig_resampled)  # (T_s, 6) -> (T_s, 6)

        signal_tensor = torch.from_numpy(sig_resampled).float()  # (T_s, 6)
        label_tensor = torch.tensor(sample.label_id, dtype=torch.long)  # ()

        return video_window, signal_tensor, label_tensor


def scan_video_samples(video_root: str | Path) -> List[ClipSample]:
    """Scan NeuroBioSense clip hierarchy and build sample index."""
    root = Path(video_root)
    if not root.exists():
        raise FileNotFoundError(f"Video root not found: {root}")

    samples: List[ClipSample] = []
    for participant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        participant_id = participant_dir.name

        for ad_dir in sorted(p for p in participant_dir.iterdir() if p.is_dir()):
            ad_code = ad_dir.name

            for emotion_dir in sorted(p for p in ad_dir.iterdir() if p.is_dir()):
                emotion_code = emotion_dir.name
                if emotion_code not in EMOTION_TO_ID:
                    continue
                label_id = EMOTION_TO_ID[emotion_code]

                # WHY case-insensitive MP4 pattern: source archives may mix .MP4/.mp4.
                clip_paths = sorted(
                    [
                        p
                        for p in emotion_dir.glob("*")
                        if p.is_file() and p.suffix.lower() == ".mp4"
                    ]
                )
                for clip_path in clip_paths:
                    samples.append(
                        ClipSample(
                            video_path=clip_path,
                            participant_id=participant_id,
                            ad_code=ad_code,
                            emotion_code=emotion_code,
                            label_id=label_id,
                        )
                    )

    if not samples:
        raise RuntimeError(f"No MP4 samples found in hierarchy under: {root}")
    return samples


def split_participants(
    samples: Sequence[ClipSample],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split participants into train/val/test without clip leakage."""
    participants = sorted({s.participant_id for s in samples})
    if len(participants) < 3:
        raise ValueError("Need at least 3 participants for train/val/test split.")

    train_ids, test_ids = train_test_split(participants, test_size=test_size, random_state=seed)

    # Adjust validation proportion relative to remaining train pool.
    rel_val_size = val_size / max(1e-8, (1.0 - test_size))
    train_ids, val_ids = train_test_split(train_ids, test_size=rel_val_size, random_state=seed)

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def _subset_by_participants(samples: Sequence[ClipSample], participants: Sequence[str]) -> List[ClipSample]:
    allowed = set(str(p) for p in participants)
    return [s for s in samples if s.participant_id in allowed]


def _build_signal_train_mask(
    signal_df: pd.DataFrame,
    participant_col: str,
    train_participants: Sequence[str],
) -> np.ndarray:
    train_set = set(str(pid) for pid in train_participants)
    mask = signal_df[participant_col].astype(str).isin(train_set).to_numpy(dtype=bool)
    return mask


def build_neurobiosense_datasets(
    video_root: str | Path,
    signal_csv_path: str | Path,
    demographics_csv_path: str | Path | None = None,
    stage: int = 3,
    t_v: int = 10,
    t_s: int = 128,
    seed: int = 42,
) -> Tuple[NeuroBioSenseDataset, NeuroBioSenseDataset, NeuroBioSenseDataset, SignalNormalizationStats]:
    """Factory for train/val/test datasets with participant-level split."""
    all_samples = scan_video_samples(video_root)
    train_ids, val_ids, test_ids = split_participants(all_samples, test_size=0.15, val_size=0.15, seed=seed)

    train_samples = _subset_by_participants(all_samples, train_ids)
    val_samples = _subset_by_participants(all_samples, val_ids)
    test_samples = _subset_by_participants(all_samples, test_ids)

    signal_df = load_32hz_csv(signal_csv_path)
    id_cols = infer_id_columns(signal_df)

    train_mask = _build_signal_train_mask(signal_df, id_cols["participant"], train_ids)
    stats = fit_signal_normalizer(signal_df, train_mask=train_mask)

    duration_df = None
    if demographics_csv_path is not None and Path(demographics_csv_path).exists():
        duration_df = load_duration_mapping(demographics_csv_path)

    train_ds = NeuroBioSenseDataset(
        samples=train_samples,
        signal_df=signal_df,
        signal_id_columns=id_cols,
        signal_stats=stats,
        duration_df=duration_df,
        split="train",
        stage=stage,
        t_v=t_v,
        t_s=t_s,
        train=True,
    )
    val_ds = NeuroBioSenseDataset(
        samples=val_samples,
        signal_df=signal_df,
        signal_id_columns=id_cols,
        signal_stats=stats,
        duration_df=duration_df,
        split="val",
        stage=stage,
        t_v=t_v,
        t_s=t_s,
        train=False,
    )
    test_ds = NeuroBioSenseDataset(
        samples=test_samples,
        signal_df=signal_df,
        signal_id_columns=id_cols,
        signal_stats=stats,
        duration_df=duration_df,
        split="test",
        stage=stage,
        t_v=t_v,
        t_s=t_s,
        train=False,
    )

    return train_ds, val_ds, test_ds, stats
