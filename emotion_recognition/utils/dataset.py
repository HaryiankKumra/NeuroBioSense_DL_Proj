"""PyTorch dataset for NeuroBioSense multimodal emotion recognition.

This dataset scans the required video structure:
{participant_id}/{ad_code}/{emotion_label}/{clip_id}.MP4
and aligns each clip with physiological rows from 32-Hertz.csv.
"""

from __future__ import annotations

import random
import warnings
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
    fit_signal_normalizer,
    infer_id_columns,
    load_32hz_csv,
    load_duration_mapping,
    lookup_duration,
    normalize_ad_code,
    normalize_signal_np,
    normalize_subject_id,
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
    category: str
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
        signal_id_columns: Dict[str, str | None],
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

        self._strict_signal_alignment = (
            self.signal_id_columns.get("participant") is not None
            and self.signal_id_columns.get("ad") is not None
        )
        self._signal_by_key: Dict[Tuple[str, str], np.ndarray] = {}
        self._signal_by_emotion: Dict[str, np.ndarray] = {}
        self._build_signal_cache()

    def _build_signal_cache(self) -> None:
        """Build fast lookup caches so __getitem__ avoids full DataFrame scans."""
        if self._strict_signal_alignment:
            p_col = self.signal_id_columns["participant"]
            a_col = self.signal_id_columns["ad"]
            assert p_col is not None and a_col is not None

            key_df = self.signal_df[[p_col, a_col] + SIGNAL_COLUMNS].copy()
            key_df["_pid"] = key_df[p_col].map(normalize_subject_id)
            key_df["_ad"] = key_df[a_col].map(normalize_ad_code)

            for (pid, ad_code), group in key_df.groupby(["_pid", "_ad"], sort=False):
                self._signal_by_key[(pid, ad_code)] = group[SIGNAL_COLUMNS].to_numpy(dtype=np.float32)

        emotion_series = self.signal_df["EMOTION"].astype(str).str.strip().str.upper()
        for emotion_code in EMOTION_TO_ID.keys():
            rows = self.signal_df.loc[emotion_series == emotion_code, SIGNAL_COLUMNS]
            if len(rows) > 0:
                self._signal_by_emotion[emotion_code] = rows.to_numpy(dtype=np.float32)

    @staticmethod
    def _slice_or_pad(signal_array: np.ndarray, target_len: int, train: bool) -> np.ndarray:
        """Extract a fixed-length segment, padding with edge samples when needed."""
        n = signal_array.shape[0]
        if n == 0:
            return np.zeros((target_len, len(SIGNAL_COLUMNS)), dtype=np.float32)

        if n >= target_len:
            if train and n > target_len:
                start = random.randint(0, n - target_len)
            else:
                start = max(0, (n - target_len) // 2)
            return signal_array[start : start + target_len]

        pad = np.repeat(signal_array[-1:, :], repeats=target_len - n, axis=0)
        return np.concatenate([signal_array, pad], axis=0)

    def _resolve_signal_segment(self, sample: ClipSample, duration_sec: float) -> np.ndarray:
        """Resolve clip-aligned signal segment.

        Priority:
        1) strict participant+ad lookup if available in 32-Hertz.csv
        2) emotion-conditioned fallback pool when keys are unavailable
        """
        target_len = max(1, int(round(duration_sec * 32.0)))

        if self._strict_signal_alignment:
            key = (
                normalize_subject_id(sample.participant_id),
                normalize_ad_code(sample.ad_code),
            )
            seq = self._signal_by_key.get(key)
            if seq is not None and len(seq) > 0:
                return self._slice_or_pad(seq, target_len, train=self.train)

        # Fallback when strict keys are not present in signal CSV.
        seq = self._signal_by_emotion.get(sample.emotion_code)
        if seq is None or len(seq) == 0:
            seq = np.zeros((target_len, len(SIGNAL_COLUMNS)), dtype=np.float32)
            return seq

        return self._slice_or_pad(seq, target_len, train=self.train)

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

        sig_segment = self._resolve_signal_segment(sample, duration_sec=duration_sec)
        # (?, 6) aligned by key if available; emotion-conditioned fallback otherwise

        sig_norm = normalize_signal_np(sig_segment, self.signal_stats)  # (?, 6) -> (?, 6)
        sig_resampled = resample_signal_to_fixed_length(sig_norm, target_length=self.t_s)  # (?,6) -> (T_s,6)

        if self.train and self.stage == 3:
            sig_resampled = augment_signal(sig_resampled)  # (T_s, 6) -> (T_s, 6)

        signal_tensor = torch.from_numpy(sig_resampled).float()  # (T_s, 6)
        label_tensor = torch.tensor(sample.label_id, dtype=torch.long)  # ()

        return video_window, signal_tensor, label_tensor


def scan_video_samples(video_root: str | Path) -> List[ClipSample]:
    """Scan NeuroBioSense clip hierarchy and build sample index.

    Supports both layouts:
    - {participant}/{ad}/{emotion}/{clip}.mp4
    - {category}/{participant}/{ad}/{emotion}/{clip}.mp4
    """
    root = Path(video_root)
    if not root.exists():
        raise FileNotFoundError(f"Video root not found: {root}")

    samples: List[ClipSample] = []

    clip_paths = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"])
    for clip_path in clip_paths:
        rel_parts = clip_path.relative_to(root).parts
        if len(rel_parts) < 4:
            continue

        emotion_code = str(rel_parts[-2]).strip().upper()
        if emotion_code not in EMOTION_TO_ID:
            continue

        ad_code = normalize_ad_code(rel_parts[-3])
        participant_id = normalize_subject_id(rel_parts[-4])
        category = rel_parts[-5] if len(rel_parts) >= 5 else "UNSPECIFIED"

        samples.append(
            ClipSample(
                video_path=clip_path,
                participant_id=participant_id,
                ad_code=ad_code,
                category=category,
                emotion_code=emotion_code,
                label_id=EMOTION_TO_ID[emotion_code],
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
    participant_col: str | None,
    train_participants: Sequence[str],
) -> np.ndarray:
    if participant_col is None:
        return np.ones(len(signal_df), dtype=bool)

    train_set = set(str(pid) for pid in train_participants)
    normalized = signal_df[participant_col].map(normalize_subject_id)
    norm_train = {normalize_subject_id(pid) for pid in train_set}
    mask = normalized.isin(norm_train).to_numpy(dtype=bool)
    if not mask.any():
        return np.ones(len(signal_df), dtype=bool)
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

    if id_cols.get("participant") is None or id_cols.get("ad") is None:
        warnings.warn(
            "32-Hertz.csv does not expose participant/ad keys. "
            "Using emotion-conditioned signal fallback (no strict clip-key alignment).",
            stacklevel=2,
        )

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
