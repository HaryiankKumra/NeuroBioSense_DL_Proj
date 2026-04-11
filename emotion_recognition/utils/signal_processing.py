"""Signal preprocessing for NeuroBioSense 32-Hertz physiological data.

This module provides:
- CSV loading and typed preprocessing
- Train-split normalization statistics
- Clip-duration alignment and fixed-length resampling
- Stage 3 physiological augmentations
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SIGNAL_COLUMNS: List[str] = ["BVP", "EDA", "TEMP", "ACC_X", "ACC_Y", "ACC_Z"]


def normalize_subject_id(value: object) -> str:
    """Normalize participant IDs across numeric/text formats."""
    s = str(value).strip()
    if not s:
        return s

    if s.endswith(".0"):
        s = s[:-2]

    # WESAD-like IDs such as S2 -> 2 for consistent matching.
    if len(s) >= 2 and s[0].upper() == "S" and s[1:].isdigit():
        return str(int(s[1:]))

    try:
        return str(int(float(s)))
    except ValueError:
        return s


def normalize_ad_code(value: object) -> str:
    """Normalize ad codes to Axx format used by NeuroBioSense clips."""
    s = str(value).strip().upper()
    if not s:
        return s

    if s.startswith("A"):
        digits = s[1:]
        if digits.isdigit():
            return f"A{int(digits):02d}"
        return s

    if s.isdigit():
        return f"A{int(s):02d}"

    try:
        return f"A{int(float(s)):02d}"
    except ValueError:
        return s


def load_32hz_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load NeuroBioSense preprocessed physiological file.

    Expected required columns include at least:
    - participant id (e.g., PARTICIPANT_ID or participant_id)
    - ad code (e.g., AD_CODE or ad_code)
    - EMOTION
    - BVP, EDA, TEMP, ACC_X, ACC_Y, ACC_Z
    """
    df = pd.read_csv(csv_path)

    # WHY canonicalization: source files may vary in exact capitalization.
    df.columns = [str(col).strip() for col in df.columns]

    # Accept raw axis names used by NeuroBioSense preprocessed files.
    alias_map = {
        "X": "ACC_X",
        "Y": "ACC_Y",
        "Z": "ACC_Z",
        "x": "ACC_X",
        "y": "ACC_Y",
        "z": "ACC_Z",
    }
    rename_map = {
        src: dst
        for src, dst in alias_map.items()
        if src in df.columns and dst not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in SIGNAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required signal columns: {missing}")

    if "EMOTION" not in df.columns:
        raise ValueError("Missing required label column: EMOTION")

    df["EMOTION"] = df["EMOTION"].astype(str).str.strip().str.upper()

    return df


def _find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise KeyError(f"None of columns found: {list(candidates)}")
    return None


def infer_id_columns(df: pd.DataFrame) -> Dict[str, str | None]:
    """Infer participant/ad/timestamp columns from dataset variants.

    Returns keys with value None when a column does not exist.
    """
    participant_col = _find_column(
        df,
        [
            "participant_id",
            "PARTICIPANT_ID",
            "Participant_ID",
            "participant",
            "PARTICIPANT",
            "SUBJECT ID",
            "SUBJECT_ID",
        ],
        required=False,
    )
    ad_col = _find_column(
        df,
        ["ad_code", "AD_CODE", "AD CODE", "ad", "AD", "stimulus_id", "STIMULUS_ID"],
        required=False,
    )
    time_col = _find_column(df, ["timestamp", "TIMESTAMP", "time", "TIME", "seconds", "SECOND"], required=False)

    return {
        "participant": participant_col,
        "ad": ad_col,
        "time": time_col,
    }


@dataclass
class SignalNormalizationStats:
    """Per-channel normalization statistics fitted on train split."""

    mean: np.ndarray  # shape (6,)
    std: np.ndarray  # shape (6,)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Sequence[float]]) -> "SignalNormalizationStats":
        return SignalNormalizationStats(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


def fit_signal_normalizer(df: pd.DataFrame, train_mask: np.ndarray) -> SignalNormalizationStats:
    """Fit per-channel zero-mean unit-variance stats on training samples only."""
    values = df.loc[train_mask, SIGNAL_COLUMNS].to_numpy(dtype=np.float32)  # (N_train, 6)
    mean = values.mean(axis=0)  # (N_train, 6) -> (6,)
    std = values.std(axis=0)  # (N_train, 6) -> (6,)
    std = np.where(std < 1e-6, 1.0, std)
    return SignalNormalizationStats(mean=mean, std=std)


def normalize_signal_np(signal: np.ndarray, stats: SignalNormalizationStats) -> np.ndarray:
    """Apply per-channel normalization."""
    normed = (signal - stats.mean[None, :]) / stats.std[None, :]  # (T, 6) -> (T, 6)
    return normed


def extract_signal_segment(
    df: pd.DataFrame,
    participant_id: str,
    ad_code: str,
    duration_sec: float,
    id_columns: Dict[str, str | None],
) -> np.ndarray:
    """Extract participant+ad physiological segment aligned to clip duration.

    If no explicit timestamp exists, this function assumes rows are already ordered
    at 32 Hz and takes a duration-matched prefix.
    """
    p_col = id_columns["participant"]
    a_col = id_columns["ad"]
    t_col = id_columns["time"]

    if p_col is None or a_col is None:
        raise KeyError("Signal DataFrame does not contain participant/ad columns for strict alignment.")

    pid_norm = normalize_subject_id(participant_id)
    ad_norm = normalize_ad_code(ad_code)

    participant_series = df[p_col].map(normalize_subject_id)
    ad_series = df[a_col].map(normalize_ad_code)

    mask = (participant_series == pid_norm) & (ad_series == ad_norm)
    sub = df.loc[mask, SIGNAL_COLUMNS + ([t_col] if t_col in df.columns else [])].copy()
    if sub.empty:
        raise KeyError(f"No signal rows for participant={participant_id}, ad={ad_code}")

    if t_col in sub.columns:
        sub = sub.sort_values(by=t_col)

    samples_needed = max(1, int(round(duration_sec * 32.0)))
    segment = sub[SIGNAL_COLUMNS].to_numpy(dtype=np.float32)

    if segment.shape[0] >= samples_needed:
        segment = segment[:samples_needed]  # (N, 6) -> (samples_needed, 6)
    else:
        pad = np.repeat(segment[-1:, :], repeats=samples_needed - segment.shape[0], axis=0)
        segment = np.concatenate([segment, pad], axis=0)  # (N,6)+(pad,6) -> (samples_needed,6)

    return segment


def resample_signal_to_fixed_length(signal: np.ndarray, target_length: int = 128) -> np.ndarray:
    """Resample a (T, 6) sequence to fixed length using linear interpolation."""
    x = torch.from_numpy(signal).float().unsqueeze(0).transpose(1, 2)  # (T, 6) -> (1, 6, T)
    y = F.interpolate(x, size=target_length, mode="linear", align_corners=False)  # (1, 6, T) -> (1, 6, target)
    out = y.transpose(1, 2).squeeze(0).numpy().astype(np.float32)  # (1, 6, target) -> (target, 6)
    return out


def augment_signal(
    signal: np.ndarray,
    gaussian_sigma: float = 0.01,
    max_shift: int = 10,
    channel_dropout_p: float = 0.1,
) -> np.ndarray:
    """Stage 3 physiological augmentations.

    Augmentations:
    - Gaussian noise
    - Random time shift
    - Channel dropout (drop one channel with probability p)
    """
    out = signal.copy()  # (T, 6)

    # WHY additive noise: improves robustness to sensor quantization and jitter.
    noise = np.random.normal(loc=0.0, scale=gaussian_sigma, size=out.shape).astype(np.float32)
    out = out + noise  # (T, 6) + (T, 6) -> (T, 6)

    # WHY time shift: mitigates imperfect clip-signal temporal alignment.
    shift = random.randint(-max_shift, max_shift)
    if shift != 0:
        out = np.roll(out, shift=shift, axis=0)  # (T, 6) -> (T, 6)

    # WHY channel dropout: simulates sensor dropout/contact artifacts.
    if random.random() < channel_dropout_p:
        ch_idx = random.randrange(out.shape[1])
        out[:, ch_idx] = 0.0

    return out


def load_duration_mapping(demographics_csv: str | Path) -> pd.DataFrame:
    """Load participant demographic info table used for alignment metadata.

    The function is tolerant to column name variants and returns a normalized
    DataFrame with columns: participant_id, ad_code, duration_sec.
    """
    path = Path(demographics_csv)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    participant_col = _find_column(
        df,
        ["participant_id", "PARTICIPANT_ID", "Participant_ID", "SUBJECT ID", "SUBJECT_ID"],
    )
    ad_col = _find_column(df, ["ad_code", "AD_CODE", "AD CODE", "ad", "AD"])
    duration_col = _find_column(df, ["Duration", "duration", "DURATION", "duration_sec", "DURATION_SEC"])

    # Forward-fill subject-level metadata rows commonly used in the workbook.
    df[participant_col] = df[participant_col].ffill()

    duration_series = pd.to_numeric(df[duration_col], errors="coerce").fillna(0.0).astype(float)

    out = pd.DataFrame(
        {
            "participant_id": df[participant_col].map(normalize_subject_id),
            "ad_code": df[ad_col].map(normalize_ad_code),
            "duration_sec": duration_series,
        }
    )

    # If duration values appear to be Excel day-fractions/timestamps, convert to seconds.
    # Typical ad clip durations should be small; values in [0, 1) imply fractions of a day.
    frac_mask = (out["duration_sec"] > 0.0) & (out["duration_sec"] < 1.0)
    if frac_mask.any():
        out.loc[frac_mask, "duration_sec"] = out.loc[frac_mask, "duration_sec"] * 86400.0

    out = out[out["duration_sec"] > 0.0].reset_index(drop=True)
    return out


def lookup_duration(
    duration_df: Optional[pd.DataFrame],
    participant_id: str,
    ad_code: str,
    fallback_duration_sec: float,
) -> float:
    """Find duration from demographics table with fallback to video-derived duration."""
    if duration_df is None:
        return float(fallback_duration_sec)

    pid_norm = normalize_subject_id(participant_id)
    ad_norm = normalize_ad_code(ad_code)

    mask = (duration_df["participant_id"].astype(str) == pid_norm) & (duration_df["ad_code"].astype(str) == ad_norm)
    sub = duration_df.loc[mask]
    if sub.empty:
        return float(fallback_duration_sec)

    dur = float(sub.iloc[0]["duration_sec"])
    # Guard against malformed metadata (e.g., timestamp-like values).
    if not math.isfinite(dur) or dur <= 0 or dur > 600:
        return float(fallback_duration_sec)
    return dur
