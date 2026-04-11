"""Single-clip inference script for multimodal emotion recognition.

This utility is deployment-friendly and supports two signal modes:
1) NeuroBioSense alignment mode (full 32-Hertz.csv + participant/ad)
2) Direct segment mode (CSV with BVP, EDA, TEMP, ACC_X, ACC_Y, ACC_Z)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from emotion_recognition.models.full_model import MultimodalEmotionModel
from emotion_recognition.utils.dataset import ID_TO_EMOTION
from emotion_recognition.utils.preprocessing import load_video_tensor, make_sliding_windows
from emotion_recognition.utils.signal_processing import (
    SIGNAL_COLUMNS,
    extract_signal_segment,
    infer_id_columns,
    load_32hz_csv,
    load_duration_mapping,
    lookup_duration,
    normalize_signal_np,
    resample_signal_to_fixed_length,
)

EMOTION_NAME = {
    0: "Joy",
    1: "Sadness",
    2: "Anger",
    3: "Disgust",
    4: "Surprise",
    5: "Neutral",
    6: "Fear",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict emotion for a single video clip")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--signal-csv", type=str, default="")
    parser.add_argument("--participant-id", type=str, default="")
    parser.add_argument("--ad-code", type=str, default="")
    parser.add_argument("--demographics-csv", type=str, default="")
    parser.add_argument("--aggregation", type=str, choices=["mean", "majority"], default="mean")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[MultimodalEmotionModel, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint_path, map_location="cpu")

    model = MultimodalEmotionModel(num_classes=7)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    mean = np.zeros((6,), dtype=np.float32)
    std = np.ones((6,), dtype=np.float32)
    if isinstance(payload, dict) and "normalization_stats" in payload:
        stats = payload["normalization_stats"]
        mean = np.asarray(stats.get("mean", mean), dtype=np.float32)
        std = np.asarray(stats.get("std", std), dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)

    return model, mean, std


def load_signal_segment(
    signal_csv: str,
    participant_id: str,
    ad_code: str,
    demographics_csv: str,
    clip_duration_sec: float,
) -> np.ndarray:
    """Load and align signal segment.

    Returns:
        Raw segment array of shape (T, 6) before normalization/resampling.
    """
    if not signal_csv:
        # Fallback for video-only inference.
        return np.zeros((128, 6), dtype=np.float32)

    csv_path = Path(signal_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Signal CSV not found: {csv_path}")

    if participant_id and ad_code:
        signal_df = load_32hz_csv(csv_path)
        id_cols = infer_id_columns(signal_df)

        duration_df = None
        if demographics_csv:
            demo_path = Path(demographics_csv)
            if demo_path.exists():
                duration_df = load_duration_mapping(demo_path)

        aligned_duration = lookup_duration(
            duration_df,
            participant_id=participant_id,
            ad_code=ad_code,
            fallback_duration_sec=clip_duration_sec,
        )
        segment = extract_signal_segment(
            signal_df,
            participant_id=participant_id,
            ad_code=ad_code,
            duration_sec=aligned_duration,
            id_columns=id_cols,
        )
        return segment

    # Direct segment mode: CSV must contain the six channels.
    direct_df = pd.read_csv(csv_path)
    missing = [c for c in SIGNAL_COLUMNS if c not in direct_df.columns]
    if missing:
        raise ValueError(
            "Direct signal mode requires columns "
            f"{SIGNAL_COLUMNS}. Missing: {missing}. "
            "Alternatively provide --participant-id and --ad-code for alignment mode."
        )
    return direct_df[SIGNAL_COLUMNS].to_numpy(dtype=np.float32)


def aggregate_probs(window_log_probs: torch.Tensor, mode: str) -> torch.Tensor:
    """Aggregate window predictions.

    Args:
        window_log_probs: Tensor of shape (N_w, 7).
        mode: mean or majority.

    Returns:
        log-prob tensor of shape (7,).
    """
    if mode == "majority":
        votes = window_log_probs.argmax(dim=1)  # (N_w, 7) -> (N_w,)
        counts = torch.bincount(votes, minlength=7).float()  # (N_w,) -> (7,)
        probs = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-9)  # (7,) -> (7,)
    else:
        probs = torch.exp(window_log_probs).mean(dim=0).clamp_min(1e-9)  # (N_w, 7) -> (7,)
    return torch.log(probs)  # (7,) -> (7,)


def main() -> None:
    args = parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, sig_mean, sig_std = load_model(args.checkpoint, device)

    # Video preprocessing and windowing.
    video_frames, clip_duration = load_video_tensor(
        args.video,
        every_n=4,
        train=False,
        stage=3,
        temporal_jitter=False,
    )
    windows = make_sliding_windows(video_frames, window_size=10, stride=5)  # (T,3,160,160) -> (N_w,10,3,160,160)

    # Signal loading and normalization.
    raw_signal = load_signal_segment(
        signal_csv=args.signal_csv,
        participant_id=args.participant_id,
        ad_code=args.ad_code,
        demographics_csv=args.demographics_csv,
        clip_duration_sec=clip_duration,
    )
    raw_signal = raw_signal.astype(np.float32)

    sig_norm = (raw_signal - sig_mean[None, :]) / sig_std[None, :]  # (T,6) -> (T,6)
    sig_128 = resample_signal_to_fixed_length(sig_norm, target_length=128)  # (T,6) -> (128,6)

    with torch.inference_mode():
        video_batch = windows.to(device)  # (N_w,10,3,160,160)
        n_w = video_batch.size(0)

        signal_one = torch.from_numpy(sig_128).float().to(device)  # (128,6)
        signal_batch = signal_one.unsqueeze(0).repeat(n_w, 1, 1)  # (128,6) -> (N_w,128,6)

        window_log_probs, _ = model(video_batch, signal_batch)  # ((N_w,10,3,160,160),(N_w,128,6)) -> (N_w,7)
        clip_log_probs = aggregate_probs(window_log_probs, mode=args.aggregation)  # (N_w,7) -> (7,)

        probs = torch.exp(clip_log_probs)  # (7,) -> (7,)
        confidence, pred_idx = probs.max(dim=0)

    pred_id = int(pred_idx.item())
    pred_code = ID_TO_EMOTION[pred_id]
    pred_name = EMOTION_NAME[pred_id]

    result = {
        "predicted_id": pred_id,
        "predicted_code": pred_code,
        "predicted_name": pred_name,
        "confidence": float(confidence.item()),
        "probabilities": probs.detach().cpu().numpy().tolist(),
        "aggregation": args.aggregation,
        "num_windows": int(windows.size(0)),
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved prediction report: {out_path}")


if __name__ == "__main__":
    main()
