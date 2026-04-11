"""Streamlit deployment app for multimodal emotion recognition.

Usage:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
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


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def persist_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    """Persist Streamlit upload to disk using content hash as filename."""
    target_dir.mkdir(parents=True, exist_ok=True)
    content = uploaded_file.getvalue()
    file_hash = hashlib.sha256(content).hexdigest()[:16]
    suffix = Path(uploaded_file.name).suffix
    out_path = target_dir / f"{file_hash}{suffix}"
    if not out_path.exists():
        out_path.write_bytes(content)
    return out_path


@st.cache_resource(show_spinner=False)
def load_checkpoint_model(checkpoint_path: str, device_name: str) -> Tuple[MultimodalEmotionModel, np.ndarray, np.ndarray]:
    """Load model + normalization stats from checkpoint."""
    device = torch.device(device_name)
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


def resolve_signal_segment(
    signal_csv_path: Path | None,
    demographics_csv_path: Path | None,
    participant_id: str,
    ad_code: str,
    clip_duration_sec: float,
) -> np.ndarray:
    """Resolve signal segment from either aligned mode or direct mode."""
    if signal_csv_path is None:
        # WHY zero-signal fallback: allows video-only sanity inference when E4 data
        # is not available yet.
        return np.zeros((128, 6), dtype=np.float32)

    if participant_id and ad_code:
        df = load_32hz_csv(signal_csv_path)
        id_cols = infer_id_columns(df)
        duration_df = load_duration_mapping(demographics_csv_path) if demographics_csv_path is not None else None

        aligned_duration = lookup_duration(
            duration_df,
            participant_id=participant_id,
            ad_code=ad_code,
            fallback_duration_sec=clip_duration_sec,
        )
        segment = extract_signal_segment(
            df,
            participant_id=participant_id,
            ad_code=ad_code,
            duration_sec=aligned_duration,
            id_columns=id_cols,
        )
        return segment.astype(np.float32)

    # Direct mode for CSV already containing 6-channel segment.
    df = pd.read_csv(signal_csv_path)
    missing = [c for c in SIGNAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Signal CSV missing required columns for direct mode: "
            f"{missing}. Provide participant/ad fields for full 32-Hertz alignment mode."
        )
    return df[SIGNAL_COLUMNS].to_numpy(dtype=np.float32)


def aggregate_window_log_probs(window_log_probs: torch.Tensor, mode: str) -> torch.Tensor:
    """Aggregate per-window log-probabilities to one clip prediction."""
    if mode == "majority":
        votes = window_log_probs.argmax(dim=1)  # (N_w, 7) -> (N_w,)
        counts = torch.bincount(votes, minlength=7).float()  # (N_w,) -> (7,)
        probs = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-9)  # (7,) -> (7,)
    else:
        probs = torch.exp(window_log_probs).mean(dim=0).clamp_min(1e-9)  # (N_w, 7) -> (7,)
    return torch.log(probs)  # (7,) -> (7,)


def main() -> None:
    st.set_page_config(page_title="NeuroBioSense Emotion Demo", layout="wide")
    st.title("NeuroBioSense Multimodal Emotion Recognition")
    st.caption("Deployment-ready Streamlit app for Stage 3 checkpoint inference")

    with st.sidebar:
        st.header("Inputs")
        ckpt_upload = st.file_uploader("Model checkpoint (.pth)", type=["pth", "pt"])
        video_upload = st.file_uploader("Video clip (.mp4)", type=["mp4", "MP4"])
        signal_upload = st.file_uploader("Signal CSV (optional)", type=["csv"])
        demographics_upload = st.file_uploader("Participant demographics CSV (optional)", type=["csv"])

        participant_id = st.text_input("Participant ID (optional, for alignment mode)", value="")
        ad_code = st.text_input("Ad Code (optional, for alignment mode)", value="")
        aggregation = st.selectbox("Window aggregation", options=["mean", "majority"], index=0)
        run_btn = st.button("Run Inference", type="primary")

    if ckpt_upload is None or video_upload is None:
        st.info("Upload at least checkpoint + video to start inference.")
        return

    cache_dir = Path("tmp_uploads")
    checkpoint_path = persist_uploaded_file(ckpt_upload, cache_dir / "checkpoints")
    video_path = persist_uploaded_file(video_upload, cache_dir / "videos")
    signal_path = persist_uploaded_file(signal_upload, cache_dir / "signals") if signal_upload is not None else None
    demographics_path = (
        persist_uploaded_file(demographics_upload, cache_dir / "demographics") if demographics_upload is not None else None
    )

    st.write(f"Using device: {get_device().type}")

    if not run_btn:
        return

    try:
        with st.spinner("Loading model..."):
            device = get_device()
            model, sig_mean, sig_std = load_checkpoint_model(str(checkpoint_path), device.type)

        with st.spinner("Preprocessing video and signal..."):
            frames, clip_duration_sec = load_video_tensor(
                video_path,
                every_n=4,
                train=False,
                stage=3,
                temporal_jitter=False,
            )
            windows = make_sliding_windows(frames, window_size=10, stride=5)  # (T,3,160,160) -> (N_w,10,3,160,160)

            signal_raw = resolve_signal_segment(
                signal_csv_path=signal_path,
                demographics_csv_path=demographics_path,
                participant_id=participant_id.strip(),
                ad_code=ad_code.strip(),
                clip_duration_sec=clip_duration_sec,
            )
            signal_norm = (signal_raw - sig_mean[None, :]) / sig_std[None, :]  # (T,6) -> (T,6)
            signal_128 = resample_signal_to_fixed_length(signal_norm.astype(np.float32), target_length=128)  # (T,6) -> (128,6)

        with st.spinner("Running inference..."):
            with torch.inference_mode():
                video_batch = windows.to(device)  # (N_w,10,3,160,160)
                n_w = video_batch.size(0)

                signal_one = torch.from_numpy(signal_128).float().to(device)  # (128,6)
                signal_batch = signal_one.unsqueeze(0).repeat(n_w, 1, 1)  # (128,6) -> (N_w,128,6)

                window_log_probs, _ = model(video_batch, signal_batch)  # ((N_w,10,3,160,160),(N_w,128,6)) -> (N_w,7)
                clip_log_probs = aggregate_window_log_probs(window_log_probs, mode=aggregation)  # (N_w,7) -> (7,)
                probs = torch.exp(clip_log_probs).detach().cpu().numpy()  # (7,) -> (7,)

        pred_id = int(np.argmax(probs))
        conf = float(probs[pred_id])

        st.success("Inference completed")
        st.subheader(f"Prediction: {EMOTION_NAME[pred_id]} ({ID_TO_EMOTION[pred_id]})")
        st.write(f"Confidence: {conf:.4f}")
        st.write(f"Windows processed: {int(windows.size(0))}")

        chart_df = pd.DataFrame(
            {
                "emotion_id": list(range(7)),
                "emotion": [EMOTION_NAME[i] for i in range(7)],
                "probability": probs,
            }
        )
        st.bar_chart(chart_df.set_index("emotion")["probability"])

        with st.expander("Detailed probabilities"):
            st.dataframe(chart_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Inference failed: {exc}")


if __name__ == "__main__":
    main()
