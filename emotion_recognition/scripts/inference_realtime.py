"""Real-time multimodal inference script for NeuroBioSense-style inputs.

Target setup:
- Device: MPS on Apple Silicon when available
- Batch size: 1
- Sliding windows: T_v=10 frames, T_s=128 timesteps
- Frame sampling: every 4th webcam frame
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import numpy as np
import torch

from emotion_recognition.models.full_model import MultimodalEmotionModel
from emotion_recognition.utils.dataset import ID_TO_EMOTION
from emotion_recognition.utils.preprocessing import build_video_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time multimodal emotion inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained Stage 3 checkpoint")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=18.0)
    return parser.parse_args()


def load_model(ckpt_path: str | Path, device: torch.device) -> Tuple[MultimodalEmotionModel, np.ndarray, np.ndarray]:
    payload = torch.load(ckpt_path, map_location="cpu")
    model = MultimodalEmotionModel(num_classes=7)

    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Defaults when normalization stats are absent.
    mean = np.zeros((6,), dtype=np.float32)
    std = np.ones((6,), dtype=np.float32)
    if isinstance(payload, dict) and "normalization_stats" in payload:
        stats = payload["normalization_stats"]
        mean = np.asarray(stats.get("mean", mean), dtype=np.float32)
        std = np.asarray(stats.get("std", std), dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)

    return model, mean, std


def mock_signal_reader() -> np.ndarray:
    """Placeholder signal acquisition function.

    Replace this with actual Empatica E4 streaming integration.
    Returns one timestep of 6-channel signal.
    """
    return np.random.randn(6).astype(np.float32)


def main() -> None:
    args = parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, sig_mean, sig_std = load_model(args.checkpoint, device)

    transform = build_video_transform(train=False, stage=3)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera id={args.camera_id}")

    # WHY circular buffers: avoid reallocations and preserve fixed-latency windows.
    frame_buffer: Deque[torch.Tensor] = collections.deque(maxlen=10)
    signal_buffer: Deque[np.ndarray] = collections.deque(maxlen=128)

    frame_index = 0
    sampled_frames = 0
    last_infer_ts = 0.0
    frame_period = 1.0 / max(args.target_fps, 1.0)

    print("Starting realtime inference. Press 'q' to quit.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            now = time.time()
            if now - last_infer_ts < frame_period:
                # Maintain target processing rate.
                cv2.imshow("NeuroBioSense Realtime", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            last_infer_ts = now

            # Acquire one new physiological sample each loop.
            sig_t = mock_signal_reader()  # (6,)
            signal_buffer.append(sig_t)

            if frame_index % 4 == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) BGR -> RGB
                face_t = transform(frame_rgb)  # (H, W, 3) -> (3, 160, 160)
                frame_buffer.append(face_t)
                sampled_frames += 1

            frame_index += 1

            if len(frame_buffer) == 10 and len(signal_buffer) == 128:
                video = torch.stack(list(frame_buffer), dim=0).unsqueeze(0).to(device)  # (10,3,160,160) -> (1,10,3,160,160)

                sig_np = np.stack(list(signal_buffer), axis=0).astype(np.float32)  # list[(6,)] -> (128, 6)
                sig_np = (sig_np - sig_mean[None, :]) / sig_std[None, :]  # (128, 6) -> (128, 6)
                signal = torch.from_numpy(sig_np).float().unsqueeze(0).to(device)  # (128, 6) -> (1, 128, 6)

                with torch.inference_mode():
                    # WHY inference_mode: reduces autograd overhead for real-time FPS.
                    output, confidence = model(video, signal)  # ((1,10,3,160,160),(1,128,6)) -> (1,7), scalar

                pred_id = int(output.argmax(dim=1).item())
                pred_code = ID_TO_EMOTION.get(pred_id, str(pred_id))
                conf_val = float(confidence)

                text = f"Emotion: {pred_code} ({pred_id})  Conf: {conf_val:.3f}"
                cv2.putText(
                    frame_bgr,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("NeuroBioSense Realtime", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
