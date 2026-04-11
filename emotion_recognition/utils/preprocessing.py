"""Video preprocessing utilities for NeuroBioSense clips.

This module handles:
- Reading MP4 clips
- Frame sampling (every 4th frame)
- Stage 3 augmentations and temporal jitter
- Sliding-window construction (T_v=10, stride=5)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T


def build_video_transform(train: bool = False, stage: int = 3) -> T.Compose:
    """Create frame transform pipeline.

    Args:
        train: Whether augmentation should be enabled.
        stage: Training stage id.

    Returns:
        torchvision Compose object.
    """
    if train and stage == 1:
        # WHY stronger geometric augmentation in Stage 1: robust face adaptation
        # on heterogeneous FER/CK+ image conditions.
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((160, 160)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomRotation(degrees=10),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                T.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ]
        )
    elif train and stage == 3:
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((160, 160)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((160, 160)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    return transform


def temporal_jitter_frames(frames: Sequence[np.ndarray], drop_prob: float = 0.15) -> List[np.ndarray]:
    """Apply temporal jitter by dropping random frames and repeating neighbors.

    Args:
        frames: List of RGB frames.
        drop_prob: Probability of replacing a frame with a neighbor.

    Returns:
        Jittered frame list with unchanged length.
    """
    if len(frames) <= 2:
        return list(frames)

    jittered: List[np.ndarray] = []
    for idx in range(len(frames)):
        if random.random() < drop_prob:
            # WHY neighbor replay: mimics dropped frames while preserving temporal
            # continuity better than inserting blank frames.
            if idx == 0:
                replacement = frames[1]
            elif idx == len(frames) - 1:
                replacement = frames[-2]
            else:
                replacement = frames[idx - 1] if random.random() < 0.5 else frames[idx + 1]
            jittered.append(replacement)
        else:
            jittered.append(frames[idx])
    return jittered


def _read_all_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    """Read all frames from a video using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0

    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) BGR -> (H, W, 3) RGB
        frames.append(frame_rgb)

    cap.release()
    if not frames:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    return frames, float(fps)


def sample_frames(frames: Sequence[np.ndarray], every_n: int = 4) -> List[np.ndarray]:
    """Sample every Nth frame from full frame sequence."""
    sampled = [frame for idx, frame in enumerate(frames) if idx % every_n == 0]
    return sampled if sampled else [frames[0]]


def load_video_tensor(
    video_path: str | Path,
    every_n: int = 4,
    train: bool = False,
    stage: int = 3,
    temporal_jitter: bool = False,
) -> Tuple[torch.Tensor, float]:
    """Load a clip and return sampled frame tensor.

    Args:
        video_path: Path to MP4 file.
        every_n: Sample every Nth frame.
        train: Whether to apply training transforms.
        stage: Stage id for augmentation policy.
        temporal_jitter: Whether to apply frame-drop jitter.

    Returns:
        frames_tensor: Tensor of shape (T, 3, 160, 160).
        duration_sec: Clip duration in seconds.
    """
    path = Path(video_path)
    raw_frames, fps = _read_all_frames(path)
    sampled_frames = sample_frames(raw_frames, every_n=every_n)

    if train and temporal_jitter:
        sampled_frames = temporal_jitter_frames(sampled_frames)

    transform = build_video_transform(train=train, stage=stage)
    frame_tensors = [transform(frame) for frame in sampled_frames]
    frames_tensor = torch.stack(frame_tensors, dim=0)  # list[(3,160,160)] -> (T, 3, 160, 160)

    duration_sec = len(raw_frames) / fps
    return frames_tensor, duration_sec


def make_sliding_windows(frames_tensor: torch.Tensor, window_size: int = 10, stride: int = 5) -> torch.Tensor:
    """Create temporal windows from sampled frames.

    Args:
        frames_tensor: Tensor of shape (T, 3, 160, 160).
        window_size: Number of frames per window.
        stride: Sliding window stride.

    Returns:
        Tensor of shape (N_w, window_size, 3, 160, 160).
    """
    total_frames = frames_tensor.size(0)

    if total_frames < window_size:
        # WHY repeat padding: maintains static shape while preserving content.
        pad_count = window_size - total_frames
        pad_frames = frames_tensor[-1:].repeat(pad_count, 1, 1, 1)  # (1, 3, H, W) -> (pad, 3, H, W)
        frames_tensor = torch.cat([frames_tensor, pad_frames], dim=0)  # (T,3,H,W)+(pad,3,H,W) -> (window_size,3,H,W)
        total_frames = frames_tensor.size(0)

    windows = []
    for start in range(0, max(1, total_frames - window_size + 1), stride):
        end = start + window_size
        if end > total_frames:
            break
        window = frames_tensor[start:end]  # (window_size, 3, 160, 160)
        windows.append(window)

    if not windows:
        windows = [frames_tensor[:window_size]]

    stacked = torch.stack(windows, dim=0)  # list[(T_v,3,160,160)] -> (N_w, T_v, 3, 160, 160)
    return stacked


def sample_training_window(frames_tensor: torch.Tensor, window_size: int = 10, stride: int = 5) -> torch.Tensor:
    """Randomly sample one temporal window for Stage 3 training."""
    windows = make_sliding_windows(frames_tensor, window_size=window_size, stride=stride)
    idx = random.randrange(windows.size(0))
    chosen = windows[idx]  # (N_w, T_v, 3, 160, 160) -> (T_v, 3, 160, 160)
    return chosen


def aggregate_window_predictions(window_log_probs: Iterable[torch.Tensor], mode: str = "mean") -> torch.Tensor:
    """Aggregate multiple window predictions into a single clip prediction.

    Args:
        window_log_probs: Iterable of tensors each shape (7,) or (1,7).
        mode: "mean" for average softmax, "majority" for vote.

    Returns:
        Tensor of shape (7,) with aggregated log-probabilities.
    """
    probs_list: List[torch.Tensor] = []
    for logp in window_log_probs:
        logp = logp.squeeze(0) if logp.dim() == 2 else logp
        probs_list.append(torch.exp(logp))  # (7,) log-prob -> (7,) prob

    if not probs_list:
        raise ValueError("window_log_probs is empty")

    probs = torch.stack(probs_list, dim=0)  # list[(7,)] -> (N_w, 7)

    if mode == "majority":
        votes = probs.argmax(dim=1)  # (N_w, 7) -> (N_w,)
        counts = torch.bincount(votes, minlength=probs.size(1)).float()  # (N_w,) -> (7,)
        agg_probs = counts / counts.sum().clamp_min(1.0)  # (7,) -> (7,)
    else:
        agg_probs = probs.mean(dim=0)  # (N_w, 7) -> (7,)

    agg_probs = agg_probs.clamp_min(1e-9)
    agg_log_probs = torch.log(agg_probs)  # (7,) -> (7,)
    return agg_log_probs
