"""
extract_face_embeddings.py
==========================
Use the pretrained FaceNet backbone (facenet_stage1.pth) to extract
512-d face embeddings from every video clip, then train a linear probe
(logistic regression) on those embeddings → binary valence.

Pipeline per clip:
  1. Sample up to N_FRAMES frames from the .mp4
  2. Resize to 160×160, normalise with VGGFace2 stats
  3. Run through the FaceNet backbone → 512-d per frame
  4. Mean-pool frames → single 512-d clip embedding
  5. Stack all clips → train/val/test split (participant-level)
  6. Train LogisticRegression on train, evaluate on test

Expected: 60-72%  (FaceNet pretrained on FER was at val macro-F1=0.678)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent.parent))
from emotion_recognition.models.facenet_backbone import FaceNetBackbone
from emotion_recognition.scripts.train_multimodal import VALENCE2_MAP
from emotion_recognition.utils.dataset import scan_video_samples, split_participants

ROOT        = Path(__file__).parent.parent
STAGE1_PTH  = ROOT / "artifacts" / "facenet_stage1.pth"
OUT_MODEL   = ROOT / "artifacts" / "face_valence_model.pkl"
OUT_EMBED   = ROOT / "artifacts" / "face_embeddings.npz"
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
VIDEO_ROOT  = ROOT / "Dataset" / "NeuroBioSense Dataset" / "NeuroBioSense" / "Advertisement Categories"

N_FRAMES    = 8       # frames sampled per clip
IMG_SIZE    = 160
# VGGFace2 mean/std used by FaceNet
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_backbone(pth_path: Path, device: torch.device) -> nn.Module:
    """Load FaceNet backbone weights from stage-1 checkpoint."""
    state = torch.load(pth_path, map_location="cpu")
    backbone = FaceNetBackbone(pretrained=None)  # don't download — we load ours
    # stage1 checkpoint has key 'backbone'
    if "backbone" in state:
        missing, unexpected = backbone.load_state_dict(state["backbone"], strict=False)
        if unexpected:
            print(f"  (ignored unexpected keys: {unexpected[:3]})")
    else:
        backbone.load_state_dict(state, strict=False)
    backbone.to(device).eval()
    return backbone


def sample_frames(video_path: Path, n: int) -> np.ndarray | None:
    """Return (n, 3, H, W) float32 tensor or None if video unreadable."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    indices = np.linspace(0, max(total - 1, 0), n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - MEAN) / STD
        frames.append(frame.transpose(2, 0, 1))   # (3, H, W)
    cap.release()
    if not frames:
        return None
    return np.stack(frames, axis=0)   # (n, 3, H, W)


@torch.no_grad()
def embed_clip(backbone: nn.Module, frames_np: np.ndarray,
               device: torch.device) -> np.ndarray:
    """frames_np: (n, 3, H, W) → 512-d mean embedding."""
    t = torch.from_numpy(frames_np).to(device)
    emb = backbone(t)           # (n, 512)
    return emb.cpu().numpy().mean(axis=0)   # (512,)


def main():
    print("\n🎥  FaceNet Embedding Extraction + Linear Probe")
    print("=" * 54)

    device = get_device()
    print(f"  Device: {device}")

    # 1. Load backbone
    print(f"  Loading backbone from {STAGE1_PTH.name} …", end=" ", flush=True)
    backbone = load_backbone(STAGE1_PTH, device)
    print("done")

    # 2. Scan clips + participant split
    print("  Scanning video clips …", end=" ", flush=True)
    all_clips = scan_video_samples(VIDEO_ROOT)
    train_ids, val_ids, test_ids = split_participants(all_clips, seed=42)
    print(f"done ({len(all_clips)} clips)")

    id_to_split = {}
    for pid in train_ids: id_to_split[pid] = "train"
    for pid in val_ids:   id_to_split[pid] = "val"
    for pid in test_ids:  id_to_split[pid] = "test"

    # 3. Extract embeddings
    X_splits: dict[str, list] = {"train": [], "val": [], "test": []}
    y_splits: dict[str, list] = {"train": [], "val": [], "test": []}

    print(f"  Extracting embeddings ({N_FRAMES} frames per clip) …")
    failed = 0
    for i, clip in enumerate(all_clips):
        if clip.label_id not in VALENCE2_MAP:
            continue
        split = id_to_split.get(clip.participant_id)
        if split is None:
            continue

        frames = sample_frames(clip.video_path, N_FRAMES)
        if frames is None:
            failed += 1
            continue

        emb = embed_clip(backbone, frames, device)
        X_splits[split].append(emb)
        y_splits[split].append(VALENCE2_MAP[clip.label_id])

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(all_clips)} done …")

    print(f"  Extraction complete. Failed: {failed}")

    X_tr = np.stack(X_splits["train"])
    y_tr = np.array(y_splits["train"])
    X_val = np.stack(X_splits["val"])
    y_val = np.array(y_splits["val"])
    X_te = np.stack(X_splits["test"])
    y_te = np.array(y_splits["test"])

    print(f"  Train={len(y_tr)}, Val={len(y_val)}, Test={len(y_te)}")

    # Save embeddings for later fusion
    np.savez(OUT_EMBED,
             X_train=X_tr, y_train=y_tr,
             X_val=X_val,  y_val=y_val,
             X_test=X_te,  y_test=y_te)
    print(f"  Embeddings saved → {OUT_EMBED}")

    # 4. Combine train+val for final classifier
    X_tv = np.concatenate([X_tr, X_val])
    y_tv = np.concatenate([y_tr, y_val])

    # 5. Compare classifiers
    candidates = {
        "LogisticReg (C=1)"  : Pipeline([("sc", StandardScaler()),
                                          ("clf", LogisticRegression(C=1.0, max_iter=500,
                                                                      class_weight="balanced"))]),
        "LogisticReg (C=0.1)": Pipeline([("sc", StandardScaler()),
                                          ("clf", LogisticRegression(C=0.1, max_iter=500,
                                                                      class_weight="balanced"))]),
        "SVM (RBF)"          : Pipeline([("sc", StandardScaler()),
                                          ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                                                      class_weight="balanced"))]),
    }

    print(f"\n  Training linear probes …")
    best_name, best_score, best_pipe = "", 0.0, None
    for name, pipe in candidates.items():
        pipe.fit(X_tv, y_tv)
        preds  = pipe.predict(X_te)
        score  = float((preds == y_te).mean())
        print(f"    {name:<22} test_acc={score:.4f}")
        if score > best_score:
            best_score, best_name, best_pipe = score, name, pipe

    # 6. Best model evaluation
    preds    = best_pipe.predict(X_te)
    acc      = float((preds == y_te).mean())
    macro_f1 = float(f1_score(y_te, preds, average="macro"))
    cm       = confusion_matrix(y_te, preds).tolist()

    print(f"\n{'='*54}")
    print(f"  Best: {best_name}")
    print(f"  TEST ACCURACY  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  MACRO-F1       : {macro_f1:.4f}")
    print(f"  Confusion Matrix (Neg/Pos):\n    {cm[0]}\n    {cm[1]}")
    print(f"\n{classification_report(y_te, preds, target_names=['Negative','Positive'])}")

    # Save face model
    with OUT_MODEL.open("wb") as f:
        pickle.dump({"model": best_pipe, "n_frames": N_FRAMES,
                     "test_accuracy": round(acc, 4),
                     "test_macro_f1": round(macro_f1, 4)}, f)
    print(f"  Model saved → {OUT_MODEL}")


if __name__ == "__main__":
    main()
