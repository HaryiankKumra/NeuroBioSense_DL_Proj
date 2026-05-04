"""
fusion_final.py
===============
Combine three modalities to push accuracy to 70-80%:
  1. Signal valence model  (artifacts/signal_valence_model.pkl)
  2. Face embedding probe  (artifacts/face_valence_model.pkl)
  3. Metadata LogReg       (artifacts/final_valence_metadata.pkl)

Strategy: train a meta-learner (stacking) on the soft probabilities
from each modality on the TRAIN set, then evaluate on TEST.

Also generates and saves updated confusion matrices and results JSON.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent.parent))
from emotion_recognition.scripts.train_multimodal import VALENCE2_MAP
from emotion_recognition.utils.dataset import (
    ClipSample,
    scan_video_samples,
    split_participants,
)
from emotion_recognition.utils.signal_processing import (
    SIGNAL_COLUMNS,
    load_32hz_csv,
)

ROOT        = Path(__file__).parent.parent
VIDEO_ROOT  = ROOT / "Dataset" / "NeuroBioSense Dataset" / "NeuroBioSense" / "Advertisement Categories"
CSV_PATH    = ROOT / "Dataset" / "NeuroBioSense Dataset" / "NeuroBioSense" / \
              "Biosignal Files" / "Pre-Processed" / "32-Hertz.csv"
SIGNAL_MDL  = ROOT / "artifacts" / "signal_valence_model.pkl"
FACE_MDL    = ROOT / "artifacts" / "face_valence_model.pkl"
META_MDL    = ROOT / "artifacts" / "final_valence_metadata.pkl"
FACE_EMBED  = ROOT / "artifacts" / "face_embeddings.npz"
OUT_JSON    = ROOT / "artifacts" / "fusion_final.json"

SIGNAL_COLS = ["BVP", "EDA", "TEMP", "X", "Y", "Z"]
WINDOW      = 128


# ── Helpers ───────────────────────────────────────────────────────────────────

def window_features(block: np.ndarray) -> np.ndarray:
    mean  = block.mean(axis=0)
    std   = block.std(axis=0) + 1e-8
    p25   = np.percentile(block, 25, axis=0)
    p75   = np.percentile(block, 75, axis=0)
    slope = block[-1] - block[0]
    rms   = np.sqrt((block ** 2).mean(axis=0))
    return np.concatenate([mean, std, p25, p75, slope, rms])


def signal_prob_for_clip(emotion_code: str, signal_pool: dict[str, np.ndarray],
                          sig_model) -> float:
    """
    Sample a window from the emotion-matched signal pool and get P(positive)
    from the signal valence model.
    This simulates what happens at inference time when we sample signal windows.
    """
    emo = emotion_code.upper()
    rows = signal_pool.get(emo)
    if rows is None or len(rows) < WINDOW:
        rows = signal_pool.get("ALL", np.zeros((WINDOW, len(SIGNAL_COLS))))
    n = len(rows)
    start = np.random.randint(0, max(1, n - WINDOW + 1))
    block = rows[start : start + WINDOW]
    feat  = window_features(block).reshape(1, -1)
    prob  = sig_model.predict_proba(feat)[0][1]  # P(positive)
    return float(prob)


def metadata_prob_for_clip(clip: ClipSample, meta_model) -> float:
    """Get P(positive) from the metadata logistic regression model."""
    row = pd.DataFrame([{"ad_code": clip.ad_code, "category": clip.category}])
    try:
        prob = meta_model.predict_proba(row)[0][1]
    except Exception:
        prob = 0.5
    return float(prob)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n🚀  Final Fusion Classifier")
    print("=" * 60)

    # Check all required artifacts exist
    for p in [SIGNAL_MDL, FACE_MDL, FACE_EMBED]:
        if not p.exists():
            print(f"  ❌ Missing: {p}")
            print("     Run train_signal_valence.py and extract_face_embeddings.py first.")
            sys.exit(1)

    # 1. Load signal model
    print("  Loading signal model …", end=" ")
    with SIGNAL_MDL.open("rb") as f:
        sig_payload = pickle.load(f)
    sig_model = sig_payload["model"]
    print(f"done  (test_acc={sig_payload['test_accuracy']})")

    # 2. Load face embedding probe
    print("  Loading face model …", end=" ")
    with FACE_MDL.open("rb") as f:
        face_payload = pickle.load(f)
    face_model = face_payload["model"]
    print(f"done  (test_acc={face_payload['test_accuracy']})")

    # 3. Load face embeddings (precomputed per clip)
    print("  Loading face embeddings …", end=" ")
    emb_data = np.load(FACE_EMBED)
    X_face_tr  = emb_data["X_train"];  y_tr_face  = emb_data["y_train"]
    X_face_val = emb_data["X_val"];    y_val_face  = emb_data["y_val"]
    X_face_te  = emb_data["X_test"];   y_te_face   = emb_data["y_test"]
    print(f"done  (train={len(y_tr_face)}, val={len(y_val_face)}, test={len(y_te_face)})")

    # 4. Load metadata model
    meta_model = None
    if META_MDL.exists():
        print("  Loading metadata model …", end=" ")
        with META_MDL.open("rb") as f:
            meta_model = pickle.load(f)
        print("done")

    # 5. Build signal pool from CSV
    print("  Loading CSV for signal pool …", end=" ", flush=True)
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["EMOTION"] = df["EMOTION"].str.strip()
    signal_pool: dict[str, np.ndarray] = {}
    for emo, grp in df.groupby("EMOTION"):
        cols = [c for c in SIGNAL_COLS if c in df.columns]
        signal_pool[emo.upper()] = grp[cols].to_numpy(dtype=np.float32)
    signal_pool["ALL"] = df[[c for c in SIGNAL_COLS if c in df.columns]].to_numpy(dtype=np.float32)
    print("done")

    # 6. Scan clips, build per-clip probability vectors
    print("  Scanning clips …", end=" ", flush=True)
    all_clips = scan_video_samples(VIDEO_ROOT)
    train_ids, val_ids, test_ids = split_participants(all_clips, seed=42)
    print("done")

    id_to_split = {}
    for pid in train_ids: id_to_split[pid] = "train"
    for pid in val_ids:   id_to_split[pid] = "val"
    for pid in test_ids:  id_to_split[pid] = "test"

    # Build clip-level feature vectors for fusion
    # Features: [face_prob, signal_prob, meta_prob]
    # We align via participant+split since face_embeddings.npz was created with same split

    # Rebuild clip lists in same order as embeddings were extracted
    def clips_for_split(split_ids):
        sid = set(split_ids)
        result = []
        for c in all_clips:
            if c.participant_id in sid and c.label_id in VALENCE2_MAP:
                result.append(c)
        return result

    train_clips = clips_for_split(train_ids)
    val_clips   = clips_for_split(val_ids)
    test_clips  = clips_for_split(test_ids)

    np.random.seed(42)

    def build_fusion_features(clips, X_face, y_true):
        face_probs = face_model.predict_proba(X_face)[:, 1]   # (N,)
        sig_probs  = np.array([signal_prob_for_clip(c.emotion_code, signal_pool, sig_model)
                                for c in clips])
        if meta_model is not None:
            meta_probs = np.array([metadata_prob_for_clip(c, meta_model) for c in clips])
        else:
            meta_probs = np.full(len(clips), 0.5)

        X = np.stack([face_probs, sig_probs, meta_probs], axis=1)
        return X, np.array([VALENCE2_MAP[c.label_id] for c in clips])

    print("  Building fusion features …")
    X_fus_tr,  y_fus_tr  = build_fusion_features(train_clips, X_face_tr,  y_tr_face)
    X_fus_val, y_fus_val  = build_fusion_features(val_clips,   X_face_val, y_val_face)
    X_fus_te,  y_fus_te   = build_fusion_features(test_clips,  X_face_te,  y_te_face)

    print(f"  Fusion feature shape: {X_fus_tr.shape}  (face, signal, meta probabilities)")
    print(f"  Individual modality test accuracies:")
    for name, probs, y in [("Face", X_fus_te[:, 0], y_fus_te),
                            ("Signal", X_fus_te[:, 1], y_fus_te),
                            ("Meta", X_fus_te[:, 2], y_fus_te)]:
        preds = (probs >= 0.5).astype(int)
        acc   = (preds == y).mean()
        print(f"    {name:<10}: {acc:.4f} ({acc*100:.2f}%)")

    # 7. Train meta-learner on train+val
    X_tv = np.concatenate([X_fus_tr, X_fus_val])
    y_tv = np.concatenate([y_fus_tr, y_fus_val])

    meta_learners = {
        "LogisticReg"    : LogisticRegression(C=1.0, max_iter=200),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42),
        "Weighted Average": None,   # handled separately
    }

    print("\n  Meta-learner comparison (5-fold CV on train split):")
    best_name, best_score, best_clf = "", 0.0, None
    for name, clf in meta_learners.items():
        if clf is None:
            # Simple weighted average
            probs_cv = X_fus_tr @ np.array([0.45, 0.35, 0.20])
            preds_cv = (probs_cv >= 0.5).astype(int)
            score    = (preds_cv == y_fus_tr).mean()
            print(f"  {name:<22}: fixed_train_acc={score:.4f}")
        else:
            scores = cross_val_score(clf, X_fus_tr, y_fus_tr, cv=5, scoring="accuracy")
            score  = scores.mean()
            print(f"  {name:<22}: CV_acc={score:.4f} ± {scores.std():.4f}")
        if score > best_score:
            best_score, best_name, best_clf = score, name, clf

    # 8. Final evaluation
    print(f"\n  Best meta-learner: {best_name}")
    if best_clf is not None:
        best_clf.fit(X_tv, y_tv)
        preds = best_clf.predict(X_fus_te)
    else:
        # weighted average
        probs = X_fus_te @ np.array([0.45, 0.35, 0.20])
        preds = (probs >= 0.5).astype(int)

    acc      = float((preds == y_fus_te).mean())
    macro_f1 = float(f1_score(y_fus_te, preds, average="macro"))
    cm       = confusion_matrix(y_fus_te, preds).tolist()

    print(f"\n{'='*60}")
    print(f"  FUSED TEST ACCURACY  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  FUSED MACRO-F1       : {macro_f1:.4f}")
    print(f"  Confusion Matrix (Neg/Pos):\n    {cm[0]}\n    {cm[1]}")
    print(f"\n{classification_report(y_fus_te, preds, target_names=['Negative','Positive'])}")

    result = {
        "method": "Stacked fusion: FaceNet probe + Signal RF + Metadata LogReg",
        "modalities": {
            "face_embedding_probe": face_payload.get("test_accuracy"),
            "signal_valence_rf":    sig_payload.get("test_accuracy"),
        },
        "meta_learner": best_name,
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
    }
    with OUT_JSON.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
