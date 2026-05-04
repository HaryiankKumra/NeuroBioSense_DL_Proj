"""
train_signal_valence.py
=======================
Train a lightweight signal → binary-valence classifier using the EMOTION
column in 32-Hertz.csv.

The EMOTION column labels each signal row with the emotion that was
being experienced at that timestamp.  There is real physiological signal
here (EDA/BVP differ between positive and negative emotions), so we can
train a genuine classifier without label leakage:

  * At training time  : row-level (BVP/EDA/TEMP/X/Y/Z window) + EMOTION label
  * At inference time : only raw signal window features → predicted valence

Strategy
--------
1. Load CSV, map EMOTION → valence (drop Neutral).
2. Extract overlapping 128-sample windows → per-window statistical features.
3. 70/30 row-level train/test split (stratified by valence).
4. Train GradientBoosting, RF, and XGBoost-style GBM; pick best by CV.
5. Save the best model → artifacts/signal_valence_model.pkl
6. Print test accuracy and confusion matrix.

Expected test accuracy: 65-75% (EDA is a real valence predictor).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT       = Path(__file__).parent.parent
CSV_PATH   = ROOT / "Dataset" / "NeuroBioSense Dataset" / "NeuroBioSense" / \
             "Biosignal Files" / "Pre-Processed" / "32-Hertz.csv"
OUT_MODEL  = ROOT / "artifacts" / "signal_valence_model.pkl"
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

SIGNAL_COLS = ["BVP", "EDA", "TEMP", "X", "Y", "Z"]
WINDOW      = 128          # ~4 sec at 32 Hz
STEP        = 64           # 50% overlap
VALENCE_MAP = {"J": 1, "SU": 1, "SA": 0, "A": 0, "D": 0, "F": 0}  # N dropped


def window_features(block: np.ndarray) -> np.ndarray:
    """block: (WINDOW, 6) → 48-d feature vector."""
    mean  = block.mean(axis=0)
    std   = block.std(axis=0) + 1e-8
    p25   = np.percentile(block, 25, axis=0)
    p75   = np.percentile(block, 75, axis=0)
    slope = block[-1] - block[0]           # overall trend per channel
    rms   = np.sqrt((block ** 2).mean(axis=0))
    return np.concatenate([mean, std, p25, p75, slope, rms])   # (48,)


def extract_windows(df: pd.DataFrame):
    """Slide a window over each emotion block and extract features."""
    print("  Extracting windows …", flush=True)
    X_list, y_list = [], []

    for emo, valence in VALENCE_MAP.items():
        rows = df[df["EMOTION"] == emo][SIGNAL_COLS].to_numpy(dtype=np.float32)
        n = len(rows)
        for start in range(0, n - WINDOW + 1, STEP):
            block = rows[start : start + WINDOW]
            X_list.append(window_features(block))
            y_list.append(valence)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    print(f"  Extracted {len(X)} windows  (pos={y.sum()}, neg={len(y)-y.sum()})")
    return X, y


def main():
    print("\n🔬  Signal Valence Classifier Training")
    print("=" * 54)

    # 1. Load CSV
    print(f"  Loading CSV …", end=" ", flush=True)
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["EMOTION"] = df["EMOTION"].str.strip()
    print(f"done ({len(df):,} rows)")

    # 2. Extract features
    X, y = extract_windows(df)

    # 3. Train / test split (30 % test, stratified)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]
    print(f"  Train={len(y_tr):,}  Test={len(y_te):,}")

    # 4. Compare classifiers via 5-fold CV on train
    candidates = {
        "LogisticRegression" : Pipeline([("sc", StandardScaler()),
                                         ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced"))]),
        "RandomForest"       : Pipeline([("sc", StandardScaler()),
                                         ("clf", RandomForestClassifier(n_estimators=400, max_depth=10,
                                                                         class_weight="balanced", n_jobs=-1, random_state=42))]),
        "GradientBoosting"   : Pipeline([("sc", StandardScaler()),
                                         ("clf", GradientBoostingClassifier(n_estimators=400, max_depth=4,
                                                                              learning_rate=0.05, subsample=0.8,
                                                                              random_state=42))]),
    }

    print(f"\n  5-fold CV on train split:")
    print(f"  {'Model':<22} {'CV Acc':>8} {'±':>6}")
    print("  " + "-" * 38)
    best_name, best_score, best_pipe = "", 0.0, None

    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_tr, y_tr, cv=5,
                                 scoring="accuracy", n_jobs=-1)
        mu, sd = scores.mean(), scores.std()
        marker = " ← best" if mu > best_score else ""
        print(f"  {name:<22} {mu:>8.4f} {sd:>6.4f}{marker}")
        if mu > best_score:
            best_score, best_name, best_pipe = mu, name, pipe

    # 5. Final fit on full train, evaluate on test
    print(f"\n  Final fit: {best_name} on {len(y_tr):,} samples …", end=" ", flush=True)
    best_pipe.fit(X_tr, y_tr)
    print("done")

    preds    = best_pipe.predict(X_te)
    acc      = float((preds == y_te).mean())
    macro_f1 = float(f1_score(y_te, preds, average="macro"))
    cm       = confusion_matrix(y_te, preds).tolist()

    print(f"\n{'='*54}")
    print(f"  TEST ACCURACY  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  MACRO-F1       : {macro_f1:.4f}")
    print(f"  Confusion Matrix (Neg/Pos):\n    {cm[0]}\n    {cm[1]}")
    print(f"\n{classification_report(y_te, preds, target_names=['Negative','Positive'])}")

    # 6. Save model
    with OUT_MODEL.open("wb") as f:
        pickle.dump({"model": best_pipe, "window": WINDOW, "step": STEP,
                     "signal_cols": SIGNAL_COLS,
                     "test_accuracy": round(acc, 4),
                     "test_macro_f1": round(macro_f1, 4)}, f)
    print(f"  Saved → {OUT_MODEL}")


if __name__ == "__main__":
    main()
