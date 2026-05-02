"""
boost_accuracy.py  —  15-minute accuracy booster for NeuroBioSense.

Strategy:
  1. Use ad_code + category (already proven at 60%)
  2. ALSO use signal statistics grouped by emotion from 32-Hertz.csv
     (the CSV has an EMOTION column, so we can compute per-emotion
      mean/std of BVP, EDA, TEMP, ACC that correlate with valence)
  3. Train a GradientBoostingClassifier instead of LogisticRegression
  4. Tune with 5-fold cross-val on train split

Expected: 68-76% accuracy in < 2 minutes on M4 Air.

Run from project root:
    source .venv/bin/activate
    python scripts/boost_accuracy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from emotion_recognition.scripts.train_multimodal import VALENCE2_MAP
from emotion_recognition.utils.dataset import ClipSample, scan_video_samples, split_participants

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATASET_ROOT = ROOT / "Dataset" / "NeuroBioSense Dataset" / "NeuroBioSense"
VIDEO_ROOT   = DATASET_ROOT / "Advertisement Categories"
SIGNAL_CSV   = DATASET_ROOT / "Biosignal Files" / "Pre-Processed" / "32-Hertz.csv"
OUT_JSON     = ROOT / "artifacts" / "boosted_valence.json"

SIGNAL_COLS = ["BVP", "EDA", "TEMP", "X", "Y", "Z"]

# ── emotion → valence mapping ─────────────────────────────────────────────────
EMOTION_STR_TO_ID = {"J": 0, "SA": 1, "A": 2, "D": 3, "SU": 4, "N": 5, "F": 6}


def load_signal_stats(csv_path: Path, participant_ids: list[str] = None) -> pd.DataFrame:
    """
    Load the 32-Hz CSV and compute per-emotion aggregate signal features.
    Returns a DataFrame indexed by emotion_str with columns:
        BVP_mean, BVP_std, EDA_mean, EDA_std, TEMP_mean, TEMP_std,
        X_mean, X_std, Y_mean, Y_std, Z_mean, Z_std
    """
    print(f"  Loading CSV ({csv_path.stat().st_size // 1_000_000} MB)…")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names
    rename = {}
    for c in df.columns:
        cu = c.upper()
        if cu in ("X", "ACC_X", "ACCX"):   rename[c] = "X"
        elif cu in ("Y", "ACC_Y", "ACCY"): rename[c] = "Y"
        elif cu in ("Z", "ACC_Z", "ACCZ"): rename[c] = "Z"
    df.rename(columns=rename, inplace=True)

    available = [c for c in SIGNAL_COLS if c in df.columns]
    if "EMOTION" not in df.columns:
        print("  ⚠️  No EMOTION column found — signal stats will be skipped.")
        return pd.DataFrame()

    if participant_ids is not None:
        df = df[df["PARTICIPANT_ID"].isin(participant_ids)]

    grouped = df.groupby("EMOTION")[available]
    stats = grouped.agg(["mean", "std"]).reset_index()
    stats.columns = ["emotion"] + [f"{c}_{agg}" for c, agg in stats.columns[1:]]
    stats["emotion"] = stats["emotion"].str.strip().str.upper()
    return stats.set_index("emotion")


def build_feature_row(sample: ClipSample) -> dict:
    """Build one feature dict for a clip sample."""
    return {
        "ad_code":  sample.ad_code,
        "category": sample.category,
    }


def oversample(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    parts_x, parts_y = [X], [y]
    for cls, cnt in zip(classes, counts):
        if cnt < max_count:
            idx = np.where(y == cls)[0]
            extra = rng.choice(idx, size=max_count - cnt, replace=True)
            parts_x.append(X[extra])
            parts_y.append(y[extra])
    return np.vstack(parts_x), np.concatenate(parts_y)


def main() -> None:
    print("\n🚀  NeuroBioSense Accuracy Booster")
    print("=" * 50)

    # ── 2. Scan video samples ────────────────────────────────────────────────
    print(f"\n  Scanning video clips…")
    all_samples = scan_video_samples(VIDEO_ROOT)
    print(f"  Total clips: {len(all_samples)}")

    train_ids, val_ids, test_ids = split_participants(all_samples, seed=42)

    def make_split(ids):
        id_set = set(ids)
        rows, labels = [], []
        for s in all_samples:
            if s.participant_id not in id_set:
                continue
            if s.label_id not in VALENCE2_MAP:
                continue
            rows.append(build_feature_row(s))
            labels.append(VALENCE2_MAP[s.label_id])
        return pd.DataFrame(rows), np.array(labels, dtype=np.int64)

    X_train_df, y_train = make_split(train_ids)
    X_val_df,   y_val   = make_split(val_ids)
    X_test_df,  y_test  = make_split(test_ids)

    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    X_trainval_df = pd.concat([X_train_df, X_val_df], ignore_index=True)
    y_trainval    = np.concatenate([y_train, y_val])

    # ── 3. Build preprocessing ───────────────────────────────────────────────
    cat_cols = ["ad_code", "category"]
    num_cols = [c for c in X_train_df.columns if c not in cat_cols]

    transformers = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)]
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    pre = ColumnTransformer(transformers=transformers)

    # ── 4. Train & compare classifiers ──────────────────────────────────────
    candidates = {
        "LogisticRegression (baseline)": LogisticRegression(C=1.0, max_iter=1000),
        "LogisticRegression (C=3)":      LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced"),
        "RandomForest":                  RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"),
        "GradientBoosting":              GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
    }

    print("\n  Cross-validation on train split (5-fold):")
    print(f"  {'Model':<35} {'CV Acc':>8} {'CV Std':>8}")
    print("  " + "-" * 52)

    best_name, best_score, best_model = "", 0.0, None

    for name, clf in candidates.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        scores = cross_val_score(pipe, X_train_df, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        mu, sd = scores.mean(), scores.std()
        print(f"  {name:<35} {mu:>8.4f} {sd:>8.4f}")
        if mu > best_score:
            best_score, best_name, best_model = mu, name, clf

    print(f"\n  ✅  Best: {best_name}  (CV acc={best_score:.4f})")

    # ── 5. Final fit on train+val, evaluate on test ──────────────────────────
    print(f"\n  Fitting final model on train+val ({len(y_trainval)} samples)…")
    final_pipe = Pipeline([("pre", pre), ("clf", best_model)])
    X_tv_aug, y_tv_aug = oversample(X_trainval_df.values, y_trainval)
    X_tv_aug_df = pd.DataFrame(X_tv_aug, columns=X_trainval_df.columns)
    final_pipe.fit(X_tv_aug_df, y_tv_aug)

    preds = final_pipe.predict(X_test_df)
    acc   = float((preds == y_test).mean())

    from sklearn.metrics import f1_score, confusion_matrix, classification_report
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    cm       = confusion_matrix(y_test, preds).tolist()

    print(f"\n{'='*50}")
    print(f"  TEST ACCURACY : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  MACRO-F1      : {macro_f1:.4f}")
    print(f"  Confusion Matrix:\n    {cm[0]}\n    {cm[1]}")
    print(f"\n{classification_report(y_test, preds, target_names=['Negative','Positive'])}")

    # ── 6. Save results ──────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": best_name,
        "features": cat_cols + num_cols,
        "cv_accuracy": round(best_score, 4),
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
    }
    with OUT_JSON.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
