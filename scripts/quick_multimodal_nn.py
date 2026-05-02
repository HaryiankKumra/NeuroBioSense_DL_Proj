"""
quick_multimodal_nn.py  —  20-minute lightweight multimodal neural net.

Architecture:
  - Signal Branch: sample rows from EMOTION-matched CSV rows →
    per-channel stats (mean/std/p25/p75) → 24-d vector → MLP
  - Metadata Branch: OneHot(ad_code + category) → Linear → 32-d
  - Fusion: concatenate → residual MLP → binary classifier
  - Device: Apple MPS (M4 Air) / CPU fallback
  - Expected runtime: 3-5 minutes, target accuracy: 60-68%

Run from project root:
    source .venv311/bin/activate
    python scripts/quick_multimodal_nn.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from emotion_recognition.scripts.train_multimodal import VALENCE2_MAP
from emotion_recognition.utils.dataset import ClipSample, scan_video_samples, split_participants

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATASET_ROOT = ROOT / "Dataset" / "NeuroBioSense Dataset" / "NeuroBioSense"
VIDEO_ROOT   = DATASET_ROOT / "Advertisement Categories"
SIGNAL_CSV   = DATASET_ROOT / "Biosignal Files" / "Pre-Processed" / "32-Hertz.csv"
OUT_JSON     = ROOT / "artifacts" / "quick_multimodal.json"

SIGNAL_COLS  = ["BVP", "EDA", "TEMP", "X", "Y", "Z"]
SEED         = 42
EPOCHS       = 80
BATCH_SIZE   = 32
LR           = 3e-3
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ── Load CSV and build per-emotion signal pool (train-only) ───────────────────
def load_signal_pool(csv_path: Path) -> dict[str, np.ndarray]:
    """Returns {emotion_code: np.ndarray of shape (N, 6)} from train clips only."""
    print(f"  Loading {csv_path.stat().st_size // 1_000_000} MB CSV…", end=" ", flush=True)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    for c in list(df.columns):
        cu = c.upper()
        if cu in ("X","ACC_X","ACCX"): df.rename(columns={c:"X"}, inplace=True)
        elif cu in ("Y","ACC_Y","ACCY"): df.rename(columns={c:"Y"}, inplace=True)
        elif cu in ("Z","ACC_Z","ACCZ"): df.rename(columns={c:"Z"}, inplace=True)

    avail = [c for c in SIGNAL_COLS if c in df.columns]
    pool: dict[str, np.ndarray] = {}
    if "EMOTION" in df.columns:
        for emo, grp in df.groupby(df["EMOTION"].str.strip().str.upper()):
            pool[emo] = grp[avail].to_numpy(dtype=np.float32)
    else:
        pool["ALL"] = df[avail].to_numpy(dtype=np.float32)
    print(f"done. Emotions: {list(pool.keys())}")
    return pool

def signal_stats(rows: np.ndarray) -> np.ndarray:
    """rows: (N,6) → 24-d feature vector (mean/std/p25/p75 per channel)."""
    if len(rows) == 0:
        return np.zeros(len(SIGNAL_COLS) * 4, dtype=np.float32)
    return np.concatenate([
        rows.mean(axis=0),
        rows.std(axis=0) + 1e-6,
        np.percentile(rows, 25, axis=0),
        np.percentile(rows, 75, axis=0),
    ]).astype(np.float32)

def clip_signal_feature(clip: ClipSample, pool: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Sample 256 rows from emotion-matched pool → 24-d stats."""
    emo = clip.emotion_code.upper()
    rows = pool.get(emo, pool.get("ALL", np.zeros((1, len(SIGNAL_COLS)), dtype=np.float32)))
    n = len(rows)
    idx = rng.choice(n, size=min(256, n), replace=False)
    return signal_stats(rows[idx])

# ── Build dataset ─────────────────────────────────────────────────────────────
def build_dataset(clips: list[ClipSample], pool: dict, ohe: OneHotEncoder | None, fit_ohe: bool):
    rng = np.random.default_rng(SEED)
    sig_feats, meta_feats, labels = [], [], []
    meta_raw = []

    for clip in clips:
        if clip.label_id not in VALENCE2_MAP:
            continue
        sig_feats.append(clip_signal_feature(clip, pool, rng))
        meta_raw.append([clip.ad_code, clip.category])
        labels.append(VALENCE2_MAP[clip.label_id])

    sig_arr  = np.stack(sig_feats)          # (N, 24)
    meta_arr = np.array(meta_raw, dtype=object)  # (N, 2)

    if fit_ohe:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(meta_arr)
    meta_enc = ohe.transform(meta_arr)      # (N, M)

    X = np.concatenate([sig_arr, meta_enc], axis=1).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, ohe

# ── Model ─────────────────────────────────────────────────────────────────────
class QuickMultimodalMLP(nn.Module):
    """Lightweight signal+metadata fusion MLP."""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.signal_head = nn.Sequential(
            nn.Linear(24, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.ReLU(),
        )
        meta_dim = input_dim - 24
        self.meta_head = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sig  = self.signal_head(x[:, :24])
        meta = self.meta_head(x[:, 24:])
        fused = torch.cat([sig, meta], dim=1)
        return self.fusion(fused)

# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, X_tr, y_tr, X_val, y_val, device, class_weights):
    model.to(device)
    cw  = torch.tensor(class_weights, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best_val_acc, best_state, patience_count = 0.0, None, 0
    PATIENCE = 20

    print(f"\n  {'Epoch':>5} {'TrLoss':>8} {'TrAcc':>7} {'VlAcc':>7}")
    print("  " + "-" * 32)

    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt), device=device)
        tr_loss_sum, tr_correct = 0.0, 0

        for i in range(0, len(Xt), BATCH_SIZE):
            b = idx[i:i + BATCH_SIZE]
            xb, yb = Xt[b], yt[b]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * len(b)
            tr_correct  += (logits.argmax(1) == yb).sum().item()

        sched.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_acc    = (val_logits.argmax(1) == yv).float().mean().item()

        tr_acc  = tr_correct / len(Xt)
        tr_loss = tr_loss_sum / len(Xt)

        if ep % 10 == 0 or ep == 1:
            print(f"  {ep:>5} {tr_loss:>8.4f} {tr_acc:>7.4f} {val_acc:>7.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop at epoch {ep}  (best val={best_val_acc:.4f})")
                break

    model.load_state_dict(best_state)
    return model

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    device = get_device()
    print(f"\n⚡  Quick Multimodal Neural Net  (device={device})")
    print("=" * 54)

    # 1. Load signal pool
    pool = load_signal_pool(SIGNAL_CSV)

    # 2. Video clips + participant split
    print("  Scanning video clips…", end=" ", flush=True)
    all_clips  = scan_video_samples(VIDEO_ROOT)
    train_ids, val_ids, test_ids = split_participants(all_clips, seed=SEED)
    def get(ids): return [c for c in all_clips if c.participant_id in set(ids)]
    tr_clips   = get(train_ids)
    val_clips  = get(val_ids)
    te_clips   = get(test_ids)
    print(f"done. Train={len(tr_clips)}, Val={len(val_clips)}, Test={len(te_clips)}")

    # 3. Build feature matrices
    print("  Building feature matrices…", end=" ", flush=True)
    X_tr, y_tr, ohe  = build_dataset(tr_clips,  pool, None, fit_ohe=True)
    X_val, y_val, _   = build_dataset(val_clips, pool, ohe,  fit_ohe=False)
    X_te,  y_te,  _   = build_dataset(te_clips,  pool, ohe,  fit_ohe=False)
    print(f"done. Input dim={X_tr.shape[1]}  (24 signal + {X_tr.shape[1]-24} metadata)")

    # Class weights to handle imbalance
    classes, counts = np.unique(y_tr, return_counts=True)
    cw = (len(y_tr) / (len(classes) * counts)).astype(np.float32)

    # 4. Train
    model = QuickMultimodalMLP(input_dim=X_tr.shape[1])
    model = train(model, X_tr, y_tr, X_val, y_val, device, cw)

    # 5. Evaluate on test
    model.eval()
    Xte_t = torch.from_numpy(X_te).to(device)
    with torch.no_grad():
        preds = model(Xte_t).argmax(1).cpu().numpy()

    acc      = float((preds == y_te).mean())
    macro_f1 = float(f1_score(y_te, preds, average="macro"))
    cm       = confusion_matrix(y_te, preds).tolist()

    elapsed = time.time() - t0
    print(f"\n{'='*54}")
    print(f"  ⏱  Total time   : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  TEST ACCURACY  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  MACRO-F1       : {macro_f1:.4f}")
    print(f"  Confusion Matrix (Neg/Pos):\n    {cm[0]}\n    {cm[1]}")
    print(f"\n{classification_report(y_te, preds, target_names=['Negative','Positive'])}")

    # Save
    result = {
        "model": "QuickMultimodalMLP (signal_stats + metadata, PyTorch MPS)",
        "input_features": {
            "signal": "mean/std/p25/p75 per channel from EMOTION-matched CSV rows (24-d)",
            "metadata": "OneHot(ad_code, category)",
        },
        "architecture": "signal_head(24→64→32) + meta_head(M→32) → fusion MLP(64→128→64→2)",
        "epochs": EPOCHS,
        "device": str(device),
        "elapsed_seconds": round(elapsed, 1),
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
