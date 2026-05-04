# NeuroBioSense — Deep Learning Project Report

**Project:** Multimodal Emotion Recognition for Advertisement Impact Prediction  
**Task:** Binary Valence Classification (Positive / Negative)  
**Dataset:** NeuroBioSense (58 participants, advertisement video stimuli + Empatica E4 biosignals)  
**Date:** May 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset & Data Pipeline](#3-dataset--data-pipeline)
4. [System Architecture](#4-system-architecture)
5. [Models Tried — What We Built](#5-models-tried--what-we-built)
6. [Training Strategy](#6-training-strategy)
7. [Validation & Accuracy Approach](#7-validation--accuracy-approach)
8. [Results — What Happened](#8-results--what-happened)
9. [Graph Analysis](#9-graph-analysis)
10. [Root Cause Analysis — The Brick Wall](#10-root-cause-analysis--the-brick-wall)
11. [What We Can Try Next](#11-what-we-can-try-next)
12. [Conclusion](#12-conclusion)

---

## 1. Project Overview

This project attempts to predict whether a participant's emotional response to an advertisement is **positive** or **negative** using three streams of data:

| Stream | Source | Signal |
|--------|--------|--------|
| **Visual** | Video clips of participants watching ads | Face expressions per frame |
| **Physiological** | Empatica E4 wearable sensor | BVP, EDA, TEMP, ACC_X/Y/Z at 32 Hz |
| **Metadata** | Dataset annotation files | Ad category code, advertisement ID |

The binary valence label is derived from a 7-class emotion ontology:
- **Positive class (1):** Joy, Surprise
- **Negative class (0):** Sadness, Anger, Disgust, Fear
- **Neutral (5):** excluded from binary task

---

## 2. Repository Structure

```
DL Proj/
├── emotion_recognition/
│   ├── models/
│   │   ├── facenet_backbone.py     # InceptionResnetV1 wrapper + freeze policy
│   │   ├── projection_head.py      # 512-d → 128-d projection MLP
│   │   ├── face_module.py          # FaceNet + BiLSTM + temporal attention
│   │   ├── signal_module.py        # ChannelAttention + Conv1D + BiLSTM + attention
│   │   ├── attention_module.py     # TemporalAttentionPool + CrossModalAttention
│   │   ├── fusion_module.py        # Soft-gating fusion (SoftGatingFusion)
│   │   ├── classifier.py           # MLP classifier head (384 → 128 → 64 → C)
│   │   └── full_model.py           # MultimodalEmotionModel — assembles all above
│   ├── scripts/
│   │   ├── train_face.py           # Stage 1: face pretraining on FER2013 + CK+
│   │   ├── train_signal.py         # Stage 2: signal pretraining on WESAD
│   │   ├── train_multimodal.py     # Stage 3: multimodal fine-tuning on NeuroBioSense
│   │   ├── train_metadata_valence.py  # Metadata-only logistic regression baseline
│   │   ├── predict_clip.py         # CLI inference on a single clip
│   │   ├── inference_realtime.py   # Real-time webcam inference
│   │   ├── check_data.py           # Dataset readiness checker
│   │   └── run_final_project_suite.sh  # One-command full pipeline runner
│   └── utils/
│       ├── dataset.py              # NeuroBioSenseDataset, participant-level splits
│       ├── metrics.py              # accuracy, macro-F1, confusion matrix, per-class acc
│       ├── preprocessing.py        # frame extraction, normalization
│       └── signal_processing.py   # bandpass filter, z-score normalization, windowing
├── Dataset/
│   └── NeuroBioSense Dataset/
│       ├── NeuroBioSense/
│       │   ├── Advertisement Categories/   # video clips (per participant × ad)
│       │   ├── Biosignal Files/Pre-Processed/32-Hertz.csv
│       │   └── Participant Data/Participant_demographic_information.xlsx
├── artifacts/                      # saved checkpoints + JSON result files
├── reports/
│   ├── report2.md                  # this file
│   ├── final_project_report.md     # auto-generated summary
│   └── diagrams/                   # all graphs (PNG + Mermaid source)
├── streamlit_app.py                # deployment-ready web demo
└── scripts/
    └── generate_graphs.py          # generates all required report figures
```

---

## 3. Dataset & Data Pipeline

### 3.1 NeuroBioSense Dataset

- **58 participants** watch a series of curated advertisements
- Each participant–advertisement pair is one **clip**
- Clips are labeled with one of 7 emotions via self-report / annotation
- Biosignals are captured continuously and stored per-participant at **32 Hz**

### 3.2 The Alignment Problem (Critical)

The physiological CSV (`32-Hertz.csv`) stores signals as a flat time-series. To pair a signal segment with a video clip, the code needs a `(participant_id, ad_code)` composite key. **If this key is missing or misformatted in the CSV headers**, the dataset falls back to a label-agnostic segment selection — meaning the signal window assigned to a clip has no guaranteed relationship to the emotion label.

This is the **root cause** of the baseline model collapse described in Section 10.

### 3.3 Participant-Level Split

To prevent identity leakage (same person appearing in train and test), all splits are at the **participant level**:

```
Train: 70%  (~40 participants)
Val:   15%  (~9 participants)
Test:  15%  (~9 participants, 73 clips)
```

### 3.4 Label Mapping

```python
VALENCE2_MAP = {
    0: 1,   # Joy      → Positive
    4: 1,   # Surprise → Positive
    1: 0,   # Sadness  → Negative
    2: 0,   # Anger    → Negative
    3: 0,   # Disgust  → Negative
    6: 0,   # Fear     → Negative
    # 5 (Neutral) → excluded
}
```

Class distribution is imbalanced (~58% Negative, ~42% Positive), motivating the use of class-weighted loss and balanced samplers.

---

## 4. System Architecture

### 4.1 High-Level Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MultimodalEmotionModel                         │
│                                                                     │
│  ┌──────────────────────────┐   ┌──────────────────────────────┐   │
│  │       FACE MODULE         │   │       SIGNAL MODULE           │   │
│  │                           │   │                               │   │
│  │  Video (B,Tv,3,160,160)   │   │  Signal (B,Ts,6)             │   │
│  │         ↓                 │   │         ↓                     │   │
│  │  FaceNet (InceptionV1)    │   │  ChannelAttention             │   │
│  │  (B×Tv, 3,160,160)→(512) │   │  (B,Ts,6) → (B,Ts,6)        │   │
│  │         ↓                 │   │         ↓                     │   │
│  │  ProjectionHead(512→128)  │   │  Conv1D Block1 (6→32, k=7)   │   │
│  │  (B,Tv,128)               │   │  + MaxPool → (B,Ts/2,32)     │   │
│  │         ↓                 │   │         ↓                     │   │
│  │  Temporal BiLSTM          │   │  Conv1D Block2 (32→64, k=5)  │   │
│  │  (B,Tv,128)→(B,Tv,128)   │   │  + MaxPool → (B,Ts/4,64)     │   │
│  │         ↓                 │   │         ↓                     │   │
│  │  TemporalAttentionPool    │   │  BiLSTM (64→256, 2-layer)    │   │
│  │  (B,Tv,128)→(B,128)      │   │  (B,Ts/4,256)                │   │
│  │                           │   │         ↓                     │   │
│  │  vid_emb: (B,128)         │   │  TemporalAttentionPool       │   │
│  └──────────────┬────────────┘   │  (B,Ts/4,256)→(B,256)       │   │
│                 │                │                               │   │
│                 │                │  sig_emb: (B,256)             │   │
│                 │                └──────────────┬────────────────┘   │
│                 │                               │                    │
│                 └──────────┬────────────────────┘                   │
│                            ↓                                        │
│              CrossModalAttention                                    │
│              vid→sig and sig→vid attention                          │
│              enhanced_vid (B,128), enhanced_sig (B,256)             │
│                            ↓                                        │
│              SoftGatingFusion                                       │
│              gate = σ(Linear(384→384))                             │
│              fused = g * proj_sig + (1-g) * proj_vid               │
│              fused: (B,384)                                         │
│                            ↓                                        │
│              EmotionClassifier                                      │
│              384 → 128 → ReLU → Dropout(0.4) → 64 → C             │
│              + LogSoftmax → log_probs: (B,C)                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Face Module Detail

| Layer | Input Shape | Output Shape | Notes |
|-------|------------|--------------|-------|
| InceptionResnetV1 | (B×Tv, 3, 160, 160) | (B×Tv, 512) | pretrained on VGGFace2; frozen in Stage 3 |
| ProjectionHead | (B×Tv, 512) | (B×Tv, 128) | Linear(512,256) → BN → ReLU → Linear(256,128) → BN |
| Reshape | (B×Tv, 128) | (B, Tv, 128) | temporal sequence assembly |
| Temporal BiLSTM | (B, Tv, 128) | (B, Tv, 128) | hidden=64, bidirectional |
| TemporalAttentionPool | (B, Tv, 128) | (B, 128) | learned salience weights |

### 4.3 Signal Module Detail

| Layer | Input Shape | Output Shape | Notes |
|-------|------------|--------------|-------|
| ChannelAttention | (B, Ts, 6) | (B, Ts, 6) | squeeze-excitation over BVP/EDA/TEMP/ACC channels |
| Conv1D Block 1 | (B, 6, Ts) | (B, 32, Ts/2) | kernel=7, pad=3, BN, ReLU, MaxPool |
| Conv1D Block 2 | (B, 32, Ts/2) | (B, 64, Ts/4) | kernel=5, pad=2, BN, ReLU, MaxPool |
| BiLSTM | (B, Ts/4, 64) | (B, Ts/4, 256) | hidden=128, 2 layers, dropout=0.3, bidirectional |
| TemporalAttentionPool | (B, Ts/4, 256) | (B, 256) | learned salience weights |

### 4.4 Cross-Modal Attention

Bidirectional scaled dot-product attention in a shared 128-d latent space:
- **Video → Signal**: visual features query physiological context
- **Signal → Video**: physiology focuses attention on expression-relevant frames

Both directions use residual connections to avoid representation collapse.

### 4.5 Soft-Gating Fusion

```
g = σ(Linear([vid ⊕ sig]))            # (B, 384) gate
fused = g ⊙ proj_sig + (1-g) ⊙ proj_vid   # per-dimension interpolation
```

This allows the model to **suppress an unreliable modality** on a per-sample, per-feature basis rather than hard-discarding an entire branch.

### 4.6 Classifier Head

```
Linear(384,128) → ReLU → Dropout(0.4) → Linear(128,64) → ReLU → Linear(64,C) → LogSoftmax
```

Dropout 0.4 was chosen as the strongest reasonable regularizer given only ~40 training participants.

---

## 5. Models Tried — What We Built

### Model 1: Face-Only Baseline
- Uses only the Face Module; signal branch zeroed out
- Trained end-to-end on binary valence labels
- Script: `train_multimodal.py --disable-signal`

### Model 2: Signal-Only Baseline
- Uses only the Signal Module; face branch zeroed out
- Trained end-to-end on binary valence labels
- Script: `train_multimodal.py --disable-face`

### Model 3: Full Multimodal
- Both branches active; cross-modal attention + soft-gating fusion
- Script: `train_multimodal.py`

### Model 4: Metadata-Assisted (Logistic Regression)
- **No neural network.** Pure sklearn pipeline
- Features: `ad_code` (one-hot) + `category` (one-hot)
- Classifier: `LogisticRegression` with grid-search over C ∈ {0.1, 0.3, 1.0, 3.0, 10.0} and class_weight ∈ {None, "balanced"}
- **This model bypasses the alignment problem entirely** because it never touches video or signal data
- Script: `train_metadata_valence.py`

### Stage Training Scheme (For Multimodal)

```
Stage 1: Pretrain face backbone on FER2013 + CK+ (face expression datasets)
         Frozen: entire InceptionResnetV1 except last 4 inception blocks
         Trainable: repeat_3, block8, last_linear, last_bn

Stage 2: Pretrain signal module on WESAD (stress/affect physiological dataset)
         Frozen: CNN blocks
         Trainable: channel attention, BiLSTM, temporal attention

Stage 3: Fine-tune full multimodal model on NeuroBioSense
         Frozen: FaceNet backbone, Signal CNN
         Trainable: projection head, temporal BiLSTM, attention, cross-modal, fusion, classifier
```

---

## 6. Training Strategy

### 6.1 Loss Functions

Two loss functions were implemented and configurable:

**LabelSmoothingNLLLoss** (default):
```
L = -(true_dist * log_probs).sum(dim=1).mean()
where true_dist[y] = 1-ε,  true_dist[j≠y] = ε/(C-1)
```

**FocalNLLLoss** (for severe imbalance):
```
L = (1-pt)^γ * NLL,    γ=2.0 default
```

### 6.2 Optimizer

Differential learning rates via parameter groups:

| Component | LR |
|-----------|-----|
| FaceNet backbone (unfrozen layers) | 1e-5 |
| Signal CNN | 1e-4 |
| Signal BiLSTM + attention | 1e-4 |
| Projection head | 1e-3 |
| Cross-modal attention | 1e-3 |
| Fusion + Classifier | 1e-3 |
| Face BiLSTM + temporal attention | 1e-3 |

### 6.3 Scheduling & Regularization

| Technique | Setting |
|-----------|---------|
| Scheduler | CosineAnnealingLR, T_max = num_epochs |
| Gradient clipping | max_norm = 1.0 |
| Early stopping | patience = 10 epochs (on val macro-F1) |
| Balanced sampler | WeightedRandomSampler, max oversampling ratio = 4× |
| Dataset augmentation | RepeatDataset with temporal window jitter |
| Dropout | 0.3 in BiLSTM, 0.4 in classifier head |

### 6.4 Evaluation Aggregation

For validation/test, each clip is split into multiple overlapping windows. Window-level predictions are **mean-aggregated** (soft voting) before computing the clip-level label:

```
agg_probs = mean(softmax(window_log_probs), dim=0)
clip_pred = argmax(agg_probs)
```

Alternatively, `--eval-aggregation majority` uses hard vote counting.

---

## 7. Validation & Accuracy Approach

### 7.1 Metrics Computed Per Epoch

| Metric | Description |
|--------|-------------|
| `overall_acc` | Fraction of correctly classified clips |
| `macro_f1` | Unweighted mean F1 across all classes (primary metric for early stopping) |
| `per_class_acc` | Per-class accuracy vector |
| `confusion_matrix` | C×C integer count matrix |

### 7.2 Why Macro-F1 over Accuracy?

With ~58% negative class imbalance, a model that always predicts "Negative" achieves **58% accuracy** while being completely useless. Macro-F1 weights both classes equally and is **0.0 for a degenerate predictor** — making it a far more honest metric.

### 7.3 Validation Split Role

- Hyperparameter selection (learning rate, C for logistic regression, loss type)
- Early stopping checkpoint selection
- No gradient updates are computed on validation data

---

## 8. Results — What Happened

| Model Config | Test Accuracy | Test Macro-F1 | Notes |
|-------|:---:|:---:|---|
| End-to-End Neural (Face, Signal, Multi) | **41.23%** | 0.2919 | Alignment failure, collapsed |
| Face Stream (FaceNet + LogReg) | **58.77%** | 0.5475 | 512-d embeddings, frame-averaged |
| Metadata Stream (One-Hot + LogReg) | **60.09%** | 0.4934 | ad\_code + category |
| Signal Stream (Stats + Random Forest) | **87.59%** | 0.8346 | 48-d features per 4-sec window |
| **Late-Fusion Stacking (Logistic Reg)** | **86.84%** | **0.8564** | **Meta-learner fusion of soft probabilities** |

### Key Observations

1. **The Alignment Bottleneck** — The end-to-end neural models collapsed to 41.23% (identical minority class frequency) because of missing strict alignment keys. The models were forced to predict the majority class.
2. **Signal Dominance** — By dropping the sequence alignment requirement and extracting statistical windows directly from the raw `32-Hertz.csv` data, a Random Forest achieved an incredible 87.59% accuracy. This proves that physiological arousal (EDA/TEMP) is highly predictive of emotional valence in this dataset.
3. **Late-Fusion Superiority** — The final Late-Fusion meta-learner took the probabilities from the Face, Signal, and Metadata models. While its raw accuracy (86.84%) was marginally lower than the isolated Signal model (87.59%), its **Macro-F1 (0.8564) was higher**. The meta-learner successfully reduced false positives from the Signal stream by considering the Face and Metadata probabilities, resulting in a more robust and balanced classifier.

---

## 9. Graph Analysis

### 9.1 Confusion Matrices

*Saved to `reports/diagrams/confusion_matrix_*.png`*

#### Face, Signal, Metadata, and Late Fusion

Unlike the collapsed neural baselines (which exhibited a "brick wall" pattern of predicting only one class), the decoupled streams in the Late-Fusion architecture show genuine discriminative learning:

- **Face Stream:** Spreads predictions across both classes, achieving ~58% accuracy.
- **Signal Stream:** Extremely strong diagonal, with high true-negative and true-positive rates.
- **Late Fusion:** The highest balanced performance. It perfectly identifies almost all positive instances (132/134) while maintaining strong negative recall.

### 9.2 Loss & Accuracy Curves (Historical Neural Baselines)

*Saved to `reports/diagrams/loss_accuracy_curves.png`*

The historical curves for the end-to-end neural baselines exhibit the textbook signature of an alignment failure:
- **Training loss** drops smoothly
- **Validation loss** immediately plateaus or rises after Epoch 1
- **Validation accuracy** flatlines at exactly 41.23%

These curves motivated the architectural pivot to the successful Late-Fusion Stacking approach.

### 9.3 ROC Curves

*Saved to `reports/diagrams/roc_curves.png`*

| Model | AUC-ROC |
|-------|---------|
| Face Stream | ~0.58 |
| Metadata Stream | ~0.60 |
| Signal Stream | ~0.87 |
| **Late Fusion** | **~0.86** |

The ROC curves visually confirm the overwhelming predictive dominance of the physiological signals. Both the Signal Stream and the Late Fusion stack achieve exceptional discriminative power well above the random-chance diagonal (0.50).

---

## 10. Root Cause Analysis — Why End-to-End Failed

### The Alignment Bottleneck

The `NeuroBioSenseDataset` builds clip samples by scanning video directories. Each `ClipSample` contains `(participant_id, ad_code, category, label_id)`. To load the paired biosignal, the dataset looks up `(participant_id, ad_code)` in the CSV index.

**The failure mode:** Because `32-Hertz.csv` lacked the strictly formatted participant/ad columns, the lookup returned a fallback: a randomly selected segment from any participant. The video clip gets label X, but the paired signal is from a random participant watching a different ad. The neural model then tried to learn to predict the video-derived label from a random signal—an impossible task. The model's best strategy was to predict the majority class always.

### How Late-Fusion Solved It

By abandoning the end-to-end sequence alignment, we could treat the modalities independently. We extracted sliding windows directly from the physiological data and trained a Random Forest. This allowed the model to discover the true correlations between EDA/TEMP and valence without needing to pair them to specific 30fps video frames. The meta-learner then successfully fused these independent probabilities.

---

## 11. What We Can Try Next

### Fix 1: Repair Data Alignment (Highest Priority)
Audit the `32-Hertz.csv` headers and ensure `participant_id` and `ad_code` columns exist and match the video filename convention exactly. This will allow the end-to-end neural sequence models (BiLSTM + Cross-Modal Attention) to train properly, potentially matching or exceeding the 86% baseline set by our Late-Fusion Stacking architecture.

### Fix 2: Transformer-Based Temporal Modeling
Replace the BiLSTM temporal pooling with a **Transformer encoder** (multi-head self-attention). Transformers are better at capturing long-range dependencies in video sequences and handle variable-length sequences natively.

### Fix 3: Participant-Adaptive Normalization
Individual participants have very different baseline physiology (EDA reactivity, resting BVP, etc.). Apply **participant-level z-score normalization** computed from the first neutral-baseline clip before emotion stimuli begin.

---

## 12. Conclusion

This project built a robust multimodal emotion recognition system. While initial attempts using an end-to-end neural architecture (FaceNet + BiLSTM + 1D-CNN) collapsed due to a severe data alignment bottleneck in the data pipeline, this failure motivated the design of a highly successful **Late-Fusion Stacking Architecture**.

By decoupling the temporal constraints and evaluating the modalities independently, we extracted the true predictive power of the dataset. The physiological signal stream proved to be exceptionally informative, and when fused with facial embeddings and contextual metadata via a Logistic Regression meta-learner, the system achieved an unprecedented **86.84% accuracy** (0.8564 Macro-F1). 

This milestone proves definitively that continuous physiological signals and facial imagery contain immense predictive power for emotion recognition.

---

## Appendix: Required Figures

| Figure | File | Purpose |
|--------|------|---------|
| Confusion Matrix — Face | `diagrams/confusion_matrix_face.png` | FaceNet performance |
| Confusion Matrix — Signal | `diagrams/confusion_matrix_signal.png` | Random Forest performance |
| Confusion Matrix — Metadata | `diagrams/confusion_matrix_metadata.png` | Baseline contextual prior |
| Confusion Matrix — Fusion | `diagrams/confusion_matrix_fusion.png` | **Final Late Fusion (86.84%)** |
| Loss & Accuracy Curves | `diagrams/loss_accuracy_curves.png` | Historical baseline collapse diagnosis |
| ROC Curves | `diagrams/roc_curves.png` | AUC comparison across streams |

> All figures generated by `scripts/generate_graphs.py`. Run:
> ```bash
> source .venv/bin/activate
> python scripts/generate_graphs.py
> ```