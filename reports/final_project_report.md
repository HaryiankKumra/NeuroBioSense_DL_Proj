# Final Project Report: Ad Impact Prediction (Binary Valence)

Goal: predict ad impact while watching video as **negative** or **positive** from face, physiology, and multimodal input.

## Final Results

| Model | Artifact | Test Accuracy | Test Macro-F1 | Best Epoch | Best Val Macro-F1 |
|---|---|---:|---:|---:|---:|
| Facial Only | final_valence_face_only | 0.4123 | 0.2919 | 1 | 0.2600 |
| Physiological Only | final_valence_signal_only | 0.4123 | 0.2919 | 1 | 0.2600 |
| Multimodal | final_valence_multimodal | 0.4123 | 0.2919 | 1 | 0.2600 |

Best current model by macro-F1: Facial Only (macro-F1=0.2919, acc=0.4123).

## Dataset Management (How labels are handled)

1. Face stream uses video clips from advertisement categories.
2. Physiology stream uses 32-Hz biosignal features: BVP, EDA, TEMP, ACC X/Y/Z.
3. Label target is binary valence mapping from emotion labels:
   - Positive: Joy, Surprise
   - Negative: Sadness, Anger, Disgust, Fear
4. Neutral is excluded for the binary task.
5. Participant-level split is used to reduce identity leakage.
6. If strict participant/ad alignment keys are missing in processed biosignal CSV, fallback segment selection is label-agnostic.

## Data Augmentation and Training Controls

1. Temporal video window sampling with jitter in training.
2. Dataset repeat expansion per epoch (augment repeats).
3. Balanced sampler with capped oversampling ratio.
4. Focal loss with label smoothing for class-imbalance robustness.
5. Modality-specific ablations for facial-only and physiological-only baselines.

## Architecture and Data Flow

Architecture and data-flow diagrams are included in the LaTeX report.

## Reproducibility

Run the full suite script:

```bash
./emotion_recognition/scripts/run_final_project_suite.sh
```
