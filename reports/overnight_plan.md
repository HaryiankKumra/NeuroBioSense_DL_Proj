# Overnight Plan (Auto-run)

This plan runs three experiments sequentially and then writes a comparison report.

## Experiments
1. Face-only baseline
   - Face branch enabled, signal branch disabled.
2. Signal-only baseline
   - Signal branch enabled, face branch disabled.
3. Multimodal tuned
   - Combined architecture with tuned loss/sampling controls.

## Pipeline Details
- Video preprocessing:
  - Sample every 4th frame.
  - Temporal jitter in training.
  - Sliding windows (T_v=10, stride=5).
- Face encoder:
  - FaceNet (InceptionResnetV1, pretrained VGGFace2) + projection + BiLSTM + temporal attention.
- Signal encoder:
  - Channel-attention + CNN + BiLSTM + temporal attention.
- Fusion/classifier:
  - Cross-modal attention + soft-gating fusion + classifier.

## Outputs
- Artifacts (JSON and checkpoint) for each run in artifacts/.
- Logs for each run in logs/.
- Comparison report:
  - reports/overnight_comparison.md
  - reports/overnight_comparison.tex

## Launch Command
Run from project root:

```bash
caffeinate -dimsu -t 32400 bash emotion_recognition/scripts/run_overnight_suite.sh
```

This keeps the laptop awake for ~9 hours while charging.
