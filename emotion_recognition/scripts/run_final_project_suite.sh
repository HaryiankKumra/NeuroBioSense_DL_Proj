#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs artifacts reports

PY=".venv311/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

run_case() {
  local name="$1"
  shift

  local out_pth="artifacts/${name}.pth"
  local out_json="artifacts/${name}.json"
  local log_file="logs/${name}.log"

  echo "[FinalSuite] Running ${name}"
  rm -f "$out_pth" "$out_json" "$log_file"
  "$PY" -u -m emotion_recognition.scripts.train_multimodal "$@" --output "$out_pth" 2>&1 | tee "$log_file"
}

# You can override any of these from shell, for example:
# EPOCHS=20 AUG_REPEATS=4 ./emotion_recognition/scripts/run_final_project_suite.sh
EPOCHS="${EPOCHS:-12}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AUG_REPEATS="${AUG_REPEATS:-2}"
SAMPLER_MAX_RATIO="${SAMPLER_MAX_RATIO:-3.0}"

COMMON_ARGS=(
  --dataset-root Dataset
  --task valence2
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --augment-repeats "$AUG_REPEATS"
  --balanced-sampler
  --sampler-max-ratio "$SAMPLER_MAX_RATIO"
  --loss-type focal
  --focal-gamma 1.5
  --label-smoothing 0.05
  --eval-aggregation majority
  --facenet-stage1 artifacts/facenet_stage1.pth
  --signal-stage2 artifacts/signal_stage2.pth
)

# 1) Facial-only baseline.
run_case \
  final_valence_face_only \
  "${COMMON_ARGS[@]}" \
  --freeze-signal-all \
  --disable-signal \
  --unfreeze-face-last-block

# 2) Physiological-only baseline.
run_case \
  final_valence_signal_only \
  "${COMMON_ARGS[@]}" \
  --freeze-face-all \
  --disable-face \
  --unfreeze-signal-cnn

# 3) Full multimodal model.
run_case \
  final_valence_multimodal \
  "${COMMON_ARGS[@]}" \
  --unfreeze-face-last-block \
  --unfreeze-signal-cnn

# 4) Metadata-assisted binary valence baseline.
"$PY" -u -m emotion_recognition.scripts.train_metadata_valence \
  --dataset-root Dataset \
  --seed 42 \
  --skip-val-tuning \
  --c 1.0 \
  --class-weight none \
  --output-json artifacts/final_valence_metadata.json \
  --output-model artifacts/final_valence_metadata.pkl 2>&1 | tee logs/final_valence_metadata.log

"$PY" -m emotion_recognition.scripts.generate_final_project_report \
  --artifacts-dir artifacts \
  --report-md reports/final_project_report.md \
  --report-tex reports/final_project_report.tex

echo "[FinalSuite] Done. Reports generated in reports/final_project_report.md and reports/final_project_report.tex"
