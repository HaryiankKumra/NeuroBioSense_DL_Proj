#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs artifacts reports

PY=".venv311/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[ERROR] Python executable not found at $PY"
  exit 1
fi

run_case() {
  local name="$1"
  shift

  local out_pth="artifacts/${name}.pth"
  local out_json="artifacts/${name}.json"
  local log_file="logs/${name}.log"

  echo "[Suite] Running ${name}"
  rm -f "$out_pth" "$out_json" "$log_file"

  "$PY" -u -m emotion_recognition.scripts.train_multimodal "$@" --output "$out_pth" 2>&1 | tee "$log_file"
}

# Common settings tuned for overnight stability.
COMMON_ARGS=(
  --dataset-root Dataset
  --epochs 25
  --patience 10
  --batch-size 8
  --num-workers 2
  --augment-repeats 2
  --balanced-sampler
  --sampler-max-ratio 3.0
  --loss-type focal
  --focal-gamma 1.5
  --label-smoothing 0.05
  --eval-aggregation majority
)

# 1) Face-only baseline (signal branch disabled).
run_case \
  multimodal_stage3_face_only_overnight \
  "${COMMON_ARGS[@]}" \
  --facenet-stage1 artifacts/facenet_stage1.pth \
  --signal-stage2 artifacts/signal_stage2.pth \
  --freeze-signal-all \
  --disable-signal \
  --unfreeze-face-last-block

# 2) Signal-only baseline (face branch disabled).
run_case \
  multimodal_stage3_signal_only_overnight \
  "${COMMON_ARGS[@]}" \
  --facenet-stage1 artifacts/facenet_stage1.pth \
  --signal-stage2 artifacts/signal_stage2.pth \
  --freeze-face-all \
  --disable-face \
  --unfreeze-signal-cnn

# 3) Multimodal tuned.
run_case \
  multimodal_stage3_multimodal_overnight \
  "${COMMON_ARGS[@]}" \
  --facenet-stage1 artifacts/facenet_stage1.pth \
  --signal-stage2 artifacts/signal_stage2.pth \
  --unfreeze-face-last-block \
  --unfreeze-signal-cnn

# Build comparison report.
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path('.')
artifacts = root / 'artifacts'
reports = root / 'reports'
reports.mkdir(parents=True, exist_ok=True)

cases = [
    ('Face Only', artifacts / 'multimodal_stage3_face_only_overnight.json'),
    ('Signal Only', artifacts / 'multimodal_stage3_signal_only_overnight.json'),
    ('Multimodal', artifacts / 'multimodal_stage3_multimodal_overnight.json'),
]

rows = []
for label, p in cases:
    if p.exists():
        data = json.loads(p.read_text())
        rows.append((
            label,
            float(data.get('test_overall_acc', 0.0)),
            float(data.get('test_macro_f1', 0.0)),
            int(data.get('best_epoch', -1)),
            float(data.get('best_val_macro_f1', 0.0)),
        ))
    else:
        rows.append((label, None, None, None, None))

md = []
md.append('# Overnight Experiment Comparison')
md.append('')
md.append('## Pipeline Summary')
md.append('- Backbone: FaceNet (InceptionResnetV1 pretrained on VGGFace2) + Projection + BiLSTM + Temporal Attention.')
md.append('- Signal branch: Channel attention + CNN + BiLSTM + Temporal attention.')
md.append('- Fusion: Cross-modal attention + soft-gating fusion + classifier.')
md.append('- Temporal modeling: frame sampling, sliding windows, BiLSTM, and clip-level aggregation.')
md.append('')
md.append('## Results')
md.append('| Model | Test Accuracy | Test Macro-F1 | Best Epoch | Best Val Macro-F1 |')
md.append('|---|---:|---:|---:|---:|')
for label, acc, f1, be, bvf1 in rows:
    if acc is None:
        md.append(f'| {label} | NA | NA | NA | NA |')
    else:
        md.append(f'| {label} | {acc:.4f} | {f1:.4f} | {be} | {bvf1:.4f} |')

md.append('')
md.append('## Notes')
md.append('- NeuroBioSense processed 32-Hz file has no participant/ad/time IDs, limiting strict signal-video alignment.')
md.append('- This comparison reflects current data constraints and tuned training settings.')

(report_md := reports / 'overnight_comparison.md').write_text('\n'.join(md), encoding='utf-8')

tex = []
tex.append('\\documentclass{article}')
tex.append('\\usepackage[margin=1in]{geometry}')
tex.append('\\begin{document}')
tex.append('\\section*{Overnight Experiment Comparison}')
tex.append('\\subsection*{Pipeline}')
tex.append('FaceNet-based video branch with temporal BiLSTM-attention, signal branch with CNN-BiLSTM-attention, and fusion classifier.')
tex.append('\\subsection*{Results}')
tex.append('\\begin{tabular}{lrrrr}')
tex.append('\\hline')
tex.append('Model & Test Acc & Test Macro-F1 & Best Epoch & Best Val Macro-F1 \\\\')
tex.append('\\hline')
for label, acc, f1, be, bvf1 in rows:
    if acc is None:
        tex.append(f'{label} & NA & NA & NA & NA \\\\')
    else:
        tex.append(f'{label} & {acc:.4f} & {f1:.4f} & {be} & {bvf1:.4f} \\\\')
tex.append('\\hline')
tex.append('\\end{tabular}')
tex.append('\\subsection*{Remarks}')
tex.append('Processed 32-Hz biosignal data lacks participant/ad/time keys, which limits strict multimodal alignment and caps achievable performance.')
tex.append('\\end{document}')

(report_tex := reports / 'overnight_comparison.tex').write_text('\n'.join(tex), encoding='utf-8')
print(f'Wrote {report_md}')
print(f'Wrote {report_tex}')
PY

echo "[Suite] Completed all experiments and generated reports in reports/."
