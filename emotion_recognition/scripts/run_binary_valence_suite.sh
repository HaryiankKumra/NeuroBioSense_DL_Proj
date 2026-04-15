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

  echo "[BinarySuite] Running ${name}"
  rm -f "$out_pth" "$out_json" "$log_file"
  "$PY" -u -m emotion_recognition.scripts.train_multimodal "$@" --output "$out_pth" 2>&1 | tee "$log_file"
}

COMMON_ARGS=(
  --dataset-root Dataset
  --task valence2
  --epochs 12
  --patience 5
  --batch-size 8
  --num-workers 0
  --augment-repeats 1
  --balanced-sampler
  --sampler-max-ratio 3.0
  --loss-type focal
  --focal-gamma 1.5
  --label-smoothing 0.05
  --eval-aggregation majority
  --facenet-stage1 artifacts/facenet_stage1.pth
  --signal-stage2 artifacts/signal_stage2.pth
)

# 1) Binary valence face-only.
run_case \
  binary_valence_face_only \
  "${COMMON_ARGS[@]}" \
  --freeze-signal-all \
  --disable-signal \
  --unfreeze-face-last-block

# 2) Binary valence multimodal.
run_case \
  binary_valence_multimodal \
  "${COMMON_ARGS[@]}" \
  --unfreeze-face-last-block \
  --unfreeze-signal-cnn

"$PY" - <<'PY'
import json
from pathlib import Path

artifacts = Path('artifacts')
reports = Path('reports')
reports.mkdir(parents=True, exist_ok=True)

cases = [
    ('Binary Face Only', artifacts / 'binary_valence_face_only.json'),
    ('Binary Multimodal', artifacts / 'binary_valence_multimodal.json'),
]

rows = []
for name, p in cases:
    if p.exists():
        data = json.loads(p.read_text())
        rows.append((
            name,
            float(data.get('test_overall_acc', 0.0)),
            float(data.get('test_macro_f1', 0.0)),
            int(data.get('best_epoch', -1)),
            float(data.get('best_val_macro_f1', 0.0)),
        ))
    else:
        rows.append((name, None, None, None, None))

md = []
md.append('# Binary Valence Comparison')
md.append('')
md.append('Target: Negative vs Positive valence (neutral excluded).')
md.append('')
md.append('| Model | Test Accuracy | Test Macro-F1 | Best Epoch | Best Val Macro-F1 |')
md.append('|---|---:|---:|---:|---:|')
for name, acc, f1, be, bv in rows:
    if acc is None:
        md.append(f'| {name} | NA | NA | NA | NA |')
    else:
        md.append(f'| {name} | {acc:.4f} | {f1:.4f} | {be} | {bv:.4f} |')

(report_md := reports / 'binary_valence_comparison.md').write_text('\n'.join(md), encoding='utf-8')

tex = []
tex.append('\\documentclass{article}')
tex.append('\\usepackage[margin=1in]{geometry}')
tex.append('\\begin{document}')
tex.append('\\section*{Binary Valence Comparison}')
tex.append('Target: Negative vs Positive valence (neutral excluded).')
tex.append('\\begin{tabular}{lrrrr}')
tex.append('\\hline')
tex.append('Model & Test Acc & Test Macro-F1 & Best Epoch & Best Val Macro-F1 \\\\')
tex.append('\\hline')
for name, acc, f1, be, bv in rows:
    if acc is None:
        tex.append(f'{name} & NA & NA & NA & NA \\\\')
    else:
        tex.append(f'{name} & {acc:.4f} & {f1:.4f} & {be} & {bv:.4f} \\\\')
tex.append('\\hline')
tex.append('\\end{tabular}')
tex.append('\\end{document}')

(report_tex := reports / 'binary_valence_comparison.tex').write_text('\n'.join(tex), encoding='utf-8')
print(f'Wrote {report_md}')
print(f'Wrote {report_tex}')
PY

echo "[BinarySuite] Completed binary valence runs and report generation."
