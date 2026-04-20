"""Generate final markdown + LaTeX report for the 5-mark project submission.

This script reads artifact JSON metrics for:
- facial only
- physiological only
- multimodal
- metadata-assisted baseline

and produces:
- reports/final_project_report.md
- reports/final_project_report.tex
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class CaseResult:
    name: str
    artifact_stem: str | None
    test_acc: float | None
    test_macro_f1: float | None
    best_epoch: int | None
    best_val_f1: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final project report from artifact JSON files.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--report-md", type=str, default="reports/final_project_report.md")
    parser.add_argument("--report-tex", type=str, default="reports/final_project_report.tex")
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def resolve_case(artifacts_dir: Path, display_name: str, candidates: Iterable[str]) -> CaseResult:
    for stem in candidates:
        payload = load_json(artifacts_dir / f"{stem}.json")
        if payload is None:
            continue

        best_epoch = payload.get("best_epoch")
        best_val_macro_f1 = payload.get("best_val_macro_f1")
        return CaseResult(
            name=display_name,
            artifact_stem=stem,
            test_acc=float(payload.get("test_overall_acc", 0.0)),
            test_macro_f1=float(payload.get("test_macro_f1", 0.0)),
            best_epoch=int(best_epoch) if best_epoch is not None else None,
            best_val_f1=float(best_val_macro_f1) if best_val_macro_f1 is not None else None,
        )

    return CaseResult(
        name=display_name,
        artifact_stem=None,
        test_acc=None,
        test_macro_f1=None,
        best_epoch=None,
        best_val_f1=None,
    )


def fmt_float(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{v:.4f}"


def fmt_int(v: int | None) -> str:
    if v is None:
        return "NA"
    return str(v)


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def pick_best_model(cases: List[CaseResult]) -> str:
    available = [c for c in cases if c.test_macro_f1 is not None]
    if not available:
        return "No completed experiments found yet."

    best = max(available, key=lambda x: x.test_macro_f1 or -1.0)
    return (
        f"Best current model by macro-F1: {best.name} "
        f"(macro-F1={fmt_float(best.test_macro_f1)}, acc={fmt_float(best.test_acc)})."
    )


def build_markdown(cases: List[CaseResult]) -> str:
    lines: List[str] = []
    lines.append("# Final Project Report: Ad Impact Prediction (Binary Valence)")
    lines.append("")
    lines.append("Goal: predict ad impact while watching video as **negative** or **positive** from face, physiology, multimodal, and metadata-assisted input.")
    lines.append("")
    lines.append("## Final Results")
    lines.append("")
    lines.append("| Model | Artifact | Test Accuracy | Test Macro-F1 | Best Epoch | Best Val Macro-F1 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for case in cases:
        artifact = case.artifact_stem if case.artifact_stem is not None else "NA"
        lines.append(
            f"| {case.name} | {artifact} | {fmt_float(case.test_acc)} | {fmt_float(case.test_macro_f1)} | "
            f"{fmt_int(case.best_epoch)} | {fmt_float(case.best_val_f1)} |"
        )
    lines.append("")
    lines.append(pick_best_model(cases))
    lines.append("")

    lines.append("## Dataset Management (How labels are handled)")
    lines.append("")
    lines.append("1. Face stream uses video clips from advertisement categories.")
    lines.append("2. Physiology stream uses 32-Hz biosignal features: BVP, EDA, TEMP, ACC X/Y/Z.")
    lines.append("3. Label target is binary valence mapping from emotion labels:")
    lines.append("   - Positive: Joy, Surprise")
    lines.append("   - Negative: Sadness, Anger, Disgust, Fear")
    lines.append("4. Neutral is excluded for the binary task.")
    lines.append("5. Participant-level split is used to reduce identity leakage.")
    lines.append("6. If strict participant/ad alignment keys are missing in processed biosignal CSV, fallback segment selection is label-agnostic.")
    lines.append("")

    lines.append("## Data Augmentation and Training Controls")
    lines.append("")
    lines.append("1. Temporal video window sampling with jitter in training.")
    lines.append("2. Dataset repeat expansion per epoch (augment repeats).")
    lines.append("3. Balanced sampler with capped oversampling ratio.")
    lines.append("4. Focal loss with label smoothing for class-imbalance robustness.")
    lines.append("5. Modality-specific ablations for facial-only and physiological-only baselines.")
    lines.append("")

    lines.append("## Architecture and Data Flow")
    lines.append("")
    lines.append("Architecture and data-flow diagrams are included in the LaTeX report.")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append("Run the full suite script:")
    lines.append("")
    lines.append("```bash")
    lines.append("./emotion_recognition/scripts/run_final_project_suite.sh")
    lines.append("```")

    return "\n".join(lines) + "\n"


def build_latex(cases: List[CaseResult]) -> str:
    best_sentence = tex_escape(pick_best_model(cases))

    row_lines: List[str] = []
    for case in cases:
        artifact = case.artifact_stem if case.artifact_stem is not None else "NA"
        row_lines.append(
            f"{tex_escape(case.name)} & {tex_escape(artifact)} & {fmt_float(case.test_acc)} & "
            f"{fmt_float(case.test_macro_f1)} & {fmt_int(case.best_epoch)} & {fmt_float(case.best_val_f1)} \\\\"  # noqa: E501
        )

    rows_block = "\n".join(row_lines)

    tex = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{tikz}}
\\usetikzlibrary{{arrows.meta,positioning,shapes.geometric,fit}}
\\usepackage{{setspace}}
\\setstretch{{1.1}}

\\title{{Final Project Report: Multimodal Ad Impact Prediction}}
\\author{{NeuroBioSense DL Project}}
\\date{{}}

\\begin{{document}}
\\maketitle

\\section*{{1. Problem Statement}}
This project predicts advertisement impact while users are watching videos. The target is binary valence:
negative vs positive response.

\\section*{{2. Dataset and Label Strategy}}
\\begin{{itemize}}
    \\item Video modality: advertisement clips and facial dynamics.
    \\item Physiological modality: 32-Hz signals (BVP, EDA, TEMP, ACC X/Y/Z).
    \\item Binary valence mapping: Positive = Joy/Surprise, Negative = Sadness/Anger/Disgust/Fear.
    \\item Neutral class excluded for binary setting.
    \\item Participant-level split is applied to limit leakage.
    \\item When strict participant-ad keys are unavailable in processed biosignal files, label-agnostic fallback segmenting is used.
\\end{{itemize}}

\\section*{{3. Data Diagram}}
\\begin{{center}}
\\begin{{tikzpicture}}[
    node distance=8mm and 10mm,
    block/.style={{rectangle, rounded corners, draw=black, minimum width=3.2cm, minimum height=8mm, align=center}},
    io/.style={{trapezium, trapezium left angle=70, trapezium right angle=110, draw=black, minimum width=3.4cm, minimum height=8mm, align=center}},
    arr/.style={{-{{Stealth[length=3mm]}}, thick}}
]
\\node[io] (video) {{Raw Video Clips}};
\\node[io, below=of video] (signal) {{32-Hz Biosignal CSV}};
\\node[io, below=of signal] (meta) {{Participant Metadata}};

\\node[block, right=25mm of signal] (prep) {{Preprocessing\\\\Frame windows + Signal resampling}};
\\node[block, right=25mm of prep] (split) {{Participant-level split\\\\Train / Val / Test}};
\\node[block, right=25mm of split] (train) {{Three training modes\\\\Face / Signal / Multimodal}};

\\draw[arr] (video.east) -- (prep.west);
\\draw[arr] (signal.east) -- (prep.west);
\\draw[arr] (meta.east) -- (prep.west);
\\draw[arr] (prep.east) -- (split.west);
\\draw[arr] (split.east) -- (train.west);
\\end{{tikzpicture}}
\\end{{center}}

\\section*{{4. Model Architecture Diagram}}
\\begin{{center}}
\\begin{{tikzpicture}}[
    node distance=9mm and 12mm,
    block/.style={{rectangle, rounded corners, draw=black, minimum width=3.1cm, minimum height=8mm, align=center}},
    arr/.style={{-{{Stealth[length=3mm]}}, thick}}
]
\\node[block] (v0) {{Video windows\\\\(T\\_v, 3, 160, 160)}};
\\node[block, right=of v0] (v1) {{FaceNet + Projection\\\\+ BiLSTM + Attention}};
\\node[block, below=13mm of v0] (s0) {{Signal sequence\\\\(T\\_s, 6)}};
\\node[block, right=of s0] (s1) {{Channel Attention + CNN\\\\+ BiLSTM + Attention}};
\\node[block, right=19mm of v1, yshift=-7mm] (fuse) {{Cross-modal Attention\\\\+ Soft Gating Fusion}};
\\node[block, right=of fuse] (clf) {{Classifier\\\\(Binary Valence)}};

\\draw[arr] (v0.east) -- (v1.west);
\\draw[arr] (s0.east) -- (s1.west);
\\draw[arr] (v1.east) -- (fuse.west);
\\draw[arr] (s1.east) -- (fuse.west);
\\draw[arr] (fuse.east) -- (clf.west);
\\end{{tikzpicture}}
\\end{{center}}

\\section*{{5. Data Augmentation and Training Policy}}
\\begin{{itemize}}
    \\item Temporal jitter and sliding windows on video stream.
    \\item Train dataset repeat expansion per epoch.
    \\item WeightedRandomSampler with capped imbalance ratio.
    \\item Focal loss + label smoothing.
    \\item Modality ablation controls for fair baselines:
    face-only and physiological-only.
\\end{{itemize}}

\\section*{{6. Final Results}}
\\begin{{center}}
\\begin{{tabular}}{{llrrrr}}
\\toprule
Model & Artifact & Test Acc & Macro-F1 & Best Epoch & Best Val Macro-F1 \\\\
\\midrule
{rows_block}
\\bottomrule
\\end{{tabular}}
\\end{{center}}

\\paragraph{{Summary.}} {best_sentence}

\\section*{{7. Discussion}}
Current performance is constrained mainly by weak cross-modal alignment when strict participant-ad-time keys are unavailable in processed biosignal files.
This can make multimodal training collapse toward one dominant class.

\section*{{8. Conclusion}}
This submission includes a complete end-to-end pipeline (facial-only, physiological-only, multimodal, metadata-assisted), data augmentation controls,
and reproducible scripts for training and report generation suitable for project evaluation.

\\end{{document}}
"""
    return tex


def main() -> None:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    report_md_path = Path(args.report_md)
    report_tex_path = Path(args.report_tex)

    cases = [
        resolve_case(artifacts_dir, "Facial Only", ["final_valence_face_only", "binary_valence_face_only"]),
        resolve_case(artifacts_dir, "Physiological Only", ["final_valence_signal_only", "binary_valence_signal_only"]),
        resolve_case(artifacts_dir, "Multimodal", ["final_valence_multimodal", "binary_valence_multimodal"]),
        resolve_case(artifacts_dir, "Metadata-Assisted", ["final_valence_metadata", "final_valence_metadata_aug"]),
    ]

    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_tex_path.parent.mkdir(parents=True, exist_ok=True)

    report_md_path.write_text(build_markdown(cases), encoding="utf-8")
    report_tex_path.write_text(build_latex(cases), encoding="utf-8")

    print(f"Wrote {report_md_path}")
    print(f"Wrote {report_tex_path}")


if __name__ == "__main__":
    main()
