"""Dataset integrity checker for NeuroBioSense multimodal training.

This script validates:
- video clip indexing
- signal CSV schema
- metadata parsing
- one-sample dataloader path through NeuroBioSenseDataset
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from emotion_recognition.utils.dataset import EMOTION_TO_ID, build_neurobiosense_datasets, scan_video_samples
from emotion_recognition.utils.signal_processing import infer_id_columns, load_32hz_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check NeuroBioSense dataset readiness")
    parser.add_argument("--dataset-root", type=str, default="Dataset")
    parser.add_argument("--video-root", type=str, default="")
    parser.add_argument("--signal-csv", type=str, default="")
    parser.add_argument("--demographics", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    video_root = (
        Path(args.video_root)
        if args.video_root
        else dataset_root / "NeuroBioSense Dataset" / "NeuroBioSense" / "Advertisement Categories"
    )
    signal_csv = (
        Path(args.signal_csv)
        if args.signal_csv
        else dataset_root / "NeuroBioSense Dataset" / "NeuroBioSense" / "Biosignal Files" / "Pre-Processed" / "32-Hertz.csv"
    )
    demographics = (
        Path(args.demographics)
        if args.demographics
        else dataset_root / "NeuroBioSense Dataset" / "NeuroBioSense" / "Participant Data" / "Participant_demographic_information.xlsx"
    )

    print(f"Video root      : {video_root}")
    print(f"Signal CSV      : {signal_csv}")
    print(f"Demographics    : {demographics if demographics.exists() else 'NOT FOUND (optional)'}")

    if not video_root.exists():
        raise FileNotFoundError(f"Video root missing: {video_root}")
    if not signal_csv.exists():
        raise FileNotFoundError(f"Signal CSV missing: {signal_csv}")

    samples = scan_video_samples(video_root)
    print(f"Total clips     : {len(samples)}")
    print(f"Participants    : {len(set(s.participant_id for s in samples))}")

    class_counts = Counter(s.label_id for s in samples)
    inv = {v: k for k, v in EMOTION_TO_ID.items()}
    print("Class distribution (label_id -> count):")
    for label_id in sorted(inv.keys()):
        print(f"  {label_id} ({inv[label_id]}): {class_counts.get(label_id, 0)}")

    signal_df = load_32hz_csv(signal_csv)
    id_cols = infer_id_columns(signal_df)
    print("Signal schema:")
    print(f"  rows          : {len(signal_df)}")
    print(f"  participant id: {id_cols.get('participant')}")
    print(f"  ad code       : {id_cols.get('ad')}")
    print(f"  time column   : {id_cols.get('time')}")

    train_ds, val_ds, test_ds, _ = build_neurobiosense_datasets(
        video_root=video_root,
        signal_csv_path=signal_csv,
        demographics_csv_path=demographics if demographics.exists() else None,
        stage=3,
        t_v=10,
        t_s=128,
        seed=42,
    )

    print(f"Split sizes     : train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # End-to-end sample check.
    video, signal, label = train_ds[0]
    print(f"Sample video    : {tuple(video.shape)}")
    print(f"Sample signal   : {tuple(signal.shape)}")
    print(f"Sample label    : {int(label.item())}")

    print("Dataset check passed.")


if __name__ == "__main__":
    main()
