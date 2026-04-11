# NeuroBioSense Multimodal Emotion Recognition

Research-grade multimodal emotion recognition system using:
- Video face dynamics (FaceNet + temporal BiLSTM + attention)
- Physiological signals (BVP/EDA/TEMP/ACC_X/ACC_Y/ACC_Z)
- Cross-modal attention + soft reliability gating

## Project Layout

- emotion_recognition/models: all model components and full assembly
- emotion_recognition/utils: preprocessing, dataset, metrics
- emotion_recognition/scripts: stage-wise training + inference utilities
- streamlit_app.py: deployment-ready web app

## Quick Start

### 1) Create environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Run smoke syntax check

```bash
python -m compileall emotion_recognition
```

### 3) Stage 1: Face pretraining (FER2013 + CK+)

```bash
python -m emotion_recognition.scripts.train_face \
  --fer-root /path/to/FER2013 \
  --ck-root /path/to/CK+ \
  --epochs 50 \
  --batch-size 64 \
  --output artifacts/facenet_stage1.pth
```

### 4) Stage 2: Signal pretraining (WESAD, optional)

```bash
python -m emotion_recognition.scripts.train_signal \
  --wesad-train-npz /path/to/wesad_train.npz \
  --wesad-val-npz /path/to/wesad_val.npz \
  --epochs 50 \
  --batch-size 32 \
  --output artifacts/signal_stage2.pth
```

If WESAD is unavailable, skip Stage 2 and proceed to Stage 3.

### 5) Stage 3: Multimodal fine-tuning (NeuroBioSense)

```bash
python -m emotion_recognition.scripts.train_multimodal \
  --video-root /path/to/NeuroBioSense/video_root \
  --signal-csv /path/to/32-Hertz.csv \
  --demographics-csv /path/to/Participant_demographic_information.csv \
  --facenet-stage1 artifacts/facenet_stage1.pth \
  --signal-stage2 artifacts/signal_stage2.pth \
  --epochs 50 \
  --batch-size 8 \
  --output artifacts/multimodal_stage3.pth
```

## Inference

### CLI single-clip inference

```bash
python -m emotion_recognition.scripts.predict_clip \
  --checkpoint artifacts/multimodal_stage3.pth \
  --video /path/to/clip.MP4 \
  --signal-csv /path/to/32-Hertz.csv \
  --participant-id P01 \
  --ad-code AD01 \
  --demographics-csv /path/to/Participant_demographic_information.csv
```

### Real-time webcam inference

```bash
python -m emotion_recognition.scripts.inference_realtime \
  --checkpoint artifacts/multimodal_stage3.pth
```

## Streamlit Deployment (Local)

```bash
streamlit run streamlit_app.py
```

Upload:
- checkpoint (.pth)
- video clip (.mp4)
- optional signal CSV and participant/ad metadata for aligned multimodal inference

## Hugging Face Spaces Deployment

Use a **Streamlit Space** and upload this repository.

Suggested settings:
- SDK: streamlit
- Python: 3.11
- Startup command: `streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0`

## GitHub Push Checklist

1. Remove large local artifacts from Git tracking (checkpoints/data are ignored by default).
2. Initialize git and commit.
3. Push to GitHub.

```bash
git init
git add .
git commit -m "Initial multimodal emotion recognition implementation"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Notes

- Validation/test split is participant-level to avoid leakage.
- Stage 3 evaluation aggregates all windows per clip (`mean` or `majority`).
- Checkpoint stores normalization stats used by deployment app.
