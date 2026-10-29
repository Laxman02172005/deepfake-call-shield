# Deepfake Call Shield

Speech deepfake detection using pretrained frozen speech encoders (HuBERT, wav2vec 2.0),
evaluated for robustness to phone-like audio degradation.

## Project Overview

| Item | Value |
|---|---|
| Dataset | ASVspoof 2019 Logical Access (LA) |
| Encoders | HuBERT-base, wav2vec 2.0 base (frozen) |
| Classifier | Logistic Regression |
| Primary metric | Equal Error Rate (EER) |
| Training conditions | Clean-only, Clean + Augmented |
| Held-out conditions | MP3, G.711 mu-law, Phone chain |

## Environment

- Python 3.11.9 (via `.venv/`)
- CPU-only (no GPU required)
- Windows 11 Pro

## Setup

### 1. Prerequisites

Install FFmpeg (required for Opus, MP3, G.711 augmentations):
```powershell
winget install --id Gyan.FFmpeg --source winget
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install PyTorch (CPU-only) — must be done first

```powershell
pip install torch==2.3.1+cpu torchaudio==2.3.1+cpu `
    --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install remaining dependencies

```powershell
pip install -r requirements.txt
# For development/testing:
pip install -r requirements-dev.txt
```

### 5. Place the dataset

Register at https://datasharing.asvspoof.org/ and download the LA partition.
Extract and place it at:
```
data/raw/LA/
├── ASVspoof2019_LA_cm_protocols/
├── ASVspoof2019_LA_train/
├── ASVspoof2019_LA_dev/
└── ASVspoof2019_LA_eval/
```

## Running the Pipeline

Execute scripts in order. Each script exits 0 on success, non-zero on failure.

```powershell
python scripts/01_verify_data.py      # Stage 3: check dataset integrity
python scripts/02_check_codecs.py     # Stage 3: verify FFmpeg codecs
python scripts/03_pilot_extract.py    # Stage 4: timing estimate (must PASS before 04)
python scripts/04_extract_features.py # Stage 4: full batched extraction
python scripts/05_train_all.py        # Stage 5: train 4 LR configurations
python scripts/06_evaluate_dev.py     # Stage 5: dev EER + threshold selection
python scripts/07_evaluate_eval.py    # Stage 6: FINAL eval (run once only)
python scripts/08_analysis.py         # Stage 6: seen/unseen + CI table
```

## Running Tests

```powershell
python -m pytest
```

## Project Structure

```
Mini_Project/
├── configs/base.yaml       Central configuration (paths, seeds, hyperparams)
├── data/
│   ├── raw/                ASVspoof 2019 LA — not committed to git
│   ├── features/           Cached embeddings (.pt) — not committed
│   └── manifests/          Lightweight CSV manifests — committed
├── src/                    Library code
├── scripts/                Numbered pipeline scripts
├── tests/                  Unit tests
└── results/
    ├── scores/             Raw score CSVs — not committed
    ├── models/             Saved LR models — not committed
    └── tables/             Final result tables — committed
```

## What is NOT committed to Git

| Item | Reason |
|---|---|
| `data/raw/` | Licensed dataset, large |
| `data/features/` | Generated, up to ~15 GB |
| `results/scores/`, `results/models/` | Fully reproducible from scripts |
| `.venv/` | Platform-specific binaries |
| `*.pt`, `*.bin`, `*.safetensors` | Model weights |
| `.env`, `secrets.yaml` | Credentials |
