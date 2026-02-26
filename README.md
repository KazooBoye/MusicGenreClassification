# Music Genre Classification

A comprehensive machine learning project for classifying music genres using both traditional ML models and deep learning approaches.

## Overview

This project implements and compares multiple machine learning models for music genre classification:
- Traditional ML: KNN, SVM, Random Forest, MLP, GMM
- Deep Learning: Convolutional Neural Network (CNN)

## Features

- Reproducible dataset preprocessing pipeline
- Multiple feature extraction methods (MFCCs, spectral features, mel-spectrograms)
- Noise reduction for inference
- Configuration-driven hyperparameters
- Comprehensive evaluation metrics
- Visualization and analysis tools
- CLI interface for training and inference

## Project Structure

```
music-genre-classification/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── data/                       # Dataset directory
│   ├── raw/                    # Raw audio files
│   ├── processed/              # Preprocessed audio
│   └── features/               # Extracted features
├── src/                        # Source code
│   ├── data/                   # Data loading and preprocessing
│   ├── features/               # Feature extraction
│   ├── models/                 # Model implementations
│   ├── evaluation/             # Metrics and visualization
│   └── inference/              # Inference pipeline
├── models/                     # Saved trained models
├── notebooks/                  # Jupyter notebooks
│   └── Report.ipynb            # Analysis and results
└── outputs/                    # Logs and results
    ├── logs/
    ├── plots/
    └── results.json
```

## Installation

1. Create and activate conda environment:
```bash
conda create -n course python=3.12.5
conda activate course
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

Place your audio dataset in `data/raw/` with the following structure:
```
data/raw/
├── genre1/
│   ├── audio1.wav
│   └── audio2.wav
├── genre2/
│   └── audio1.wav
...
```

### Preprocessing (Optional)

The training scripts process audio on-the-fly, but you can optionally preprocess and cache data:

**Preprocess audio segments:**
```bash
python src/data/preprocess_dataset.py
```
This saves preprocessed audio segments to `data/processed/`.

**Extract and save features:**
```bash
python src/data/extract_all_features.py
```
This extracts features and saves them to `data/features/` for faster loading.

### Training Models

Train traditional ML models:
```bash
python src/models/train_knn.py
python src/models/train_svm.py
python src/models/train_rf.py
python src/models/train_mlp.py
python src/models/train_gmm.py
```

Train CNN model:
```bash
python src/models/train_cnn.py
```

### Inference

Classify a new audio file:
```bash
python src/inference/infer.py --audio path/to/file.wav --model svm
```

Available models: `knn`, `svm`, `rf`, `mlp`, `gmm`, `cnn`

## Configuration

All hyperparameters and settings are controlled via `config.yaml`. Modify this file to adjust:
- Audio preprocessing parameters
- Feature extraction settings
- Model hyperparameters
- Training parameters
- File paths

## Results

Analysis and results are documented in `notebooks/Report.ipynb`, including:
- Data exploration and visualization
- Feature analysis (PCA, t-SNE)
- Model comparison
- Confusion matrices
- Performance metrics
