# Music Genre Classification - Quick Start Guide

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

## Dataset Preparation

Place your audio dataset in `data/raw/` with genre folders:
```
data/raw/
├── blues/
│   ├── blues001.wav
│   └── blues002.wav
├── classical/
│   ├── classical001.wav
│   └── classical002.wav
└── ... (other genres)
```

## Preprocessing (Required)

Before training any models, you must preprocess the audio data:

### 1. Preprocess Audio Segments
```bash
python src/data/preprocess_dataset.py
```
This will:
- Load all audio files from `data/raw/`
- Apply normalization and segmentation
- Save preprocessed segments to `data/processed/`
- Each audio file is split into 3-second segments

### 2. Extract Features
```bash
python src/data/extract_all_features.py
```
This will:
- Extract MFCC and spectral features from all audio
- Save features as numpy arrays to `data/features/`
- Create a metadata CSV file
- Features can be loaded quickly for training

Note: All training scripts require preprocessed data. They will not process audio on-the-fly.

## Training Models

Train each model individually from the project root directory:

```bash
python src/models/train_knn.py
python src/models/train_svm.py
python src/models/train_rf.py
python src/models/train_mlp.py
python src/models/train_gmm.py
python src/models/train_cnn.py
```

Each script will:
- Load preprocessed features from `data/features/` or `data/processed/`
- Train the model
- Save the trained model to `models/`
- Generate evaluation metrics and plots in `outputs/`

Note: Ensure preprocessing steps are completed before training.

## Making Predictions

Use the inference script to classify new audio files:

```bash
python src/inference/infer.py --audio path/to/song.wav --model svm
```

Available models: `knn`, `svm`, `rf`, `mlp`, `gmm`, `cnn`

Example:
```bash
python src/inference/infer.py --audio data/raw/blues/blues001.wav --model cnn
```

Output will show:
- Predicted genre
- Probability distribution across all genres

## Analysis

Open and run the Jupyter notebook for comprehensive analysis:

```bash
jupyter notebook notebooks/Report.ipynb
```

The notebook includes:
- Data exploration and visualization
- Waveform and spectrogram plots
- Feature analysis (PCA, t-SNE)
- Model comparison
- Confusion matrices
- Training histories

## Configuration

All hyperparameters are controlled via `config.yaml`:

- Audio settings (sample rate, segment length)
- Feature extraction parameters
- Model hyperparameters
- Training parameters
- File paths

Modify the config file to experiment with different settings.

## Project Structure

```
music-genre-classification/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── QUICKSTART.md           # This file
├── data/
│   ├── raw/                # Raw audio files (add your dataset here)
│   ├── processed/          # Preprocessed audio
│   └── features/           # Extracted features
├── src/
│   ├── data/              # Data loading and preprocessing
│   │   ├── dataset_loader.py
│   │   ├── preprocess_audio.py
│   │   └── noise_reduction.py
│   ├── features/          # Feature extraction
│   │   ├── extract_mfcc_features.py
│   │   ├── extract_spectral_features.py
│   │   └── extract_melspectrogram.py
│   ├── models/            # Model implementations
│   │   ├── train_knn.py
│   │   ├── train_svm.py
│   │   ├── train_rf.py
│   │   ├── train_mlp.py
│   │   ├── train_gmm.py
│   │   ├── train_cnn.py
│   │   ├── cnn_model.py
│   │   └── utils.py
│   ├── evaluation/        # Metrics and visualization
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── inference/         # Inference pipeline
│       └── infer.py
├── models/                # Saved trained models
├── notebooks/             # Jupyter notebooks
│   └── Report.ipynb       # Analysis notebook
└── outputs/               # Logs, plots, and results
    ├── logs/
    ├── plots/
    └── results files (.json)
```

## Troubleshooting

**No audio files found:**
- Ensure audio files are placed in `data/raw/` with genre subfolders
- Supported formats: WAV, MP3, AU

**Out of memory:**
- Reduce batch size in `config.yaml`
- Process fewer files initially

**CUDA errors (for MLP/CNN):**
- Models automatically use CPU if CUDA is unavailable
- Install PyTorch with CUDA support for GPU acceleration

**Import errors:**
- Ensure you're running scripts from the project root directory
- Activate the correct conda environment

## Tips

1. Start with a small subset of data to test the pipeline
2. Monitor training progress in log files: `outputs/logs/`
3. Compare models using the Jupyter notebook
4. Adjust hyperparameters in `config.yaml` for better results
5. Use noise reduction during inference for real-world audio

## Expected Output

After training all models, you should have:
- Trained models in `models/` directory
- Evaluation results in `outputs/` (JSON files)
- Confusion matrices in `outputs/plots/`
- Training history plots for MLP and CNN
- Comprehensive analysis in the Jupyter notebook

## Performance Expectations

Model performance depends on:
- Dataset size and quality
- Genre diversity and complexity
- Feature extraction parameters
- Model hyperparameters

Typical accuracy ranges from 60-85% on standard datasets like GTZAN.

## Next Steps

1. Experiment with different hyperparameters
2. Try feature augmentation techniques
3. Implement ensemble methods
4. Test on your own audio files
5. Extend to more genres or audio classification tasks
