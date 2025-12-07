# Data Processing Pipeline

This project uses a mandatory two-stage preprocessing pipeline before training:

## Required Preprocessing Steps

### Step 1: Preprocess Audio
```bash
python src/data/preprocess_dataset.py
```

**What it does:**
- Loads all audio files from `data/raw/`
- Applies preprocessing (normalization, segmentation)
- Saves segments as `.npy` files to `data/processed/`
- Organizes by genre

**Output structure:**
```
data/processed/
├── blues/
│   ├── blues001_seg0.npy
│   ├── blues001_seg1.npy
│   └── ...
├── classical/
│   └── ...
└── ...
```

### Step 2: Extract Features
```bash
python src/data/extract_all_features.py
```

**What it does:**
- Processes all preprocessed audio
- Extracts MFCC and spectral features
- Saves features as numpy arrays
- Creates metadata CSV

**Output files:**
```
data/features/
├── features.npy      # Feature matrix (N x D)
├── labels.npy        # Label array (N,)
└── metadata.csv      # Filename and genre mapping
```

### Step 3: Train Models
```bash
python src/models/train_knn_fast.py
```

This loads preprocessed features directly, skipping audio processing entirely.

**Advantages:**
- Much faster training
- Efficient for hyperparameter tuning
- Lower memory usage
- Can inspect/analyze features separately

**Disadvantages:**
- Requires more disk space
- Two-step process
- Must re-run if preprocessing parameters change

## When to Use Each Approach

### Use On-the-Fly (Approach 1) when:
All training models
- Hyperparameter tuning
- Model comparison experiments
- Feature analysis

## Disk Space Requirements

**Required storage:**
- Raw audio: ~1.2GB (GTZAN dataset)
- Processed segments: ~2.5GB (as .npy files)
- Extracted features: ~50MB
- Trained models: ~100MB
- Total: ~3.9GB

## Performance Benefits

For GTZAN dataset (1000 audio files):

**Preprocessing (one-time cost):**
- Audio segmentation: ~10 minutes
- Feature extraction: ~15 minutes
- Total: ~25 minutes

**Training (per model):**
- Training time: ~30 seconds to 5 minutes
- Multiple experiments: no reprocessing needed

**Speedup for 5 models:**
- Preprocessing once: 25 minutes
- Training 5 models: ~2-10 minutes total
- No redundant feature extraction across models

## Important Notes

- All training scripts require preprocessed data
- Training will fail with clear error messages if preprocessing is skipped
- Run preprocessing scripts before any model training
- CNN uses segments from `data/processed/`
- Other models use aggregated features from `data/features/`
