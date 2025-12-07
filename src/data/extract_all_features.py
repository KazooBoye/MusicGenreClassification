"""
Extract features from preprocessed audio and save to data/features/.
This script extracts MFCC and spectral features for classical ML models.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.utils import load_config, setup_logging
from src.data.dataset_loader import DatasetLoader
from src.data.preprocess_audio import preprocess_audio
from src.features.extract_mfcc_features import extract_mfcc_features
from src.features.extract_spectral_features import extract_all_spectral_features

logger = logging.getLogger(__name__)


def extract_combined_features(audio_segment, sr, config):
    """Extract combined MFCC and spectral features."""
    mfcc_features = extract_mfcc_features(
        audio_segment, sr,
        n_mfcc=config['features']['mfcc']['n_mfcc'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        n_mels=config['features']['mfcc']['n_mels']
    )
    
    spectral_features = extract_all_spectral_features(
        audio_segment, sr,
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        n_chroma=config['features']['chroma']['n_chroma'],
        n_bands=config['features']['spectral']['n_bands']
    )
    
    return np.concatenate([mfcc_features, spectral_features])


def main():
    config = load_config('config.yaml')
    
    setup_logging(
        log_file=f"{config['paths']['logs']}/extract_features.log",
        level=config['logging']['level']
    )
    
    logger.info("Starting feature extraction")
    
    dataset_loader = DatasetLoader(config['paths']['data_raw'])
    file_list = dataset_loader.get_file_list()
    
    if len(file_list) == 0:
        logger.error("No audio files found. Please add data to data/raw/")
        return
    
    features_dir = Path(config['paths']['data_features'])
    features_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {len(file_list)} audio files")
    logger.info(f"Output directory: {features_dir}")
    
    all_features = []
    all_labels = []
    all_filenames = []
    
    for file_path, genre, label in tqdm(file_list, desc="Extracting features"):
        try:
            segments, sr = preprocess_audio(
                file_path,
                sr=config['audio']['sample_rate'],
                segment_length=config['audio']['segment_length'],
                mono=config['audio']['mono']
            )
            
            segment_features = []
            for segment in segments:
                features = extract_combined_features(segment, sr, config)
                segment_features.append(features)
            
            avg_features = np.mean(segment_features, axis=0)
            
            all_features.append(avg_features)
            all_labels.append(label)
            all_filenames.append(Path(file_path).name)
            
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            continue
    
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    np.save(features_dir / 'features.npy', features_array)
    np.save(features_dir / 'labels.npy', labels_array)
    
    metadata_df = pd.DataFrame({
        'filename': all_filenames,
        'label': labels_array,
        'genre': [dataset_loader.idx_to_genre[label] for label in labels_array]
    })
    metadata_df.to_csv(features_dir / 'metadata.csv', index=False)
    
    logger.info(f"Feature extraction complete!")
    logger.info(f"Features shape: {features_array.shape}")
    logger.info(f"Labels shape: {labels_array.shape}")
    logger.info(f"Saved to {features_dir}")
    
    print(f"\nFeature extraction complete!")
    print(f"Features saved: {features_dir / 'features.npy'}")
    print(f"Labels saved: {features_dir / 'labels.npy'}")
    print(f"Metadata saved: {features_dir / 'metadata.csv'}")
    print(f"Features shape: {features_array.shape}")
    print(f"Number of samples: {len(labels_array)}")


if __name__ == '__main__':
    main()
