"""
Preprocess all audio files and save to data/processed/.
This script loads raw audio, applies preprocessing, and saves the results.
"""

import sys
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.utils import load_config, setup_logging
from src.data.dataset_loader import DatasetLoader
from src.data.preprocess_audio import preprocess_audio, save_processed_audio

logger = logging.getLogger(__name__)


def main():
    config = load_config('config.yaml')
    
    setup_logging(
        log_file=f"{config['paths']['logs']}/preprocess_dataset.log",
        level=config['logging']['level']
    )
    
    logger.info("Starting dataset preprocessing")
    
    dataset_loader = DatasetLoader(config['paths']['data_raw'])
    file_list = dataset_loader.get_file_list()
    
    if len(file_list) == 0:
        logger.error("No audio files found. Please add data to data/raw/")
        return
    
    output_dir = Path(config['paths']['data_processed'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {len(file_list)} audio files")
    logger.info(f"Output directory: {output_dir}")
    
    processed_count = 0
    error_count = 0
    
    for file_path, genre, label in tqdm(file_list, desc="Preprocessing"):
        try:
            segments, sr = preprocess_audio(
                file_path,
                sr=config['audio']['sample_rate'],
                segment_length=config['audio']['segment_length'],
                mono=config['audio']['mono']
            )
            
            genre_dir = output_dir / genre
            genre_dir.mkdir(parents=True, exist_ok=True)
            
            file_stem = Path(file_path).stem
            
            for idx, segment in enumerate(segments):
                output_path = genre_dir / f"{file_stem}_seg{idx}.npy"
                np.save(output_path, segment)
            
            processed_count += 1
            
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            error_count += 1
            continue
    
    logger.info(f"Preprocessing complete!")
    logger.info(f"Successfully processed: {processed_count} files")
    logger.info(f"Errors: {error_count} files")
    logger.info(f"Total segments saved: {len(list(output_dir.rglob('*.npy')))}")


if __name__ == '__main__':
    main()
