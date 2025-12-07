"""
Dataset loader for music genre classification.
Handles loading audio files from directory structure.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Load audio dataset organized by genre folders.
    
    Expected structure:
    data_path/
        genre1/
            audio1.wav
            audio2.wav
        genre2/
            audio1.wav
    """
    
    def __init__(self, data_path: str):
        """
        Initialize dataset loader.
        
        Args:
            data_path: Path to directory containing genre folders
        """
        self.data_path = Path(data_path)
        self.genres = []
        self.genre_to_idx = {}
        self.idx_to_genre = {}
        
        if self.data_path.exists():
            self._discover_genres()
    
    def _discover_genres(self):
        """Discover available genres from directory structure."""
        self.genres = sorted([
            d.name for d in self.data_path.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}
        
        logger.info(f"Discovered {len(self.genres)} genres: {self.genres}")
    
    def get_file_list(self) -> List[Tuple[str, str, int]]:
        """
        Get list of all audio files with their genres.
        
        Returns:
            List of tuples (file_path, genre_name, genre_idx)
        """
        file_list = []
        
        for genre in self.genres:
            genre_path = self.data_path / genre
            genre_idx = self.genre_to_idx[genre]
            
            audio_files = list(genre_path.glob('*.wav')) + \
                         list(genre_path.glob('*.mp3')) + \
                         list(genre_path.glob('*.au'))
            
            for audio_file in audio_files:
                file_list.append((str(audio_file), genre, genre_idx))
        
        logger.info(f"Found {len(file_list)} audio files")
        return file_list
    
    def get_genre_distribution(self) -> Dict[str, int]:
        """
        Get distribution of files per genre.
        
        Returns:
            Dictionary mapping genre to file count
        """
        file_list = self.get_file_list()
        distribution = {genre: 0 for genre in self.genres}
        
        for _, genre, _ in file_list:
            distribution[genre] += 1
        
        return distribution
    
    def get_num_classes(self) -> int:
        """Get number of genre classes."""
        return len(self.genres)
