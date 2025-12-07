"""
Audio preprocessing utilities.
Handles loading, normalization, resampling, and segmentation.
"""

import librosa
import numpy as np
import logging
from typing import Tuple, List
import soundfile as sf
from pathlib import Path

logger = logging.getLogger(__name__)


def load_audio(file_path: str, sr: int = 22050, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio file with specified sample rate.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=mono)
        return audio, sample_rate
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to [-1, 1] range.
    
    Args:
        audio: Audio signal
        
    Returns:
        Normalized audio signal
    """
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio))
    return audio


def segment_audio(audio: np.ndarray, sr: int, segment_length: float) -> List[np.ndarray]:
    """
    Split audio into fixed-length segments.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        segment_length: Length of each segment in seconds
        
    Returns:
        List of audio segments
    """
    segment_samples = int(segment_length * sr)
    segments = []
    
    for start in range(0, len(audio), segment_samples):
        end = start + segment_samples
        if end <= len(audio):
            segments.append(audio[start:end])
        elif len(audio) - start > segment_samples // 2:
            segment = audio[start:]
            padded = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
            segments.append(padded)
    
    return segments


def preprocess_audio(
    file_path: str,
    sr: int = 22050,
    segment_length: float = 3.0,
    mono: bool = True
) -> Tuple[List[np.ndarray], int]:
    """
    Complete preprocessing pipeline for audio file.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        segment_length: Length of segments in seconds
        mono: Convert to mono if True
        
    Returns:
        Tuple of (list of audio segments, sample_rate)
    """
    audio, sample_rate = load_audio(file_path, sr=sr, mono=mono)
    audio = normalize_audio(audio)
    segments = segment_audio(audio, sample_rate, segment_length)
    
    return segments, sample_rate


def save_processed_audio(audio: np.ndarray, output_path: str, sr: int):
    """
    Save processed audio to file.
    
    Args:
        audio: Audio signal to save
        output_path: Output file path
        sr: Sample rate
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)
