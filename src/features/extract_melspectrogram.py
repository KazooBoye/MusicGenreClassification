"""
Mel-spectrogram extraction for CNN model.
"""

import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)


def extract_melspectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0,
    fmax: float = 8000
) -> np.ndarray:
    """
    Extract mel-spectrogram from audio signal.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Mel-spectrogram (2D array)
    """
    melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    
    return log_melspec


def normalize_melspectrogram(melspec: np.ndarray) -> np.ndarray:
    """
    Normalize mel-spectrogram to [0, 1] range.
    
    Args:
        melspec: Mel-spectrogram
        
    Returns:
        Normalized mel-spectrogram
    """
    melspec_min = melspec.min()
    melspec_max = melspec.max()
    
    if melspec_max - melspec_min > 0:
        normalized = (melspec - melspec_min) / (melspec_max - melspec_min)
    else:
        normalized = melspec - melspec_min
    
    return normalized


def extract_melspectrogram_normalized(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0,
    fmax: float = 8000
) -> np.ndarray:
    """
    Extract and normalize mel-spectrogram.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Normalized mel-spectrogram
    """
    melspec = extract_melspectrogram(audio, sr, n_mels, n_fft, hop_length, fmin, fmax)
    normalized = normalize_melspectrogram(melspec)
    
    return normalized


def pad_melspectrogram(melspec: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate mel-spectrogram to target length.
    
    Args:
        melspec: Mel-spectrogram (n_mels x time)
        target_length: Target time dimension
        
    Returns:
        Padded/truncated mel-spectrogram
    """
    if melspec.shape[1] < target_length:
        pad_width = target_length - melspec.shape[1]
        melspec = np.pad(melspec, ((0, 0), (0, pad_width)), mode='constant')
    elif melspec.shape[1] > target_length:
        melspec = melspec[:, :target_length]
    
    return melspec
