"""
Spectral feature extraction for music genre classification.
"""

import numpy as np
import librosa
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def extract_chroma_features(
    audio: np.ndarray,
    sr: int,
    n_chroma: int = 12,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract chroma features from audio signal.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_chroma: Number of chroma bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Chroma feature vector
    """
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_chroma=n_chroma,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    return np.concatenate([chroma_mean, chroma_std])


def extract_spectral_centroid(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract spectral centroid features.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Spectral centroid feature vector
    """
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return np.array([np.mean(centroid), np.std(centroid)])


def extract_spectral_bandwidth(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract spectral bandwidth features.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Spectral bandwidth feature vector
    """
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return np.array([np.mean(bandwidth), np.std(bandwidth)])


def extract_spectral_contrast(
    audio: np.ndarray,
    sr: int,
    n_bands: int = 6,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract spectral contrast features.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_bands: Number of frequency bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Spectral contrast feature vector
    """
    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr,
        n_bands=n_bands,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    
    return np.concatenate([contrast_mean, contrast_std])


def extract_zero_crossing_rate(
    audio: np.ndarray,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract zero crossing rate features.
    
    Args:
        audio: Audio signal
        hop_length: Hop length for frame analysis
        
    Returns:
        Zero crossing rate feature vector
    """
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)
    
    return np.array([np.mean(zcr), np.std(zcr)])


def extract_rms_energy(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract RMS energy features.
    
    Args:
        audio: Audio signal
        n_fft: FFT window size
        hop_length: Hop length for frame analysis
        
    Returns:
        RMS energy feature vector
    """
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
    
    return np.array([np.mean(rms), np.std(rms)])


def extract_all_spectral_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_chroma: int = 12,
    n_bands: int = 6
) -> np.ndarray:
    """
    Extract all spectral features in one function.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_chroma: Number of chroma bins
        n_bands: Number of spectral contrast bands
        
    Returns:
        Combined spectral feature vector
    """
    chroma = extract_chroma_features(audio, sr, n_chroma, n_fft, hop_length)
    centroid = extract_spectral_centroid(audio, sr, n_fft, hop_length)
    bandwidth = extract_spectral_bandwidth(audio, sr, n_fft, hop_length)
    contrast = extract_spectral_contrast(audio, sr, n_bands, n_fft, hop_length)
    zcr = extract_zero_crossing_rate(audio, hop_length)
    rms = extract_rms_energy(audio, n_fft, hop_length)
    
    features = np.concatenate([chroma, centroid, bandwidth, contrast, zcr, rms])
    
    return features
