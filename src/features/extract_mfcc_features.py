"""
MFCC feature extraction for music genre classification.
"""

import numpy as np
import librosa
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def extract_mfcc_features(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128
) -> np.ndarray:
    """
    Extract MFCC features from audio signal.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        
    Returns:
        MFCC feature vector (statistical aggregation)
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    features = np.concatenate([mfcc_mean, mfcc_std])
    
    return features


def extract_mfcc_features_with_delta(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128
) -> np.ndarray:
    """
    Extract MFCC features with delta and delta-delta coefficients.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        
    Returns:
        Extended MFCC feature vector with deltas
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    delta_mean = np.mean(delta_mfccs, axis=1)
    delta_std = np.std(delta_mfccs, axis=1)
    delta2_mean = np.mean(delta2_mfccs, axis=1)
    delta2_std = np.std(delta2_mfccs, axis=1)
    
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        delta2_mean, delta2_std
    ])
    
    return features


def extract_mfcc_statistics(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128
) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive MFCC statistics.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        
    Returns:
        Dictionary with different MFCC representations
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    return {
        'mean': np.mean(mfccs, axis=1),
        'std': np.std(mfccs, axis=1),
        'min': np.min(mfccs, axis=1),
        'max': np.max(mfccs, axis=1),
        'median': np.median(mfccs, axis=1),
        'raw': mfccs
    }
