"""
Noise reduction pipeline for audio inference.
Applies spectral gating and high-pass filtering.
"""

import numpy as np
import noisereduce as nr
from scipy import signal
import logging

logger = logging.getLogger(__name__)


def apply_highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 80) -> np.ndarray:
    """
    Apply high-pass filter to remove low-frequency noise.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        cutoff: Cutoff frequency in Hz
        
    Returns:
        Filtered audio signal
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    
    b, a = signal.butter(5, normalized_cutoff, btype='high', analog=False)
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def apply_spectral_gating(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.03
) -> np.ndarray:
    """
    Apply spectral gating noise reduction.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        threshold: Threshold for noise gate
        
    Returns:
        Denoised audio signal
    """
    try:
        reduced_noise = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=threshold
        )
        return reduced_noise
    except Exception as e:
        logger.warning(f"Spectral gating failed: {e}. Returning original audio.")
        return audio


def median_filter_smoothing(audio: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter for smoothing.
    
    Args:
        audio: Audio signal
        kernel_size: Size of median filter kernel
        
    Returns:
        Smoothed audio signal
    """
    return signal.medfilt(audio, kernel_size=kernel_size)


def noise_reduction_pipeline(
    audio: np.ndarray,
    sr: int,
    enable: bool = True,
    highpass_cutoff: float = 80,
    gate_threshold: float = 0.03,
    apply_smoothing: bool = False
) -> np.ndarray:
    """
    Complete noise reduction pipeline.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        enable: Enable noise reduction
        highpass_cutoff: High-pass filter cutoff frequency
        gate_threshold: Spectral gate threshold
        apply_smoothing: Apply median filter smoothing
        
    Returns:
        Processed audio signal
    """
    if not enable:
        return audio
    
    processed = audio.copy()
    
    processed = apply_highpass_filter(processed, sr, highpass_cutoff)
    processed = apply_spectral_gating(processed, sr, gate_threshold)
    
    if apply_smoothing:
        processed = median_filter_smoothing(processed)
    
    if np.max(np.abs(processed)) > 0:
        processed = processed / np.max(np.abs(processed))
    
    return processed
