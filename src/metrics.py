"""
Image quality assessment metrics for compression evaluation.

This module provides various metrics to evaluate the quality of compressed images
compared to their original versions.
"""

import numpy as np
from typing import Tuple, Dict, Any
import time
import os
import psutil
from functools import wraps

def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original: Original image as numpy array
        compressed: Compressed/reconstructed image as numpy array
        
    Returns:
        float: PSNR value in decibels (dB)
    """
    # Ensure images have the same shape
    if original.shape != compressed.shape:
        raise ValueError("Input images must have the same dimensions")
    
    # Convert to float64 to avoid overflow
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    
    # Calculate mean squared error
    mse = np.mean((original - compressed) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Maximum pixel value (255 for 8-bit images)
    max_pixel = 255.0
    
    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr

def calculate_ssim(original: np.ndarray, compressed: np.ndarray, 
                  window_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        original: Original image as numpy array
        compressed: Compressed/reconstructed image as numpy array
        window_size: Size of the sliding window
        k1, k2: Algorithm parameters (default values from original paper)
        
    Returns:
        float: SSIM index (higher is better, range [-1, 1])
    """
    from scipy.signal import fftconvolve
    
    # Ensure images have the same shape
    if original.shape != compressed.shape:
        raise ValueError("Input images must have the same dimensions")
    
    # Convert to float64
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    
    # Constants
    C1 = (k1 * 255) ** 2
    C2 = (k2 * 255) ** 2
    
    # Create 2D Gaussian window
    window = np.outer(
        np.hanning(window_size),
        np.hanning(window_size)
    )
    window = window / np.sum(window)
    
    # Apply window to both images
    mu1 = fftconvolve(original, window, mode='valid')
    mu2 = fftconvolve(compressed, window, mode='valid')
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = fftconvolve(original ** 2, window, mode='valid') - mu1_sq
    sigma2_sq = fftconvolve(compressed ** 2, window, mode='valid') - mu2_sq
    sigma12 = fftconvolve(original * compressed, window, mode='valid') - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)

def compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Calculate compression ratio.
    
    Args:
        original_size: Size of original data in bytes
        compressed_size: Size of compressed data in bytes
        
    Returns:
        float: Compression ratio (original_size / compressed_size)
    """
    if compressed_size == 0:
        return float('inf')
    return original_size / compressed_size

def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)

def measure_time(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

def measure_memory():
    """Context manager to measure memory usage."""
    process = psutil.Process()
    start_memory = process.memory_info().rss
    
    class MemoryContext:
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.peak_memory = process.memory_info().rss - start_memory
    
    return MemoryContext()

def evaluate_compression(original: np.ndarray, compressed: np.ndarray,
                       original_size: int, compressed_size: int) -> Dict[str, Any]:
    """
    Evaluate compression performance with multiple metrics.
    
    Args:
        original: Original image
        compressed: Compressed/reconstructed image
        original_size: Size of original data in bytes
        compressed_size: Size of compressed data in bytes
        
    Returns:
        Dict containing evaluation metrics
    """
    return {
        'psnr': calculate_psnr(original, compressed),
        'ssim': calculate_ssim(original, compressed),
        'compression_ratio': compression_ratio(original_size, compressed_size),
        'bpp': (compressed_size * 8) / (original.shape[0] * original.shape[1])
    }
