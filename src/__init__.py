"""
Main package for the QuadtreeXR project:

- quadtree: Implementation of quadtree data structure and image compression algorithm
- jpeg_compressor: Wrapper for JPEG image compression for performance comparison
- bitstream: Binary data read/write utilities
- huffman: Huffman coding implementation
- data_loader: Image data loading and processing utilities
- metrics: Image quality assessment and performance metrics
"""

import numpy as np

from .quadtree import QuadTree, QuadTreeNode, compress_image, visualize_quadtree
from .jpeg_compressor import JPEGCompressor, CompressionBenchmark
from .bitstream import BitStreamWriter, BitStreamReader
from .huffman import HuffmanCoding
from .data_loader import load_image, save_image
from .metrics import (
    calculate_psnr, calculate_ssim, compression_ratio,
    get_file_size, measure_time, measure_memory, evaluate_compression
)

def map_quality_to_quadtree_params(quality_level: int) -> dict:
    """
    Map JPEG quality levels to QuadTree parameters for consistent quality comparison.

    Args:
        quality_level: JPEG quality level (10, 30, 50, 70, 90)

    Returns:
        Dictionary containing QuadTree parameters for equivalent quality
    """
    # Validate input
    if quality_level not in [10, 30, 50, 70, 90]:
        raise ValueError("Quality level must be one of: 10, 30, 50, 70, 90")

    # Quality mapping based on empirical testing and literature
    quality_mapping = {
        10: {  # Very low quality - high compression
            'variance_threshold': 200.0,  # High threshold = more aggressive merging
            'quantize_bits': 4,           # Low precision
            'min_size': 8,                # Larger blocks
            'max_depth': 6                 # Shallower tree
        },
        30: {  # Low quality
            'variance_threshold': 100.0,
            'quantize_bits': 5,
            'min_size': 4,
            'max_depth': 8
        },
        50: {  # Medium quality
            'variance_threshold': 50.0,
            'quantize_bits': 6,
            'min_size': 2,
            'max_depth': 10
        },
        70: {  # High quality
            'variance_threshold': 20.0,
            'quantize_bits': 7,
            'min_size': 2,
            'max_depth': 12
        },
        90: {  # Very high quality
            'variance_threshold': 5.0,    # Low threshold = more splitting
            'quantize_bits': 8,           # High precision
            'min_size': 2,                # Smaller blocks
            'max_depth': 14                # Deeper tree
        }
    }

    params = quality_mapping[quality_level].copy()

    # Add common parameters
    params.update({
        'use_quantization': True,
        'use_huffman': True
    })

    return params

def create_quadtree_with_quality(image: np.ndarray, quality_level: int) -> QuadTree:
    """
    Create a QuadTree with parameters mapped from JPEG quality level.

    Args:
        image: Input image as numpy array
        quality_level: JPEG quality level (10, 30, 50, 70, 90)

    Returns:
        QuadTree instance with appropriate parameters for the quality level
    """
    params = map_quality_to_quadtree_params(quality_level)

    return QuadTree(
        image=image,
        min_size=params['min_size'],
        variance_threshold=params['variance_threshold'],
        max_depth=params['max_depth'],
        quantize_bits=params['quantize_bits'],
        use_quantization=params['use_quantization'],
        use_huffman=params['use_huffman']
    )

__all__ = [
    'QuadTree',
    'QuadTreeNode',
    'compress_image',
    'visualize_quadtree',
    'load_image',
    'save_image',
    'JPEGCompressor',
    'CompressionBenchmark',
    'calculate_psnr',
    'calculate_ssim',
    'compression_ratio',
    'get_file_size',
    'measure_time',
    'measure_memory',
    'evaluate_compression',
    'BitStreamWriter',
    'BitStreamReader',
    'HuffmanCoding',
    'map_quality_to_quadtree_params',
    'create_quadtree_with_quality'
]
