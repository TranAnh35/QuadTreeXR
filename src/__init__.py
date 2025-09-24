"""
Main package for the QuadtreeXR project:

- quadtree: Implementation of quadtree data structure and image compression algorithm
- jpeg_compressor: Wrapper for JPEG image compression for performance comparison
- bitstream: Binary data read/write utilities
- huffman: Huffman coding implementation
- data_loader: Image data loading and processing utilities
- metrics: Image quality assessment and performance metrics
"""

from .quadtree import QuadTree, QuadTreeNode, compress_image, visualize_quadtree
from .jpeg_compressor import JPEGCompressor, CompressionBenchmark
from .bitstream import BitStreamWriter, BitStreamReader
from .huffman import HuffmanCoding
from .data_loader import load_image, save_image
from .metrics import (
    calculate_psnr, calculate_ssim, compression_ratio,
    get_file_size, measure_time, measure_memory, evaluate_compression
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
    'HuffmanCoding'
]
