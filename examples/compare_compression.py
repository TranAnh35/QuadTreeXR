"""
Script to compare performance between Quadtree and JPEG compression
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import json
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import (
    compress_image, load_image, save_image, 
    JPEGCompressor, CompressionBenchmark,
    calculate_psnr, calculate_ssim, compression_ratio,
    measure_time, measure_memory, evaluate_compression
)

def save_comparison_figure(original: np.ndarray, quadtree_img: np.ndarray, 
                          jpeg_img: np.ndarray, title: str, output_path: str):
    """Save a comparison figure with original and compressed images."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Quadtree
    axes[1].imshow(quadtree_img, cmap='gray' if len(quadtree_img.shape) == 2 else None)
    axes[1].set_title('Quadtree Compressed')
    axes[1].axis('off')
    
    # JPEG
    axes[2].imshow(jpeg_img, cmap='gray' if len(jpeg_img.shape) == 2 else None)
    axes[2].set_title('JPEG Compressed')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def compare_compression(image_path: str, output_dir: str):
    """Compare Quadtree and JPEG compression on an image."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    print(f"\n=== Loading image: {image_path} ===")
    image = load_image(image_path)
    
    # Convert to 8-bit grayscale if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Ensure dimensions are multiples of 16 for quadtree
    h, w = image.shape[:2]
    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    if new_h != h or new_w != w:
        print(f"Resizing image from {h}x{w} to {new_h}x{new_w} for quadtree compatibility")
        image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS))
    
    # Test different quality settings
    # Parameters: (name, min_size, var_thresh, jpeg_quality)
    test_cases = [
        ("high_quality", 2, 10, 95),    # Higher quality with more details
        ("medium_quality", 4, 25, 85),  # Balanced quality
        ("low_quality", 8, 50, 75)      # More aggressive compression
    ]
    
    results = []
    
    for name, min_size, var_thresh, jpeg_quality in test_cases:
        print(f"\n=== Testing {name} ===")
        print(f"Quadtree: min_size={min_size}, var_thresh={var_thresh}")
        print(f"JPEG quality: {jpeg_quality}")
        
        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # ===== Quadtree Compression =====
            print("\n[Quadtree Compression]")
            quadtree = None
            quadtree_time = 0
            
            # Time and memory measurement for quadtree
            with measure_memory() as mem_ctx:
                start_time = time.time()
                quadtree = compress_image(
                    image, 
                    min_size=min_size, 
                    variance_threshold=var_thresh
                )
                quadtree_time = time.time() - start_time
            
            # Reconstruct image
            reconstructed_qt = quadtree.reconstruct().astype(np.uint8)
            
            # Save quadtree to a temporary file and measure its size
            qt_path = temp_dir / f"{name}_quadtree.qtr"
            quadtree.save(str(qt_path), format='bin')
            
            # Get the size of the saved file
            qt_size = os.path.getsize(qt_path) / 1024.0  # Convert to KB
            
            # Read the file back to get the compressed data
            with open(qt_path, 'rb') as f:
                compressed_qt = f.read()
            
            # ===== JPEG Compression =====
            print("\n[JPEG Compression]")
            jpeg = JPEGCompressor(quality=jpeg_quality)
            compressed_jpeg = None
            jpeg_time = 0
            
            with measure_memory() as mem_ctx_jpeg:
                start_time = time.time()
                compressed_jpeg = jpeg.compress(image)
                jpeg_time = time.time() - start_time
            
            # Get JPEG size in KB
            jpeg_size = len(compressed_jpeg) / 1024.0
            
            # Save JPEG to file for reference
            jpeg_path = temp_dir / f"{name}_jpeg.jpg"
            with open(jpeg_path, 'wb') as f:
                f.write(compressed_jpeg)
            
            # Decompress for comparison
            reconstructed_jpeg = jpeg.decompress(compressed_jpeg)
            
            # ===== Evaluation =====
            # Save images for visualization
            original_path = temp_dir / f"{name}_original.png"
            qt_img_path = temp_dir / f"{name}_quadtree.png"
            jpeg_img_path = temp_dir / f"{name}_jpeg.png"
            
            save_image(image, str(original_path))
            save_image(reconstructed_qt, str(qt_img_path))
            save_image(reconstructed_jpeg, str(jpeg_img_path))
            
            # Load images to ensure we're comparing the exact pixel data
            original_img = load_image(str(original_path))
            qt_img = load_image(str(qt_img_path))
            jpeg_img = load_image(str(jpeg_img_path))
            
            # Get original image size in KB
            original_size = original_path.stat().st_size / 1024.0
            
            # Calculate metrics for quadtree
            qt_metrics = evaluate_compression(
                original_img, 
                qt_img,
                original_path.stat().st_size,  # Original size in bytes
                qt_size * 1024  # Compressed size in bytes
            )
            
            # Calculate metrics for JPEG
            jpeg_metrics = evaluate_compression(
                original_img, 
                jpeg_img,
                original_path.stat().st_size,  # Original size in bytes
                jpeg_size * 1024  # Compressed size in bytes
            )
            
            # Store results
            results.append({
                'name': name,
                'parameters': {
                    'quadtree': {'min_size': min_size, 'var_thresh': var_thresh},
                    'jpeg': {'quality': jpeg_quality}
                },
                'original_size': original_path.stat().st_size,
                'quadtree': {
                    'size': qt_path.stat().st_size,
                    'compression_ratio': qt_metrics['compression_ratio'],
                    'psnr': qt_metrics['psnr'],
                    'ssim': qt_metrics['ssim'],
                    'bpp': qt_metrics['bpp'],
                    'time': quadtree_time,
                    'memory': mem_ctx.peak_memory / (1024 * 1024)  # MB
                },
                'jpeg': {
                    'size': jpeg_path.stat().st_size,
                    'compression_ratio': jpeg_metrics['compression_ratio'],
                    'psnr': jpeg_metrics['psnr'],
                    'ssim': jpeg_metrics['ssim'],
                    'bpp': jpeg_metrics['bpp'],
                    'time': jpeg_time,
                    'memory': mem_ctx_jpeg.peak_memory / (1024 * 1024)  # MB
                }
            })
            
            # Save comparison figure
            fig_path = os.path.join(output_dir, f"{name}_comparison.png")
            save_comparison_figure(
                original_img, qt_img, jpeg_img,
                f"{name.replace('_', ' ').title()} (QT: {qt_metrics['psnr']:.2f}dB, "
                f"JPEG: {jpeg_metrics['psnr']:.2f}dB)",
                fig_path
            )
            
            # Print summary
            print(f"\n{name.upper()} Results:")
            print(f"Original size: {original_path.stat().st_size / 1024:.2f} KB")
            print("\nQuadtree:")
            print(f"  Size: {qt_path.stat().st_size / 1024:.2f} KB")
            print(f"  Compression ratio: {qt_metrics['compression_ratio']:.2f}x")
            print(f"  PSNR: {qt_metrics['psnr']:.2f} dB")
            print(f"  SSIM: {qt_metrics['ssim']:.4f}")
            print(f"  bpp: {qt_metrics['bpp']:.2f}")
            print(f"  Time: {quadtree_time:.2f}s")
            print(f"  Memory: {mem_ctx.peak_memory / (1024 * 1024):.2f} MB")
            
            print("\nJPEG:")
            print(f"  Size: {jpeg_path.stat().st_size / 1024:.2f} KB")
            print(f"  Compression ratio: {jpeg_metrics['compression_ratio']:.2f}x")
            print(f"  PSNR: {jpeg_metrics['psnr']:.2f} dB")
            print(f"  SSIM: {jpeg_metrics['ssim']:.4f}")
            print(f"  bpp: {jpeg_metrics['bpp']:.2f}")
            print(f"  Time: {jpeg_time:.2f}s")
            print(f"  Memory: {mem_ctx_jpeg.peak_memory / (1024 * 1024):.2f} MB")
    
    # Generate summary report
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, 'w') as f:
        f.write("=== Compression Comparison Report ===\n\n")
        
        for result in results:
            f.write(f"\n=== {result['name'].upper()} ===\n")
            f.write(f"Original size: {result['original_size'] / 1024:.2f} KB\n\n")
            
            f.write("Quadtree:\n")
            f.write(f"  Size: {result['quadtree']['size'] / 1024:.2f} KB\n")
            f.write(f"  Compression ratio: {result['quadtree']['compression_ratio']:.2f}x\n")
            f.write(f"  PSNR: {result['quadtree']['psnr']:.2f} dB\n")
            f.write(f"  SSIM: {result['quadtree']['ssim']:.4f}\n")
            f.write(f"  bpp: {result['quadtree']['bpp']:.2f}\n")
            f.write(f"  Time: {result['quadtree']['time']:.2f}s\n")
            f.write(f"  Memory: {result['quadtree']['memory']:.2f} MB\n\n")
            
            f.write("JPEG:\n")
            f.write(f"  Size: {result['jpeg']['size'] / 1024:.2f} KB\n")
            f.write(f"  Compression ratio: {result['jpeg']['compression_ratio']:.2f}x\n")
            f.write(f"  PSNR: {result['jpeg']['psnr']:.2f} dB\n")
            f.write(f"  SSIM: {result['jpeg']['ssim']:.4f}\n")
            f.write(f"  bpp: {result['jpeg']['bpp']:.2f}\n")
            f.write(f"  Time: {result['jpeg']['time']:.2f}s\n")
            f.write(f"  Memory: {result['jpeg']['memory']:.2f} MB\n")
    
    # Save detailed results to JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"- Comparison images: {output_dir}/*_comparison.png")
    print(f"- Detailed report: {report_path}")
    print(f"- Raw results: {results_path}")
    
    # Plot comparison charts
    plot_comparison_charts(results, output_dir)

def plot_comparison_charts(results: list, output_dir: str):
    """Generate comparison charts for the results."""
    # Prepare data
    names = [r['name'].replace('_', ' ').title() for r in results]
    
    # PSNR Comparison
    plt.figure(figsize=(10, 6))
    qt_psnr = [r['quadtree']['psnr'] for r in results]
    jpeg_psnr = [r['jpeg']['psnr'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, qt_psnr, width, label='Quadtree')
    plt.bar(x + width/2, jpeg_psnr, width, label='JPEG')
    
    plt.xlabel('Quality Setting')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Comparison')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    psnr_path = os.path.join(output_dir, 'psnr_comparison.png')
    plt.savefig(psnr_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # File Size Comparison
    plt.figure(figsize=(10, 6))
    qt_sizes = [r['quadtree']['size'] / 1024 for r in results]  # KB
    jpeg_sizes = [r['jpeg']['size'] / 1024 for r in results]    # KB
    
    plt.bar(x - width/2, qt_sizes, width, label='Quadtree')
    plt.bar(x + width/2, jpeg_sizes, width, label='JPEG')
    
    plt.xlabel('Quality Setting')
    plt.ylabel('File Size (KB)')
    plt.title('File Size Comparison')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    size_path = os.path.join(output_dir, 'size_comparison.png')
    plt.savefig(size_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compression Ratio Comparison
    plt.figure(figsize=(10, 6))
    qt_ratios = [r['quadtree']['compression_ratio'] for r in results]
    jpeg_ratios = [r['jpeg']['compression_ratio'] for r in results]
    
    plt.bar(x - width/2, qt_ratios, width, label='Quadtree')
    plt.bar(x + width/2, jpeg_ratios, width, label='JPEG')
    
    plt.xlabel('Quality Setting')
    plt.ylabel('Compression Ratio')
    plt.title('Compression Ratio Comparison')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    ratio_path = os.path.join(output_dir, 'ratio_comparison.png')
    plt.savefig(ratio_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison charts saved to {output_dir}")
    print(f"- PSNR comparison: {psnr_path}")
    print(f"- File size comparison: {size_path}")
    print(f"- Compression ratio comparison: {ratio_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Quadtree and JPEG compression')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output', '-o', default='results', 
                        help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    compare_compression(args.image_path, args.output)
