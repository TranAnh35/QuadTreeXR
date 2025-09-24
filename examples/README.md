# Compression Comparison Examples

This directory contains example scripts for comparing the Quadtree compression algorithm with JPEG.

## compare_compression.py

This script compares the performance of Quadtree compression with standard JPEG compression on X-ray images.

### Features

- Supports various image formats (PNG, JPG, DICOM)
- Multiple quality presets for comparison
- Visual comparison of results
- Detailed metrics (compression ratio, PSNR, processing time)

### Usage

```bash
python examples/compare_compression.py path/to/your/image.png --output results
```

### Arguments

- `image_path`: Path to the input image file (required)
- `--output`: Output directory for results (default: 'results')

### Example Output

The script will generate:
1. Comparison images showing original, Quadtree, and JPEG results
2. Console output with detailed metrics

```
=== KẾT QUẢ SO SÁNH ===
--------------------------------------------------------------------------------
CẤu hình          | Phương pháp  | Kích thước (KB) | Tỷ lệ nén | Thời gian (s) | PSNR (dB)
--------------------------------------------------------------------------------
high_quality      | Quadtree        |         123.4 KB |      5.67x |      0.0456 |    42.35
high_quality      | JPEG            |          98.7 KB |      7.12x |      0.0123 |    45.21
--------------------------------------------------------------------------------
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Notes

- For DICOM files, make sure `pydicom` is installed
- Higher compression ratios typically result in lower image quality
- Quadtree compression may be slower but can preserve more details in medical images
