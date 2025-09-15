# Implementation Plan

## 1. Dataset Preparation (Week 1-2)

### 1.1 Dataset Selection
- [X] Download MIMIC-CXR dataset (PNG/JPG versions)
- [X] Create data loading utilities for PNG and JPG formats

### 1.2 Preprocessing Pipeline
- [X] Support for 8-bit and 16-bit grayscale images
- [X] Implement dynamic range mapping (for 16-bit → 8-bit conversion if needed)
- [X] Add padding/cropping to power-of-two dimensions
- [X] Implement optional denoising (Gaussian, bilateral)
- [X] Create data split (train/val/test)

## 2. Core Algorithm (Week 3-4)

### 2.1 Quadtree Implementation
- [X] Design QuadNode and Quadtree data structures
- [X] Implement basic recursive quadtree decomposition
- [X] Add variance-based splitting criterion
- [X] Implement minimum block size constraint
- [X] Add mean value approximation for leaf nodes
- [X] Optimize with integral images for O(1) block statistics
- [X] Add visualization for quadtree decomposition
- [X] Implement serialization/deserialization for quadtree

### 2.2 Optimization
- [ ] Integral images for O(1) block statistics
- [ ] Bitstream encoding/decoding
- [ ] Quantization of node values
- [ ] Entropy coding (Huffman/Arithmetic)

## 3. Baseline Implementation (Week 4)
- [ ] JPEG compression wrapper
- [ ] Quality parameter mapping
- [ ] Consistent interface for both methods

## 4. Evaluation Framework (Week 5)
- [ ] PSNR calculation
- [ ] SSIM implementation
- [ ] Compression ratio metrics
- [ ] Runtime performance measurement
- [ ] Memory usage tracking

## 5. Experiments (Week 6-7)

### 5.1 Parameter Sweep
- [ ] Variance threshold (τ)
- [ ] Minimum block size (1x1, 2x2, 4x4, 8x8)
- [ ] Quantization bits (8, 6, 4)
- [ ] JPEG quality levels (10, 30, 50, 70, 90)

### 5.2 Evaluation
- [ ] Rate-distortion curves
- [ ] Quality vs. compression ratio
- [ ] Computational complexity analysis
- [ ] Memory usage comparison

## 6. Analysis & Reporting (Week 7-8)
- [ ] Statistical analysis of results
- [ ] Generate visual comparisons
- [ ] Prepare final report
- [ ] Create presentation slides

## Technical Specifications

### Quadtree Node Structure
```python
class QuadTreeNode:
    def __init__(self, x, y, size):
        self.x = x          # x-coordinate of top-left corner
        self.y = y          # y-coordinate of top-left corner
        self.size = size    # Size of the block (power of 2)
        self.is_leaf = True
        self.mean_value = 0
        self.children = [None, None, None, None]  # TL, TR, BL, BR
```

### File Format Specification
```
[Header (13 bytes)]
- Magic number: 4 bytes (0x51 0x54 0x52 0x45)  # 'QTRE'
- Width: 4 bytes (big-endian)
- Height: 4 bytes (big-endian)
- Bit depth: 1 byte
- Min block size: 1 byte (log2 of min block dimension)
- Flags: 1 byte (reserved for future use)

[Node Data]
- Bitstream of nodes (pre-order traversal)
- Leaf nodes: 0 + mean value
- Internal nodes: 1

[Optional Entropy Coding Tables]
- If entropy coding is used
```

## Dependencies
- Python 3.9+
- NumPy
- OpenCV (optional, for advanced image processing)
- Pillow
- scikit-image
- Matplotlib (for visualization)
- tqdm (for progress bars)

## Expected Outcomes
1. Implementation of quadtree-based X-ray image compression
2. Comparison with JPEG at equivalent bitrates
3. Analysis of compression performance and image quality
4. Documentation of findings and implementation details
