# Quadtree-based X-ray Image Compression

This project implements and evaluates a quadtree-based compression algorithm for X-ray images, comparing its performance against standard JPEG compression in terms of compression ratio, image quality metrics (PSNR, SSIM), and computational efficiency.

## Project Structure

```
legalor-quadtree-cxr/
├── data/                # Scripts to download datasets and sample data
├── notebooks/           # Jupyter notebooks for EDA and visualization
├── results/             # Outputs, CSV of experiments, and plots
├── slides/              # Presentation materials
└── src/                 # Source code
    ├── loader.py         # Dataset loader (DICOM/PNG/JPG)
    ├── preprocess.py     # Image preprocessing (resize, denoise, pad)
    ├── quadtree.py       # Core quadtree implementation
    ├── encoder.py        # Binary bitstream writer/reader
    ├── baseline_jpeg.py  # JPEG baseline implementation
    ├── metrics.py        # Evaluation metrics (PSNR, SSIM)
    ├── experiments.py    # Experiment runner and logging
    └── utils.py          # Utility functions
```

## Getting Started

### Prerequisites
- Python 3.9+
- Required packages: `numpy`, `opencv-python`, `pillow`, `pydicom`, `scikit-image`, `numba`

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/quadtree-xray-compression.git
cd quadtree-xray-compression

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Download the desired dataset (MIMIC-CXR, CheXpert, or NIH ChestX-ray14)
2. Place the data in the appropriate directory structure
3. Run preprocessing scripts to prepare the data

### Running Experiments
```bash
# Run quadtree compression with default parameters
python src/experiments.py --method quadtree --dataset_path data/chestxray

# Compare with JPEG baseline
python src/experiments.py --method jpeg --quality 75 --dataset_path data/chestxray

# Run parameter sweep
python src/experiments.py --sweep --dataset_path data/chestxray
```

## Results

Performance metrics and visual comparisons will be saved in the `results/` directory.

## References
- MIMIC-CXR Database (Johnson et al., 2019)
- SSIM: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004)
