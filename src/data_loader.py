"""
Data loading and preprocessing utilities for X-ray image datasets.
Supports loading from Hugging Face datasets and local files.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any

import numpy as np
from PIL import Image, ImageOps
from datasets import load_dataset, Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Try to import OpenCV, but make it optional
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False



class XRayDataset:
    def __init__(
        self,
        data_dir: str = "data",
        split: str = 'train',
        image_size: Tuple[int, int] = (256, 256),
        bit_depth: int = 8,
        pad_to_pow2: bool = True,
        denoising: Optional[str] = None,
        denoising_params: Optional[Dict[str, float]] = None,
        random_seed: int = 42,
        use_cache: bool = True,
    ):
        """Initialize the X-ray dataset loader.

        Args:
            data_dir: Directory containing 'train', 'val', 'test' subdirectories
            split: Which split to use ('train', 'val', 'test')
            image_size: Target image size (height, width)
            bit_depth: Bit depth for image normalization (8 or 16)
            pad_to_pow2: Whether to pad images to power-of-two dimensions
            denoising: Type of denoising to apply ('gaussian', 'bilateral', or None)
            denoising_params: Parameters for denoising
            random_seed: Random seed for reproducibility
            use_cache: Whether to use cached dataset if available
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.bit_depth = bit_depth
        self.pad_to_pow2 = pad_to_pow2
        self.denoising = denoising
        self.denoising_params = denoising_params or {}
        self.split = split
        self.random_seed = random_seed
        self.use_cache = use_cache
        
        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'")
            
        # Set up paths
        self.split_dir = self.data_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        # Define image transforms
        self.transform = self._get_transforms()
        
        # Load dataset
        self.dataset = self._load_dataset()
    
    def _get_power_of_two_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate the nearest power of two dimensions that maintains aspect ratio."""
        height, width = size
        
        def nearest_power_of_two(n):
            return 2 ** max(4, int(np.ceil(np.log2(n))))  # Minimum size of 16x16
            
        # Maintain aspect ratio
        if width > height:
            new_width = nearest_power_of_two(width)
            new_height = max(16, 2 ** int(np.ceil(np.log2(height * new_width / width))))
        else:
            new_height = nearest_power_of_two(height)
            new_width = max(16, 2 ** int(np.ceil(np.log2(width * new_height / height))))
            
        return new_height, new_width
    
    def _pad_to_power_of_two(self, image: Image.Image) -> Image.Image:
        """Pad image to nearest power of two dimensions."""
        width, height = image.size
        target_height, target_width = self._get_power_of_two_size((height, width))
        
        # Calculate padding
        pad_width = max(0, (target_width - width) // 2)
        pad_height = max(0, (target_height - height) // 2)
        padding = (
            pad_width, 
            pad_height, 
            (target_width - width - pad_width), 
            (target_height - height - pad_height)
        )
        
        # Apply padding with edge extension
        return ImageOps.expand(image, border=padding, fill=0)
    
    def _crop_to_power_of_two(self, image: Image.Image) -> Image.Image:
        """Crop image to largest power of two dimensions."""
        width, height = image.size
        
        # Find largest power of two that fits within the image
        crop_width = 2 ** int(np.floor(np.log2(width)))
        crop_height = 2 ** int(np.floor(np.log2(height)))
        
        # Center crop
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        return image.crop((left, top, right, bottom))
    
    def _apply_denoising(self, image: Image.Image) -> Image.Image:
        """Apply denoising to the image if specified.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Denoised PIL Image
        """
        if not self.denoising:
            return image
            
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        try:
            import cv2
            
            if self.denoising == 'gaussian':
                # Default parameters for Gaussian blur
                ksize = self.denoising_params.get('ksize', 5)
                sigma = self.denoising_params.get('sigma', 1.0)
                
                # Apply Gaussian blur
                denoised = cv2.GaussianBlur(
                    img_array, 
                    ksize=(ksize, ksize), 
                    sigmaX=sigma
                )
                
            elif self.denoising == 'bilateral':
                # Default parameters for bilateral filter
                d = self.denoising_params.get('d', 9)
                sigma_color = self.denoising_params.get('sigma_color', 75)
                sigma_space = self.denoising_params.get('sigma_space', 75)
                
                # Apply bilateral filter
                denoised = cv2.bilateralFilter(
                    img_array,
                    d=d,
                    sigmaColor=sigma_color,
                    sigmaSpace=sigma_space
                )
            else:
                raise ValueError(f"Unsupported denoising method: {self.denoising}")
                
            return Image.fromarray(denoised)
            
        except ImportError:
            print("Warning: OpenCV is required for denoising. Please install it with: pip install opencv-python")
            return image
    
    def map_dynamic_range(
        self, 
        image: Union[np.ndarray, Image.Image], 
        input_bits: int = 16, 
        output_bits: int = 8, 
        min_percentile: float = 0.5, 
        max_percentile: float = 99.5
    ) -> np.ndarray:
        """Map the dynamic range of an image.
        
        Args:
            image: Input image as numpy array or PIL Image
            input_bits: Bit depth of the input image
            output_bits: Desired output bit depth
            min_percentile: Minimum percentile for contrast stretching
            max_percentile: Maximum percentile for contrast stretching
            
        Returns:
            Numpy array with mapped pixel values
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Calculate input range based on bit depth
        input_max = (1 << input_bits) - 1
        output_max = (1 << output_bits) - 1
        
        # Convert to float for calculations
        img_float = image.astype(np.float32)
        
        # Calculate percentiles for contrast stretching
        if min_percentile > 0 or max_percentile < 100:
            min_val = np.percentile(img_float, min_percentile)
            max_val = np.percentile(img_float, max_percentile)
            
            # Apply contrast stretching
            img_float = np.clip(img_float, min_val, max_val)
            img_float = (img_float - min_val) / (max_val - min_val) * input_max
        
        # Normalize to output range
        if input_max != output_max:
            img_float = img_float / input_max * output_max
        
        # Clip to valid range and convert to integer
        result = np.clip(img_float, 0, output_max)
        return result.astype(np.uint16 if output_bits > 8 else np.uint8)
    
    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load an image from file, supporting PNG and JPG formats."""
        image_path = Path(image_path)
        
        try:
            # Check file extension
            if image_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                raise ValueError(f"Unsupported image format: {image_path.suffix}. Only PNG and JPG are supported.")
                
            # Open image file
            img = Image.open(image_path)
            
            # Convert to grayscale if not already
            if img.mode != 'L':
                img = img.convert('L')
                
            # Convert to numpy array for processing
            img_array = np.array(img)
            
            # Apply dynamic range mapping if needed
            if self.bit_depth < 16 and img_array.dtype == np.uint16:
                img_array = self.map_dynamic_range(
                    img_array,
                    input_bits=16,
                    output_bits=self.bit_depth,
                    min_percentile=0.5,
                    max_percentile=99.5
                )
                return Image.fromarray(img_array, 'L')
                
            return img
            
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
    
    def _apply_padding(self, img):
        """Apply padding if enabled."""
        return self._pad_to_power_of_two(img) if self.pad_to_pow2 else img

    def _get_transforms(self):
        """Get image transformations based on configuration."""
        # Calculate target size as power of two
        target_height, target_width = self._get_power_of_two_size(self.image_size)
        
        transform_list = []
        
        # Add padding if enabled
        if self.pad_to_pow2:
            transform_list.append(transforms.Lambda(self._apply_padding))
            
        # Add resizing
        transform_list.extend([
            transforms.Resize((target_height, target_width)),
            transforms.Grayscale(num_output_channels=1)
        ])
        
        # Add denoising if enabled
        if self.denoising:
            transform_list.append(transforms.Lambda(self._apply_denoising))
            
        # Add final transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        return transforms.Compose(transform_list)
    
    def _create_data_splits(self, dataset: Dataset) -> Dict[str, Dataset]:
        """Split the dataset into train/val/test sets."""
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Shuffle the dataset
        indices = np.random.permutation(total_size)
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subsets
        return {
            'train': dataset.select(train_indices),
            'val': dataset.select(val_indices),
            'test': dataset.select(test_indices)
        }
    
    def _load_dataset(self) -> Dataset:
        """Load the dataset from the specified split directory."""
        print(f"Loading {self.split} dataset from {self.split_dir}")
        
        # Find all image files in the split directory
        image_paths = []
        for ext in ['.png', '.jpg', '.jpeg', '.dcm', '.dicom']:
            # Use glob with case-insensitive matching
            pattern = f'*{ext}'
            if ext[0] == '.':  # If extension starts with dot
                pattern = f'*{ext.lower()}' + '|' + f'*{ext.upper()}'
            
            for p in self.split_dir.glob('**/*'):
                if p.suffix.lower() == ext.lower() and p.is_file():
                    image_paths.append(p)
            
        if not image_paths:
            raise FileNotFoundError(f"No image files found in {self.split_dir}")
            
        print(f"Found {len(image_paths)} images in {self.split} split")
        
        # Create a dataset from the image paths
        dataset = Dataset.from_dict({"image_path": [str(p) for p in image_paths]})
        return dataset
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an item from the dataset."""
        item = self.dataset[idx]
        
        # Handle different dataset formats
        if 'image' in item:
            # Already loaded image
            img = item['image']
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
        elif 'image_path' in item:
            # Need to load image from path
            img = self._load_image(item['image_path'])
        else:
            raise ValueError("Dataset must contain either 'image' or 'image_path' field")
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Prepare output
        output = {'image': img}
        
        # Add any additional fields
        for key in item:
            if key not in ['image', 'image_path']:
                output[key] = item[key]
                
        return output
    
    def get_dataloader(self, batch_size: int = 32, **kwargs) -> DataLoader:
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=(self.split == 'train'),
            num_workers=os.cpu_count(),
            **kwargs
        )


def create_dataset(
    data_dir: str = "data",
    split: str = 'train',
    image_size: Tuple[int, int] = (256, 256),
    bit_depth: int = 8,
    **kwargs
) -> XRayDataset:
    """Create an X-ray dataset from the specified directory.
    
    Args:
        data_dir: Directory containing 'train', 'val', 'test' subdirectories
        split: Which split to load ('train', 'val', or 'test')
        image_size: Target image size (height, width)
        bit_depth: Bit depth for image normalization (8 or 16)
        **kwargs: Additional arguments to pass to XRayDataset
        
    Returns:
        XRayDataset instance
    """
    return XRayDataset(
        data_dir=data_dir,
        split=split,
        image_size=image_size,
        bit_depth=bit_depth,
        **kwargs
    )


def load_image(file_path: str, target_size: Optional[Tuple[int, int]] = None, 
              bit_depth: int = 8) -> np.ndarray:
    """
    Load an image from file and convert to grayscale.
    
    Args:
        file_path: Path to the image file
        target_size: Optional target size as (height, width)
        bit_depth: Bit depth of output image (8 or 16)
        
    Returns:
        Grayscale image as numpy array
    """
    file_path = str(file_path)
    
    # Handle DICOM files
    if file_path.lower().endswith(('.dcm', '.dicom')):
        if not HAS_PYDICOM:
            raise ImportError("pydicom is required for DICOM support. Install with: pip install pydicom")
        image = load_dicom_image(file_path)
    else:
        # Handle regular image files
        image = Image.open(file_path)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = ImageOps.grayscale(image)
            
        image = np.array(image)
    
    # Resize if needed
    if target_size is not None and target_size != image.shape[:2]:
        image = cv2.resize(image, (target_size[1], target_size[0]), 
                         interpolation=cv2.INTER_AREA)
    
    # Convert bit depth
    if bit_depth == 8 and image.dtype != np.uint8:
        image = (image / 256).astype(np.uint8)
    elif bit_depth == 16 and image.dtype != np.uint16:
        image = (image.astype(np.float32) * 65535 / 255).astype(np.uint16)
    
    return image


def save_image(image: np.ndarray, file_path: str) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image data as numpy array (HxW or HxWxC)
        file_path: Path to save the image
        
    Raises:
        ImportError: If OpenCV is not installed and required
        ValueError: If image is not a numpy array
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(image)}")
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Convert to 8-bit if needed
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    elif image.dtype != np.uint8:
        # Normalize to 0-255 if not already in uint8 or uint16
        image = ((image - image.min()) * (255.0 / (image.max() - image.min() + 1e-7))).astype(np.uint8)
    
    # Ensure 2D or 3D array
    if len(image.shape) == 2:
        # Grayscale
        if HAS_OPENCV:
            cv2.imwrite(str(file_path), image)
        else:
            Image.fromarray(image).save(file_path)
    elif len(image.shape) == 3:
        # Color - ensure RGB order for OpenCV
        if image.shape[2] == 3:  # RGB
            if HAS_OPENCV:
                cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                Image.fromarray(image, 'RGB').save(file_path)
        elif image.shape[2] == 4:  # RGBA
            if HAS_OPENCV:
                cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
            else:
                Image.fromarray(image, 'RGBA').save(file_path)
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


if __name__ == "__main__":
    # Example usage
    print("Loading X-ray dataset...")
    dataset = XRayDataset(
        data_dir="data",
        split='train',
        image_size=(512, 512),
        bit_depth=8,
        pad_to_pow2=True,
        denoising='gaussian',
        denoising_params={'ksize': 5, 'sigma': 1.0}
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create a DataLoader
    dataloader = dataset.get_dataloader(batch_size=4)
    
    # Iterate through the dataset
    for batch in dataloader:
        images = batch['image']
        print(f"Batch shape: {images.shape}")
        break  # Just show the first batch
