import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional
import io

class JPEGCompressor:
    """
    Lớp wrapper cho nén ảnh JPEG với OpenCV
    Hỗ trợ mức chất lượng từ 0-100 và tối ưu hóa cho ảnh y tế
    """
    
    def __init__(self, quality: int = 95, optimize: bool = True):
        """
        Khởi tạo bộ nén JPEG
        
        Args:
            quality: Chất lượng nén (0-100), mặc định 95
            optimize: Có tối ưu hóa quá trình nén không
        """
        self.quality = max(0, min(100, quality))
        self.optimize = optimize
        self.params = [
            int(cv2.IMWRITE_JPEG_QUALITY), self.quality,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1 if self.optimize else 0
        ]
    
    def compress(self, image: np.ndarray) -> bytes:
        """
        Nén ảnh đầu vào thành dạng JPEG
        
        Args:
            image: Ảnh đầu vào (numpy array, uint8 hoặc uint16)
            
        Returns:
            Dữ liệu ảnh đã nén dạng bytes
        """
        # Chuyển đổi về dạng 8-bit nếu cần
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        
        # Mã hóa ảnh thành JPEG
        success, buffer = cv2.imencode('.jpg', image, self.params)
        
        if not success:
            raise ValueError("Không thể nén ảnh")
            
        return buffer.tobytes()
    
    def decompress(self, data: bytes) -> np.ndarray:
        """
        Giải nén dữ liệu JPEG thành ảnh
        
        Args:
            data: Dữ liệu ảnh đã nén dạng bytes
            
        Returns:
            Ảnh đã giải nén dạng numpy array (uint8)
        """
        # Chuyển đổi bytes thành numpy array
        buffer = np.frombuffer(data, dtype=np.uint8)
        
        # Giải mã ảnh
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError("Không thể giải nén ảnh")
            
        return image
    
    def compress_to_file(self, image: np.ndarray, file_path: Union[str, Path]) -> None:
        """
        Nén và lưu ảnh vào file
        
        Args:
            image: Ảnh đầu vào
            file_path: Đường dẫn file đích
        """
        compressed = self.compress(image)
        with open(file_path, 'wb') as f:
            f.write(compressed)
    
    @staticmethod
    def decompress_from_file(file_path: Union[str, Path]) -> np.ndarray:
        """
        Đọc và giải nén ảnh từ file
        
        Args:
            file_path: Đường dẫn file ảnh JPEG
            
        Returns:
            Ảnh đã giải nén
        """
        with open(file_path, 'rb') as f:
            data = f.read()
        return JPEGCompressor().decompress(data)
    
    @staticmethod
    def get_compression_ratio(original_size: int, compressed_size: int) -> float:
        """
        Tính tỷ lệ nén
        
        Args:
            original_size: Kích thước gốc (bytes)
            compressed_size: Kích thước sau khi nén (bytes)
            
        Returns:
            Tỷ lệ nén (original/compressed)
        """
        if compressed_size == 0:
            return float('inf')
        return original_size / compressed_size
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
        """
        Tính PSNR (Peak Signal-to-Noise Ratio) giữa ảnh gốc và ảnh đã nén
        
        Args:
            original: Ảnh gốc
            compressed: Ảnh đã nén và giải nén
            
        Returns:
            Giá trị PSNR (dB)
        """
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))

# Lớp tiện ích để so sánh với quadtree
class CompressionBenchmark:
    """Lớp tiện ích để so sánh hiệu suất giữa JPEG và Quadtree"""
    
    @staticmethod
    def compare_compression(image: np.ndarray,
                           compressor_func,
                           decompressor_func,
                           quality: int = 95) -> dict:
        """
        So sánh nén JPEG và Quadtree

        Args:
            image: Ảnh đầu vào
            compressor_func: Hàm nén QuadTree (nhận image, trả về bytes)
            decompressor_func: Hàm giải nén QuadTree (nhận bytes, trả về image)
            quality: Chất lượng nén JPEG (0-100)

        Returns:
            Dictionary chứa kết quả so sánh
        """
        # Nén bằng JPEG
        jpeg = JPEGCompressor(quality=quality)
        jpeg_data = jpeg.compress(image)
        jpeg_ratio = jpeg.get_compression_ratio(
            image.nbytes,
            len(jpeg_data)
        )

        # Giải nén và tính PSNR
        jpeg_decompressed = jpeg.decompress(jpeg_data)
        jpeg_psnr = jpeg.calculate_psnr(image, jpeg_decompressed)

        # Nén bằng Quadtree
        quadtree_data = compressor_func(image)
        quadtree_ratio = jpeg.get_compression_ratio(
            image.nbytes,
            len(quadtree_data)
        )
        quadtree_decompressed = decompressor_func(quadtree_data)
        quadtree_psnr = jpeg.calculate_psnr(image, quadtree_decompressed)

        return {
            'jpeg': {
                'size': len(jpeg_data),
                'ratio': jpeg_ratio,
                'psnr': jpeg_psnr
            },
            'quadtree': {
                'size': len(quadtree_data),
                'ratio': quadtree_ratio,
                'psnr': quadtree_psnr
            },
            'original_size': image.nbytes
        }
