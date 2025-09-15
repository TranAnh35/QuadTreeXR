"""
Quadtree-based image compression implementation.

This module provides the core quadtree data structure and algorithms
for image compression using variance-based splitting.
"""

import numpy as np
import cv2
import json
import base64
import pickle
import zlib
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union, Any, Dict, BinaryIO, Iterator
from dataclasses import dataclass, asdict, is_dataclass

# Import utilities
from .bitstream import BitStreamWriter, BitStreamReader
from .huffman import HuffmanCoding


@dataclass
class QuadTreeNode:
    """Node in the quadtree structure."""
    
    x: int                    # x-coordinate of top-left corner
    y: int                    # y-coordinate of top-left corner
    size: int                 # Size of the block (must be power of 2)
    is_leaf: bool = True      # Whether this is a leaf node
    mean_value: float = 0.0   # Mean intensity (for leaf nodes)
    variance: float = 0.0     # Variance of the block
    children: List[Optional['QuadTreeNode']] = None  # 4 children: TL, TR, BL, BR
    
    def __post_init__(self):
        if self.children is None:
            self.children = [None, None, None, None]
    
    def split(self) -> None:
        """Split the node into 4 children."""
        if self.size <= 1:
            raise ValueError("Cannot split node of size 1")
            
        self.is_leaf = False
        half = self.size // 2
        
        # Create four child nodes (TL, TR, BL, BR)
        self.children[0] = QuadTreeNode(self.x, self.y, half)                    # TL
        self.children[1] = QuadTreeNode(self.x + half, self.y, half)             # TR
        self.children[2] = QuadTreeNode(self.x, self.y + half, half)             # BL
        self.children[3] = QuadTreeNode(self.x + half, self.y + half, half)      # BR


class QuadTree:
    """Quadtree implementation for image compression."""
    
    def __init__(self, 
                 image: np.ndarray,
                 min_size: int = 2,
                 variance_threshold: float = 10.0,
                 max_depth: int = 10,
                 quantize_bits: int = 8,
                 use_quantization: bool = True,
                 use_huffman: bool = True):
        """Initialize the quadtree.
        
        Args:
            image: Input grayscale image (2D numpy array)
            min_size: Minimum block size (must be power of 2)
            variance_threshold: Maximum allowed variance in a leaf node
            max_depth: Maximum depth of the quadtree
            quantize_bits: Number of bits to use for quantization (1-16)
            use_quantization: Whether to use quantization for node values
        """
        self.image = image.astype(np.float32)
        self.height, self.width = image.shape
        self.min_size = min_size
        self.variance_threshold = variance_threshold
        self.max_depth = max_depth
        self.quantize_bits = min(max(1, quantize_bits), 16)  # Clamp to 1-16 bits
        self.use_quantization = use_quantization
        self.use_huffman = use_huffman
        
        # Initialize entropy coder
        self.huffman = HuffmanCoding() if use_huffman else None
        
        # Precompute quantization parameters
        self._init_quantization()
        
        # Precompute integral images for fast block statistics
        self._compute_integral_images()
        
        # Build the quadtree
        self.root = self._build_quadtree(0, 0, self._next_power_of_two(max(self.width, self.height)))
        
    def _init_quantization(self) -> None:
        """Initialize quantization parameters."""
        if self.use_quantization:
            self.quant_scale = (1 << self.quantize_bits) - 1
            self.quant_min = 0.0
            self.quant_max = 255.0  # For 8-bit grayscale
            self.quant_range = self.quant_max - self.quant_min
    
    def _quantize_value(self, value: float) -> float:
        """Quantize a floating-point value."""
        if not self.use_quantization:
            return value
            
        # Scale to [0, 2^bits - 1], round, then scale back
        normalized = (value - self.quant_min) / self.quant_range
        quantized = round(normalized * self.quant_scale)
        dequantized = (quantized / self.quant_scale) * self.quant_range + self.quant_min
        return dequantized
    
    def _quantize_node(self, node: 'QuadTreeNode') -> None:
        """Quantize the mean value of a node."""
        if node is None or not self.use_quantization:
            return
            
        node.mean_value = self._quantize_value(node.mean_value)
        
        # Recursively quantize children
        if not node.is_leaf and node.children:
            for child in node.children:
                self._quantize_node(child)
    
    def _next_power_of_two(self, n: int) -> int:
        """Find the smallest power of 2 >= n."""
        return 1 << (n - 1).bit_length() if n > 1 else 1
    
    def _compute_integral_images(self) -> None:
        """Precompute integral images for sum and sum of squares."""
        # Pad image to handle non-power-of-two dimensions
        size = self._next_power_of_two(max(self.width, self.height))
        padded = np.pad(self.image, 
                       ((0, size - self.height), 
                        (0, size - self.width)), 
                       'edge')
        
        # Compute integral images
        self.integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        self.integral_sq = np.cumsum(np.cumsum(padded**2, axis=0), axis=1)
        
        # Add zero padding for easier indexing
        self.integral = np.pad(self.integral, ((1, 0), (1, 0)), 'constant')
        self.integral_sq = np.pad(self.integral_sq, ((1, 0), (1, 0)), 'constant')
    
    def _block_stats(self, x: int, y: int, size: int) -> Tuple[float, float]:
        """Compute mean and variance of a block using direct computation."""
        # Extract the block from the image
        block = self.image[y:y+size, x:x+size]
        
        # Calculate mean and variance directly from the block
        mean = np.mean(block)
        variance = np.var(block) if block.size > 1 else 0.0
        
        # Apply quantization if enabled
        if self.use_quantization:
            mean = self._quantize_value(mean)
            
        return mean, max(0, variance)  # Ensure variance is non-negative
    
    def _build_quadtree(self, x: int, y: int, size: int, depth: int = 0) -> Optional[QuadTreeNode]:
        """Recursively build the quadtree with improved splitting and merging."""
        # Check if the current node is completely outside the image
        if x >= self.width or y >= self.height or size <= 0:
            return None
            
        # Calculate actual size within image bounds
        actual_size = min(size, self.width - x, self.height - y)
        if actual_size <= 0:
            return None
            
        # Calculate mean and variance for this block
        mean, variance = self._block_stats(x, y, actual_size)
        
        # Create node
        node = QuadTreeNode(x, y, actual_size)
        node.mean_value = mean
        node.variance = variance
        
        # Check stopping conditions
        if (size <= self.min_size or 
            variance <= self.variance_threshold or 
            depth >= self.max_depth):
            return node
            
        # Split the node if it's large enough
        half = max(1, size // 2)  # Ensure at least 1x1
        
        # Check if splitting would produce blocks that are too small
        if half < self.min_size // 2:
            return node
            
        node.split()
        
        # Recurse on children
        child_means = []
        all_leaves = True
        has_children = False
        
        for i, (dx, dy) in enumerate([(0, 0), (half, 0), (0, half), (half, half)]):
            child_x = x + dx
            child_y = y + dy
            
            # Skip if child is outside image bounds
            if child_x >= self.width or child_y >= self.height:
                continue
                
            child = self._build_quadtree(child_x, child_y, half, depth + 1)
            if child is not None:
                node.children[i] = child
                has_children = True
                all_leaves = all_leaves and child.is_leaf
                if child.is_leaf:
                    child_means.append(child.mean_value)
        
        # If all children are leaves, check if we should merge them
        if all_leaves and len(child_means) > 0:
            child_variance = np.var(child_means)
            
            # If variance is below threshold, merge the children
            if child_variance <= self.variance_threshold * 0.5:  # Slightly more aggressive merging
                node.is_leaf = True
                node.children = [None, None, None, None]
                node.mean_value = np.mean(child_means)
                node.variance = child_variance
                return node
        
        # If we have at least one valid child, keep the node
        if has_children:
            return node
            
        # If no valid children, return a leaf node
        return node
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct the image from the quadtree with optimized performance."""
        # Create output image with original dimensions
        result = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Use a stack for iterative traversal to avoid recursion depth issues
        stack = [(self.root, False)]
        
        while stack:
            node, processed = stack.pop()
            if node is None:
                continue
                
            if node.is_leaf:
                # Calculate actual bounds to prevent out-of-bounds access
                y_end = min(node.y + node.size, self.height)
                x_end = min(node.x + node.size, self.width)
                
                # Only process if within bounds
                if node.y < self.height and node.x < self.width:
                    result[node.y:y_end, node.x:x_end] = node.mean_value
            else:
                if not processed:
                    # Push back the node as processed
                    stack.append((node, True))
                    # Push children in reverse order (to process them in order)
                    for child in reversed(node.children):
                        if child is not None:
                            stack.append((child, False))
        
        return result
    
    def get_compression_ratio(self) -> float:
        """Calculate the compression ratio (original_size / compressed_size)."""
        original_bits = self.width * self.height * 8  # 8 bits per pixel
        compressed_bits = self._count_nodes(self.root) * 9  # 1 bit flag + 8 bits mean
        return original_bits / compressed_bits if compressed_bits > 0 else float('inf')
    
    def _count_nodes(self, node: QuadTreeNode) -> int:
        """Count the number of nodes in the quadtree."""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + sum(self._count_nodes(child) for child in node.children if child is not None)
    
    def get_leaf_nodes(self) -> List[QuadTreeNode]:
        """Get all leaf nodes in the quadtree."""
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves
    
    def _collect_leaves(self, node: QuadTreeNode, leaves: List[QuadTreeNode]) -> None:
        """Collect all leaf nodes."""
        if node is None:
            return
        if node.is_leaf:
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)
                
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the quadtree to a dictionary.
        
        Returns:
            A dictionary containing the quadtree data
        """
        def node_to_dict(node: Optional[QuadTreeNode]) -> Optional[Dict[str, Any]]:
            if node is None:
                return None
            
            # Convert node to dict, handling children recursively
            node_dict = asdict(node)
            node_dict['children'] = [node_to_dict(child) for child in node.children]
            return node_dict
            
        # Convert numpy arrays to lists for JSON serialization
        image_data = {
            'data': base64.b64encode(
                zlib.compress(
                    pickle.dumps(self.image, protocol=pickle.HIGHEST_PROTOCOL)
                )
            ).decode('utf-8'),
            'dtype': str(self.image.dtype),
            'shape': list(self.image.shape)
        }
        
        return {
            'image': image_data,
            'min_size': self.min_size,
            'variance_threshold': self.variance_threshold,
            'max_depth': self.max_depth,
            'root': node_to_dict(self.root)
        }
        
    def to_json(self, file_path: Optional[str] = None) -> Optional[str]:
        """Serialize the quadtree to JSON.
        
        Args:
            file_path: Optional path to save the JSON file. If None, returns the JSON string.
            
        Returns:
            JSON string if file_path is None, otherwise None
        """
        import json
        
        def default_serializer(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
        data = self.to_dict()
        json_str = json.dumps(data, default=default_serializer, indent=2)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
            return None
        return json_str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuadTree':
        """Deserialize a quadtree from a dictionary.
        
        Args:
            data: Dictionary containing quadtree data
            
        Returns:
            A new QuadTree instance
        """
        # Reconstruct the numpy array
        image_data = data['image']
        compressed_data = base64.b64decode(image_data['data'].encode('utf-8'))
        image = pickle.loads(zlib.decompress(compressed_data))
        
        # Create a new QuadTree instance
        quadtree = cls.__new__(cls)
        quadtree.image = image
        quadtree.height, quadtree.width = image.shape
        quadtree.min_size = data['min_size']
        quadtree.variance_threshold = data['variance_threshold']
        quadtree.max_depth = data['max_depth']
        
        # Rebuild the quadtree structure
        def dict_to_node(node_data: Optional[Dict[str, Any]]) -> Optional[QuadTreeNode]:
            if node_data is None:
                return None
                
            # Create node from dict
            node = QuadTreeNode(
                x=node_data['x'],
                y=node_data['y'],
                size=node_data['size'],
                is_leaf=node_data['is_leaf'],
                mean_value=node_data['mean_value'],
                variance=node_data['variance']
            )
            
            # Rebuild children recursively
            if node_data['children'] is not None:
                node.children = [dict_to_node(child) for child in node_data['children']]
                
            return node
            
        quadtree.root = dict_to_node(data['root'])
        return quadtree
    
    @classmethod
    def from_json(cls, json_str_or_file: str) -> 'QuadTree':
        """Deserialize a quadtree from a JSON string or file.
        
        Args:
            json_str_or_file: JSON string or path to a JSON file
            
        Returns:
            A new QuadTree instance
        """
        import json
        
        # Check if input is a file path
        try:
            with open(json_str_or_file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, OSError):
            # If not a file, treat as JSON string
            data = json.loads(json_str_or_file)
            
        return cls.from_dict(data)
    
    def save(self, file_path: str, format: str = 'pickle') -> None:
        """Save the quadtree to a file.
        
        Args:
            file_path: Path to save the file
            format: Serialization format ('pickle', 'json', 'bin' for binary bitstream, 
                   or 'huff' for Huffman compressed binary)
        """
        if format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == 'json':
            self.to_json(file_path)
        elif format == 'bin':
            with open(file_path, 'wb') as f:
                self._write_to_bitstream(f)
        elif format == 'huff':
            with open(file_path, 'wb') as f:
                self._write_with_huffman(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _write_with_huffman(self, file: BinaryIO) -> None:
        """Write the quadtree to a binary file with Huffman compression."""
        print("  - In _write_with_huffman method")
        try:
            if not self.use_huffman:
                print("    - Initializing Huffman coder")
                self.huffman = HuffmanCoding()
                
            # First, write a temporary bitstream to memory
            import io
            print("    - Creating temporary buffer for bitstream")
            temp_stream = io.BytesIO()
            print("    - Writing to bitstream...")
            self._write_to_bitstream(temp_stream, include_huffman=False)
            print("    - Seeking to start of buffer")
            temp_stream.seek(0)
            
            # Read the data and compress it
            print("    - Reading data from buffer")
            data = temp_stream.read()
            print(f"    - Read {len(data)} bytes from buffer")
            
            if not data:
                print("    - Warning: No data read from buffer!")
                file.write(b'')
                return
                
            print("    - Compressing data with Huffman")
            compressed = self.huffman.compress(np.frombuffer(data, dtype=np.uint8))
            print(f"    - Compressed size: {len(compressed)} bytes")
            
            # Write the compressed data
            print("    - Writing compressed data to file")
            file.write(compressed)
            print("    - Successfully wrote compressed data")
            
        except Exception as e:
            import traceback
            print("    - Error in _write_with_huffman:")
            traceback.print_exc()
            raise
            
    def _write_to_bitstream(self, file: BinaryIO, include_huffman: bool = True) -> None:
        """Write the quadtree to a binary bitstream.
        
        Args:
            file: Binary file-like object to write to
            include_huffman: Whether to include Huffman coding info in the header
        """
        writer = BitStreamWriter(file)
        
        # Write header
        writer.write_uint(0x51545245)  # 'QTRE' magic number
        writer.write_uint(self.width, 2)  # 2 bytes for width
        writer.write_uint(self.height, 2)  # 2 bytes for height
        
        # Write flags (1 byte)
        flags = 0
        if self.use_quantization:
            flags |= 0x01
        if include_huffman and self.use_huffman:
            flags |= 0x02
        writer.write_bits(flags, 8)
        
        # Write quantization info (1 byte, only if used)
        if self.use_quantization:
            writer.write_bits(self.quantize_bits, 8)
        
        # Write tree structure and values
        self._write_node_bitstream(self.root, writer)
        
        # Flush any remaining bits
        writer.close()
    
    def _write_node_bitstream(self, node: 'QuadTreeNode', writer: BitStreamWriter) -> None:
        """Recursively write a node to the bitstream with enhanced data handling."""
        if node is None:
            # Write a marker for None nodes
            writer.write_bit(0)  # Not a leaf
            for _ in range(4):
                writer.write_bit(0)  # All children are None
            return
            
        # Write node type (0=internal, 1=leaf)
        is_leaf = node.is_leaf
        writer.write_bit(1 if is_leaf else 0)
        
        if is_leaf:
            # For leaf nodes, write the mean value
            if self.use_quantization:
                # Scale to [0, 2^bits - 1]
                normalized = (node.mean_value - self.quant_min) / self.quant_range
                quantized = int(np.clip(round(normalized * self.quant_scale), 0, self.quant_scale))
                writer.write_bits(quantized, self.quantize_bits)
                
                # For debugging: write the original mean value (32-bit float)
                # writer.write_float(node.mean_value)
            else:
                # Fallback to 8-bit if quantization is disabled
                quantized_mean = int(np.clip(node.mean_value, 0, 255))
                writer.write_bits(quantized_mean, 8)
        else:
            # For internal nodes, write which children are present
            child_mask = 0
            for i, child in enumerate(node.children):
                if child is not None:
                    child_mask |= (1 << (3 - i))  # Use MSB for first child
            
            writer.write_bits(child_mask, 4)  # 4 bits for 4 children
            
            # Recursively write non-None children in order
            for i in range(4):
                if node.children[i] is not None:
                    self._write_node_bitstream(node.children[i], writer)
    
    @classmethod
    def load(cls, file_path: str) -> 'QuadTree':
        """Load a quadtree from a file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            A new QuadTree instance
        """
        if file_path.endswith('.json'):
            return cls.from_json(file_path)
        elif file_path.endswith('.huff'):
            with open(file_path, 'rb') as f:
                return cls._read_from_huffman(f)
        elif file_path.endswith('.bin'):
            with open(file_path, 'rb') as f:
                return cls._read_from_bitstream(f)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
    @classmethod
    def _read_from_huffman(cls, file: BinaryIO) -> 'QuadTree':
        """Read a quadtree from a Huffman-compressed file."""
        # Read the entire compressed data
        compressed_data = file.read()
        
        # Create a temporary quadtree instance to get the Huffman coder
        temp_quadtree = cls(np.zeros((1, 1), dtype=np.uint8))
        
        # Decompress the data
        decompressed = temp_quadtree.huffman.decompress(compressed_data)
        
        # Read from the decompressed data
        import io
        return cls._read_from_bitstream(io.BytesIO(decompressed))
    
    @classmethod
    def _read_from_bitstream(cls, file: BinaryIO) -> 'QuadTree':
        """Read a quadtree from a binary bitstream."""
        reader = BitStreamReader(file)
        
        # Read header
        magic = reader.read_uint(4)
        if magic != 0x51545245:  # 'QTRE'
            raise ValueError("Invalid file format")
            
        width = reader.read_uint(2)
        height = reader.read_uint(2)
        
        # Read flags
        flags = reader.read_bits(8)
        use_quantization = (flags & 0x01) != 0
        use_huffman = (flags & 0x02) != 0
        
        # Read quantization info (if used)
        quantize_bits = 8  # Default
        if use_quantization:
            try:
                quantize_bits = reader.read_bits(8)
                if not (1 <= quantize_bits <= 16):
                    raise ValueError("Invalid quantization bits")
            except EOFError:
                raise ValueError("Unexpected end of file while reading quantization bits")
        
        # Create a dummy image (will be replaced by reconstruction)
        dummy_image = np.zeros((height, width), dtype=np.float32)
        quadtree = cls(dummy_image, 
                      quantize_bits=quantize_bits,
                      use_quantization=use_quantization,
                      use_huffman=use_huffman)
        
        # Read tree structure
        quadtree.root = cls._read_node_bitstream(reader, 0, 0, max(width, height))
        
        # Reconstruct the image
        quadtree.image = quadtree.reconstruct()
        
        return quadtree

    def _write_to_bitstream(self, file: BinaryIO, include_huffman: bool = True) -> None:
        """Write the quadtree to a binary bitstream.
        
        Args:
            file: Binary file-like object to write to
            include_huffman: Whether to include Huffman coding info in the header
        """
        writer = BitStreamWriter(file)
        
        # Write header
        writer.write_uint(0x51545245)  # 'QTRE' magic number
        writer.write_uint(self.width, 2)  # 2 bytes for width
        writer.write_uint(self.height, 2)  # 2 bytes for height
        
        # Write flags (1 byte)
        flags = 0
        if self.use_quantization:
            flags |= 0x01
        if include_huffman and self.use_huffman:
            flags |= 0x02
        writer.write_bits(flags, 8)
        
        # Write quantization info (1 byte, only if used)
        if self.use_quantization:
            writer.write_bits(self.quantize_bits, 8)
        
        # Write tree structure and values
        self._write_node_bitstream(self.root, writer)
        
        # Flush any remaining bits
        writer.close()
    
    def _write_node_bitstream(self, node: 'QuadTreeNode', writer: BitStreamWriter) -> None:
        """Recursively write a node to the bitstream with enhanced data handling."""
        if node is None:
            # Write a marker for None nodes
            writer.write_bit(0)  # Not a leaf
            for _ in range(4):
                writer.write_bit(0)  # All children are None
            return
            
        # Write node type (0=internal, 1=leaf)
        is_leaf = node.is_leaf
        writer.write_bit(1 if is_leaf else 0)
        
        if is_leaf:
            # For leaf nodes, write the mean value
            if self.use_quantization:
                # Scale to [0, 2^bits - 1]
                normalized = (node.mean_value - self.quant_min) / self.quant_range
                quantized = int(np.clip(round(normalized * self.quant_scale), 0, self.quant_scale))
                writer.write_bits(quantized, self.quantize_bits)
                
                # For debugging: write the original mean value (32-bit float)
                # writer.write_float(node.mean_value)
            else:
                # Fallback to 8-bit if quantization is disabled
                quantized_mean = int(np.clip(node.mean_value, 0, 255))
                writer.write_bits(quantized_mean, 8)
        else:
            # For internal nodes, write which children are present
            child_mask = 0
            for i, child in enumerate(node.children):
                if child is not None:
                    child_mask |= (1 << (3 - i))  # Use MSB for first child
            
            writer.write_bits(child_mask, 4)  # 4 bits for 4 children
            
            # Recursively write non-None children in order
            for i in range(4):
                if node.children[i] is not None:
                    self._write_node_bitstream(node.children[i], writer)
    
@classmethod
def load(cls, file_path: str) -> 'QuadTree':
    """Load a quadtree from a file.
    
    Args:
        file_path: Path to the file to load
            
    Returns:
        A new QuadTree instance
    """
    if file_path.endswith('.json'):
        return cls.from_json(file_path)
    elif file_path.endswith('.huff'):
        with open(file_path, 'rb') as f:
            return cls._read_from_huffman(f)
    elif file_path.endswith('.bin'):
        with open(file_path, 'rb') as f:
            return cls._read_from_bitstream(f)
    else:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
                
@classmethod
def _read_from_huffman(cls, file: BinaryIO) -> 'QuadTree':
    """Read a quadtree from a Huffman-compressed file."""
    # Read the entire compressed data
    compressed_data = file.read()
    
    # Create a temporary quadtree instance to get the Huffman coder
    temp_quadtree = cls(np.zeros((1, 1), dtype=np.uint8))
        
    # Decompress the data
    decompressed = temp_quadtree.huffman.decompress(compressed_data)
        
    # Read from the decompressed data
    import io
    return cls._read_from_bitstream(io.BytesIO(decompressed))

def compress(self) -> bytes:
    """
    Compress the quadtree into a binary format.
    
    Returns:
        bytes: Compressed binary data
        """
    print("Starting compression...")
    if self.root is None:
        print("  - Root is None, returning empty bytes")
        return b''
            
    try:
        # Use in-memory bytes buffer
        import io
        print("  - Creating in-memory buffer...")
        with io.BytesIO() as buffer:
            if self.use_huffman:
                print("  - Using Huffman compression...")
                self._write_with_huffman(buffer)
            else:
                print("  - Using bitstream without Huffman...")
                self._write_to_bitstream(buffer, include_huffman=False)
            compressed_data = buffer.getvalue()
            print(f"  - Initial compressed data size: {len(compressed_data)} bytes")
                
        # If using Huffman, we've already written the compressed data
        if self.use_huffman:
            print(f"  - Returning Huffman compressed data: {len(compressed_data)} bytes")
            return compressed_data
                
        # If not using Huffman, we need to compress the bitstream
        if self.huffman is None:
            print("  - Initializing Huffman coder...")
            self.huffman = HuffmanCoding()
                
        # Compress the raw bitstream
        if len(compressed_data) > 0:
            print("  - Compressing with Huffman...")
            compressed = self.huffman.compress(np.frombuffer(compressed_data, dtype=np.uint8))
            print(f"  - Final compressed size: {len(compressed)} bytes")
            return compressed
        print("  - No data to compress, returning empty bytes")
        return compressed_data
            
    except Exception as e:
        import traceback
        print("Error during compression:")
        traceback.print_exc()
        raise

def decompress(self, data: bytes) -> np.ndarray:
    """
    Decompress the quadtree and reconstruct the image.
    
    Args:
        data: Compressed binary data
            
    Returns:
        np.ndarray: Reconstructed image
    """
    # Create a new quadtree from the compressed data
    import io
    with io.BytesIO(data) as buffer:
        # Try to determine the format
        magic = buffer.read(4)
        buffer.seek(0)
            
        if magic == b'QTHF':  # Huffman compressed
            quadtree = self.__class__._read_from_huffman(buffer)
        else:  # Assume raw bitstream
            quadtree = self.__class__._read_from_bitstream(buffer)
                
        # Return the reconstructed image
        return quadtree.reconstruct()
    
def visualize(self, show_grid: bool = True, show_mean: bool = True, 
                     figsize: Tuple[int, int] = (10, 10)):
        """Visualize the quadtree decomposition.
        
        Args:
            show_grid: Whether to show the quadtree grid lines
            show_mean: Whether to show mean intensity values in leaf nodes
            figsize: Figure size (width, height) in inches
        """
        # Create a copy of the image to draw on
        vis_image = self.image.astype(np.uint8)
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
            
        # Draw the quadtree
        self._draw_quadtree(self.root, vis_image, show_grid, show_mean)
        
        # Show the result
        plt.figure(figsize=figsize)
        plt.imshow(vis_image)
        plt.title(f'Quadtree Decomposition (CR: {self.get_compression_ratio():.2f})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
def _draw_quadtree(self, node: QuadTreeNode, image: np.ndarray, 
                      show_grid: bool, show_mean: bool) -> None:
    """Recursively draw quadtree on the image.
    
    Args:
        node: Current node to draw
        image: Image to draw on
        show_grid: Whether to show grid lines
        show_mean: Whether to show mean values
    """
    if node is None:
        return
            
    if node.is_leaf:
        # Draw leaf node
        if show_grid:
            # Draw rectangle around the node
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, 
                            (node.x, node.y), 
                            (node.x + node.size - 1, node.y + node.size - 1),
                            color, 1)
        
            # Draw mean value if requested
            if show_mean and node.size > 4:  # Only draw text if node is large enough
                mean_str = f"{node.mean_value:.1f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = min(0.5, 0.5 * node.size / 100)  # Scale font with node size
                text_size = cv2.getTextSize(mean_str, font, font_scale, 1)[0]
                text_x = node.x + (node.size - text_size[0]) // 2
                text_y = node.y + (node.size + text_size[1]) // 2
            
                # Add semi-transparent background for better text visibility
                overlay = image.copy()
                cv2.rectangle(overlay, 
                            (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + 2),
                            (0, 0, 0), -1)
                alpha = 0.6  # Transparency factor
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                    
                # Draw text
                cv2.putText(image, mean_str, (text_x, text_y),
                            font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        # Draw internal node (recursively process children)
        for child in node.children:
            if child is not None:
                self._draw_quadtree(child, image, show_grid, show_mean)
            
        # Draw split lines for internal nodes
        if show_grid and node.size > 1:
            half = node.size // 2
            color = (255, 0, 0)  # Red
            # Vertical line
            cv2.line(image, 
                        (node.x + half, node.y),
                        (node.x + half, node.y + node.size - 1),
                        color, 1)
            # Horizontal line
            cv2.line(image,
                        (node.x, node.y + half),
                        (node.x + node.size - 1, node.y + half),
                        color, 1)


def compress_image(image: np.ndarray, 
                  min_size: int = 2,
                  variance_threshold: float = 10.0,
                  max_depth: int = 10) -> QuadTree:
    """Compress an image using quadtree decomposition with optimized parameters.
    
    Args:
        image: Input grayscale image (2D numpy array)
        min_size: Minimum block size (must be power of 2)
        variance_threshold: Maximum allowed variance in a leaf node
        max_depth: Maximum depth of the quadtree
        
    Returns:
        QuadTree: The constructed quadtree with optimized compression
    """
    # Calculate image statistics for adaptive parameters
    img_mean = np.mean(image)
    img_std = np.std(image)
    
    # Adaptive variance threshold based on image content
    if variance_threshold is None:
        # Use image standard deviation to determine threshold
        variance_threshold = max(5.0, img_std * 0.5)
    
    # Adjust min_size based on image dimensions
    min_dim = min(image.shape)
    if min_size < 2:
        min_size = 2
    elif min_size > min_dim // 8:  # Don't let min_size be too large
        min_size = max(2, min_dim // 8)
    
    # Create quadtree with optimized parameters
    return QuadTree(
        image=image,
        min_size=min_size,
        variance_threshold=variance_threshold,
        max_depth=max_depth,
        quantize_bits=8,  # Standard 8-bit quantization
        use_quantization=True,
        use_huffman=True  # Enable Huffman coding for better compression
    )


def visualize_quadtree(image: np.ndarray, 
                     min_size: int = 2,
                     variance_threshold: float = 10.0,
                     max_depth: int = 10,
                     show_grid: bool = True,
                     show_mean: bool = True,
                     figsize: Tuple[int, int] = (10, 10)) -> None:
    """Helper function to create and visualize a quadtree from an image.
    
    Args:
        image: Input grayscale image (2D numpy array)
        min_size: Minimum block size (must be power of 2)
        variance_threshold: Maximum allowed variance in a leaf node
        max_depth: Maximum depth of the quadtree
        show_grid: Whether to show the quadtree grid lines
        show_mean: Whether to show mean intensity values in leaf nodes
        figsize: Figure size (width, height) in inches
    """
    # Create and visualize quadtree
    quadtree = QuadTree(image, min_size, variance_threshold, max_depth)
    quadtree.visualize(show_grid, show_mean, figsize)
    
    # Print compression info
    print(f"Original size: {image.shape[0]}x{image.shape[1]} = {image.size} pixels")
    print(f"Leaf nodes: {len(quadtree.get_leaf_nodes())}")
    print(f"Compression ratio: {quadtree.get_compression_ratio():.2f}")
    print(f"Min size: {min_size}, Max depth: {max_depth}, Variance threshold: {variance_threshold}")
