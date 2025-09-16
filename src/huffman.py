"""
Huffman coding implementation for entropy encoding/decoding of quadtree data.
"""
from collections import defaultdict, deque
import heapq
from typing import Dict, List, Tuple, Optional, BinaryIO, Union
import numpy as np

class HuffmanNode:
    """Node in the Huffman tree."""
    
    def __init__(self, 
                 value: Optional[int] = None, 
                 freq: int = 0,
                 left: Optional['HuffmanNode'] = None,
                 right: Optional['HuffmanNode'] = None):
        self.value = value  # None for internal nodes
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other: 'HuffmanNode') -> bool:
        # For priority queue ordering
        return self.freq < other.freq
    
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return self.left is None and self.right is None


class HuffmanCoding:
    """Huffman coding implementation for compressing quadtree data."""
    
    def __init__(self):
        self.codes = {}
        self.reverse_mapping = {}
    
    def build_frequency_dict(self, data: np.ndarray) -> Dict[int, int]:
        """Build frequency dictionary from input data."""
        freq = defaultdict(int)
        for value in np.nditer(data):
            freq[int(value)] += 1
        return dict(freq)
    
    def build_huffman_tree(self, freq_dict: Dict[int, int]) -> Optional[HuffmanNode]:
        """Build Huffman tree from frequency dictionary."""
        if not freq_dict:
            return None
            
        # Create a priority queue of nodes
        heap = []
        for value, freq in freq_dict.items():
            heapq.heappush(heap, HuffmanNode(value=value, freq=freq))
        
        # Special case: single value in frequency dict
        if len(heap) == 1:
            node = heapq.heappop(heap)
            return HuffmanNode(freq=node.freq, left=node)
        
        # Build the Huffman tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Create a new node with these two as children
            merged = HuffmanNode(
                freq=left.freq + right.freq,
                left=left,
                right=right
            )
            heapq.heappush(heap, merged)
        
        return heap[0] if heap else None
    
    def _build_codes_helper(self, node: Optional[HuffmanNode], current_code: str) -> None:
        """Helper function to build the code dictionary."""
        if node is None:
            return
            
        if node.is_leaf():
            if current_code:  # Only add non-empty codes
                self.codes[node.value] = current_code
                self.reverse_mapping[current_code] = node.value
            return
            
        self._build_codes_helper(node.left, current_code + '0')
        self._build_codes_helper(node.right, current_code + '1')
    
    def build_codes(self, root: Optional[HuffmanNode]) -> None:
        """Build code dictionary from Huffman tree."""
        self.codes = {}
        self.reverse_mapping = {}
        self._build_codes_helper(root, '')
    
    def encode_data(self, data: np.ndarray) -> str:
        """Encode data using the current code dictionary."""
        encoded_bits = []
        for value in np.nditer(data):
            encoded_bits.append(self.codes[int(value)])
        return ''.join(encoded_bits)
    
    def pad_encoded_data(self, encoded_bits: str) -> str:
        """Pad the encoded bit string to make its length a multiple of 8."""
        extra_padding = 8 - len(encoded_bits) % 8
        if extra_padding > 0:
            encoded_bits += '0' * extra_padding
        
        # Add padding info at the beginning
        padding_info = "{0:08b}".format(extra_padding)
        return padding_info + encoded_bits
    
    def get_byte_array(self, padded_encoded_data: str) -> bytearray:
        """Convert bit string to byte array."""
        if len(padded_encoded_data) % 8 != 0:
            raise ValueError("Encoded data must be padded to multiple of 8 bits")
            
        b = bytearray()
        for i in range(0, len(padded_encoded_data), 8):
            byte = padded_encoded_data[i:i+8]
            b.append(int(byte, 2))
        return b
    
    def compress(self, data: np.ndarray) -> bytes:
        """Compress data using Huffman coding.
        
        Returns:
            bytes: Compressed data including the Huffman tree and encoded data
        """
        if data.size == 0:
            return b''
            
        # Build Huffman tree and codes
        freq_dict = self.build_frequency_dict(data)
        root = self.build_huffman_tree(freq_dict)
        self.build_codes(root)
        
        # Encode the data
        encoded_data = self.encode_data(data)
        padded_data = self.pad_encoded_data(encoded_data)
        
        # Serialize the Huffman tree
        tree_bytes = self._serialize_tree(root)
        
        # Combine tree and data
        return len(tree_bytes).to_bytes(4, 'big') + tree_bytes + bytes(self.get_byte_array(padded_data))
    
    def _serialize_tree(self, node: Optional[HuffmanNode]) -> bytes:
        """Serialize Huffman tree to bytes."""
        if node is None:
            return b''
            
        # Pre-order traversal
        if node.is_leaf():
            # For leaf nodes: 1 bit flag + 8 bits value
            return b'\x01' + node.value.to_bytes(1, 'big')
        else:
            # For internal nodes: 0 bit flag + left subtree + right subtree
            left = self._serialize_tree(node.left)
            right = self._serialize_tree(node.right)
            return b'\x00' + left + right
    
    @classmethod
    def _deserialize_tree(cls, data: bytes, index: int = 0) -> Tuple[Optional['HuffmanNode'], int]:
        """Deserialize Huffman tree from bytes."""
        if index >= len(data):
            return None, index
            
        is_leaf = data[index] == 1
        index += 1
        
        if is_leaf:
            if index >= len(data):
                raise ValueError("Invalid Huffman tree data")
            value = data[index]
            index += 1
            return HuffmanNode(value=value), index
        else:
            left, index = cls._deserialize_tree(data, index)
            right, index = cls._deserialize_tree(data, index)
            return HuffmanNode(left=left, right=right), index
    
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """Decompress data using Huffman coding."""
        if not compressed_data:
            return np.array([], dtype=np.uint8)
            
        # Read tree size
        if len(compressed_data) < 4:
            raise ValueError("Invalid compressed data")
            
        tree_size = int.from_bytes(compressed_data[:4], 'big')
        
        # Deserialize Huffman tree
        tree_data = compressed_data[4:4+tree_size]
        root, _ = self._deserialize_tree(tree_data)
        
        # Rebuild codes from the tree
        self.build_codes(root)
        
        # Read and decode the data
        encoded_data = compressed_data[4+tree_size:]
        bit_string = ''.join(f'{byte:08b}' for byte in encoded_data)
        
        # Remove padding
        padding_info = bit_string[:8]
        padding = int(padding_info, 2)
        bit_string = bit_string[8:-padding] if padding > 0 else bit_string[8:]
        
        # Decode the data
        current_code = ""
        decoded_values = []
        
        for bit in bit_string:
            current_code += bit
            if current_code in self.reverse_mapping:
                decoded_values.append(self.reverse_mapping[current_code])
                current_code = ""
        
        return np.array(decoded_values, dtype=np.uint8)
