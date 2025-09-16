"""
Bitstream utilities for efficient binary data encoding/decoding.
"""
import struct
from typing import BinaryIO, List, Tuple, Union, Optional

class BitStreamWriter:
    """Helper class for writing bits to a binary stream."""
    
    def __init__(self, file: Optional[BinaryIO] = None):
        self.file = file
        self.buffer = 0
        self.bits_in_buffer = 0
        self.total_bits = 0
    
    def write_bit(self, bit: int) -> None:
        """Write a single bit (0 or 1) to the stream."""
        if bit not in (0, 1):
            raise ValueError("Bit must be 0 or 1")
            
        self.buffer = (self.buffer << 1) | bit
        self.bits_in_buffer += 1
        self.total_bits += 1
        
        if self.bits_in_buffer == 8:
            self._flush_byte()
    
    def write_bits(self, value: int, num_bits: int) -> None:
        """Write multiple bits from an integer value."""
        if num_bits < 0 or num_bits > 64:
            raise ValueError("Number of bits must be between 0 and 64")
            
        # Write bits from MSB to LSB
        for i in range(num_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)
    
    def write_uint(self, value: int, num_bytes: int = 4) -> None:
        """Write an unsigned integer with a fixed number of bytes."""
        if value < 0:
            raise ValueError("Value must be non-negative")
            
        for _ in range(num_bytes):
            self.write_bits(value & 0xFF, 8)
            value >>= 8
    
    def write_float(self, value: float) -> None:
        """Write a 32-bit floating point number."""
        # Convert float to bytes and write as bits
        packed = struct.pack('!f', value)
        for byte in packed:
            self.write_bits(byte, 8)
    
    def _flush_byte(self) -> None:
        """Flush the buffer to the output file."""
        if self.bits_in_buffer > 0:
            byte = self.buffer << (8 - self.bits_in_buffer)
            if self.file:
                self.file.write(bytes([byte]))
            self.buffer = 0
            self.bits_in_buffer = 0
    
    def close(self) -> int:
        """Close the writer and return the total number of bits written."""
        self._flush_byte()
        return self.total_bits


class BitStreamReader:
    """Helper class for reading bits from a binary stream."""
    
    def __init__(self, file: BinaryIO):
        self.file = file
        self.buffer = 0
        self.bits_in_buffer = 0
        self.total_bits = 0
        self.eof = False
    
    def read_bit(self) -> int:
        """Read a single bit (0 or 1) from the stream."""
        if self.bits_in_buffer == 0:
            self._fill_buffer()
            if self.eof:
                raise EOFError("End of bit stream reached")
        
        bit = (self.buffer >> (self.bits_in_buffer - 1)) & 1
        self.bits_in_buffer -= 1
        self.total_bits += 1
        return bit
    
    def read_bits(self, num_bits: int) -> int:
        """Read multiple bits and return as an integer."""
        if num_bits < 0 or num_bits > 64:
            raise ValueError("Number of bits must be between 0 and 64")
            
        result = 0
        for _ in range(num_bits):
            result = (result << 1) | self.read_bit()
        return result
    
    def read_uint(self, num_bytes: int = 4) -> int:
        """Read an unsigned integer with a fixed number of bytes."""
        value = 0
        for _ in range(num_bytes):
            value = (value << 8) | self.read_bits(8)
        return value
    
    def read_float(self) -> float:
        """Read a 32-bit floating point number."""
        # Read 4 bytes and unpack as float
        packed = bytes([self.read_bits(8) for _ in range(4)])
        return struct.unpack('!f', packed)[0]
    
    def _fill_buffer(self) -> None:
        """Fill the buffer from the input file."""
        byte = self.file.read(1)
        if not byte:
            self.eof = True
            return
            
        self.buffer = byte[0]
        self.bits_in_buffer = 8
    
    def close(self) -> int:
        """Close the reader and return the total number of bits read."""
        return self.total_bits


def write_bitstream_to_file(bitstream: List[int], filename: str) -> int:
    """Write a list of bits to a binary file."""
    with open(filename, 'wb') as f:
        writer = BitStreamWriter(f)
        for bit in bitstream:
            writer.write_bit(bit)
        return writer.close()


def read_bitstream_from_file(filename: str) -> List[int]:
    """Read a binary file and return as a list of bits."""
    bits = []
    with open(filename, 'rb') as f:
        reader = BitStreamReader(f)
        try:
            while True:
                bits.append(reader.read_bit())
        except EOFError:
            pass
    return bits
