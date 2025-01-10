# Compressor Framework

from importlib import import_module

import numpy as np

class LZWCompressor:
    """LZW Compressor"""
    def compress(self, uncompressed: bytes) -> bytes:
        """Compress a string to a list of output symbols."""
        dict_size = 256
        dictionary = {bytes([i]): i for i in range(dict_size)}
        w = bytes()
        result = []
        for c in uncompressed:
            wc = w + bytes([c])
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = bytes([c])
        if w:
            result.append(dictionary[w])
        compressed_data = bytearray()
        for number in result:
            compressed_data.extend(number.to_bytes((number.bit_length() + 7) // 8, byteorder='big') or b'\0')
        return bytes(compressed_data)

class DefaultCompressor:
    """For non-neural-based compressor"""

    def __init__(self, compressor, typ="text"):
        if compressor == "lzw":
            self.compressor = LZWCompressor()
        else:
            try:
                self.compressor = import_module(compressor)
            except ModuleNotFoundError:
                raise RuntimeError("Unsupported compressor")
        self.type = typ

    def get_compressed_len(self, x: str) -> int:
        """
        Calculates the size of `x` once compressed.

        Arguments:
            x (str): String to be compressed.

        Returns:
            int: Length of x after compression.
        """
        if self.type == "text":
            return len(self.compressor.compress(x.encode("utf-8")))
        else:
            return len(self.compressor.compress(np.array(x).tobytes()))

    def get_bits_per_character(self, original_fn: str) -> float:
        """
        Returns the compressed size of the original function
        in bits.

        Arguments:
            original_fn (str): Function name to be compressed.

        Returns:
            int: Compressed size of original_fn content in bits.
        """
        with open(original_fn) as fo:
            data = fo.read()
            compressed_str = self.compressor.compress(data.encode("utf-8"))
            return len(compressed_str) * 8 / len(data)


"""Test Compressors"""
if __name__ == "__main__":
    comp = DefaultCompressor("gzip")
    print(comp.get_compressed_len("Hello world"))
