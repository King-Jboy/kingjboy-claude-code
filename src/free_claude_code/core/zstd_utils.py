"""High-performance Zstandard compression utilities."""

import zstandard as zstd

_COMPRESSOR = zstd.ZstdCompressor(level=3)
_DECOMPRESSOR = zstd.ZstdDecompressor()


def compress_zstd(data: bytes) -> bytes:
    """Compress raw bytes using Zstandard."""
    return _COMPRESSOR.compress(data)


def decompress_zstd(data: bytes) -> bytes:
    """Decompress Zstandard bytes back to raw data."""
    return _DECOMPRESSOR.decompress(data)
