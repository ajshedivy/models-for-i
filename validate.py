#!/usr/bin/env python3
"""
validate_gguf.py

Simple validator for a GGUF file header.
Exits with code 0 on success, 1 on failure.
"""

import sys
import struct

class GGUFValidationError(Exception):
    pass

def read_exact(f, n):
    """Read exactly n bytes or raise EOFError."""
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"Expected {n} bytes, got {len(data)}")
    return data

def validate_header(path):
    with open(path, 'rb') as f:
        # 1. Magic number (4 bytes): must be ASCII 'GGUF'
        magic = read_exact(f, 4)
        if magic != b'GGUF':
            raise GGUFValidationError(f"Bad magic: {magic!r}, expected b'GGUF'")
        # 2. Version (4 bytes, little-endian uint32)
        version, = struct.unpack('<I', read_exact(f, 4))
        if version < 1:
            raise GGUFValidationError(f"Unsupported GGUF version: {version}")
        # 3. Tensor count (8 bytes, little-endian uint64)
        n_tensors, = struct.unpack('<Q', read_exact(f, 8))
        # 4. Metadata KV count (8 bytes, little-endian uint64)
        n_kv, = struct.unpack('<Q', read_exact(f, 8))

    # If we reach here without error, basic header is valid
    print("GGUF header is valid.")
    print(f"  Version       : {version}")
    print(f"  # Tensors     : {n_tensors}")
    print(f"  # Metadata KV : {n_kv}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/model.gguf", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    try:
        validate_header(path)
    except (EOFError, struct.error, GGUFValidationError) as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
