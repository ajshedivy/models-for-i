#!/usr/bin/env python3
"""
validate_gguf.py

Validate the header of a GGUF file, auto-detecting little vs. big endian.

Exit code 0 on success, 1 on failure.
"""

import sys
import struct

SPEC_VERSION = 3

class GGUFValidationError(Exception):
    pass

def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"Expected {n} bytes, got {len(data)}")
    return data

def detect_endianness(version_bytes):
    """Return ('<', version) if little-endian version matches,
       ('>', version) if big-endian version matches, else error."""
    ver_le = struct.unpack('<I', version_bytes)[0]
    if ver_le == SPEC_VERSION:
        return '<', ver_le
    ver_be = struct.unpack('>I', version_bytes)[0]
    if ver_be == SPEC_VERSION:
        return '>', ver_be
    raise GGUFValidationError(
        f"Unrecognized version (LE={ver_le}, BE={ver_be}); expected {SPEC_VERSION}"
    )

def validate_header(path):
    with open(path, 'rb') as f:
        # 1) Magic: must be ASCII 'GGUF'
        magic = read_exact(f, 4)
        if magic != b'GGUF':
            raise GGUFValidationError(f"Bad magic: {magic!r}; expected b'GGUF'")

        # 2) Version: sniff endianness
        version_bytes = read_exact(f, 4)
        endian, version = detect_endianness(version_bytes)

        # 3) Tensor count
        tensor_bytes = read_exact(f, 8)
        n_tensors = struct.unpack(endian + 'Q', tensor_bytes)[0]

        # 4) Metadata KV count
        kv_bytes = read_exact(f, 8)
        n_kv = struct.unpack(endian + 'Q', kv_bytes)[0]

    # Basic sanity checks
    if n_tensors > 1_000_000_000:
        raise GGUFValidationError(f"Unrealistic tensor count: {n_tensors}")
    if n_kv > 100_000:
        raise GGUFValidationError(f"Unrealistic metadata count: {n_kv}")

    # All good!
    print("GGUF header is valid.")
    print(f"  Endianness    : {'little' if endian=='<' else 'big'}")
    print(f"  Version       : {version}")
    print(f"  # Tensors     : {n_tensors}")
    print(f"  # Metadata KV : {n_kv}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/model.gguf", file=sys.stderr)
        sys.exit(1)
    try:
        validate_header(sys.argv[1])
    except (EOFError, struct.error, GGUFValidationError) as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
