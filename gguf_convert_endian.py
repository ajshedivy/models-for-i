#!/usr/bin/env python3
"""Convert a GGUF file between little- and big-endian representations.

The conversion is in-place.  A JSON manifest is written beside a successfully
converted model so the exact source and resulting artifacts can
be audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

import gguf

logger = logging.getLogger("gguf-convert-endian")
CONVERTER_VERSION = "2.0.1"

# (block size in bytes, byte offsets of 16-bit fields).  All other bytes are
# packed quantization data and must be left exactly as they are.
QUANT_LAYOUTS = {
    gguf.GGMLQuantizationType.Q4_0: (18, (0,)),
    gguf.GGMLQuantizationType.Q4_1: (20, (0, 2)),
    gguf.GGMLQuantizationType.Q8_0: (34, (0,)),
    # block_q4_K: d (f16), dmin (f16), scales[12], qs[128].
    gguf.GGMLQuantizationType.Q4_K: (144, (0, 2)),
    # block_q5_K: d (f16), dmin (f16), scales[12], qh[32], qs[128].
    gguf.GGMLQuantizationType.Q5_K: (176, (0, 2)),
    # block_q6_K: ql[128], qh[64], scales[16], d (f16).
    gguf.GGMLQuantizationType.Q6_K: (210, (208,)),
}
# Block types whose leading d/dmin FP16 scales are sampled after conversion.
# Real d/dmin scales are tiny; a mis-ordered FP16 is often NaN/Inf or huge.
SCALE_SAMPLE_BLOCKS = 256
SCALE_LIMIT = 10.0
MAX_BAD_SCALE_FRACTION = 0.05
K_QUANT_SCALE_TYPES = {gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K}
NO_SWAP_QUANT_TYPES = {gguf.GGMLQuantizationType.MXFP4}
SCALAR_TYPES = {
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.BF16,
}


def _bytes(tensor) -> np.ndarray:
    """Return a one-dimensional, writable byte view without changing shape."""
    return np.asarray(tensor.data).view(np.uint8).reshape(-1)


def _swap_u16_fields(blocks: np.ndarray, offsets: tuple[int, ...]) -> None:
    """Swap selected 16-bit lanes in every block without per-block Python work."""
    for offset in offsets:
        lane = blocks[:, offset:offset + 2].copy()
        lane.view(np.uint16).byteswap(inplace=True)
        blocks[:, offset:offset + 2] = lane


def byteswap_quantized_data(data: np.ndarray, tensor_type: gguf.GGMLQuantizationType) -> None:
    """Byte-swap the floating-point lanes of a quantized tensor in place."""
    if tensor_type in NO_SWAP_QUANT_TYPES:
        return
    try:
        block_size, offsets = QUANT_LAYOUTS[tensor_type]
    except KeyError as exc:
        raise ValueError(f"Cannot byte-swap quantization type {tensor_type.name}") from exc
    raw = np.asarray(data).view(np.uint8).reshape(-1)
    if raw.size % block_size:
        raise ValueError(
            f"{tensor_type.name} data is {raw.size} bytes, not divisible by its "
            f"{block_size}-byte block size"
        )
    _swap_u16_fields(raw.reshape(-1, block_size), offsets)


def bad_scale_fraction(reader: gguf.GGUFReader, byteorder: str) -> float | None:
    """Fraction of sampled Q4_K/Q5_K ``d``/``dmin`` values that are implausible.

    ``byteorder`` is the order the file will be read in ('BIG' or 'LITTLE').
    Returns None when the file has no Q4_K/Q5_K tensors.
    """
    dtype = np.dtype(">f2" if byteorder == "BIG" else "<f2")
    bad = total = 0
    for tensor in reader.tensors:
        if tensor.tensor_type not in K_QUANT_SCALE_TYPES:
            continue
        block_size = QUANT_LAYOUTS[tensor.tensor_type][0]
        blocks = _bytes(tensor)[:block_size * SCALE_SAMPLE_BLOCKS].reshape(-1, block_size)
        scales = blocks[:, 0:4].copy().view(dtype).astype(np.float32)
        with np.errstate(invalid="ignore"):
            bad += int(np.count_nonzero(~np.isfinite(scales) | (np.abs(scales) > SCALE_LIMIT)))
        total += scales.size
    return bad / total if total else None


def _check_scales(reader: gguf.GGUFReader, byteorder: str) -> None:
    fraction = bad_scale_fraction(reader, byteorder)
    if fraction is None:
        return
    logger.info("* Implausible Q4_K/Q5_K d/dmin values read as %s endian: %.1f%%", byteorder, 100 * fraction)
    if fraction > MAX_BAD_SCALE_FRACTION:
        raise ValueError(
            f"{100 * fraction:.1f}% of sampled Q4_K/Q5_K d/dmin values are implausible as "
            f"{byteorder} endian; the output is corrupt. Re-convert from the original file."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _type_counts(reader: gguf.GGUFReader) -> dict[str, int]:
    return dict(sorted(Counter(t.tensor_type.name for t in reader.tensors).items()))


def _check_tensors(reader: gguf.GGUFReader) -> list[str]:
    """Reject unsupported tensors and malformed quantized byte streams early.

    Returns the names of Q4_K/Q5_K tensors.
    """
    k_quant_names = []
    for tensor in reader.tensors:
        tensor_type = tensor.tensor_type
        if tensor_type not in set(QUANT_LAYOUTS) | NO_SWAP_QUANT_TYPES | SCALAR_TYPES:
            raise ValueError(f"Cannot handle type {tensor_type.name} for tensor {tensor.name!r}")
        if tensor_type in QUANT_LAYOUTS:
            block_size = QUANT_LAYOUTS[tensor_type][0]
            n_bytes = _bytes(tensor).size
            if n_bytes % block_size:
                raise ValueError(
                    f"Tensor {tensor.name!r} ({tensor_type.name}) has {n_bytes} bytes; "
                    f"expected a multiple of {block_size}"
                )
        if tensor_type in K_QUANT_SCALE_TYPES:
            k_quant_names.append(str(tensor.name))
    return k_quant_names


def _write_manifest(path: Path, manifest_path: Path, *, source_checksum: str,
                    target_endianness: str, type_counts: dict[str, int]) -> None:
    manifest = {
        "converter": "gguf_convert_endian.py",
        "converter_version": CONVERTER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(path),
        "source_sha256": source_checksum,
        "output_path": str(path),
        "output_sha256": _sha256(path),
        "target_endianness": target_endianness.lower(),
        "quantization_mix": type_counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("* Wrote conversion manifest: %s", manifest_path)


def _confirm(args: argparse.Namespace) -> None:
    if args.yes:
        return
    logger.warning("*** Warning: this modifies the GGUF file in place. Keep a backup. ***")
    response = input("Enter exactly YES to proceed> ")
    if response != "YES":
        logger.warning("No changes made.")
        raise SystemExit(0)


def _convert_tensor_data(tensor) -> None:
    if tensor.tensor_type in set(QUANT_LAYOUTS) | NO_SWAP_QUANT_TYPES:
        byteswap_quantized_data(tensor.data, tensor.tensor_type)
    elif tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        _bytes(tensor).view(np.uint16).byteswap(inplace=True)
    else:
        tensor.data.byteswap(inplace=True)


def convert_byteorder(reader: gguf.GGUFReader, args: argparse.Namespace) -> None:
    path = Path(args.model)
    file_endian = reader.endianess.name
    host_endian = ('BIG' if file_endian == 'LITTLE' else 'LITTLE') if reader.byte_order == 'S' else file_endian
    order = host_endian if args.order == "native" else args.order.upper()
    logger.info("* Host is %s endian, GGUF file is %s endian", host_endian, file_endian)
    k_quant_names = _check_tensors(reader)
    counts = _type_counts(reader)
    logger.info("* Tensor type counts: %s", counts)
    logger.info("* Q4_K/Q5_K tensors in this conversion: %d", len(k_quant_names))
    for name in k_quant_names:
        logger.debug("  - %s", name)
    if file_endian == order:
        logger.info("* File is already %s endian. Nothing to do.", order)
        return
    if args.dry_run:
        logger.info("* Dry run complete; no changes made.")
        return

    _confirm(args)
    source_checksum = _sha256(path)
    logger.info("* Converting %d metadata fields", len(reader.fields))
    for field in reader.fields.values():
        for part in field.parts:
            part.byteswap(inplace=True)
    logger.info("* Converting %d tensors", len(reader.tensors))
    for tensor in tqdm(reader.tensors, desc="Converting tensors"):
        for part in tensor.field.parts:
            part.byteswap(inplace=True)
        _convert_tensor_data(tensor)
    _check_scales(reader, order)
    _write_manifest(path, args.manifest or path.with_suffix(path.suffix + ".endian-manifest.json"),
                    source_checksum=source_checksum, target_endianness=order,
                    type_counts=counts)
    logger.info("* Completion")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GGUF file byte order in place")
    parser.add_argument("model", help="GGUF model filename")
    parser.add_argument("order", choices=["big", "little", "native"], help="requested byte order")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without modifying")
    parser.add_argument("--yes", action="store_true", help="skip the destructive-operation confirmation")
    parser.add_argument("--manifest", type=Path, help="path for the JSON conversion manifest")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    logger.info("* Loading: %s", args.model)
    reader = gguf.GGUFReader(args.model, "r" if args.dry_run else "r+")
    convert_byteorder(reader, args)


if __name__ == "__main__":
    main()
