# Extending gguf_convert_endian.py

When the compatibility check or `--dry-run` reports `Cannot handle type <QTYPE>`, add a handler for that type to `gguf_convert_endian.py`. Every block-quantized GGML type has the same two-step pattern: byte-swap any `f16`/`f32` header fields, leave the `uint8` payload alone.

## Pattern

1. Write a `byteswap_<qtype>(tensor, block_offs)` function that swaps the float fields at their known offsets within the block.
2. Register it in the `byteswap_tensors` dict with the matching `gguf.GGMLQuantizationType` key.

Example (already in-tree, added for Q5_K):

```python
def byteswap_q5_k(tensor, block_offs):
    # Each block_q5_k (176 bytes) starts with two f16 values (d, dmin) followed by
    # 12 + 32 + 128 = 172 uint8 values. Only the two f16 fields need byteswapping.

    delta = tensor.data[block_offs:block_offs + 2].view(dtype=np.uint16)
    delta.byteswap(inplace=True)

    delta = tensor.data[block_offs + 2:block_offs + 4].view(dtype=np.uint16)
    delta.byteswap(inplace=True)


byteswap_tensors = {
    ...
    gguf.GGMLQuantizationType.Q5_K:  byteswap_q5_k,
    ...
}
```

The `block_size` is looked up automatically via `gguf.constants.GGML_QUANT_SIZES[tensor_type][1]` — you don't need to hardcode it, only the offsets within a block.

## Block layouts (K-quants, QK_K=256)

Source: `llama.cpp/ggml/src/ggml-common.h`. All K-quants use `QK_K=256` elements per block. `ggml_half` = f16 = 2 bytes. `K_SCALE_SIZE = 12`.

| Type   | Total bytes | Float fields (all f16)    | Byte-swap offsets |
|--------|-------------|---------------------------|-------------------|
| Q2_K   | 84          | `d` at 80, `dmin` at 82   | 80, 82            |
| Q3_K   | 110         | `d` at 108                | 108               |
| Q4_K   | 144         | `d` at 0, `dmin` at 2     | 0, 2              |
| Q5_K   | 176         | `d` at 0, `dmin` at 2     | 0, 2              |
| Q6_K   | 210         | `d` at 208                | 208               |

C struct orders from `ggml-common.h` (for cross-checking):

- **Q2_K**: `{ scales[16], qs[64], d, dmin }` → d@80, dmin@82
- **Q3_K**: `{ hmask[32], qs[64], scales[12], d }` → d@108
- **Q4_K**: `{ d, dmin, scales[12], qs[128] }` → d@0, dmin@2
- **Q5_K**: `{ d, dmin, scales[12], qh[32], ql[128] }` → d@0, dmin@2
- **Q6_K**: `{ ql[128], qh[64], scales[16], d }` → d@208

**Non-K quants:**

| Type   | Block (elems) | Total bytes | Float field(s) | Byte-swap offsets |
|--------|---------------|-------------|----------------|-------------------|
| Q4_0   | 32            | 18          | `d` (f16) at 0 | 0                 |
| Q4_1   | 32            | 20          | `d`, `m` (both f16) at 0, 2 | 0, 2     |
| Q5_0   | 32            | 22          | `d` (f16) at 0, then `qh` uint32 at 2 | 0, and `tensor.data[2:6].view(dtype=np.uint32).byteswap(inplace=True)` |
| Q5_1   | 32            | 24          | `d`, `m` (f16) at 0, 2; `qh` uint32 at 4 | 0, 2, and uint32 at 4 |
| Q8_0   | 32            | 34          | `d` (f16) at 0 | 0                 |
| Q8_1   | 32            | 36          | `d`, `s` (f16) at 0, 2 | 0, 2       |

Q5_0 and Q5_1 also have a 4-byte `qh` bitmask that is serialized as a little-endian uint32 on little-endian hosts; that needs byte-swapping too. This is different from the K-quant pattern and easy to miss.

## Verification

After adding a handler, dry-run against a file of the newly-supported type:

```bash
uv run gguf_convert_endian.py models/<name>/<file>.gguf big --dry-run --verbose
```

It should log `Preparing to convert from LITTLE to BIG` and exit 0 without any `Cannot handle type` errors.

If extending the in-tree `gguf_convert_endian.py` (the top-level copy used by the project), the same fix should be applied to `gguf-py/gguf/scripts/gguf_convert_endian.py` if that copy is also used, but in this project only the top-level script is invoked via `uv run`.
