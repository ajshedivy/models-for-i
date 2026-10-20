# Extending gguf_convert_endian.py

When the compatibility check or `--dry-run` reports `Cannot handle type <QTYPE>`, add a handler for that type to `gguf_convert_endian.py`. Every block-quantized GGML type has the same two-step pattern: byte-swap any `f16`/`f32` header fields, leave the `uint8` payload alone.

## Pattern

Add one entry to the `QUANT_LAYOUTS` table: `(block size in bytes, byte offsets of the 16-bit fields)`. Every byte not listed is left untouched. Take the offsets from the C struct in `ggml-common.h` (tables below), never from memory.

```python
QUANT_LAYOUTS = {
    ...
    # block_q5_K: d (f16), dmin (f16), scales[12], qh[32], qs[128].
    gguf.GGMLQuantizationType.Q5_K: (176, (0, 2)),
    ...
}
```

Types whose blocks contain only single-byte fields (e.g. MXFP4) go in `NO_SWAP_QUANT_TYPES` instead. `_swap_u16_fields` swaps 16-bit lanes only, so a type with a 32-bit field (Q5_0/Q5_1 `qh`) also needs a 32-bit swap added.

An unreleased build once got the Q4_K/Q5_K offsets wrong (it used the last four bytes) and its own unit test encoded the same mistake. Verify against the struct and against an independent converter, not only against the in-tree tests.

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

It should list the tensor type counts, log `Dry run complete; no changes made.` and exit 0 without any `Cannot handle type` errors.

Then prove the byte layout. Convert a copy with the in-tree script and with upstream `.venv/lib/python3.12/site-packages/gguf/scripts/gguf_convert_endian.py` (if it supports the type) and compare SHA-256 values; also convert back to little-endian and check that the original SHA-256 returns. Add the new type to `tests/test_gguf_convert_endian.py`.

If extending the in-tree `gguf_convert_endian.py` (the top-level copy used by the project), the same fix should be applied to `gguf-py/gguf/scripts/gguf_convert_endian.py` if that copy is also used, but in this project only the top-level script is invoked via `uv run`.
