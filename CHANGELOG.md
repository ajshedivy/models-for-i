# Change record

## 2.0.1 — 2026-10-20

- Byte-swap quantized tensors with vectorized NumPy block views instead of a
  per-block Python loop. Output is byte-identical to the previous converter.
- Added Q4_1 support.
- Validate quantized block sizes and report tensor type counts before writing.
- Write a JSON manifest beside each converted model with source and output
  SHA-256 values, quantization mix, converter version and target byte order.
- Added `--yes` to skip the confirmation prompt and `--manifest PATH`.
- After writing, sample Q4_K/Q5_K `d`/`dmin` scales in the target byte order
  and fail if more than 5% are NaN, Inf or above 10.
- Added byte-layout and command-line regression tests.
