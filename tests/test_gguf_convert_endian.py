"""Byte-layout regression tests for gguf_convert_endian."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import gguf

from gguf_convert_endian import byteswap_quantized_data

SCRIPT = Path(__file__).resolve().parent.parent / "gguf_convert_endian.py"
T = gguf.GGMLQuantizationType


class QuantizedEndianTests(unittest.TestCase):
    def round_trip(self, tensor_type, block_size, blocks=3):
        original = np.arange(block_size * blocks, dtype=np.uint8)
        converted = original.copy()
        byteswap_quantized_data(converted, tensor_type)
        byteswap_quantized_data(converted, tensor_type)
        np.testing.assert_array_equal(converted, original)

    def test_quantized_round_trips(self):
        for tensor_type, size in (
            (gguf.GGMLQuantizationType.Q4_0, 18),
            (gguf.GGMLQuantizationType.Q4_1, 20),
            (gguf.GGMLQuantizationType.Q4_K, 144),
            (gguf.GGMLQuantizationType.Q5_K, 176),
            (gguf.GGMLQuantizationType.Q6_K, 210),
        ):
            with self.subTest(tensor_type=tensor_type.name):
                self.round_trip(tensor_type, size)

    def test_bf16_round_trip(self):
        original = np.arange(32, dtype=np.uint8)
        converted = original.copy()
        converted.view(np.uint16).byteswap(inplace=True)
        converted.view(np.uint16).byteswap(inplace=True)
        np.testing.assert_array_equal(converted, original)

    def test_k_quants_only_swap_leading_fp16_values(self):
        # ggml-common.h: block_q4_K/block_q5_K begin with d and dmin.
        for tensor_type, size in (
            (gguf.GGMLQuantizationType.Q4_K, 144),
            (gguf.GGMLQuantizationType.Q5_K, 176),
        ):
            with self.subTest(tensor_type=tensor_type.name):
                original = np.arange(size, dtype=np.uint8)
                converted = original.copy()
                byteswap_quantized_data(converted, tensor_type)
                np.testing.assert_array_equal(converted[:4], np.array([1, 0, 3, 2], dtype=np.uint8))
                np.testing.assert_array_equal(converted[4:], original[4:])

    def test_rejects_incomplete_quantized_block(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            byteswap_quantized_data(np.zeros(143, dtype=np.uint8), gguf.GGMLQuantizationType.Q4_K)


class MetadataEndianTests(unittest.TestCase):
    def test_metadata_style_numeric_parts_round_trip(self):
        # GGUFReader field parts are NumPy numeric arrays; this is the same
        # operation used for metadata fields and tensor descriptors.
        original = np.array([0x01020304, 0xA0B0C0D0], dtype=np.uint32)
        part = original.copy()
        part.byteswap(inplace=True)
        part.byteswap(inplace=True)
        np.testing.assert_array_equal(part, original)


def write_fixture(path, tensor_types=(T.Q4_0, T.Q8_0, T.Q4_K, T.Q5_K, T.Q6_K, T.MXFP4)):
    """Little-endian GGUF with metadata and one tensor of every supported type."""
    rng = np.random.default_rng(0)
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_uint32("test.u32", 0x01020304)
    writer.add_array("test.arr", [1, 2, 3])
    writer.add_string("test.str", "hello")
    writer.add_tensor("f32", rng.standard_normal((4, 64)).astype(np.float32))
    writer.add_tensor("f16", rng.standard_normal((4, 64)).astype(np.float16))
    writer.add_tensor("bf16", rng.integers(0, 256, (4, 128), dtype=np.uint8), raw_dtype=T.BF16)
    for tensor_type in tensor_types:
        block_elems, block_size = gguf.constants.GGML_QUANT_SIZES[tensor_type]
        blocks = rng.integers(0, 256, (4 * max(1, 256 // block_elems), block_size), dtype=np.uint8)
        # Small, realistic FP16 scales so the converter's plausibility check passes.
        blocks[:, 0:4] = (rng.random((len(blocks), 2)) * 0.01).astype("<f2").view(np.uint8)
        writer.add_tensor(tensor_type.name.lower(), blocks.reshape(4, -1), raw_dtype=tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class CommandLineTests(unittest.TestCase):
    """Behaviour of the previous converter that must keep working."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name) / "model.gguf"
        write_fixture(self.model)
        self.original = self.model.read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args, answer="YES\n"):
        return subprocess.run([sys.executable, str(SCRIPT), str(self.model), *args],
                              input=answer, capture_output=True, text=True)

    def test_round_trip_restores_original_bytes(self):
        self.assertEqual(self.run_cli("big").returncode, 0)
        self.assertNotEqual(self.model.read_bytes(), self.original)
        self.assertEqual(gguf.GGUFReader(self.model).endianess.name, "BIG")
        self.assertEqual(self.run_cli("little").returncode, 0)
        self.assertEqual(self.model.read_bytes(), self.original)

    def test_dry_run_and_declined_prompt_leave_file_unchanged(self):
        self.assertEqual(self.run_cli("big", "--dry-run").returncode, 0)
        self.assertEqual(self.run_cli("big", answer="no\n").returncode, 0)
        self.assertEqual(self.model.read_bytes(), self.original)

    def test_already_in_requested_order_is_a_no_op(self):
        result = self.run_cli("little")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Nothing to do", result.stderr)
        self.assertEqual(self.model.read_bytes(), self.original)

    def test_unsupported_type_fails_before_writing(self):
        write_fixture(self.model, tensor_types=(T.Q5_0,))
        before = self.model.read_bytes()
        result = self.run_cli("big")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot handle type Q5_0", result.stderr)
        self.assertEqual(self.model.read_bytes(), before)

    def test_missing_order_is_a_usage_error(self):
        self.assertEqual(self.run_cli().returncode, 2)


if __name__ == "__main__":
    unittest.main()
