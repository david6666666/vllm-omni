# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static checks for diffusion FP8 QKV routing.

These checks intentionally avoid importing vllm_omni because the lightweight
local test environment used for this PR review may not have optional runtime
dependencies such as OpenCV, aenum, FlashAttention, or MindIE-SD installed.
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


class FP8BackendRoutingStaticTest(unittest.TestCase):

    def read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_layer_only_propagates_fp8_intent(self):
        layer = self.read("vllm_omni/diffusion/attention/layer.py")

        self.assertNotIn("quantize_qkv_fp8", layer)
        self.assertNotIn("torch.float8_e4m3fn", layer)
        self.assertIn("attn_metadata.kv_cache_dtype = \"fp8\"", layer)

    def test_metadata_carries_kv_cache_dtype(self):
        abstract = self.read("vllm_omni/diffusion/attention/backends/abstract.py")

        self.assertIn("kv_cache_dtype: str | None = None", abstract)

    def test_sdpa_does_not_receive_prequantized_fp8(self):
        sdpa = self.read("vllm_omni/diffusion/attention/backends/sdpa.py")

        self.assertNotIn("torch.float8_e4m3fn", sdpa)
        self.assertNotIn("dequantize_fp8", sdpa)

    def test_flash_backend_owns_fp8_qkv_preparation(self):
        flash_attn = self.read("vllm_omni/diffusion/attention/backends/flash_attn.py")

        self.assertIn("def _maybe_prepare_fp8_qkv", flash_attn)
        self.assertIn("quantize_qkv_fp8_fast", flash_attn)
        self.assertIn("attn_metadata.kv_cache_dtype == \"fp8\"", flash_attn)


if __name__ == "__main__":
    unittest.main()
