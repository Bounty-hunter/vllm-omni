# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for fused prompt||image KV cat + GQA repeat."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [pytest.mark.core_model]

CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="CUDA required for Triton fused cat/repeat")
@pytest.mark.parametrize("n_rep", [1, 4])
@pytest.mark.parametrize("batch", [1, 2])
def test_fused_cat_repeat_kv_matches_torch(n_rep: int, batch: int):
    from vllm_omni.diffusion.layers.triton.fused_cat_repeat_kv import (
        TRITON_FUSED_CAT_REPEAT_KV_AVAILABLE,
        fused_cat_repeat_kv,
        fused_cat_repeat_kv_ref,
    )

    if not TRITON_FUSED_CAT_REPEAT_KV_AVAILABLE:
        pytest.skip("Triton fused cat/repeat unavailable")

    os.environ["VLLM_OMNI_HUNYUAN_FUSED_CAT_REPEAT_KV"] = "1"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    prompt_len, image_len = 128, 1024
    num_kv_heads, head_dim = 2, 128

    torch.manual_seed(0)
    prompt = torch.randn(batch, prompt_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    image = torch.randn(batch, image_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    ref = fused_cat_repeat_kv_ref(prompt, image, n_rep=n_rep)
    out = fused_cat_repeat_kv(prompt, image, n_rep=n_rep)

    assert out.shape == ref.shape == (batch, prompt_len + image_len, num_kv_heads * n_rep, head_dim)
    assert out.is_contiguous()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_fused_cat_repeat_kv_env_disable_falls_back():
    from vllm_omni.diffusion.layers.triton.fused_cat_repeat_kv import (
        fused_cat_repeat_kv,
        fused_cat_repeat_kv_ref,
    )

    os.environ["VLLM_OMNI_HUNYUAN_FUSED_CAT_REPEAT_KV"] = "0"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    prompt = torch.randn(1, 16, 2, 64, device=device, dtype=dtype)
    image = torch.randn(1, 32, 2, 64, device=device, dtype=dtype)
    ref = fused_cat_repeat_kv_ref(prompt, image, n_rep=4)
    out = fused_cat_repeat_kv(prompt, image, n_rep=4)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    os.environ["VLLM_OMNI_HUNYUAN_FUSED_CAT_REPEAT_KV"] = "1"
