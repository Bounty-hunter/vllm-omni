# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for Hunyuan fused packed-QKV NeoX-RoPE + QK-RMSNorm."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [pytest.mark.core_model]

CUDA = torch.cuda.is_available()


def _torch_golden(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    tokens = batch_size * seq_len
    q = qkv[:, :q_size].reshape(batch_size, seq_len, num_heads, head_dim)
    k = qkv[:, q_size : q_size + kv_size].reshape(batch_size, seq_len, num_kv_heads, head_dim)
    v = qkv[:, q_size + kv_size :].reshape(tokens, num_kv_heads, head_dim)

    # Keep rope math in fp32 so golden tracks the fused kernel accumulate path.
    q = apply_rotary_emb_torch(q.float(), cos, sin, interleaved=False)
    k = apply_rotary_emb_torch(k.float(), cos, sin, interleaved=False)

    def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (out * weight.float()).to(qkv.dtype)

    q = _rms(q, q_weight).reshape(tokens, num_heads, head_dim)
    k = _rms(k, k_weight).reshape(tokens, num_kv_heads, head_dim)
    return q, k, v


@pytest.mark.skipif(not CUDA, reason="CUDA required for Triton fused epilogue")
def test_fused_qkv_neox_rope_qk_rmsnorm_matches_torch_golden():
    from vllm_omni.diffusion.layers.triton.fused_qkv_neox_rope_qk_rmsnorm import (
        TRITON_FUSED_QKV_ROPE_QKNORM_AVAILABLE,
        fused_qkv_neox_rope_qk_rmsnorm,
    )

    if not TRITON_FUSED_QKV_ROPE_QKNORM_AVAILABLE:
        pytest.skip("Triton fused kernel unavailable")

    os.environ["VLLM_OMNI_HUNYUAN_FUSED_ATTN_EPILOGUE"] = "1"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, seq_len = 2, 64
    num_heads, num_kv_heads, head_dim = 8, 2, 128
    eps = 1e-5
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    torch.manual_seed(0)
    qkv = torch.randn(batch_size * seq_len, q_size + 2 * kv_size, device=device, dtype=dtype)
    ang = torch.rand(batch_size, seq_len, head_dim // 2, device=device)
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)

    q_ref, k_ref, v_ref = _torch_golden(
        qkv,
        cos,
        sin,
        q_weight,
        k_weight,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=eps,
    )
    q, k, v = fused_qkv_neox_rope_qk_rmsnorm(
        qkv,
        cos,
        sin,
        q_weight,
        k_weight,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=eps,
    )

    assert torch.equal(v, v_ref)
    # bf16 store ULP (~0.0625) is expected; cosine should stay near 1.
    assert torch.allclose(q.float(), q_ref.float(), atol=8e-2, rtol=2e-2)
    assert torch.allclose(k.float(), k_ref.float(), atol=8e-2, rtol=2e-2)
    q_cos = torch.nn.functional.cosine_similarity(q.float().flatten(), q_ref.float().flatten(), dim=0)
    k_cos = torch.nn.functional.cosine_similarity(k.float().flatten(), k_ref.float().flatten(), dim=0)
    assert q_cos.item() > 0.9999
    assert k_cos.item() > 0.9999


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_disable_env_falls_back_to_ref_path():
    from vllm_omni.diffusion.layers.triton.fused_qkv_neox_rope_qk_rmsnorm import (
        fused_qkv_neox_rope_qk_rmsnorm,
        fused_qkv_neox_rope_qk_rmsnorm_ref,
        is_hunyuan_fused_attn_epilogue_enabled,
    )

    os.environ["VLLM_OMNI_HUNYUAN_FUSED_ATTN_EPILOGUE"] = "0"
    assert is_hunyuan_fused_attn_epilogue_enabled() is False

    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, seq_len = 1, 8
    num_heads, num_kv_heads, head_dim = 4, 2, 64
    eps = 1e-5
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    torch.manual_seed(1)
    qkv = torch.randn(batch_size * seq_len, q_size + 2 * kv_size, device=device, dtype=dtype)
    ang = torch.rand(batch_size, seq_len, head_dim // 2, device=device)
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    q_weight = torch.ones(head_dim, device=device, dtype=dtype)
    k_weight = torch.ones(head_dim, device=device, dtype=dtype)

    q_ref, k_ref, v_ref = fused_qkv_neox_rope_qk_rmsnorm_ref(
        qkv,
        cos,
        sin,
        q_weight,
        k_weight,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=eps,
    )
    q, k, v = fused_qkv_neox_rope_qk_rmsnorm(
        qkv,
        cos,
        sin,
        q_weight,
        k_weight,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=eps,
    )
    assert torch.equal(q, q_ref)
    assert torch.equal(k, k_ref)
    assert torch.equal(v, v_ref)
    os.environ["VLLM_OMNI_HUNYUAN_FUSED_ATTN_EPILOGUE"] = "1"
