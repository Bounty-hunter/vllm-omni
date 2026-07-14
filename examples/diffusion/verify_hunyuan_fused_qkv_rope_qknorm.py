# SPDX-License-Identifier: Apache-2.0
"""Micro-check for fused packed-QKV NeoX-RoPE + QK-RMSNorm.

Usage:
  python examples/diffusion/verify_hunyuan_fused_qkv_rope_qknorm.py
  VLLM_OMNI_HUNYUAN_FUSED_ATTN_EPILOGUE=0 python ...  # force PyTorch ref path only
"""

from __future__ import annotations

import os

import torch


def main() -> None:
    from vllm_omni.diffusion.layers.triton.fused_qkv_neox_rope_qk_rmsnorm import (
        TRITON_FUSED_QKV_ROPE_QKNORM_AVAILABLE,
        fused_qkv_neox_rope_qk_rmsnorm,
        fused_qkv_neox_rope_qk_rmsnorm_ref,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    bsz, seq_len, num_heads, num_kv_heads, head_dim = 2, 64, 8, 2, 128
    eps = 1e-5
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    torch.manual_seed(0)
    qkv = torch.randn(bsz * seq_len, q_size + 2 * kv_size, device=device, dtype=dtype)
    ang = torch.rand(bsz, seq_len, head_dim // 2, device=device)
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    q_w = torch.randn(head_dim, device=device, dtype=dtype)
    k_w = torch.randn(head_dim, device=device, dtype=dtype)

    from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch

    q_size_i = num_heads * head_dim
    kv_size_i = num_kv_heads * head_dim
    qg = qkv[:, :q_size_i].reshape(bsz, seq_len, num_heads, head_dim)
    kg = qkv[:, q_size_i : q_size_i + kv_size_i].reshape(bsz, seq_len, num_kv_heads, head_dim)
    qg = apply_rotary_emb_torch(qg.float(), cos, sin, interleaved=False)
    kg = apply_rotary_emb_torch(kg.float(), cos, sin, interleaved=False)

    def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (out * weight.float()).to(dtype)

    q_ref = _rms(qg, q_w).reshape(bsz * seq_len, num_heads, head_dim)
    k_ref = _rms(kg, k_w).reshape(bsz * seq_len, num_kv_heads, head_dim)
    v_ref = qkv[:, q_size_i + kv_size_i :].reshape(bsz * seq_len, num_kv_heads, head_dim)

    os.environ["VLLM_OMNI_HUNYUAN_FUSED_ATTN_EPILOGUE"] = "1"
    q, k, v = fused_qkv_neox_rope_qk_rmsnorm(
        qkv,
        cos,
        sin,
        q_w,
        k_w,
        batch_size=bsz,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=eps,
    )

    def _stats(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
        diff = (a.float() - b.float()).abs()
        cos_sim = torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
        print(f"{name}: max={diff.max().item():.6g} mean={diff.mean().item():.6g} cos={cos_sim.item():.8f}")

    print(f"device={device} dtype={dtype} triton_available={TRITON_FUSED_QKV_ROPE_QKNORM_AVAILABLE}")
    _stats("q", q, q_ref)
    _stats("k", k, k_ref)
    _stats("v", v, v_ref)
    assert torch.equal(v, v_ref), "V must be a view/equal to packed slice"
    atol = 8e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(q.float(), q_ref.float(), atol=atol, rtol=2e-2)
    assert torch.allclose(k.float(), k_ref.float(), atol=atol, rtol=2e-2)
    assert (
        torch.nn.functional.cosine_similarity(q.float().flatten(), q_ref.float().flatten(), dim=0).item() > 0.9999
    )
    _ = fused_qkv_neox_rope_qk_rmsnorm_ref  # keep import used for discovery
    print("OK")


if __name__ == "__main__":
    main()
