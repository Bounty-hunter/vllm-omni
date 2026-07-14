# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused packed-QKV epilogue: NeoX-RoPE + QK-RMSNorm → FA layout.

Reads ``qkv`` as ``[T, q_size + kv_size + kv_size]``, applies NeoX RoPE then
per-head RMSNorm on Q/K in registers, and stores ``q/k`` as ``[T, H, D]``.
``V`` is a zero-copy view into the packed buffer (no RoPE / Norm).

Order matches HunyuanImage3 ``HunYuanAttention``: RoPE → QK-RMSNorm.
"""

from __future__ import annotations

import os

import torch

from vllm_omni.diffusion.layers.triton.neox_rope import (
    TRITON_NEOX_ROPE_AVAILABLE,
    _normalize_cos_sin,
    apply_neox_rope,
)

TRITON_FUSED_QKV_ROPE_QKNORM_AVAILABLE = False

try:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _fused_qkv_neox_rope_qk_rmsnorm_kernel(
        qkv_ptr,
        q_out_ptr,
        k_out_ptr,
        cos_ptr,
        sin_ptr,
        q_w_ptr,
        k_w_ptr,
        stride_qkv_tok,
        stride_q_tok,
        stride_q_head,
        stride_k_tok,
        stride_k_head,
        stride_cos_tok,
        num_q_heads,
        num_kv_heads,
        HEAD_DIM: tl.constexpr,
        HALF: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
        Q_SIZE: tl.constexpr,
        EPS: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        nheads = num_q_heads + num_kv_heads
        token_idx = pid // nheads
        head_pid = pid - token_idx * nheads
        is_q = head_pid < num_q_heads
        head_idx = tl.where(is_q, head_pid, head_pid - num_q_heads)

        feat_base = tl.where(is_q, head_idx * HEAD_DIM, Q_SIZE + head_idx * HEAD_DIM)
        x_base = qkv_ptr + token_idx * stride_qkv_tok + feat_base

        w_ptr = tl.where(is_q, q_w_ptr, k_w_ptr)
        out_base = tl.where(
            is_q,
            q_out_ptr + token_idx * stride_q_tok + head_idx * stride_q_head,
            k_out_ptr + token_idx * stride_k_tok + head_idx * stride_k_head,
        )
        cos_base = cos_ptr + token_idx * stride_cos_tok
        sin_base = sin_ptr + token_idx * stride_cos_tok

        offs = tl.arange(0, BLOCK_HALF)
        mask = offs < HALF

        cos = tl.load(cos_base + offs, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_base + offs, mask=mask, other=0.0).to(tl.float32)

        x1 = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin

        sumsq = tl.sum(tl.where(mask, y1 * y1, 0.0), axis=0) + tl.sum(tl.where(mask, y2 * y2, 0.0), axis=0)

        if HEAD_DIM > HALF * 2:
            tail_offs = offs + HALF * 2
            tail_mask = tail_offs < HEAD_DIM
            y_tail = tl.load(x_base + tail_offs, mask=tail_mask, other=0.0).to(tl.float32)
            sumsq += tl.sum(tl.where(tail_mask, y_tail * y_tail, 0.0), axis=0)
        else:
            y_tail = tl.zeros([BLOCK_HALF], dtype=tl.float32)
            tail_offs = offs
            tail_mask = offs < 0

        inv_rms = tl.rsqrt(sumsq / HEAD_DIM + EPS)

        w1 = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + offs, mask=mask, other=0.0).to(tl.float32)
        o1 = y1 * inv_rms * w1
        o2 = y2 * inv_rms * w2

        tl.store(out_base + offs, o1.to(q_out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_base + HALF + offs, o2.to(q_out_ptr.dtype.element_ty), mask=mask)

        if HEAD_DIM > HALF * 2:
            w_tail = tl.load(w_ptr + tail_offs, mask=tail_mask, other=0.0).to(tl.float32)
            o_tail = y_tail * inv_rms * w_tail
            tl.store(out_base + tail_offs, o_tail.to(q_out_ptr.dtype.element_ty), mask=tail_mask)

    def _launch_fused(
        qkv: torch.Tensor,
        cos_flat: torch.Tensor,
        sin_flat: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = qkv.shape[0]
        q_size = num_heads * head_dim
        half = head_dim // 2
        block_half = triton.next_power_of_2(half)

        q_out = torch.empty((num_tokens, num_heads, head_dim), device=qkv.device, dtype=qkv.dtype)
        k_out = torch.empty((num_tokens, num_kv_heads, head_dim), device=qkv.device, dtype=qkv.dtype)

        grid = (num_tokens * (num_heads + num_kv_heads),)
        num_warps = 4 if block_half <= 64 else 8

        _fused_qkv_neox_rope_qk_rmsnorm_kernel[grid](
            qkv,
            q_out,
            k_out,
            cos_flat,
            sin_flat,
            q_weight,
            k_weight,
            qkv.stride(0),
            q_out.stride(0),
            q_out.stride(1),
            k_out.stride(0),
            k_out.stride(1),
            cos_flat.stride(0),
            num_heads,
            num_kv_heads,
            HEAD_DIM=head_dim,
            HALF=half,
            BLOCK_HALF=block_half,
            Q_SIZE=q_size,
            EPS=eps,
            num_warps=num_warps,
        )
        return q_out, k_out

    TRITON_FUSED_QKV_ROPE_QKNORM_AVAILABLE = True
except Exception:
    triton = None  # type: ignore[assignment,misc]
    tl = None  # type: ignore[assignment,misc]


def is_hunyuan_fused_attn_epilogue_enabled() -> bool:
    """Resolve whether fused QKV→RoPE→QK-RMSNorm epilogue is enabled.

    Precedence:
    1. Env ``VLLM_OMNI_HUNYUAN_FUSED_ATTN_EPILOGUE`` (0/false/off disables)
    2. ``OmniDiffusionConfig.enable_hunyuan_fused_attn_epilogue``
    3. Default ``True``
    """
    env = os.environ.get("VLLM_OMNI_HUNYUAN_FUSED_ATTN_EPILOGUE")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "off", "no"}

    try:
        from vllm_omni.diffusion.config import get_current_diffusion_config_or_none

        cfg = get_current_diffusion_config_or_none()
        if cfg is not None:
            return bool(getattr(cfg, "enable_hunyuan_fused_attn_epilogue", True))
    except Exception:
        pass

    try:
        from vllm_omni.diffusion.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if is_forward_context_available():
            cfg = get_forward_context().omni_diffusion_config
            if cfg is not None:
                return bool(getattr(cfg, "enable_hunyuan_fused_attn_epilogue", True))
    except Exception:
        pass

    return True


def fused_qkv_neox_rope_qk_rmsnorm_ref(
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
    """PyTorch reference: split → NeoX-RoPE → RMSNorm, FA layout ``[T, H, D]``."""
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    if qkv.dim() != 2 or qkv.shape[-1] != q_size + 2 * kv_size:
        raise ValueError(f"qkv expected [T, {q_size + 2 * kv_size}], got {tuple(qkv.shape)}")

    tokens = batch_size * seq_len
    if qkv.shape[0] != tokens:
        raise ValueError(f"qkv tokens {qkv.shape[0]} != B*S {tokens}")

    q = qkv[:, :q_size].reshape(batch_size, seq_len, num_heads, head_dim)
    k = qkv[:, q_size : q_size + kv_size].reshape(batch_size, seq_len, num_kv_heads, head_dim)
    v = qkv[:, q_size + kv_size :].reshape(tokens, num_kv_heads, head_dim)

    q = apply_neox_rope(q, cos, sin)
    k = apply_neox_rope(k, cos, sin)

    def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x_f = x.to(torch.float32)
        out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
        return (out * weight.to(torch.float32)).to(orig)

    q = _rms(q, q_weight).reshape(tokens, num_heads, head_dim)
    k = _rms(k, k_weight).reshape(tokens, num_kv_heads, head_dim)
    return q, k, v


def fused_qkv_neox_rope_qk_rmsnorm(
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
    """Fused packed-QKV → NeoX-RoPE → QK-RMSNorm.

    Returns:
        q: ``[T, Hq, D]``, k: ``[T, Hkv, D]``, v: ``[T, Hkv, D]`` (V is a view).
    """
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    tokens = batch_size * seq_len
    if qkv.dim() != 2 or qkv.shape[0] != tokens or qkv.shape[-1] != q_size + 2 * kv_size:
        raise ValueError(
            f"qkv shape {tuple(qkv.shape)} incompatible with "
            f"T={tokens}, q_size={q_size}, kv_size={kv_size}"
        )
    if q_weight.numel() != head_dim or k_weight.numel() != head_dim:
        raise ValueError("q/k RMSNorm weights must be length head_dim")

    v = qkv[:, q_size + kv_size :].reshape(tokens, num_kv_heads, head_dim)

    use_triton = (
        is_hunyuan_fused_attn_epilogue_enabled()
        and TRITON_FUSED_QKV_ROPE_QKNORM_AVAILABLE
        and TRITON_NEOX_ROPE_AVAILABLE
        and qkv.is_cuda
        and qkv.dtype in (torch.float16, torch.bfloat16)
        and qkv.stride(-1) == 1
        and not torch.compiler.is_compiling()
    )
    if use_triton:
        cos_flat, sin_flat = _normalize_cos_sin(cos, sin, batch_size=batch_size, seq_len=seq_len)
        q, k = _launch_fused(
            qkv.contiguous(),
            cos_flat,
            sin_flat,
            q_weight.contiguous(),
            k_weight.contiguous(),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
        )
        return q, k, v

    return fused_qkv_neox_rope_qk_rmsnorm_ref(
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
