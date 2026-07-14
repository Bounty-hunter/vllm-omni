# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPT-NeoX style RoPE in Triton (bf16/fp16), avoiding fp32 round-trips."""

from __future__ import annotations

import torch

TRITON_NEOX_ROPE_AVAILABLE = False

try:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _neox_rope_kernel(
        x_ptr,
        out_ptr,
        cos_ptr,
        sin_ptr,
        stride_x_token,
        stride_x_head,
        stride_cos_token,
        num_heads,
        head_dim: tl.constexpr,
        HALF_DIM: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        token_idx = pid // num_heads
        head_idx = pid - token_idx * num_heads

        x_base = x_ptr + token_idx * stride_x_token + head_idx * stride_x_head
        out_base = out_ptr + token_idx * stride_x_token + head_idx * stride_x_head
        cos_base = cos_ptr + token_idx * stride_cos_token
        sin_base = sin_ptr + token_idx * stride_cos_token

        offs = tl.arange(0, BLOCK_HALF)
        mask = offs < HALF_DIM

        cos = tl.load(cos_base + offs, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_base + offs, mask=mask, other=0.0).to(tl.float32)

        x1_raw = tl.load(x_base + offs, mask=mask, other=0.0)
        x2_raw = tl.load(x_base + HALF_DIM + offs, mask=mask, other=0.0)
        x1 = x1_raw.to(tl.float32)
        x2 = x2_raw.to(tl.float32)

        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin

        tl.store(out_base + offs, o1.to(x1_raw.dtype), mask=mask)
        tl.store(out_base + HALF_DIM + offs, o2.to(x2_raw.dtype), mask=mask)

        if head_dim > HALF_DIM * 2:
            tail_offs = offs + HALF_DIM * 2
            tail_mask = tail_offs < head_dim
            tail = tl.load(x_base + tail_offs, mask=tail_mask, other=0.0)
            tl.store(out_base + tail_offs, tail, mask=tail_mask)

    def _launch_neox_rope(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # x: [num_tokens, num_heads, head_dim]
        num_tokens, num_heads, head_dim = x.shape
        half_dim = head_dim // 2
        block_half = triton.next_power_of_2(half_dim)

        out = torch.empty_like(x)
        grid = (num_tokens * num_heads,)
        num_warps = 4 if block_half <= 64 else 8

        _neox_rope_kernel[grid](
            x,
            out,
            cos,
            sin,
            x.stride(0),
            x.stride(1),
            cos.stride(0),
            num_heads,
            head_dim=head_dim,
            HALF_DIM=half_dim,
            BLOCK_HALF=block_half,
            num_warps=num_warps,
        )
        return out

    TRITON_NEOX_ROPE_AVAILABLE = True
except Exception:
    triton = None  # type: ignore[assignment,misc]
    tl = None  # type: ignore[assignment,misc]


def _normalize_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = cos.shape[-1]
    if cos.dim() == 2:
        if cos.shape[0] != seq_len:
            raise ValueError(f"cos seq {cos.shape[0]} != seq_len {seq_len}")
        cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
        sin = sin.unsqueeze(0).expand(batch_size, -1, -1)
    elif cos.dim() == 3:
        if cos.shape[0] == 1 and batch_size > 1:
            cos = cos.expand(batch_size, -1, -1)
            sin = sin.expand(batch_size, -1, -1)
        if cos.shape[0] != batch_size or cos.shape[1] != seq_len:
            raise ValueError(f"cos shape {cos.shape} incompatible with B={batch_size}, S={seq_len}")
    else:
        raise ValueError(f"cos/sin must be 2D or 3D, got {cos.dim()}D")

    cos_flat = cos.reshape(batch_size * seq_len, half).contiguous()
    sin_flat = sin.reshape(batch_size * seq_len, half).contiguous()
    return cos_flat, sin_flat


def apply_neox_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply GPT-NeoX RoPE to ``x`` with shape ``[B, S, H, D]``.

    ``cos``/``sin`` may be ``[S, D/2]`` or ``[B, S, D/2]``. When ``D > ro_dim``,
    tail dimensions are copied unchanged (matches ``apply_rotary_emb_torch``).
    """
    if x.dim() != 4:
        raise ValueError(f"x must be [B, S, H, D], got shape {x.shape}")

    batch_size, seq_len, _num_heads, head_dim = x.shape
    x = x.contiguous()
    cos_flat, sin_flat = _normalize_cos_sin(
        cos,
        sin,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    if TRITON_NEOX_ROPE_AVAILABLE and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        tokens = x.reshape(batch_size * seq_len, x.shape[2], head_dim)
        return _launch_neox_rope(tokens, cos_flat, sin_flat).reshape_as(x)

    from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch

    return apply_rotary_emb_torch(x, cos, sin, interleaved=False)
