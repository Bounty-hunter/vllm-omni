# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused prompt||image KV cat + GQA repeat into FA layout.

Merges ``cached [B,P,Hkv,D]`` with ``image [B,I,Hkv,D]`` and expands KV heads
to ``Hq = Hkv * n_rep``, writing contiguous ``out [B,P+I,Hq,D]`` in one pass.
Avoids intermediate ``torch.cat`` + ``expand``/``reshape`` materializations.
"""

from __future__ import annotations

import os

import torch

TRITON_FUSED_CAT_REPEAT_KV_AVAILABLE = False


def _torch_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA head expand (same semantics as Hunyuan ``repeat_kv``)."""
    if n_rep == 1:
        return hidden_states
    batch, slen, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_kv_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_kv_heads * n_rep, head_dim)

try:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _fused_cat_repeat_kv_kernel(
        prompt_ptr,
        image_ptr,
        out_ptr,
        stride_p_b,
        stride_p_t,
        stride_p_h,
        stride_i_b,
        stride_i_t,
        stride_i_h,
        stride_o_b,
        stride_o_t,
        stride_o_h,
        prompt_len,
        image_len,
        num_kv_heads,
        n_rep,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        total_len = prompt_len + image_len
        num_q_heads = num_kv_heads * n_rep
        toks_heads = total_len * num_q_heads

        batch_idx = pid // toks_heads
        rem = pid - batch_idx * toks_heads
        tok_idx = rem // num_q_heads
        q_head = rem - tok_idx * num_q_heads
        kv_head = q_head // n_rep

        offs = tl.arange(0, BLOCK_D)
        mask = offs < HEAD_DIM

        out_base = out_ptr + batch_idx * stride_o_b + tok_idx * stride_o_t + q_head * stride_o_h

        if tok_idx < prompt_len:
            src = prompt_ptr + batch_idx * stride_p_b + tok_idx * stride_p_t + kv_head * stride_p_h
            vals = tl.load(src + offs, mask=mask, other=0.0)
        else:
            img_tok = tok_idx - prompt_len
            src = image_ptr + batch_idx * stride_i_b + img_tok * stride_i_t + kv_head * stride_i_h
            vals = tl.load(src + offs, mask=mask, other=0.0)

        tl.store(out_base + offs, vals, mask=mask)

    def _launch_fused_cat_repeat(
        prompt: torch.Tensor,
        image: torch.Tensor,
        *,
        n_rep: int,
    ) -> torch.Tensor:
        if prompt.dim() != 4 or image.dim() != 4:
            raise ValueError(f"expected 4D tensors, got {prompt.dim()}/{image.dim()}")
        if prompt.shape[0] != image.shape[0] or prompt.shape[2:] != image.shape[2:]:
            raise ValueError(f"shape mismatch prompt={tuple(prompt.shape)} image={tuple(image.shape)}")
        if n_rep < 1:
            raise ValueError(f"n_rep must be >= 1, got {n_rep}")

        batch, prompt_len, num_kv_heads, head_dim = prompt.shape
        image_len = image.shape[1]
        num_q_heads = num_kv_heads * n_rep
        total_len = prompt_len + image_len

        out = torch.empty(
            (batch, total_len, num_q_heads, head_dim),
            device=prompt.device,
            dtype=prompt.dtype,
        )
        if total_len == 0 or batch == 0:
            return out

        block_d = triton.next_power_of_2(head_dim)
        grid = (batch * total_len * num_q_heads,)
        num_warps = 4 if block_d <= 128 else 8

        _fused_cat_repeat_kv_kernel[grid](
            prompt,
            image,
            out,
            prompt.stride(0),
            prompt.stride(1),
            prompt.stride(2),
            image.stride(0),
            image.stride(1),
            image.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            prompt_len,
            image_len,
            num_kv_heads,
            n_rep,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        return out

    TRITON_FUSED_CAT_REPEAT_KV_AVAILABLE = True
except Exception:
    triton = None  # type: ignore[assignment,misc]
    tl = None  # type: ignore[assignment,misc]


def is_hunyuan_fused_cat_repeat_kv_enabled() -> bool:
    """Resolve whether fused prompt||image KV cat + GQA repeat is enabled.

    Precedence:
    1. Env ``VLLM_OMNI_HUNYUAN_FUSED_CAT_REPEAT_KV`` (0/false/off disables)
    2. ``OmniDiffusionConfig.enable_hunyuan_fused_cat_repeat_kv``
    3. Default ``True``
    """
    env = os.environ.get("VLLM_OMNI_HUNYUAN_FUSED_CAT_REPEAT_KV")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "off", "no"}

    try:
        from vllm_omni.diffusion.config import get_current_diffusion_config_or_none

        cfg = get_current_diffusion_config_or_none()
        if cfg is not None:
            return bool(getattr(cfg, "enable_hunyuan_fused_cat_repeat_kv", True))
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
                return bool(getattr(cfg, "enable_hunyuan_fused_cat_repeat_kv", True))
    except Exception:
        pass

    return True


def fused_cat_repeat_kv_ref(
    prompt: torch.Tensor,
    image: torch.Tensor,
    *,
    n_rep: int,
) -> torch.Tensor:
    """PyTorch reference: ``cat(dim=1)`` then GQA ``repeat_kv``."""
    merged = torch.cat([prompt, image], dim=1)
    if n_rep == 1:
        return merged.contiguous()
    return _torch_repeat_kv(merged, n_rep).contiguous()


def fused_cat_repeat_kv(
    prompt: torch.Tensor,
    image: torch.Tensor,
    *,
    n_rep: int,
) -> torch.Tensor:
    """Fused prompt||image cat + GQA repeat → contiguous ``[B,P+I,Hq,D]``."""
    use_triton = (
        is_hunyuan_fused_cat_repeat_kv_enabled()
        and TRITON_FUSED_CAT_REPEAT_KV_AVAILABLE
        and prompt.is_cuda
        and image.is_cuda
        and prompt.dtype == image.dtype
        and prompt.dtype in (torch.float16, torch.bfloat16)
        and prompt.stride(-1) == 1
        and image.stride(-1) == 1
        and not torch.compiler.is_compiling()
    )
    if use_triton:
        # Ensure token/head dims are contiguous in the trailing dimension only;
        # avoid full `.contiguous()` when already D-major.
        p = prompt if prompt.stride(-1) == 1 else prompt.contiguous()
        i = image if image.stride(-1) == 1 else image.contiguous()
        return _launch_fused_cat_repeat(p, i, n_rep=n_rep)
    return fused_cat_repeat_kv_ref(prompt, image, n_rep=n_rep)
