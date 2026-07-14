# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Hunyuan Image 3.0 2D RoPE helpers (model-level cos/sin cache + apply)."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.layers.triton.neox_rope import apply_neox_rope


class HunyuanImageRoPECache:
    """Cache device-resident cos/sin for one denoise forward (shared by all layers)."""

    __slots__ = ("_cos", "_sin", "_key")

    def __init__(self) -> None:
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._key: tuple[int, int, torch.device] | None = None

    def prepare(
        self,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_in, sin_in = custom_pos_emb
        key = (cos_in.data_ptr(), sin_in.data_ptr(), device)
        if self._key != key:
            self._cos = cos_in.to(device, non_blocking=True)
            self._sin = sin_in.to(device, non_blocking=True)
            self._key = key
        assert self._cos is not None and self._sin is not None
        return self._cos, self._sin

    def clear(self) -> None:
        self._cos = None
        self._sin = None
        self._key = None


def apply_hunyuan_image_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D RoPE to flattened Q/K from HunYuanAttention.

    Args:
        q: ``[B*S, H*D]``
        k: ``[B*S, Hkv*D]``
        cos, sin: ``[B, S, D/2]`` or ``[S, D/2]`` (pipeline ``get_pos_emb`` output)
    """
    q_4d = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k_4d = k.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    q_4d = apply_neox_rope(q_4d, cos, sin)
    k_4d = apply_neox_rope(k_4d, cos, sin)
    flat_q = batch_size * seq_len
    return (
        q_4d.reshape(flat_q, num_heads * head_dim),
        k_4d.reshape(flat_q, num_kv_heads * head_dim),
    )
