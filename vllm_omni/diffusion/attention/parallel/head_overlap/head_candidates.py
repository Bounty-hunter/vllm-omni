# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Conservative dense layouts and pure head partitions for real-layer trials."""

import math

import torch

_PACKED_FIELDS = frozenset(("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"))
# These are head-independent producer hints ignored by CUDA packed FlashAttention.
# State-bearing reference-cache, gate, mask and quantization extras remain excluded.
_PACKED_HINTS = frozenset(
    (
        "valid_kv_length",
        "npu_attn_varlen",
        "laser_input_scale",
        "vsa_h3_tile_size",
        "vsa_h3_ref_policy",
        "vsa_h3_ref_keep_rate",
        "vsa_h3_sparsity",
    )
)


def _packed_padding_eligible(metadata, rows):
    padding, extra = metadata.packed_padding, metadata.extra
    if not _PACKED_FIELDS <= extra.keys() or extra.keys() - _PACKED_FIELDS - _PACKED_HINTS:
        return False
    if not (
        type(padding.q_length) is int
        and type(padding.kv_length) is int
        and 0 < padding.q_length == padding.kv_length <= rows
        and extra["max_seqlen_q"] == padding.q_length
        and extra["max_seqlen_k"] == padding.kv_length
        and extra.get("valid_kv_length", padding.kv_length) == padding.kv_length
        and all(x is None for x in (metadata.joint_query, metadata.joint_key, metadata.joint_value))
    ):
        return False
    # The producer supplies canonical [0, valid] views of the complete
    # [real, pad] document boundaries. Heads are independent; retain the
    # original cu_seqlens and varlen dispatcher for every partition.
    for key, view in (("cu_seqlens_q", padding.cu_seqlens_q), ("cu_seqlens_k", padding.cu_seqlens_k)):
        cu = extra[key]
        if not (
            isinstance(cu, torch.Tensor)
            and isinstance(view, torch.Tensor)
            and cu.ndim == view.ndim == 1
            and cu.numel() == (2 if padding.q_length == rows else 3)
            and view.numel() == 2
            and cu.dtype == view.dtype == torch.int32
            and cu.is_contiguous()
            and view.is_contiguous()
            and cu.device == view.device
            and cu.data_ptr() == view.data_ptr()
        ):
            return False
    return True


def eligible(kernel, metadata, distributed_rows):
    owner = getattr(kernel, "__self__", None)
    if getattr(owner, "causal", None) is not False or not hasattr(owner, "effective_attention"):
        return False
    impl, backend, _ = owner.effective_attention()
    if backend is None or backend.get_name() != "FLASH_ATTN":
        return False
    if metadata is None:
        return True
    if any(
        x is not None
        for x in (
            metadata.attn_mask,
            metadata.joint_attn_mask,
            metadata.full_attn_spans,
            metadata.query_ranges,
            metadata.cross_query_mask,
        )
    ):
        return False
    if metadata.packed_padding is not None:
        return getattr(impl, "stable_packed_head_partitions", False) and _packed_padding_eligible(
            metadata, distributed_rows
        )
    layout = metadata.video_layout
    if layout is None:
        if metadata.extra:
            return False
    else:
        # Dense FA3 ignores grid geometry. Only admit a single fully occupied
        # video grid; masks, packed rows and sparse backends keep their routes.
        grid = metadata.extra.get("vsa_dit_seq_shape")
        if (
            layout.prefix_len != 0
            or layout.video_spans
            or layout.used_len != distributed_rows
            or layout.latent_grid is None
            or math.prod(layout.latent_grid) != distributed_rows
            or len(metadata.extra) != 1
            or type(grid) is not tuple
            or any(type(dim) is not int for dim in grid)
            or grid != layout.latent_grid
            or metadata.joint_query is not None
        ):
            return False
    return (
        (metadata.joint_query is None or metadata.joint_query.shape[1] == 0)
        and metadata.attn_mask is None
        and metadata.joint_attn_mask is None
        and metadata.full_attn_spans is None
        and metadata.query_ranges is None
        and metadata.packed_padding is None
        and metadata.cross_query_mask is None
    )


def estimated_waves(widths, rows, sms):
    """SM90 candidate-ordering estimate, not an observed kernel launch grid."""
    tiles = math.ceil(rows / 128)
    return sum(math.ceil(tiles * width / sms) for width in widths)


def candidates(heads, rows, sms):
    """Rank all small-first pairs by estimated waves; measurements decide.

    Bound the experiment to at most sixteen pairs. Larger head counts retain
    native attention until a separately validated search budget is available.
    """
    if heads < 2 or heads > 32 or estimated_waves((heads,), rows, sms) < 2:
        return ()
    pairs = [(small, heads - small) for small in range(1, heads // 2 + 1)]
    return tuple(sorted(pairs, key=lambda parts: (estimated_waves(parts, rows, sms), parts[0])))
