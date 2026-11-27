# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Conservative dense layouts and pure head partitions for real-layer trials."""

import math


def eligible(kernel, metadata, distributed_rows):
    owner = getattr(kernel, "__self__", None)
    if getattr(owner, "causal", None) is not False or not hasattr(owner, "effective_attention"):
        return False
    _, backend, _ = owner.effective_attention()
    if backend is None or backend.get_name() != "FLASH_ATTN":
        return False
    if metadata is None:
        return True
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


def candidates(heads, rows, sms):
    """Try small/large/small heads first, then small/large, then stay native.

    This SM90 estimate is only a candidate filter, not a measured launch grid.
    Keep the smallest edge groups that do not add estimated compute waves.
    Every candidate still has to beat native CE in real-layer measurements;
    the estimate alone cannot establish that an edge hides enough transfer.
    """

    def waves(width):
        return math.ceil(math.ceil(rows / 128) * width / sms)

    choices = []
    for groups in (3, 2):
        for edge in range(1, heads // groups + 1):
            widths = (edge, heads - 2 * edge, edge) if groups == 3 else (edge, heads - edge)
            if sum(waves(width) for width in widths) <= waves(heads):
                choices.append(widths)
                break
    return tuple(choices)
