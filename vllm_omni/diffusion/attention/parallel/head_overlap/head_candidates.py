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


def estimated_waves(widths, rows, sms):
    """SM90 candidate-ordering estimate, not an observed kernel launch grid."""
    tiles = math.ceil(rows / 128)
    return sum(math.ceil(tiles * width / sms) for width in widths)


def candidates(heads, rows, sms):
    """Try small/large/small heads first, then small/large, then stay native.

    This SM90 estimate is only a candidate filter, not a measured launch grid.
    Measure the first five symmetric partitions, including wave-count bumps.
    This bounded experiment does not exclude a triple based on its wave cost:
    extra compute may still buy more hidden communication.
    Every candidate still has to beat native CE in real-layer measurements;
    the estimate alone cannot establish that an edge hides enough transfer.
    """

    choices = []
    whole = estimated_waves((heads,), rows, sms)
    for groups in (3, 2):
        if heads < groups or whole < groups:
            continue
        partitions = [
            (edge, heads - 2 * edge, edge) if groups == 3 else (edge, heads - edge)
            for edge in range(1, heads // groups + 1)
        ]
        if groups == 3:
            choices.extend(partitions[:5])
        else:
            choices.append(min(partitions, key=lambda parts: estimated_waves(parts, rows, sms)))
    return tuple(choices)
