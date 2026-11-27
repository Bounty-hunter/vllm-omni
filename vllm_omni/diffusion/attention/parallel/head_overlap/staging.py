# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage this rank's BF16 Q/K/V and replicated prefix rows for one CE group."""

from itertools import accumulate

from vllm.triton_utils import tl, triton


@triton.jit(
    do_not_specialize=["sources", "destinations", "rows", "ends"],
    do_not_specialize_on_alignment=["rows", "ends"],
)
def _stage(
    sources,
    destinations,
    widths: tl.constexpr,
    pitches: tl.constexpr,
    rows,
    ends,
    column: tl.constexpr,
    wide_offsets: tl.constexpr,
    block: tl.constexpr,
):
    tile = tl.program_id(0)
    for i in tl.static_range(len(widths)):
        if (tile >= ends[i]) & (tile < ends[i + 1]):
            local_tile = tile - ends[i]
            # A fused V row can have three times the packed pitch. Large video
            # layouts can cross 2^31 elements even when one destination cannot.
            if wide_offsets[i]:
                local_tile = local_tile.to(tl.int64)
            offset = local_tile * block + tl.arange(0, block)
            row, col = offset // widths[i], offset % widths[i]
            src = sources[i].to(tl.pointer_type(tl.uint16))
            dst = destinations[i].to(tl.pointer_type(tl.uint16))
            value = tl.load(src + row * pitches[i] + column + col, row < rows[i], other=0)
            tl.store(dst + offset, value, row < rows[i])


def prepare_staging(targets, pitches, column):
    """Prepare a launcher without specializing on query or prefix KV length.

    Widths and pitches retain constant-division optimization. Runtime row counts
    and tile boundaries let another prompt reuse the compiled kernel, including
    across alignment boundaries. Only crossing the 32-bit address limit changes
    the index-width specialization.
    """
    block = 1024
    destinations = tuple(t.data_ptr() for t in targets)
    widths = tuple(t.shape[2] * t.shape[3] for t in targets)
    rows = tuple(t.shape[1] for t in targets)
    ends = tuple(accumulate((triton.cdiv(w * r, block) for w, r in zip(widths, rows, strict=True)), initial=0))
    wide_offsets = tuple(
        r * w + block >= 2**31 or r * p + column + w >= 2**31 for r, w, p in zip(rows, widths, pitches, strict=True)
    )
    args = (destinations, widths, pitches, rows, ends, column, wide_offsets, block)
    grid = (ends[-1], 1, 1)
    # A mapped workspace address has the same 64-bit argument type as each
    # incoming tensor pointer. warmup loads the module without a probe launch.
    kernel = _stage.warmup(destinations, *args, grid=grid, num_warps=4)
    launch = kernel[grid]

    def stage(sources):
        launch(sources, *args)

    return stage
