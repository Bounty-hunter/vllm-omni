# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage this rank's BF16 Q/K/V and replicated prefix rows for one CE group."""

from vllm.triton_utils import tl, triton


@triton.jit(
    do_not_specialize=["sources", "destinations", "query_rows", "prefix_rows", "query_tiles", "prefix_tiles"],
    do_not_specialize_on_alignment=["query_rows", "prefix_rows", "query_tiles", "prefix_tiles"],
)
def _stage(
    sources,
    destinations,
    width: tl.constexpr,
    pitches: tl.constexpr,
    query_rows,
    prefix_rows,
    query_tiles,
    prefix_tiles,
    column: tl.constexpr,
    wide_offsets: tl.constexpr,
    block: tl.constexpr,
):
    tile = tl.program_id(0)
    for i in tl.static_range(len(pitches)):
        if i < 3:
            first, count, rows = i * query_tiles, query_tiles, query_rows
        else:
            first, count, rows = 3 * query_tiles + (i - 3) * prefix_tiles, prefix_tiles, prefix_rows
        if (tile >= first) & (tile < first + count):
            local_tile = tile - first
            # A fused V row can have three times the packed pitch. Large video
            # layouts can cross 2^31 elements even when one destination cannot.
            if wide_offsets[i]:
                local_tile = local_tile.to(tl.int64)
            offset = local_tile * block + tl.arange(0, block)
            row, col = offset // width, offset % width
            src = sources[i].to(tl.pointer_type(tl.uint16))
            dst = destinations[i].to(tl.pointer_type(tl.uint16))
            value = tl.load(src + row * pitches[i] + column + col, row < rows, other=0)
            tl.store(dst + offset, value, row < rows)


def prepare_staging(targets, pitches, column):
    """Prepare a launcher without specializing on query or prefix KV length.

    Width and pitches retain constant-division optimization. Runtime row counts
    and tile boundaries let another prompt reuse the compiled kernel, including
    across alignment boundaries. Only crossing the 32-bit address limit changes
    the index-width specialization.
    """
    block = 1024
    destinations = tuple(t.data_ptr() for t in targets)
    width = targets[0].shape[2] * targets[0].shape[3]
    rows = tuple(t.shape[1] for t in targets)
    query_rows, prefix_rows = rows[0], rows[3] if len(rows) == 5 else 0
    query_tiles, prefix_tiles = (triton.cdiv(width * r, block) for r in (query_rows, prefix_rows))
    wide_offsets = tuple(
        r * width + block >= 2**31 or r * p + column + width >= 2**31 for r, p in zip(rows, pitches, strict=True)
    )
    # Scalars also avoid Triton versions that still specialize tuple elements
    # on alignment despite do_not_specialize_on_alignment on the tuple itself.
    args = (
        destinations,
        width,
        pitches,
        query_rows,
        prefix_rows,
        query_tiles,
        prefix_tiles,
        column,
        wide_offsets,
        block,
    )
    grid = (3 * query_tiles + 2 * prefix_tiles, 1, 1)
    # A mapped workspace address has the same 64-bit argument type as each
    # incoming tensor pointer. warmup loads the module without a probe launch.
    kernel = _stage.warmup(destinations, *args, grid=grid, num_warps=4)
    launch = kernel[grid]

    def stage(sources):
        launch(sources, *args)

    return stage
