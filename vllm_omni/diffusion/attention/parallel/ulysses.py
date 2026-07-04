# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.distributed.comm import AllToAll4DPending, SeqAllToAll4D, all_to_all_4D_finalize, all_to_all_4D_launch
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator
from vllm_omni.diffusion.distributed.overlap import (
    AttentionOverlapConfig,
    ChunkOverlapScheduler,
    CommStreamContext,
    get_active_parallel_config,
    resolve_attention_overlap_config,
)
from vllm_omni.diffusion.forward_context import get_ulysses_mode


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _positive_divisors(n: int) -> list[int]:
    if n <= 0:
        return []
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs)


@torch.compiler.disable
def _all_gather_int(pg: dist.ProcessGroup, value: int, *, device: torch.device) -> list[int]:
    """All-gather a scalar int across pg.

    Note: we use a device tensor so this works for NCCL subgroups (e.g. Ulysses/Ring).
    """
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return [int(value)]

    t = torch.tensor([int(value)], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t, group=pg)
    return [int(x.item()) for x in gathered]


def _ulysses_all_to_all_any_qkv(
    pg: dist.ProcessGroup,
    x: torch.Tensor,  # (B, S_local, H, D)
    *,
    seq_lens: list[int],
    use_sync: bool,
) -> tuple[torch.Tensor, int]:
    """UAA forward all-to-all: (B, S_local, H, D) -> (B, S_global, H_local, D).

    Returns:
        (resharded, orig_head_cnt)
    """
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return x, int(x.shape[2])

    bsz, s_local, head_cnt, head_dim = x.shape
    orig_head_cnt = int(head_cnt)
    padded_head_cnt = _ceil_div(orig_head_cnt, world_size) * world_size
    head_pad = padded_head_cnt - orig_head_cnt
    if head_pad:
        x = F.pad(x, (0, 0, 0, head_pad))

    head_cnt_local = padded_head_cnt // world_size

    # (B, S_local, H, D) -> (world_size, S_local, B, H_local, D)
    x_t = x.reshape(bsz, s_local, world_size, head_cnt_local, head_dim).permute(2, 1, 0, 3, 4).contiguous()
    # (world_size, S_local, B, H_local, D) -> (world_size * S_local, B, H_local, D)
    x_t = x_t.flatten(0, 1)

    input_split_sizes = [s_local] * world_size
    output_split_sizes = seq_lens
    s_global = int(sum(output_split_sizes))

    out = torch.empty((s_global, bsz, head_cnt_local, head_dim), device=x.device, dtype=x.dtype)
    dist.all_to_all_single(
        out,
        x_t,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=pg,
    )
    if use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()

    # (S_global, B, H_local, D) -> (B, S_global, H_local, D)
    out = out.permute(1, 0, 2, 3).contiguous()
    return out, orig_head_cnt


def _ulysses_all_to_all_any_o(
    pg: dist.ProcessGroup,
    x: torch.Tensor,  # (B, S_global, H_local, D)
    *,
    seq_lens: list[int],
    local_seq_len: int,
    orig_head_cnt: int,
    use_sync: bool,
) -> torch.Tensor:
    """UAA reverse all-to-all: (B, S_global, H_local, D) -> (B, S_local, H, D)."""
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return x

    bsz, s_global, head_cnt_local, head_dim = x.shape
    s_local = int(local_seq_len)

    # (B, S_global, H_local, D) -> (S_global, B, H_local, D)
    x_t = x.permute(1, 0, 2, 3).contiguous()

    input_split_sizes = seq_lens
    output_split_sizes = [s_local] * world_size

    out = torch.empty((world_size * s_local, bsz, head_cnt_local, head_dim), device=x.device, dtype=x.dtype)
    dist.all_to_all_single(
        out,
        x_t,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=pg,
    )
    if use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()

    # (world_size * S_local, B, H_local, D) -> (B, S_local, H, D)
    out = out.reshape(world_size, s_local, bsz, head_cnt_local, head_dim).permute(2, 1, 0, 3, 4).contiguous()
    out = out.reshape(bsz, s_local, world_size * head_cnt_local, head_dim)

    if out.shape[2] != orig_head_cnt:
        out = out[:, :, :orig_head_cnt, :].contiguous()
    return out


@dataclass(frozen=True, slots=True)
class _UlyssesCtx(ParallelAttentionContext):
    """Per-forward context for Ulysses sequence-parallel attention."""

    ulysses_pg: dist.ProcessGroup
    scatter_idx: int
    gather_idx: int
    use_sync: bool
    joint_len: int = 0
    joint_strategy: str = "front"
    # UAA (Ulysses Anything Attention) metadata
    use_uaa: bool = False
    uaa_seq_lens: tuple[int, ...] = ()
    uaa_local_seq_len: int = 0
    orig_head_cnt: int = 0
    joint_orig_head_cnt: int = 0


@dataclass(frozen=True, slots=True)
class _JointOverlapState:
    """Rank-local joint Q/K/V prepared for head-chunk overlap."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    joint_len: int
    joint_strategy: str
    heads_per_rank: int


@dataclass(frozen=True, slots=True)
class _PostChunkState:
    image_pending: AllToAll4DPending
    joint_output: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ChunkPostResult:
    image: torch.Tensor
    joint: torch.Tensor


class UlyssesParallelAttention:
    """Ulysses sequence-parallel strategy (all-to-all over seq/head dims).

    This preserves the semantics previously implemented in
    `Attention._forward_ulysses`:
    - If `AttentionMetadata.joint_*` is provided, joint_query/key/value are
      concatenated *after* all-to-all.
    - joint_key/value are assumed to be replicated across SP ranks and are sliced
      by ulysses head rank before concatenation.
    """

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool,
    ) -> None:
        self._sp_group = sp_group
        self._ulysses_pg = sp_group.ulysses_group
        self._scatter_idx = scatter_idx
        self._gather_idx = gather_idx
        self._use_sync = use_sync

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ulysses"

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        mode = get_ulysses_mode(default="strict")
        joint_tensor_query = joint_tensor_key = joint_tensor_value = None
        joint_strategy = "front"
        joint_len = 0
        joint_orig_head_cnt = 0

        if attn_metadata is not None:
            joint_tensor_query = attn_metadata.joint_query
            joint_tensor_key = attn_metadata.joint_key
            joint_tensor_value = attn_metadata.joint_value
            joint_strategy = attn_metadata.joint_strategy

        is_joint = False
        if joint_tensor_query is not None and joint_tensor_key is not None and joint_tensor_value is not None:
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supported."
                    f" supported joint strategy: {supported_joint_strategy}"
                )

            # Slice joint_query for this Ulysses rank
            # joint_query is (B, S, H, D). We split H (dim 2).
            ulysses_world_size = self._sp_group.ulysses_world_size
            ulysses_rank = self._sp_group.ulysses_rank
            joint_head_cnt = int(joint_tensor_query.shape[-2])
            joint_orig_head_cnt = joint_head_cnt

            if mode == "advanced_uaa":
                padded_joint_head_cnt = _ceil_div(joint_head_cnt, ulysses_world_size) * ulysses_world_size
                joint_head_pad = padded_joint_head_cnt - joint_head_cnt
                if joint_head_pad:
                    joint_tensor_query = F.pad(joint_tensor_query, (0, 0, 0, joint_head_pad))
                    joint_tensor_key = F.pad(joint_tensor_key, (0, 0, 0, joint_head_pad))
                    joint_tensor_value = F.pad(joint_tensor_value, (0, 0, 0, joint_head_pad))
                joint_head_cnt = padded_joint_head_cnt
            else:
                if joint_head_cnt % ulysses_world_size != 0:
                    supported = _positive_divisors(joint_head_cnt)
                    raise ValueError(
                        "Ulysses-SP strict mode requires joint head_cnt divisible by ulysses_degree. "
                        f"joint_head_cnt={joint_head_cnt}, ulysses_degree={ulysses_world_size}. "
                        f"Try ulysses_degree in {supported}, or set ulysses_mode='advanced_uaa'."
                    )

            attn_heads_per_ulysses_rank = joint_head_cnt // ulysses_world_size

            # Note: We use the same heads for Q/K/V
            joint_tensor_query = joint_tensor_query[
                ...,
                attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
                :,
            ]

            joint_len = joint_tensor_query.shape[1]

            is_joint = True
        elif joint_tensor_query is None and joint_tensor_key is None and joint_tensor_value is None:
            pass
        else:
            raise ValueError("joint_query, joint_key, and joint_value should be None or not None simultaneously.")

        if is_joint:
            # Slice joint key/value heads for this ulysses rank.
            # Using same slicing logic as query
            attn_heads_per_ulysses_rank_kv = joint_tensor_key.shape[-2] // ulysses_world_size

            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank_kv * ulysses_rank : attn_heads_per_ulysses_rank_kv * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank_kv * ulysses_rank : attn_heads_per_ulysses_rank_kv * (ulysses_rank + 1),
                :,
            ]

            # Update metadata with sliced tensors so Ring attention can use them if needed
            if attn_metadata is not None:
                attn_metadata.joint_key = joint_tensor_key
                attn_metadata.joint_value = joint_tensor_value

        ulysses_world_size = self._sp_group.ulysses_world_size
        if mode == "advanced_uaa":
            if self._scatter_idx != 2 or self._gather_idx != 1:
                raise ValueError(
                    "ulysses_mode='advanced_uaa' currently only supports scatter_idx=2, gather_idx=1 "
                    f"(got scatter_idx={self._scatter_idx}, gather_idx={self._gather_idx})."
                )

            local_seq_len = int(query.shape[1])
            seq_lens = _all_gather_int(self._ulysses_pg, local_seq_len, device=query.device)
            s_global = int(sum(seq_lens))

            # In hybrid Ulysses+Ring, Ring attention uses P2P send/recv with fixed-shape
            # buffers. This requires all ring ranks to have the same seq_len after the
            # Ulysses all-to-all (i.e. per-ring-rank S_global must match).
            if self._sp_group.ring_world_size > 1:
                ring_s_globals = _all_gather_int(self._sp_group.ring_group, s_global, device=query.device)
                if len(set(ring_s_globals)) != 1:
                    raise ValueError(
                        "ulysses_mode='advanced_uaa' with hybrid Ulysses+Ring requires the "
                        "post-Ulysses seq_len to be equal across ring ranks, but got "
                        f"{ring_s_globals} (ring_degree={self._sp_group.ring_world_size}). "
                        "This typically means the input sequence was not evenly shardable across the ring. "
                        "Try setting ring_degree=1, or choose a sequence length divisible by ring_degree."
                    )
            query, orig_head_cnt = _ulysses_all_to_all_any_qkv(
                self._ulysses_pg, query, seq_lens=seq_lens, use_sync=self._use_sync
            )
            key, _ = _ulysses_all_to_all_any_qkv(self._ulysses_pg, key, seq_lens=seq_lens, use_sync=self._use_sync)
            value, _ = _ulysses_all_to_all_any_qkv(self._ulysses_pg, value, seq_lens=seq_lens, use_sync=self._use_sync)
        else:
            # Strict mode: fail fast with actionable errors for head divisibility.
            for name, t in (("query", query), ("key", key), ("value", value)):
                head_cnt = int(t.shape[2])
                if head_cnt % ulysses_world_size != 0:
                    supported = _positive_divisors(head_cnt)
                    raise ValueError(
                        "Ulysses-SP strict mode requires head_cnt divisible by ulysses_degree. "
                        f"{name}_head_cnt={head_cnt}, ulysses_degree={ulysses_world_size}. "
                        f"Try ulysses_degree in {supported}, or set ulysses_mode='advanced_uaa'."
                    )

            # (bs, seq_len/P, head_cnt, head_size) -> (bs, seq_len, head_cnt/P, head_size)
            query = SeqAllToAll4D.apply(self._ulysses_pg, query, self._scatter_idx, self._gather_idx, self._use_sync)
            key = SeqAllToAll4D.apply(self._ulysses_pg, key, self._scatter_idx, self._gather_idx, self._use_sync)
            value = SeqAllToAll4D.apply(self._ulysses_pg, value, self._scatter_idx, self._gather_idx, self._use_sync)
            seq_lens = []
            local_seq_len = 0
            orig_head_cnt = 0

        if is_joint:
            # Concatenate joint query AFTER AllToAll
            # Image query is now (B, S, H/P, D). Joint query is (B, S_txt, H/P, D).
            # This is dimensionally consistent.
            if joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)

        # Check if Ring Attention is also active (Hybrid mode)
        # If Ring is active, we should NOT concatenate joint_key/value to k/v here.
        # Instead, they should remain in attn_metadata and be passed to the Ring kernel.
        use_ring = self._sp_group.ring_world_size > 1

        if is_joint and not use_ring:
            # Concatenate joint key/value after all-to-all ONLY for pure Ulysses (Local Attention).
            if joint_strategy == "front":
                key = torch.cat([joint_tensor_key, key], dim=1)
                value = torch.cat([joint_tensor_value, value], dim=1)
            else:  # "rear"
                key = torch.cat([key, joint_tensor_key], dim=1)
                value = torch.cat([value, joint_tensor_value], dim=1)

        ctx = _UlyssesCtx(
            name=self.name,
            ulysses_pg=self._ulysses_pg,
            scatter_idx=self._scatter_idx,
            gather_idx=self._gather_idx,
            use_sync=self._use_sync,
            joint_len=joint_len,
            joint_strategy=joint_strategy,
            use_uaa=(mode == "advanced_uaa"),
            uaa_seq_lens=tuple(int(x) for x in seq_lens) if mode == "advanced_uaa" else (),
            uaa_local_seq_len=int(local_seq_len) if mode == "advanced_uaa" else 0,
            orig_head_cnt=int(orig_head_cnt) if mode == "advanced_uaa" else 0,
            joint_orig_head_cnt=int(joint_orig_head_cnt) if mode == "advanced_uaa" else 0,
        )
        use_2d_mask = False
        if attn_metadata is not None:
            if attn_metadata.attn_mask is not None and attn_metadata.attn_mask.ndim == 2:
                use_2d_mask = True
            if attn_metadata.joint_attn_mask is not None and attn_metadata.joint_attn_mask.ndim == 2:
                use_2d_mask = True

        if attn_metadata is not None and use_2d_mask:
            if is_joint:
                if attn_metadata.joint_attn_mask is None and attn_metadata.attn_mask is None:
                    attn_metadata.attn_mask = None
                else:
                    if attn_metadata.attn_mask is None:
                        attn_metadata.attn_mask = torch.ones(
                            [query.shape[0], query.shape[1] - attn_metadata.joint_attn_mask.shape[1]],
                            dtype=torch.bool,
                            device=query.device,
                        )
                    elif attn_metadata.joint_attn_mask is None:
                        attn_metadata.joint_attn_mask = torch.ones(
                            [query.shape[0], query.shape[1] - attn_metadata.attn_mask.shape[1]],
                            dtype=torch.bool,
                            device=query.device,
                        )
                    attn_metadata.attn_mask = (
                        torch.cat([attn_metadata.joint_attn_mask, attn_metadata.attn_mask], dim=1)
                        if joint_strategy == "front"
                        else torch.cat([attn_metadata.attn_mask, attn_metadata.joint_attn_mask], dim=1)
                    )

            if attn_metadata.attn_mask is not None:
                # the final attn_mask is ready, the length should be aligedn with query length
                assert attn_metadata.attn_mask.shape[1] == query.shape[1], (
                    f"attn_mask length: {attn_metadata.attn_mask.shape[1]} != query length: {query.shape[1]}"
                )
                attn_metadata.attn_mask = attn_metadata.attn_mask.bool().contiguous()
        return query, key, value, attn_metadata, ctx

    def post_attention(self, attn_output: torch.Tensor, ctx: ParallelAttentionContext | None) -> torch.Tensor:
        assert isinstance(ctx, _UlyssesCtx), f"Unexpected ctx type: {type(ctx)!r}"

        if ctx.joint_len > 0:
            joint_len = ctx.joint_len

            if ctx.joint_strategy == "front":
                output_joint = attn_output[:, :joint_len]
                output_img = attn_output[:, joint_len:]
            else:
                output_img = attn_output[:, :-joint_len]
                output_joint = attn_output[:, -joint_len:]

            # 1. Process Image part: Standard Ulysses Reverse (AllToAll)
            # (bs, seq_len, head_cnt/P, head_size) -> (bs, seq_len/P, head_cnt, head_size)
            # SeqAllToAll4D handles: Scatter gather_idx, Gather scatter_idx.
            # Forward: Scatter 2 (H), Gather 1 (S).
            # Reverse: Scatter 1 (S), Gather 2 (H).
            if ctx.use_uaa:
                output_img = _ulysses_all_to_all_any_o(
                    ctx.ulysses_pg,
                    output_img,
                    seq_lens=list(ctx.uaa_seq_lens),
                    local_seq_len=ctx.uaa_local_seq_len,
                    orig_head_cnt=ctx.orig_head_cnt,
                    use_sync=ctx.use_sync,
                )
            else:
                output_img = SeqAllToAll4D.apply(
                    ctx.ulysses_pg, output_img, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
                )

            # 2. Process Joint part: AllGather on Heads
            # Input: (B, JointLen, H/P, D). Output: (B, JointLen, H, D).
            # AllGather along dim 2.
            # Ensure tensor is contiguous for all_gather (slicing may create non-contiguous views)
            output_joint = output_joint.contiguous()
            gathered_joint = [torch.zeros_like(output_joint) for _ in range(dist.get_world_size(ctx.ulysses_pg))]
            dist.all_gather(gathered_joint, output_joint, group=ctx.ulysses_pg)
            output_joint = torch.cat(gathered_joint, dim=2)
            if ctx.use_uaa and ctx.joint_orig_head_cnt > 0 and output_joint.shape[2] != ctx.joint_orig_head_cnt:
                output_joint = output_joint[:, :, : ctx.joint_orig_head_cnt, :].contiguous()

            # 3. Recombine
            if ctx.joint_strategy == "front":
                return torch.cat([output_joint, output_img], dim=1)
            else:
                return torch.cat([output_img, output_joint], dim=1)

        # Standard Ulysses Reverse
        if ctx.use_uaa:
            return _ulysses_all_to_all_any_o(
                ctx.ulysses_pg,
                attn_output,
                seq_lens=list(ctx.uaa_seq_lens),
                local_seq_len=ctx.uaa_local_seq_len,
                orig_head_cnt=ctx.orig_head_cnt,
                use_sync=ctx.use_sync,
            )
        return SeqAllToAll4D.apply(ctx.ulysses_pg, attn_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync)

    def _resolve_overlap_config(self, num_heads: int) -> AttentionOverlapConfig:
        return resolve_attention_overlap_config(get_active_parallel_config(), num_heads=num_heads)

    def _heads_per_chunk(self, head_count: int, num_chunks: int) -> int:
        if head_count % num_chunks != 0:
            raise ValueError(f"Head count {head_count} is not divisible by num_chunks={num_chunks}.")
        return head_count // num_chunks

    def _slice_head_chunk(self, tensor: torch.Tensor, chunk_index: int, heads_per_chunk: int) -> torch.Tensor:
        start = chunk_index * heads_per_chunk
        end = start + heads_per_chunk
        return tensor[:, :, start:end, :].contiguous()

    def _prepare_joint_overlap_state(
        self,
        attn_metadata: AttentionMetadata | None,
        *,
        num_chunks: int,
    ) -> _JointOverlapState | None:
        if attn_metadata is None:
            return None
        joint_query = attn_metadata.joint_query
        joint_key = attn_metadata.joint_key
        joint_value = attn_metadata.joint_value
        if joint_query is None and joint_key is None and joint_value is None:
            return None
        if joint_query is None or joint_key is None or joint_value is None:
            raise ValueError("joint_query, joint_key, and joint_value should be None or not None simultaneously.")

        joint_strategy = attn_metadata.joint_strategy
        if joint_strategy not in ("front", "rear"):
            raise ValueError(
                f"joint_strategy: {joint_strategy} not supported. supported joint strategy: ['front', 'rear']"
            )

        ulysses_world_size = self._sp_group.ulysses_world_size
        ulysses_rank = self._sp_group.ulysses_rank
        joint_head_cnt = int(joint_query.shape[-2])
        if joint_head_cnt % (ulysses_world_size * num_chunks) != 0:
            supported = _positive_divisors(joint_head_cnt)
            raise ValueError(
                "Ulysses head-chunk overlap requires joint head_cnt divisible by "
                f"ulysses_degree * attention_head_chunks, but got joint_head_cnt={joint_head_cnt}, "
                f"ulysses_degree={ulysses_world_size}, attention_head_chunks={num_chunks}. "
                f"Try attention_head_chunks in {supported}."
            )

        heads_per_rank = joint_head_cnt // ulysses_world_size
        rank_head_start = heads_per_rank * ulysses_rank
        rank_head_end = heads_per_rank * (ulysses_rank + 1)
        joint_query = joint_query[..., rank_head_start:rank_head_end, :]
        joint_key = joint_key[..., rank_head_start:rank_head_end, :]
        joint_value = joint_value[..., rank_head_start:rank_head_end, :]

        return _JointOverlapState(
            query=joint_query,
            key=joint_key,
            value=joint_value,
            joint_len=int(joint_query.shape[1]),
            joint_strategy=joint_strategy,
            heads_per_rank=heads_per_rank,
        )

    def _apply_overlap_joint_mask(
        self,
        attn_metadata: AttentionMetadata | None,
        joint_state: _JointOverlapState | None,
        query: torch.Tensor,
    ) -> None:
        if attn_metadata is None or joint_state is None:
            return

        use_2d_mask = False
        if attn_metadata.attn_mask is not None and attn_metadata.attn_mask.ndim == 2:
            use_2d_mask = True
        if attn_metadata.joint_attn_mask is not None and attn_metadata.joint_attn_mask.ndim == 2:
            use_2d_mask = True
        if not use_2d_mask:
            return

        if attn_metadata.joint_attn_mask is None and attn_metadata.attn_mask is None:
            attn_metadata.attn_mask = None
            return

        if attn_metadata.attn_mask is None:
            attn_metadata.attn_mask = torch.ones(
                [query.shape[0], query.shape[1] - attn_metadata.joint_attn_mask.shape[1]],
                dtype=torch.bool,
                device=query.device,
            )
        elif attn_metadata.joint_attn_mask is None:
            attn_metadata.joint_attn_mask = torch.ones(
                [query.shape[0], query.shape[1] - attn_metadata.attn_mask.shape[1]],
                dtype=torch.bool,
                device=query.device,
            )

        if joint_state.joint_strategy == "front":
            attn_metadata.attn_mask = torch.cat(
                [attn_metadata.joint_attn_mask, attn_metadata.attn_mask],
                dim=1,
            )
        else:
            attn_metadata.attn_mask = torch.cat(
                [attn_metadata.attn_mask, attn_metadata.joint_attn_mask],
                dim=1,
            )
        assert attn_metadata.attn_mask.shape[1] == query.shape[1], (
            f"attn_mask length: {attn_metadata.attn_mask.shape[1]} != query length: {query.shape[1]}"
        )
        attn_metadata.attn_mask = attn_metadata.attn_mask.bool().contiguous()

    def _concat_joint_and_image_qkv(
        self,
        joint_state: _JointOverlapState,
        chunk_index: int,
        num_chunks: int,
        image_q: torch.Tensor,
        image_k: torch.Tensor,
        image_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joint_heads_per_chunk = self._heads_per_chunk(joint_state.heads_per_rank, num_chunks)
        joint_q = self._slice_head_chunk(joint_state.query, chunk_index, joint_heads_per_chunk)
        joint_k = self._slice_head_chunk(joint_state.key, chunk_index, joint_heads_per_chunk)
        joint_v = self._slice_head_chunk(joint_state.value, chunk_index, joint_heads_per_chunk)

        if joint_state.joint_strategy == "rear":
            query = torch.cat([image_q, joint_q], dim=1)
            key = torch.cat([image_k, joint_k], dim=1)
            value = torch.cat([image_v, joint_v], dim=1)
        else:
            query = torch.cat([joint_q, image_q], dim=1)
            key = torch.cat([joint_k, image_k], dim=1)
            value = torch.cat([joint_v, image_v], dim=1)
        return query, key, value

    def _split_joint_and_image_output(
        self,
        joint_state: _JointOverlapState,
        attn_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_len = joint_state.joint_len
        if joint_state.joint_strategy == "front":
            return attn_output[:, :joint_len], attn_output[:, joint_len:]
        return attn_output[:, :-joint_len], attn_output[:, -joint_len:]

    def _all_gather_joint_heads(self, joint_output: torch.Tensor) -> torch.Tensor:
        output_joint = joint_output.contiguous()
        gathered_joint = [torch.zeros_like(output_joint) for _ in range(dist.get_world_size(self._ulysses_pg))]
        dist.all_gather(gathered_joint, output_joint, group=self._ulysses_pg)
        return torch.cat(gathered_joint, dim=2)

    def supports_head_chunk_overlap(
        self,
        query: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> bool:
        overlap_cfg = self._resolve_overlap_config(int(query.shape[2]))
        if not overlap_cfg.active:
            return False
        if get_ulysses_mode(default="strict") != "strict":
            return False
        if self._sp_group.ring_world_size > 1:
            return False
        if self._use_sync:
            return False

        if attn_metadata is not None and attn_metadata.joint_query is not None:
            joint_head_cnt = int(attn_metadata.joint_query.shape[-2])
            divisor = self._sp_group.ulysses_world_size * overlap_cfg.head_chunks
            if joint_head_cnt % divisor != 0:
                return False
        return True

    def forward_with_head_chunk_overlap(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        overlap_cfg = self._resolve_overlap_config(int(query.shape[2]))
        num_chunks = overlap_cfg.head_chunks
        image_heads_per_chunk = self._heads_per_chunk(int(query.shape[2]), num_chunks)
        joint_state = self._prepare_joint_overlap_state(attn_metadata, num_chunks=num_chunks)

        scheduler = ChunkOverlapScheduler(num_chunks, device=query.device)
        mask_prepared = False

        def launch_pre(
            chunk_index: int,
        ) -> tuple[AllToAll4DPending, AllToAll4DPending, AllToAll4DPending]:
            return (
                all_to_all_4D_launch(
                    self._slice_head_chunk(query, chunk_index, image_heads_per_chunk),
                    self._scatter_idx,
                    self._gather_idx,
                    group=self._ulysses_pg,
                ),
                all_to_all_4D_launch(
                    self._slice_head_chunk(key, chunk_index, image_heads_per_chunk),
                    self._scatter_idx,
                    self._gather_idx,
                    group=self._ulysses_pg,
                ),
                all_to_all_4D_launch(
                    self._slice_head_chunk(value, chunk_index, image_heads_per_chunk),
                    self._scatter_idx,
                    self._gather_idx,
                    group=self._ulysses_pg,
                ),
            )

        def finalize_pre(
            chunk_index: int,
            pending: tuple[AllToAll4DPending, AllToAll4DPending, AllToAll4DPending],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            nonlocal mask_prepared
            stream = scheduler.compute_stream
            q_pending, k_pending, v_pending = pending
            image_q = all_to_all_4D_finalize(q_pending, stream=stream)
            image_k = all_to_all_4D_finalize(k_pending, stream=stream)
            image_v = all_to_all_4D_finalize(v_pending, stream=stream)

            if joint_state is None:
                return image_q, image_k, image_v

            q_chunk, k_chunk, v_chunk = self._concat_joint_and_image_qkv(
                joint_state,
                chunk_index,
                num_chunks,
                image_q,
                image_k,
                image_v,
            )
            if not mask_prepared:
                self._apply_overlap_joint_mask(attn_metadata, joint_state, q_chunk)
                mask_prepared = True
            return q_chunk, k_chunk, v_chunk

        def compute(
            chunk_index: int,
            qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            del chunk_index
            q_chunk, k_chunk, v_chunk = qkv
            return attention_fn(q_chunk, k_chunk, v_chunk)

        def launch_post(chunk_index: int, attn_out: torch.Tensor) -> AllToAll4DPending | _PostChunkState:
            del chunk_index
            if joint_state is None:
                return all_to_all_4D_launch(
                    attn_out,
                    self._gather_idx,
                    self._scatter_idx,
                    group=self._ulysses_pg,
                )

            output_joint, output_img = self._split_joint_and_image_output(joint_state, attn_out)
            return _PostChunkState(
                image_pending=all_to_all_4D_launch(
                    output_img,
                    self._gather_idx,
                    self._scatter_idx,
                    group=self._ulysses_pg,
                ),
                joint_output=output_joint,
            )

        def finalize_post(chunk_index: int, pending: AllToAll4DPending | _PostChunkState) -> torch.Tensor | _ChunkPostResult:
            del chunk_index
            stream = scheduler.compute_stream
            if joint_state is None:
                assert isinstance(pending, AllToAll4DPending)
                return all_to_all_4D_finalize(pending, stream=stream)

            assert isinstance(pending, _PostChunkState)
            image = all_to_all_4D_finalize(pending.image_pending, stream=stream)
            joint = self._all_gather_joint_heads(pending.joint_output)
            return _ChunkPostResult(image=image, joint=joint)

        def merge(outputs: Sequence[torch.Tensor | _ChunkPostResult]) -> torch.Tensor:
            if joint_state is None:
                tensors = [output for output in outputs]
                if len(tensors) == 1:
                    return tensors[0]
                return torch.cat(tensors, dim=2)

            chunk_results = [output for output in outputs]
            assert all(isinstance(result, _ChunkPostResult) for result in chunk_results)
            images = torch.cat([result.image for result in chunk_results], dim=2)
            joints = torch.cat([result.joint for result in chunk_results], dim=2)
            if joint_state.joint_strategy == "front":
                return torch.cat([joints, images], dim=1)
            return torch.cat([images, joints], dim=1)

        with CommStreamContext(device=query.device, enabled=True):
            return scheduler.run(
                launch_pre=launch_pre,
                finalize_pre=finalize_pre,
                compute=compute,
                launch_post=launch_post,
                finalize_post=finalize_post,
                merge=merge,
            )
