# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
from torch import Tensor

from vllm_omni.diffusion.distributed.comm import AllToAll4DPending, all_to_all_4D_finalize, all_to_all_4D_launch
from vllm_omni.diffusion.distributed.overlap.stream import CommStreamManager


@dataclass
class _PreChunkPending:
    query: AllToAll4DPending
    key: AllToAll4DPending
    value: AllToAll4DPending


@dataclass
class _ChunkSlot:
    pre_ready: torch.Event
    post_ready: torch.Event
    pre: _PreChunkPending | None = None
    post: AllToAll4DPending | None = None


AttentionFn = Callable[[Tensor, Tensor, Tensor], Tensor]


class HeadChunkUlyssesRunner:
    """Head-chunk Ulysses pipeline: FA(i) || pre-A2A(i+1), FA(i+1) || post-A2A(i)."""

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        *,
        scatter_idx: int,
        gather_idx: int,
        num_chunks: int,
    ) -> None:
        if num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
        self._pg = process_group
        self._scatter_idx = scatter_idx
        self._gather_idx = gather_idx
        self._num_chunks = num_chunks

    @property
    def num_chunks(self) -> int:
        return self._num_chunks

    def run(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_fn: AttentionFn,
    ) -> Tensor:
        head_count = int(query.shape[2])
        if head_count % self._num_chunks != 0:
            raise ValueError(
                f"Head count {head_count} is not divisible by num_chunks={self._num_chunks}."
            )
        chunk_heads = head_count // self._num_chunks

        compute_stream = CommStreamManager.get_compute_stream(query.device)
        comm_stream = CommStreamManager.get(query.device)
        slots = [_ChunkSlot(pre_ready=torch.Event(), post_ready=torch.Event()) for _ in range(self._num_chunks)]

        def slice_chunk(tensor: Tensor, index: int) -> Tensor:
            start = index * chunk_heads
            end = start + chunk_heads
            return tensor[:, :, start:end, :].contiguous()

        def launch_pre(index: int) -> None:
            slot = slots[index]
            slot.pre = _PreChunkPending(
                query=all_to_all_4D_launch(
                    slice_chunk(query, index),
                    self._scatter_idx,
                    self._gather_idx,
                    group=self._pg,
                ),
                key=all_to_all_4D_launch(
                    slice_chunk(key, index),
                    self._scatter_idx,
                    self._gather_idx,
                    group=self._pg,
                ),
                value=all_to_all_4D_launch(
                    slice_chunk(value, index),
                    self._scatter_idx,
                    self._gather_idx,
                    group=self._pg,
                ),
            )
            with torch.cuda.stream(comm_stream):
                slot.pre_ready.record(comm_stream)

        def finalize_pre(index: int) -> tuple[Tensor, Tensor, Tensor]:
            pending = slots[index].pre
            assert pending is not None
            return (
                all_to_all_4D_finalize(pending.query, stream=compute_stream),
                all_to_all_4D_finalize(pending.key, stream=compute_stream),
                all_to_all_4D_finalize(pending.value, stream=compute_stream),
            )

        def launch_post(index: int, attn_out: Tensor) -> None:
            slot = slots[index]
            slot.post = all_to_all_4D_launch(
                attn_out,
                self._gather_idx,
                self._scatter_idx,
                group=self._pg,
            )
            with torch.cuda.stream(comm_stream):
                slot.post_ready.record(comm_stream)

        def finalize_post(index: int) -> Tensor:
            pending = slots[index].post
            assert pending is not None
            return all_to_all_4D_finalize(pending, stream=compute_stream)

        launch_pre(0)
        compute_stream.wait_event(slots[0].pre_ready)
        chunk_qkv = [finalize_pre(0)]

        post_outputs: list[Tensor] = []
        for index in range(self._num_chunks):
            if index + 1 < self._num_chunks:
                launch_pre(index + 1)

            q_chunk, k_chunk, v_chunk = chunk_qkv[index]
            attn_out = attention_fn(q_chunk, k_chunk, v_chunk)
            launch_post(index, attn_out)

            if index + 1 < self._num_chunks:
                compute_stream.wait_event(slots[index + 1].pre_ready)
                chunk_qkv.append(finalize_pre(index + 1))

        for index in range(self._num_chunks):
            compute_stream.wait_event(slots[index].post_ready)
            post_outputs.append(finalize_post(index))

        if len(post_outputs) == 1:
            return post_outputs[0]
        return torch.cat(post_outputs, dim=2)
