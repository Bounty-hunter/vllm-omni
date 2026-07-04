# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

import torch
from torch import Tensor

from vllm_omni.diffusion.distributed.overlap.stream import CommStreamManager

ComputeInput = TypeVar("ComputeInput")
CommState = TypeVar("CommState")

LaunchPreFn = Callable[[int], CommState]
LaunchPostFn = Callable[[int, Tensor], CommState]
FinalizePreFn = Callable[[int, CommState], ComputeInput]
ComputeFn = Callable[[int, ComputeInput], Tensor]
FinalizePostFn = Callable[[int, CommState], Tensor]
MergeFn = Callable[[Sequence[Tensor]], Tensor]


class ChunkOverlapScheduler(Generic[ComputeInput, CommState]):
    """Generic chunk pipeline scheduler for comm/compute overlap.

    Each chunk goes through three caller-defined phases:

    - **stage0 (pre-comm)**: ``launch_pre`` enqueues async comm; ``finalize_pre`` waits
      and returns compute inputs (e.g. pre-attention AllToAll).
    - **stage1 (compute)**: ``compute`` runs on the main stream (e.g. FlashAttention).
    - **stage2 (post-comm)**: ``launch_post`` enqueues async comm; ``finalize_post``
      waits and returns the chunk result (e.g. post-attention AllToAll).

    Overlap schedule (``num_chunks >= 2``):

    .. code-block:: text

        comm:  pre(c0) | pre(c1)     | post(c0) | post(c1)
        main:  wait    | FA(c0)      | FA(c1)   | wait/merge
                      └ FA(c0) ∥ pre(c1) ─┘└ FA(c1) ∥ post(c0) ─┘

    Callers define chunking and comm/compute details; this class only orchestrates
    CUDA stream dependencies and iteration order.
    """

    def __init__(self, num_chunks: int, *, device: torch.device | int) -> None:
        if num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
        self._num_chunks = num_chunks
        self._device = torch.device(device)
        self._compute_stream = CommStreamManager.get_compute_stream(self._device)
        self._comm_stream = CommStreamManager.get(self._device)

    @property
    def num_chunks(self) -> int:
        return self._num_chunks

    @property
    def compute_stream(self) -> torch.cuda.Stream:
        return self._compute_stream

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        return self._comm_stream

    def _mark_comm_done(self) -> torch.Event:
        ready = torch.Event()
        with torch.cuda.stream(self._comm_stream):
            ready.record(self._comm_stream)
        return ready

    def _launch_pre(self, chunk_index: int, launch_pre: Callable[[int], CommState]) -> tuple[torch.Event, CommState]:
        state = launch_pre(chunk_index)
        return self._mark_comm_done(), state

    def _launch_post(
        self,
        chunk_index: int,
        compute_output: Tensor,
        launch_post: Callable[[int, Tensor], CommState],
    ) -> tuple[torch.Event, CommState]:
        state = launch_post(chunk_index, compute_output)
        return self._mark_comm_done(), state

    def run(
        self,
        *,
        launch_pre: Callable[[int], CommState],
        finalize_pre: Callable[[int, CommState], ComputeInput],
        compute: Callable[[int, ComputeInput], Tensor],
        launch_post: Callable[[int, Tensor], CommState],
        finalize_post: Callable[[int, CommState], Tensor],
        merge: MergeFn,
    ) -> Tensor:
        if self._num_chunks == 1:
            pre_ready, pre_state = self._launch_pre(0, launch_pre)
            self._compute_stream.wait_event(pre_ready)
            compute_input = finalize_pre(0, pre_state)
            compute_output = compute(0, compute_input)
            post_ready, post_state = self._launch_post(0, compute_output, launch_post)
            self._compute_stream.wait_event(post_ready)
            return finalize_post(0, post_state)

        pre_ready, pre_state = self._launch_pre(0, launch_pre)
        self._compute_stream.wait_event(pre_ready)
        compute_inputs: list[ComputeInput] = [finalize_pre(0, pre_state)]

        post_states: list[tuple[torch.Event, CommState]] = []
        for chunk_index in range(self._num_chunks):
            pending_pre: tuple[torch.Event, CommState] | None = None
            if chunk_index + 1 < self._num_chunks:
                pending_pre = self._launch_pre(chunk_index + 1, launch_pre)

            compute_output = compute(chunk_index, compute_inputs[chunk_index])
            post_states.append(self._launch_post(chunk_index, compute_output, launch_post))

            if pending_pre is not None:
                pending_ready, pending_state = pending_pre
                self._compute_stream.wait_event(pending_ready)
                compute_inputs.append(finalize_pre(chunk_index + 1, pending_state))

        post_outputs: list[Tensor] = []
        for chunk_index, (post_ready, post_state) in enumerate(post_states):
            self._compute_stream.wait_event(post_ready)
            post_outputs.append(finalize_post(chunk_index, post_state))

        return merge(post_outputs)
