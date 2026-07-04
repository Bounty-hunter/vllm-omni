# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionParallelConfig


@dataclass(frozen=True, slots=True)
class AttentionOverlapConfig:
    enabled: bool = False
    head_chunks: int = 2

    @property
    def active(self) -> bool:
        return self.enabled and self.head_chunks > 1


def resolve_attention_overlap_config(
    parallel_config: DiffusionParallelConfig | None,
    *,
    num_heads: int | None = None,
) -> AttentionOverlapConfig:
    if parallel_config is None:
        return AttentionOverlapConfig()

    enabled = bool(getattr(parallel_config, "attention_comm_overlap", False))
    head_chunks = int(getattr(parallel_config, "attention_head_chunks", 2))
    if head_chunks < 1:
        head_chunks = 1

    if not enabled:
        return AttentionOverlapConfig(enabled=False, head_chunks=head_chunks)

    ulysses_degree = int(getattr(parallel_config, "ulysses_degree", 1))
    if ulysses_degree <= 1:
        return AttentionOverlapConfig(enabled=False, head_chunks=head_chunks)

    if num_heads is not None and head_chunks > 1:
        divisor = ulysses_degree * head_chunks
        if num_heads % divisor != 0:
            raise ValueError(
                "attention_head_chunks requires num_heads divisible by "
                f"ulysses_degree * attention_head_chunks, but got "
                f"num_heads={num_heads}, ulysses_degree={ulysses_degree}, "
                f"attention_head_chunks={head_chunks}."
            )

    return AttentionOverlapConfig(enabled=True, head_chunks=head_chunks)
