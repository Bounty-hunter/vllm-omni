# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.distributed.overlap.config import resolve_attention_overlap_config


def test_overlap_disabled_by_default():
    cfg = DiffusionParallelConfig(ulysses_degree=2)
    resolved = resolve_attention_overlap_config(cfg, num_heads=16)
    assert resolved.enabled is False
    assert resolved.active is False


def test_overlap_active_requires_flag_and_chunks():
    cfg = DiffusionParallelConfig(
        attention_comm_overlap=True,
        attention_head_chunks=2,
        ulysses_degree=2,
    )
    resolved = resolve_attention_overlap_config(cfg, num_heads=16)
    assert resolved.enabled is True
    assert resolved.active is True
    assert resolved.head_chunks == 2


def test_overlap_validates_head_divisibility():
    cfg = DiffusionParallelConfig(
        attention_comm_overlap=True,
        attention_head_chunks=2,
        ulysses_degree=2,
    )
    with pytest.raises(ValueError, match="attention_head_chunks requires"):
        resolve_attention_overlap_config(cfg, num_heads=15)


def test_overlap_supports_joint_when_divisible():
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import set_forward_context

    od_config = OmniDiffusionConfig(
        parallel_config=DiffusionParallelConfig(
            attention_comm_overlap=True,
            attention_head_chunks=2,
            ulysses_degree=2,
            sequence_parallel_size=2,
        )
    )
    query = torch.zeros(1, 4, 32, 8)
    metadata = AttentionMetadata(
        joint_query=torch.zeros(1, 3, 32, 8),
        joint_key=torch.zeros(1, 3, 32, 8),
        joint_value=torch.zeros(1, 3, 32, 8),
        joint_strategy="front",
    )

    class _FakeSPGroup:
        ulysses_world_size = 2
        ring_world_size = 1
        ulysses_rank = 0
        ulysses_group = None

    strategy = UlyssesParallelAttention(
        sp_group=_FakeSPGroup(),  # type: ignore[arg-type]
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
    )

    with set_forward_context(omni_diffusion_config=od_config), set_current_diffusion_config(od_config):
        assert strategy.supports_head_chunk_overlap(query, metadata)


def test_single_chunk_is_not_active():
    cfg = DiffusionParallelConfig(
        attention_comm_overlap=True,
        attention_head_chunks=1,
        ulysses_degree=2,
    )
    resolved = resolve_attention_overlap_config(cfg, num_heads=16)
    assert resolved.enabled is True
    assert resolved.active is False
