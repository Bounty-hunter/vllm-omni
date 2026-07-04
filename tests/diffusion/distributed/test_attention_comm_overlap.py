# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

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


def test_single_chunk_is_not_active():
    cfg = DiffusionParallelConfig(
        attention_comm_overlap=True,
        attention_head_chunks=1,
        ulysses_degree=2,
    )
    resolved = resolve_attention_overlap_config(cfg, num_heads=16)
    assert resolved.enabled is True
    assert resolved.active is False
