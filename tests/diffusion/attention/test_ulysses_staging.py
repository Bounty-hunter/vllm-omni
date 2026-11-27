# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Changing prompt lengths must reuse staging code and copy exact head slices."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.parallel.head_overlap import staging
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "H100"}, num_cards=1)]


@pytest.mark.parametrize("widths", [(1, 15), (5, 11), (8, 8)])
@pytest.mark.parametrize("rank", [0, 1])
def test_prompt_lengths_reuse_staging_kernel(monkeypatch, widths, rank):
    device = current_omni_platform.device_type
    torch.manual_seed(23)
    packed = torch.randn(1, 129, 3, 32, 128, dtype=torch.bfloat16, device=device)
    prefix = tuple(torch.randn(1, 1025, 32, 128, dtype=torch.bfloat16, device=device) for _ in range(2))
    kernels = []
    original = staging._stage.warmup

    def observe(*args, **kwargs):
        kernel = original(*args, **kwargs)
        kernels.append(kernel)
        return kernel

    monkeypatch.setattr(staging._stage, "warmup", observe)
    for rows, kv_rows in ((127, 19), (128, 32), (129, 33), (127, 1024), (129, 1025)):
        inputs = (*packed[:, :rows].unbind(2), *(p[:, :kv_rows] for p in prefix))
        column = rank * sum(widths)
        for width in widths:
            # Offset contiguous destinations match this rank's region within
            # the larger attention input. Sources retain packed Q/K/V strides.
            targets = tuple(
                torch.empty(1, x.shape[1] + 6, width, 128, dtype=x.dtype, device=device)[:, 3 : 3 + x.shape[1]]
                for x in inputs
            )
            stage = staging.prepare_staging(targets, tuple(x.stride(1) for x in inputs), column * 128)
            stage(tuple(x.data_ptr() for x in inputs))
            for actual, source in zip(targets, inputs, strict=True):
                torch.testing.assert_close(actual, source[:, :, column : column + width], rtol=0, atol=0)
            column += width
    # Each head group reuses the exact CompiledKernel, even when lengths change
    # divisibility/alignment and the launch grid. Preparation launches no probe.
    for group in range(2):
        assert all(kernel is kernels[group] for kernel in kernels[group::2])
