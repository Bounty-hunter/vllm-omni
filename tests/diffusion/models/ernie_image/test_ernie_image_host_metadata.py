# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import (
    _attention_mask_mode_from_host_lengths,
)
from vllm_omni.diffusion.models.ernie_image.pipeline_ernie_image import ErnieImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [
        ([4, 4], "none"),
        ([4, 2], "padding"),
        ([0, 0], "padding"),
    ],
)
def test_attention_mask_mode_from_host_lengths(lengths, expected) -> None:
    assert _attention_mask_mode_from_host_lengths(lengths, batch_size=2, max_text_tokens=4) == expected


@pytest.mark.parametrize("lengths", [[4], [-1, 4], [5, 4]])
def test_attention_mask_mode_rejects_invalid_host_lengths(lengths) -> None:
    with pytest.raises(ValueError):
        _attention_mask_mode_from_host_lengths(lengths, batch_size=2, max_text_tokens=4)


def test_pad_text_returns_the_cpu_known_lengths() -> None:
    text_bth, text_lens, host_text_lens = ErnieImagePipeline._pad_text(
        [torch.ones(2, 3), torch.ones(4, 3)],
        device=torch.device("cpu"),
        dtype=torch.float32,
        text_in_dim=3,
    )

    assert text_bth.shape == (2, 4, 3)
    torch.testing.assert_close(text_lens, torch.tensor([2, 4]))
    assert host_text_lens == [2, 4]


def test_pad_text_empty_batch_keeps_host_metadata_empty() -> None:
    text_bth, text_lens, host_text_lens = ErnieImagePipeline._pad_text(
        [],
        device=torch.device("cpu"),
        dtype=torch.float32,
        text_in_dim=3,
    )

    assert text_bth.shape == (0, 0, 3)
    assert text_lens.shape == (0,)
    assert host_text_lens == []
