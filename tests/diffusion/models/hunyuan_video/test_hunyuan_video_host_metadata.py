# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import _trim_encoder_context
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    HunyuanVideo15Pipeline,
    _combine_valid_token_counts,
)
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v import HunyuanVideo15I2VPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("counts", [None, (3, 2)])
def test_trim_encoder_context_host_metadata_matches_fallback(counts) -> None:
    hidden_states = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    attention_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]
    )

    trimmed_states, trimmed_mask = _trim_encoder_context(hidden_states, attention_mask, counts)

    torch.testing.assert_close(trimmed_states, hidden_states[:, :3])
    assert trimmed_mask is not None
    torch.testing.assert_close(trimmed_mask, attention_mask[:, :3])


def test_trim_encoder_context_drops_uniform_packed_mask_from_host_counts() -> None:
    hidden_states = torch.randn(2, 5, 3)
    attention_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, False, False],
        ]
    )

    trimmed_states, trimmed_mask = _trim_encoder_context(hidden_states, attention_mask, (3, 3))

    torch.testing.assert_close(trimmed_states, hidden_states[:, :3])
    assert trimmed_mask is None


@pytest.mark.parametrize("counts", [(1,), (-1, 2), (6, 2)])
def test_trim_encoder_context_rejects_invalid_host_counts(counts) -> None:
    hidden_states = torch.randn(2, 5, 3)
    attention_mask = torch.ones(2, 5, dtype=torch.bool)

    with pytest.raises(ValueError):
        _trim_encoder_context(hidden_states, attention_mask, counts)


def test_combine_valid_token_counts_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError):
        _combine_valid_token_counts((1, 2), (3,))


@pytest.mark.parametrize("pipeline_cls", [HunyuanVideo15Pipeline, HunyuanVideo15I2VPipeline])
def test_encode_prompt_preserves_default_result_and_optionally_returns_host_counts(pipeline_cls, monkeypatch) -> None:
    pipeline = object.__new__(pipeline_cls)

    def get_mllm(prompt, device, dtype):
        count = 2 if prompt[0] == "positive" else 1
        batch_size = len(prompt)
        return (
            torch.zeros(batch_size, 4, 2, dtype=dtype),
            torch.ones(batch_size, 4),
            (count,) * batch_size,
        )

    def get_byte5(prompt, device, dtype):
        count = 3 if prompt[0] == "positive" else 2
        batch_size = len(prompt)
        return (
            torch.zeros(batch_size, 4, 2, dtype=dtype),
            torch.ones(batch_size, 4),
            (count,) * batch_size,
        )

    monkeypatch.setattr(pipeline, "_get_mllm_prompt_embeds", get_mllm)
    monkeypatch.setattr(pipeline, "_get_byte5_prompt_embeds", get_byte5)

    default_result = pipeline.encode_prompt(
        ["positive", "positive"],
        device=torch.device("cpu"),
        dtype=torch.float32,
        negative_prompt=["negative", "negative"],
        do_classifier_free_guidance=True,
    )
    result_with_counts = pipeline.encode_prompt(
        ["positive", "positive"],
        device=torch.device("cpu"),
        dtype=torch.float32,
        negative_prompt=["negative", "negative"],
        do_classifier_free_guidance=True,
        return_token_counts=True,
    )

    assert len(default_result) == 8
    assert len(result_with_counts) == 10
    assert result_with_counts[-2] == (5, 5)
    assert result_with_counts[-1] == (3, 3)
