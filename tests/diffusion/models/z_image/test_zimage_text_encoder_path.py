# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.z_image.pipeline_z_image import _encode_qwen3_hidden_states

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(hidden_states=(torch.tensor(-1), torch.tensor(7), torch.tensor(11)))


class _Qwen3CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = _RecordingModel()

    def forward(self, **_kwargs):
        raise AssertionError("the unused CausalLM logits path should not run")


def test_qwen3_conditioning_uses_base_model_without_cache_or_logits():
    encoder = _Qwen3CausalLM()
    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.tensor([[True, True]])

    result = _encode_qwen3_hidden_states(encoder, input_ids, attention_mask)

    assert result.item() == 7
    assert encoder.model.kwargs["input_ids"] is input_ids
    assert encoder.model.kwargs["attention_mask"] is attention_mask
    assert encoder.model.kwargs["output_hidden_states"] is True
    assert encoder.model.kwargs["use_cache"] is False


def test_non_qwen3_preserves_causal_lm_path():
    encoder = _RecordingModel()
    encoder.config = SimpleNamespace(model_type="other")
    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.tensor([[True, True]])

    result = _encode_qwen3_hidden_states(encoder, input_ids, attention_mask)

    assert result.item() == 7
    assert encoder.kwargs["input_ids"] is input_ids
    assert encoder.kwargs["attention_mask"] is attention_mask
    assert encoder.kwargs["output_hidden_states"] is True
    assert "use_cache" not in encoder.kwargs
