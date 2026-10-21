# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch.nn as nn

from vllm_omni.diffusion.models.z_image import pipeline_z_image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _TextEncoder(nn.Module):
    def __init__(self, model_type="qwen3", layers=2):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(layers)])


def test_compile_qwen3_text_encoder_layers_is_regional_and_static(monkeypatch):
    encoder = _TextEncoder()
    calls = []

    def fake_compile(function, **kwargs):
        calls.append((function, kwargs))
        return function

    monkeypatch.setattr(pipeline_z_image.torch, "compile", fake_compile)

    assert pipeline_z_image._compile_qwen3_text_encoder_layers(encoder) == 2
    assert len(calls) == 2
    assert all(
        kwargs == {
            "mode": "default",
            "fullgraph": False,
            "dynamic": False,
        }
        for _, kwargs in calls
    )


def test_compile_text_encoder_skips_non_qwen3(monkeypatch):
    encoder = _TextEncoder(model_type="other")

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("non-Qwen3 text encoder should not be compiled")

    monkeypatch.setattr(pipeline_z_image.torch, "compile", unexpected_compile)
    assert pipeline_z_image._compile_qwen3_text_encoder_layers(encoder) == 0
