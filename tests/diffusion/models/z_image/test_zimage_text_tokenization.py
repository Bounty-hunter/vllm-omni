# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.models.z_image.pipeline_z_image import _tokenize_prompt_batch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _RecordingTokenizer:
    def __init__(self):
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return object()


def test_prompt_tokenization_truncates_but_avoids_fixed_length_padding():
    tokenizer = _RecordingTokenizer()
    prompts = ["short", "a somewhat longer prompt"]

    result = _tokenize_prompt_batch(tokenizer, prompts, 512)

    assert result is not None
    assert tokenizer.args == (prompts,)
    assert tokenizer.kwargs == {
        "padding": "longest",
        "max_length": 512,
        "truncation": True,
        "return_tensors": "pt",
    }
