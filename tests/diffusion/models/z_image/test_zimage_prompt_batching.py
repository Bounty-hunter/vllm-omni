# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import types

import pytest
import torch

from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_cfg_prompts_share_one_text_encoder_batch():
    pipeline = object.__new__(ZImagePipeline)
    object.__setattr__(pipeline, "_execution_device", torch.device("cpu"))
    object.__setattr__(pipeline, "_negative_prompt_cache", {})
    calls = []

    def encode(self, prompt, **_kwargs):
        calls.append(prompt)
        return [torch.tensor(index) for index in range(len(prompt))]

    object.__setattr__(pipeline, "_encode_prompt", types.MethodType(encode, pipeline))
    positive, negative = pipeline.encode_prompt(
        ["first", "second"],
        negative_prompt=["bad first", "bad second"],
        do_classifier_free_guidance=True,
    )

    assert calls == [["first", "second", "bad first", "bad second"]]
    assert [item.item() for item in positive] == [0, 1]
    assert [item.item() for item in negative] == [2, 3]


def test_precomputed_cfg_embeddings_preserve_separate_inputs():
    pipeline = object.__new__(ZImagePipeline)
    object.__setattr__(pipeline, "_execution_device", torch.device("cpu"))
    object.__setattr__(pipeline, "_negative_prompt_cache", {})
    calls = []

    def encode(self, prompt, prompt_embeds=None, **_kwargs):
        calls.append((prompt, prompt_embeds))
        return prompt_embeds

    object.__setattr__(pipeline, "_encode_prompt", types.MethodType(encode, pipeline))
    positive_input = [torch.tensor(1)]
    negative_input = [torch.tensor(2)]
    positive, negative = pipeline.encode_prompt(
        ["first"],
        negative_prompt=["bad first"],
        prompt_embeds=positive_input,
        negative_prompt_embeds=negative_input,
        do_classifier_free_guidance=True,
    )

    assert calls == [(["first"], positive_input), (["bad first"], negative_input)]
    assert positive is positive_input
    assert negative is negative_input


def test_negative_prompt_embedding_cache_reuses_exact_condition():
    pipeline = object.__new__(ZImagePipeline)
    object.__setattr__(pipeline, "_execution_device", torch.device("cpu"))
    object.__setattr__(pipeline, "_negative_prompt_cache", {})
    calls = []

    def encode(self, prompt, **_kwargs):
        calls.append(prompt)
        return [torch.tensor(index) for index in range(len(prompt))]

    object.__setattr__(pipeline, "_encode_prompt", types.MethodType(encode, pipeline))
    _, first_negative = pipeline.encode_prompt(
        ["first"],
        negative_prompt=["same negative"],
        do_classifier_free_guidance=True,
    )
    _, second_negative = pipeline.encode_prompt(
        ["second"],
        negative_prompt=["same negative"],
        do_classifier_free_guidance=True,
    )

    assert calls == [["first", "same negative"], ["second"]]
    assert second_negative is first_negative
