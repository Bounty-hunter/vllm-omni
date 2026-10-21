# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.z_image import pipeline_z_image
from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    LEARNED_PADDING,
    ZImageTransformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _IdentityRope(nn.Module):
    def forward(self, ids: torch.Tensor):
        values = ids[:, :1].to(torch.float32)
        return values, -values


class _CountingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, *_args):
        self.calls += 1
        return x


def _make_caption_only_transformer() -> ZImageTransformer2DModel:
    transformer = object.__new__(ZImageTransformer2DModel)
    nn.Module.__init__(transformer)
    transformer.cap_embedder = nn.Identity()
    transformer.cap_pad_token = nn.Parameter(torch.full((1, 4), 7.0), requires_grad=False)
    transformer.rope_embedder = _IdentityRope()
    transformer.context_refiner = nn.ModuleList([_CountingBlock()])
    transformer.alignment_padding_mode = LEARNED_PADDING
    return transformer


def test_prepare_caption_context_refines_once_and_preserves_padding():
    transformer = _make_caption_only_transformer()
    caption = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    cap_feats, cap_cos, cap_sin, seq_lens, attn_mask = transformer.prepare_caption_context(
        [caption],
        torch.device("cpu"),
    )

    assert transformer.context_refiner[0].calls == 1
    assert seq_lens == [32]
    assert cap_feats.shape == (1, 32, 4)
    assert torch.equal(cap_feats[0, :3], caption)
    assert torch.equal(cap_feats[0, 3:], torch.full((29, 4), 7.0))
    assert cap_cos.shape == cap_sin.shape == (1, 32, 1)
    assert torch.equal(attn_mask, torch.ones((1, 32), dtype=torch.bool))


def test_request_cache_reuses_cfg_variants_and_skips_subclasses(monkeypatch):
    class FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def prepare_caption_context(self, prompt_embeds, device):
            self.calls += 1
            return (self.calls, prompt_embeds, device)

    class FakeSubclass(FakeTransformer):
        pass

    monkeypatch.setattr(pipeline_z_image, "ZImageTransformer2DModel", FakeTransformer)
    transformer = FakeTransformer()
    cache = {}
    prompt_embeds = [torch.zeros((3, 4))]
    device = torch.device("cpu")

    first = pipeline_z_image._get_or_prepare_caption_context(transformer, cache, False, prompt_embeds, device)
    second = pipeline_z_image._get_or_prepare_caption_context(transformer, cache, False, [torch.ones((3, 4))], device)
    cfg = pipeline_z_image._get_or_prepare_caption_context(transformer, cache, True, prompt_embeds, device)

    assert first is second
    assert cfg is not first
    assert transformer.calls == 2

    subclass = FakeSubclass()
    assert pipeline_z_image._get_or_prepare_caption_context(subclass, {}, False, prompt_embeds, device) is None
    assert subclass.calls == 0
