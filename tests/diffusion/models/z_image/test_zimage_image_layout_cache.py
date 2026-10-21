# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.z_image import pipeline_z_image
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _bare_transformer() -> ZImageTransformer2DModel:
    transformer = object.__new__(ZImageTransformer2DModel)
    nn.Module.__init__(transformer)
    return transformer


def test_patchify_reuses_static_image_layout_but_not_latent_values(monkeypatch):
    transformer = _bare_transformer()
    coordinate_calls = []

    def counting_coordinate_grid(size, start=None, device=None):
        coordinate_calls.append((size, start))
        return ZImageTransformer2DModel.create_coordinate_grid(size, start, device)

    monkeypatch.setattr(transformer, "create_coordinate_grid", counting_coordinate_grid)
    caption = [torch.arange(12, dtype=torch.float32).reshape(3, 4)]
    cache = {}

    first = transformer.patchify_and_embed(
        [torch.zeros((1, 1, 2, 2))],
        caption,
        patch_size=1,
        f_patch_size=1,
        image_layout_cache=cache,
    )
    first_calls = len(coordinate_calls)
    first_position_ids = first[3][0]
    first_padding_mask = first[5][0]

    coordinate_calls.clear()
    second = transformer.patchify_and_embed(
        [torch.ones((1, 1, 2, 2))],
        caption,
        patch_size=1,
        f_patch_size=1,
        image_layout_cache=cache,
    )

    assert first_calls == 3
    assert len(coordinate_calls) == 1
    assert second[3][0] is first_position_ids
    assert second[5][0] is first_padding_mask
    assert torch.count_nonzero(first[0][0][:4]) == 0
    assert torch.equal(second[0][0][:4], torch.ones((4, 1)))


def test_request_layout_cache_is_cfg_partitioned_and_native_only():
    transformer = _bare_transformer()
    caches = {}

    positive = pipeline_z_image._get_image_layout_cache(transformer, caches, False)
    assert positive is pipeline_z_image._get_image_layout_cache(transformer, caches, False)
    assert positive is not pipeline_z_image._get_image_layout_cache(transformer, caches, True)

    class ZImageSubclass(ZImageTransformer2DModel):
        pass

    subclass = object.__new__(ZImageSubclass)
    nn.Module.__init__(subclass)
    assert pipeline_z_image._get_image_layout_cache(subclass, {}, False) is None
