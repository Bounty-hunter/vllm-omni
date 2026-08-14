# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ResBlock must stay a drop-in replacement after the fused-op rewrite.

Two things are checked for each of the two HunyuanImage3 ResBlock copies (the
DiT-stage one in ``diffusion/`` and the AR-stage one in ``model_executor/``):

1. ``forward`` still matches the original eager formulation, recomputed here
   from the block's *own* submodules so the comparison cannot drift.
2. The state_dict keys are byte-for-byte what an unfused block produces. This
   is the failure mode that matters most: an earlier attempt at this fusion
   replaced the ``nn.Sequential`` containers with named submodules, which
   silently renamed ``in_layers.0`` -> ``in_norm`` and would have loaded
   published checkpoints incorrectly rather than loudly failing.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernels",
)


def _dit_res_block():
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import ResBlock

    return ResBlock


def _ar_res_block():
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import ResBlock

    return ResBlock


def _eager_forward(block, x, emb):
    """The pre-fusion forward, expressed with the block's own submodules."""
    h = block.in_layers(x)

    emb_out = block.emb_layers(emb)
    while len(emb_out.shape) < len(h.shape):
        emb_out = emb_out[..., None]

    out_norm, out_rest = block.out_layers[0], block.out_layers[1:]
    scale, shift = torch.chunk(emb_out, 2, dim=1)
    h = out_norm(h) * (1.0 + scale) + shift
    h = out_rest(h)

    return block.skip_connection(x) + h


EXPECTED_KEYS = {
    "in_layers.0.weight",
    "in_layers.0.bias",
    "in_layers.2.weight",
    "in_layers.2.bias",
    "emb_layers.1.weight",
    "emb_layers.1.bias",
    "out_layers.0.weight",
    "out_layers.0.bias",
    "out_layers.3.weight",
    "out_layers.3.bias",
}


@pytest.mark.parametrize("factory", [_dit_res_block, _ar_res_block], ids=["dit", "ar"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_res_block_matches_eager(factory, batch_size):
    res_block_cls = factory()

    torch.manual_seed(0)
    in_channels, emb_channels, out_channels = 64, 128, 64
    block = res_block_cls(
        in_channels=in_channels,
        emb_channels=emb_channels,
        out_channels=out_channels,
        device="cuda",
        dtype=torch.float32,
    ).eval()

    # ``zero_module`` zeroes the final conv, which would mask any error in the
    # AdaGN branch by multiplying it away. Give it real weights.
    torch.nn.init.normal_(block.out_layers[3].weight, std=0.05)
    torch.nn.init.normal_(block.out_layers[3].bias, std=0.05)

    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda")
    emb = torch.randn(batch_size, emb_channels, device="cuda")

    with torch.no_grad():
        fused_out = block(x, emb)
        eager_out = _eager_forward(block, x, emb)

    torch.testing.assert_close(fused_out, eager_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("factory", [_dit_res_block, _ar_res_block], ids=["dit", "ar"])
def test_res_block_state_dict_keys_unchanged(factory):
    res_block_cls = factory()

    block = res_block_cls(
        in_channels=64,
        emb_channels=128,
        out_channels=64,
        device="cuda",
        dtype=torch.float32,
    )

    keys = set(block.state_dict().keys())
    assert keys == EXPECTED_KEYS, (
        "ResBlock state_dict keys changed -- published checkpoints would load "
        f"incorrectly.\nmissing: {sorted(EXPECTED_KEYS - keys)}\n"
        f"unexpected: {sorted(keys - EXPECTED_KEYS)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
