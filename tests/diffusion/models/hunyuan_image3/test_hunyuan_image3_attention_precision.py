# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunYuanAttention,
    HunYuanRotary2DEmbedder,
)
from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia import _cudnn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("use_qk_norm", "dtype"), [(True, torch.bfloat16), (False, torch.float16)], ids=["qk-norm", "no-qk-norm"]
)
def test_image_attention_qk_precision(use_qk_norm, dtype):
    generator = torch.Generator().manual_seed(42)
    seq_len, head_dim = 4, 8
    qkv = torch.randn(seq_len, 3 * head_dim, generator=generator).to(dtype)
    angles = torch.randn(seq_len, head_dim // 2, generator=generator)
    cos, sin = angles.cos(), angles.sin()
    rope = HunYuanRotary2DEmbedder(num_heads=1, num_kv_heads=1, head_dim=head_dim)
    rope.rope = rope.rope.forward_native  # Exercise the CPU operator even on a GPU host.
    q_norm = RMSNorm(head_dim, dtype=torch.bfloat16)
    k_norm = RMSNorm(head_dim, dtype=torch.bfloat16)
    with torch.no_grad():
        q_norm.weight.copy_(torch.linspace(0.5, 1.5, head_dim))
        k_norm.weight.copy_(torch.linspace(1.5, 0.5, head_dim))

    hidden = qkv[:, :head_dim].unsqueeze(0)
    attention = SimpleNamespace(
        num_heads=1,
        num_kv_heads=1,
        head_dim=head_dim,
        q_size=head_dim,
        kv_size=head_dim,
        use_qk_norm=use_qk_norm,
        image_rope2d_emb=rope,
        query_layernorm=q_norm,
        key_layernorm=k_norm,
        qkv_proj=Mock(return_value=(qkv, None)),
        image_attn=Mock(return_value=hidden),
        o_proj=Mock(return_value=(hidden, None)),
    )
    HunYuanAttention.forward(
        attention,
        torch.arange(seq_len),
        hidden,
        custom_pos_emb=(cos, sin),
        mode="gen_image",
        first_step=True,
        query_lens=[seq_len],
    )

    q, k, v = attention.image_attn.call_args.args
    assert v.dtype == dtype
    query, key, _ = qkv.split(head_dim, dim=-1)
    for actual, source, norm in ((q, query, q_norm), (k, key, k_norm)):
        left, right = source.float().chunk(2, dim=-1)
        rotated = source.float() * cos.repeat(1, 2) + torch.cat((-right, left), dim=-1) * sin.repeat(1, 2)
        if use_qk_norm:
            assert norm.weight.dtype == torch.bfloat16
            rotated = rotated * torch.rsqrt(rotated.square().mean(-1, keepdim=True) + norm.variance_epsilon)
            rotated = rotated * norm.weight.float()
        torch.testing.assert_close(actual.reshape(seq_len, head_dim), rotated.to(v.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("benchmark", [False, True], ids=["benchmark-off", "benchmark-on"])
def test_deterministic_cudnn_preserves_caller_flags(benchmark):
    cudnn = torch.backends.cudnn
    with cudnn.flags(benchmark=benchmark, deterministic=False):
        with pytest.raises(RuntimeError, match="convolution failed"), _cudnn.deterministic_cudnn():
            assert cudnn.benchmark is benchmark
            assert cudnn.deterministic is True
            raise RuntimeError("convolution failed")
        assert cudnn.benchmark is benchmark
        assert cudnn.deterministic is False
