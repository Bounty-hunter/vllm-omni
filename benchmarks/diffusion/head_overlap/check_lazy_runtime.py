# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fast scheduled GPU API checks, outside model inference or policy learning."""

import importlib
from dataclasses import replace
from types import MethodType, SimpleNamespace

import torch
from execution_counters import install
from lazy_head_policy import LazyHeadPolicy


def check_video_query_tail():
    from query_tail import eligible

    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend

    owner = SimpleNamespace(causal=False, effective_attention=lambda: (None, FlashAttentionBackend, None))
    kernel = MethodType(lambda self: None, owner)
    grid = (9, 30, 52)
    rows = 14040
    layout = VideoTokenLayout(prefix_len=0, latent_grid=grid, used_len=rows)
    md = AttentionMetadata(video_layout=layout, extra={"vsa_dit_seq_shape": grid})
    assert eligible(kernel, md, rows)
    assert eligible(kernel, AttentionMetadata(), rows)
    assert not eligible(kernel, md, rows + 4)
    for unsupported in (
        replace(md, attn_mask=torch.ones(1, rows, dtype=torch.bool)),
        replace(md, extra={**md.extra, "kv_cache_dtype": "fp8"}),
        replace(md, extra={"vsa_dit_seq_shape": (9, 30, 51)}),
        replace(md, video_layout=replace(layout, prefix_len=1)),
        replace(md, video_layout=replace(layout, used_len=rows - 1)),
    ):
        assert not eligible(kernel, unsupported, rows)
    owner.causal = True
    assert not eligible(kernel, md, rows)
    owner.causal = False
    owner.effective_attention = lambda: (None, None, None)
    assert not eligible(kernel, md, rows)


def main():
    counters = install()
    policy = LazyHeadPolicy(None)
    importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    check_video_query_tail()
    from native_prefilled import layout

    from vllm_omni.diffusion.cuda_graph import clone_static

    prefix = torch.empty(1, 1100, 32, 128)[:, :17].clone()
    captured_prefix = clone_static(prefix)
    assert prefix.stride() != captured_prefix.stride()
    assert layout((prefix,)) == layout((captured_prefix,))
    assert policy.sms > 0 and policy.identity[-1]
    plain = torch.cuda.Event()
    timer = torch.cuda.Event(enable_timing=True)
    timer._hcms_lazy_timer = True
    plain.record()
    timer.record()
    torch.accelerator.synchronize()
    assert counters["event_records"] == 2 and counters["timed_event_records"] == 1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        value = torch.ones(4, device="cuda")
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin(pool=None)
        result = value + 2
        graph.capture_end()
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(result, torch.full_like(value, 3), rtol=0, atol=0)
    print("PASS backend identity, timer tagging/counters, manual capture API", policy.identity, flush=True)


if __name__ == "__main__":
    main()
