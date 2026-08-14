# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark for the HunyuanImage3 normalization fusions.

Compares the two fused operators against the eager sequences they replace, at
the activation shapes the HunyuanImage3 ResBlock actually sees.

Usage::

    python benchmarks/kernels/hunyuan_image3_norm_fusion.py
"""

import argparse

import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.ops import (
    fused_adaptive_group_norm,
    fused_group_norm_silu,
)

NUM_GROUPS = 32
EPS = 1e-5


def _time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _check(fused, ref_fp32, dtype):
    """Compare against an fp32 reference, not against the eager low-precision
    result: GroupNorm's affine can cancel to near zero, and a bf16 "reference"
    is then less accurate than the kernel it is supposed to validate."""
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(fused.float(), ref_fp32, rtol=tol, atol=tol)


def bench_group_norm_silu(batch, channels, spatial, dtype):
    kw = dict(device="cuda", dtype=dtype)
    x = torch.randn(batch, channels, spatial, spatial, **kw)
    weight = torch.randn(channels, **kw)
    bias = torch.randn(channels, **kw)

    eager = lambda: F.silu(F.group_norm(x, NUM_GROUPS, weight, bias, EPS))  # noqa: E731
    fused = lambda: fused_group_norm_silu(x, weight, bias, NUM_GROUPS, EPS)  # noqa: E731

    ref = F.silu(
        F.group_norm(x.float(), NUM_GROUPS, weight.float(), bias.float(), EPS)
    )
    _check(fused(), ref, dtype)
    return _time_ms(eager), _time_ms(fused)


def bench_adaptive_group_norm(batch, channels, spatial, dtype):
    kw = dict(device="cuda", dtype=dtype)
    x = torch.randn(batch, channels, spatial, spatial, **kw)
    weight = torch.randn(channels, **kw)
    bias = torch.randn(channels, **kw)
    emb_out = torch.randn(batch, 2 * channels, 1, 1, **kw)
    scale, shift = torch.chunk(emb_out, 2, dim=1)

    eager = lambda: F.group_norm(x, NUM_GROUPS, weight, bias, EPS) * (1.0 + scale) + shift  # noqa: E731
    fused = lambda: fused_adaptive_group_norm(  # noqa: E731
        x, weight, bias, scale, shift, NUM_GROUPS, EPS
    )

    ref = F.group_norm(
        x.float(), NUM_GROUPS, weight.float(), bias.float(), EPS
    ) * (1.0 + scale.float()) + shift.float()
    _check(fused(), ref, dtype)
    return _time_ms(eager), _time_ms(fused)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    # (batch, channels, spatial) triples spanning the ResBlock activations:
    # UNetDown/UNetUp run at the latent resolution (64x64 for a 1024px image)
    # with a few hundred channels.
    configs = [
        (1, 256, 32),
        (1, 256, 64),
        (2, 256, 64),
        (1, 512, 64),
        (2, 512, 32),
    ]

    header = (
        f"{'shape':>22} | {'op':>18} | {'eager ms':>9} | {'fused ms':>9} | "
        f"{'speedup':>7} | {'CTAs':>5}"
    )
    print(f"dtype = {args.dtype}, num_groups = {NUM_GROUPS}, SMs = {num_sms}")
    print("both kernels launch batch*num_groups programs -- see the CTAs column")
    print(header)
    print("-" * len(header))

    for name, fn in [
        ("group_norm_silu", bench_group_norm_silu),
        ("adaptive_group_norm", bench_adaptive_group_norm),
    ]:
        for batch, channels, spatial in configs:
            eager_ms, fused_ms = fn(batch, channels, spatial, dtype)
            shape = f"({batch}, {channels}, {spatial}, {spatial})"
            print(
                f"{shape:>22} | {name:>18} | {eager_ms:9.3f} | {fused_ms:9.3f} | "
                f"{eager_ms / fused_ms:6.2f}x | {batch * NUM_GROUPS:5d}"
            )


if __name__ == "__main__":
    main()
