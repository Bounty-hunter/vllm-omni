# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy and micro-perf checks for Triton fused GroupNorm(+Ada)+SiLU."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DEVICE = torch.device("cuda:0")
EPS = 1e-6
NUM_GROUPS = 32

# Hunyuan-like UNet ResBlock activation; matches blog microbench shape.
PERF_SHAPE = (2, 4096, 64, 64)
# Soft floor: CI GPUs vary; measured L20X was ~3× / ~5×.
MIN_SPEEDUP = float(os.environ.get("VLLM_OMNI_FUSED_GN_MIN_SPEEDUP", "1.5"))


def _aten_gn_silu(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    return F.silu(F.group_norm(x, num_groups, weight, bias, eps))


def _aten_gn_ada_silu(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    h = F.group_norm(x, num_groups, weight, bias, eps)
    return F.silu(h * (1.0 + scale) + shift)


def _cuda_sync_ms(fn, warmup: int = 10, iters: int = 50) -> float:
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 8, 8),
        (2, 128, 16, 16),
        (1, 256, 4, 4, 4),  # 5D (N,C,T,H,W)
    ],
)
def test_fused_groupnorm_silu_matches_aten(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    from vllm_omni.diffusion.layers.vae.fused_groupnorm import FusedGroupNormSiLU

    torch.manual_seed(0)
    c = shape[1]
    assert c % NUM_GROUPS == 0
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    op = FusedGroupNormSiLU(num_channels=c, num_groups=NUM_GROUPS, eps=EPS).to(DEVICE, dtype)
    op._use_triton = True

    ref = _aten_gn_silu(x, op.weight, op.bias, NUM_GROUPS, EPS)
    out = op.forward_cuda(x)

    atol = 2e-3 if dtype == torch.bfloat16 else 1e-4
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_groupnorm_ada_silu_matches_aten(dtype: torch.dtype) -> None:
    from vllm_omni.diffusion.layers.vae.fused_groupnorm import FusedGroupNormAdaSiLU

    torch.manual_seed(1)
    n, c, h, w = 2, 128, 16, 16
    x = torch.randn(n, c, h, w, device=DEVICE, dtype=dtype)
    scale = torch.randn(n, c, 1, 1, device=DEVICE, dtype=dtype)
    shift = torch.randn(n, c, 1, 1, device=DEVICE, dtype=dtype)
    op = FusedGroupNormAdaSiLU(num_channels=c, num_groups=NUM_GROUPS, eps=EPS).to(DEVICE, dtype)
    op._use_triton = True

    ref = _aten_gn_ada_silu(x, op.weight, op.bias, NUM_GROUPS, EPS, scale, shift)
    out = op.forward_cuda(x, scale, shift)

    atol = 2e-3 if dtype == torch.bfloat16 else 1e-4
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def test_fused_groupnorm_silu_faster_than_aten() -> None:
    from vllm_omni.diffusion.layers.vae.fused_groupnorm import FusedGroupNormSiLU

    torch.manual_seed(2)
    dtype = torch.bfloat16
    x = torch.randn(*PERF_SHAPE, device=DEVICE, dtype=dtype)
    op = FusedGroupNormSiLU(num_channels=PERF_SHAPE[1], num_groups=NUM_GROUPS, eps=EPS).to(
        DEVICE, dtype
    )
    op._use_triton = True

    # Autotune / compile warmup outside timed region.
    _ = op.forward_cuda(x)
    torch.cuda.synchronize()

    aten_ms = _cuda_sync_ms(lambda: _aten_gn_silu(x, op.weight, op.bias, NUM_GROUPS, EPS))
    fused_ms = _cuda_sync_ms(lambda: op.forward_cuda(x))
    speedup = aten_ms / fused_ms
    print(
        f"[GN+SiLU] shape={PERF_SHAPE} aten={aten_ms:.3f}ms "
        f"fused={fused_ms:.3f}ms speedup={speedup:.2f}x"
    )
    assert speedup >= MIN_SPEEDUP, (
        f"expected fused GN+SiLU speedup >= {MIN_SPEEDUP}, got {speedup:.2f}x "
        f"(aten={aten_ms:.3f}ms, fused={fused_ms:.3f}ms)"
    )


def test_fused_groupnorm_ada_silu_faster_than_aten() -> None:
    from vllm_omni.diffusion.layers.vae.fused_groupnorm import FusedGroupNormAdaSiLU

    torch.manual_seed(3)
    dtype = torch.bfloat16
    n, c, h, w = PERF_SHAPE
    x = torch.randn(n, c, h, w, device=DEVICE, dtype=dtype)
    scale = torch.randn(n, c, 1, 1, device=DEVICE, dtype=dtype)
    shift = torch.randn(n, c, 1, 1, device=DEVICE, dtype=dtype)
    op = FusedGroupNormAdaSiLU(num_channels=c, num_groups=NUM_GROUPS, eps=EPS).to(DEVICE, dtype)
    op._use_triton = True

    _ = op.forward_cuda(x, scale, shift)
    torch.cuda.synchronize()

    aten_ms = _cuda_sync_ms(
        lambda: _aten_gn_ada_silu(x, op.weight, op.bias, NUM_GROUPS, EPS, scale, shift)
    )
    fused_ms = _cuda_sync_ms(lambda: op.forward_cuda(x, scale, shift))
    speedup = aten_ms / fused_ms
    print(
        f"[GN+Ada+SiLU] shape={PERF_SHAPE} aten={aten_ms:.3f}ms "
        f"fused={fused_ms:.3f}ms speedup={speedup:.2f}x"
    )
    assert speedup >= MIN_SPEEDUP, (
        f"expected fused GN+Ada+SiLU speedup >= {MIN_SPEEDUP}, got {speedup:.2f}x "
        f"(aten={aten_ms:.3f}ms, fused={fused_ms:.3f}ms)"
    )
