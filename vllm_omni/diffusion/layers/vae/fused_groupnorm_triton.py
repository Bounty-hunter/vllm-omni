"""Triton fused GroupNorm(+Ada)+SiLU for contiguous channel-first activations.

Each program owns one (batch, group). Group elements are a contiguous NCS slice
``[g * cpg * S : (g+1) * cpg * S]``, so reductions use coalesced vector loads.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK": 8192}, num_warps=16, num_stages=2),
    ],
    key=["S", "groups"],
)
@triton.jit
def _fused_group_norm_silu_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    Scale_ptr,
    Shift_ptr,
    N,
    C,
    S,
    groups,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_ADA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // groups
    g = pid % groups
    cpg = C // groups
    numel = cpg * S
    # Contiguous group base in NCS layout (stride_c = S, stride_s = 1).
    base = n * C * S + g * cpg * S

    # Pass 1: mean / var
    sum_x = tl.zeros([BLOCK], dtype=tl.float32)
    sum_x2 = tl.zeros([BLOCK], dtype=tl.float32)
    off = 0
    while off < numel:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < numel
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        sum_x += x
        sum_x2 += x * x
        off += BLOCK

    mean = tl.sum(sum_x) / numel
    var = tl.sum(sum_x2) / numel - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Pass 2: normalize + affine (+ada) + silu
    off = 0
    while off < numel:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < numel
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        c = g * cpg + idx // S
        if HAS_WEIGHT:
            w = tl.load(W_ptr + c, mask=mask, other=1.0).to(tl.float32)
            y = y * w
        if HAS_BIAS:
            b = tl.load(B_ptr + c, mask=mask, other=0.0).to(tl.float32)
            y = y + b
        if HAS_ADA:
            # scale/shift are contiguous (N, C)
            ada_off = n * C + c
            scale = tl.load(Scale_ptr + ada_off, mask=mask, other=0.0).to(tl.float32)
            shift = tl.load(Shift_ptr + ada_off, mask=mask, other=0.0).to(tl.float32)
            y = y * (1.0 + scale) + shift
        y = y * tl.sigmoid(y)
        tl.store(Y_ptr + base + idx, y.to(X_ptr.dtype.element_ty), mask=mask)
        off += BLOCK


def _as_ncs(x: torch.Tensor) -> tuple[torch.Tensor, int, int, int, tuple[int, ...]]:
    """View channel-first activation as contiguous (N, C, S)."""
    if x.ndim == 4:
        n, c, h, w = x.shape
        s = h * w
        x_ncs = x.reshape(n, c, s)
        out_shape = (n, c, h, w)
    elif x.ndim == 5:
        n, c, t, h, w = x.shape
        s = t * h * w
        x_ncs = x.reshape(n, c, s)
        out_shape = (n, c, t, h, w)
    else:
        raise ValueError(f"Fused GroupNorm SiLU expects 4D/5D, got {x.ndim}D")
    if not x_ncs.is_contiguous():
        x_ncs = x_ncs.contiguous()
    return x_ncs, n, c, s, out_shape


def _ada_as_nc(t: torch.Tensor, n: int, c: int) -> torch.Tensor:
    """Broadcast ada scale/shift to contiguous (N, C)."""
    t = t.expand(n, c, *([1] * (t.ndim - 2))).reshape(n, c)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def fused_group_norm_silu_triton(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch Triton fused GroupNorm(+Ada)+SiLU. ``x`` must be CUDA channel-first."""
    if x.ndim not in (4, 5):
        raise ValueError(f"unsupported ndim={x.ndim}")
    if x.shape[1] % num_groups != 0:
        raise ValueError(f"channels {x.shape[1]} not divisible by groups {num_groups}")

    x_ncs, n, c, s, out_shape = _as_ncs(x)
    y_ncs = torch.empty_like(x_ncs)

    has_weight = weight is not None
    has_bias = bias is not None
    has_ada = scale is not None and shift is not None

    if has_weight:
        weight = weight.contiguous()
    if has_bias:
        bias = bias.contiguous()

    if has_ada:
        scale_ptr = _ada_as_nc(scale, n, c)
        shift_ptr = _ada_as_nc(shift, n, c)
    else:
        scale_ptr = y_ncs
        shift_ptr = y_ncs

    w_ptr = weight if has_weight else y_ncs
    b_ptr = bias if has_bias else y_ncs

    # Flatten pointers for contiguous NCS addressing inside the kernel.
    x_flat = x_ncs.view(-1)
    y_flat = y_ncs.view(-1)

    grid = (n * num_groups,)
    _fused_group_norm_silu_kernel[grid](
        x_flat,
        y_flat,
        w_ptr,
        b_ptr,
        scale_ptr,
        shift_ptr,
        n,
        c,
        s,
        num_groups,
        float(eps),
        HAS_WEIGHT=has_weight,
        HAS_BIAS=has_bias,
        HAS_ADA=has_ada,
    )
    return y_ncs.view(out_shape)
