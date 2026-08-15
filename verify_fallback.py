"""Prove the non-CUDA path is sound -- the user's concern with the split.

Two things to establish:
  1. Static: the default block modules pull in nothing CUDA/Triton-specific.
     If they did, a non-CUDA backend would fail at import.
  2. Functional: force the dispatch down the else-branch and run both blocks
     on CPU.
"""

import ast
import pathlib
import sys

import torch
from torch import nn

PKG = pathlib.Path("vllm_omni/diffusion/models/hunyuan_image3")

print("=" * 70)
print("1. STATIC: default modules must not import CUDA/Triton-only code")
print("=" * 70)

FORBIDDEN = ("triton", "common.ops", "fused_")
bad = []
for fname in ("autoencoder_blocks.py", "transformer_blocks.py"):
    tree = ast.parse((PKG / fname).read_text())
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
            mods += [f"{node.module}.{a.name}" for a in node.names]
    hits = [m for m in mods if any(f in m for f in FORBIDDEN)]
    print(f"  {fname}: imports = {sorted(set(m for m in mods if m))}")
    if hits:
        bad.append((fname, hits))
assert not bad, f"default modules import CUDA-only code: {bad}"
print("  OK: both default modules are pure torch")

print()
print("=" * 70)
print("2. FUNCTIONAL: force the non-CUDA branch, run on CPU")
print("=" * 70)

# Force is_cuda() -> False before blocks.py is imported, so it takes the else.
import vllm_omni.platforms as plat

_real = plat.current_omni_platform
_real.__class__.is_cuda = lambda self: False

for m in list(sys.modules):
    if "hunyuan_image3.blocks" in m:
        del sys.modules[m]

from vllm_omni.diffusion.models.hunyuan_image3 import blocks

print(f"  ResnetBlock -> {blocks.ResnetBlock.__module__}")
print(f"  ResBlock    -> {blocks.ResBlock.__module__}")
assert ".nvidia." not in blocks.ResnetBlock.__module__, "still resolved to nvidia!"
assert ".nvidia." not in blocks.ResBlock.__module__, "still resolved to nvidia!"
print("  OK: dispatch fell through to the default blocks")

torch.manual_seed(0)
rb = blocks.ResnetBlock(64, 64)
x = torch.randn(1, 64, 2, 8, 8)
y = rb(x)
assert y.shape == x.shape, y.shape
assert torch.isfinite(y).all()
print(f"  ResnetBlock CPU forward OK: {tuple(x.shape)} -> {tuple(y.shape)}")

rb2 = blocks.ResBlock(64, 128)
x2 = torch.randn(2, 64, 8, 8)
emb = torch.randn(2, 128)
y2 = rb2(x2, emb)
assert y2.shape == x2.shape, y2.shape
assert torch.isfinite(y2).all()
print(f"  ResBlock CPU forward OK: {tuple(x2.shape)} -> {tuple(y2.shape)}")

# Channel-changing variants exercise nin_shortcut / skip_connection.
rb3 = blocks.ResnetBlock(64, 128)
y3 = rb3(torch.randn(1, 64, 2, 8, 8))
assert y3.shape == (1, 128, 2, 8, 8), y3.shape
assert isinstance(rb3.nin_shortcut, nn.Module)
print(f"  ResnetBlock 64->128 (nin_shortcut) OK: {tuple(y3.shape)}")

rb4 = blocks.ResBlock(64, 128, out_channels=96)
y4 = rb4(torch.randn(2, 64, 8, 8), torch.randn(2, 128))
assert y4.shape == (2, 96, 8, 8), y4.shape
print(f"  ResBlock 64->96 (skip_connection) OK: {tuple(y4.shape)}")

print()
print("=" * 70)
print("NON-CUDA PATH OK")
print("=" * 70)
