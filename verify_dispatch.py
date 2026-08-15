"""Verify the hunyuan_image3 hardware dispatch actually resolves.

The failure mode we are guarding against is the one from commit 8c2ec769:
nvidia/__init__.py re-exported names that did not exist, the try/except
swallowed the ImportError, and every CUDA machine silently ran the default
blocks while looking fine. So we assert on the resolved __module__, not on
"did it import without raising".
"""

import torch

from vllm_omni.platforms import current_omni_platform

print("=" * 70)
print(f"platform.is_cuda() = {current_omni_platform.is_cuda()}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
print("=" * 70)

# 1. Leaf dispatch module
from vllm_omni.diffusion.models.hunyuan_image3.blocks import ResBlock, ResnetBlock

print(f"blocks.ResnetBlock -> {ResnetBlock.__module__}.{ResnetBlock.__name__}")
print(f"blocks.ResBlock    -> {ResBlock.__module__}.{ResBlock.__name__}")

expect_nvidia = current_omni_platform.is_cuda()
want = "nvidia" if expect_nvidia else "not nvidia"
for name, cls in [("ResnetBlock", ResnetBlock), ("ResBlock", ResBlock)]:
    is_nv = ".nvidia." in cls.__module__
    assert is_nv == expect_nvidia, f"{name}: resolved to {cls.__module__}, expected {want}"
print(f"OK: both blocks resolved to the {want} implementation")

# 2. Package __init__ re-export is the same object (no second dispatch)
import vllm_omni.diffusion.models.hunyuan_image3 as pkg

assert pkg.ResnetBlock is ResnetBlock, "package __init__ ResnetBlock differs from blocks.py"
assert pkg.ResBlock is ResBlock, "package __init__ ResBlock differs from blocks.py"
print("OK: package __init__ re-exports the identical objects")

# 3. autoencoder.py consumed the dispatched class (8c2ec769's other bug:
#    Encoder/Decoder resolved ResnetBlock from their own module globals)
from vllm_omni.diffusion.models.hunyuan_image3 import autoencoder as ae

assert ae.ResnetBlock is ResnetBlock, f"autoencoder.py has a different ResnetBlock: {ae.ResnetBlock.__module__}"
print("OK: autoencoder.py uses the dispatched ResnetBlock")

# 4. hunyuan_image3_transformer.py consumed the dispatched ResBlock
from vllm_omni.diffusion.models.hunyuan_image3 import hunyuan_image3_transformer as xf

assert xf.ResBlock is ResBlock, f"transformer has a different ResBlock: {xf.ResBlock.__module__}"
print("OK: hunyuan_image3_transformer.py uses the dispatched ResBlock")

# 5. state_dict keys identical between default and nvidia variants
from vllm_omni.diffusion.models.hunyuan_image3.autoencoder_blocks import ResnetBlock as DefRes
from vllm_omni.diffusion.models.hunyuan_image3.transformer_blocks import ResBlock as DefResB

pairs = [("ResnetBlock", DefRes(128, 128), None), ("ResBlock", DefResB(128, 256), None)]
if expect_nvidia:
    from vllm_omni.diffusion.models.hunyuan_image3.nvidia.autoencoder_blocks import ResnetBlock as NvRes
    from vllm_omni.diffusion.models.hunyuan_image3.nvidia.transformer_blocks import ResBlock as NvResB

    pairs = [("ResnetBlock", DefRes(128, 128), NvRes(128, 128)), ("ResBlock", DefResB(128, 256), NvResB(128, 256))]

for name, a, b in pairs:
    ka = sorted(a.state_dict().keys())
    if b is None:
        print(f"  {name}: {len(ka)} keys (nvidia variant not applicable on this platform)")
        continue
    kb = sorted(b.state_dict().keys())
    assert ka == kb, f"{name} state_dict mismatch:\n  default={ka}\n  nvidia ={kb}"
    print(f"  {name}: {len(ka)} keys identical between default and nvidia")

# 6. Numerical agreement default vs nvidia
if expect_nvidia:
    torch.manual_seed(0)
    dev = "cuda"

    d = DefRes(128, 128).to(dev).eval()
    n = NvRes(128, 128).to(dev).eval()
    n.load_state_dict(d.state_dict())
    x = torch.randn(1, 128, 2, 32, 32, device=dev)
    with torch.no_grad():
        diff = (d(x) - n(x)).abs().max().item()
    print(f"  ResnetBlock  max|default - nvidia| = {diff:.3e}")
    assert diff < 1e-2, f"ResnetBlock diverged: {diff}"

    d2 = DefResB(128, 256).to(dev).eval()
    n2 = NvResB(128, 256).to(dev).eval()
    n2.load_state_dict(d2.state_dict())
    x2 = torch.randn(2, 128, 32, 32, device=dev)
    emb = torch.randn(2, 256, device=dev)
    with torch.no_grad():
        diff2 = (d2(x2, emb) - n2(x2, emb)).abs().max().item()
    print(f"  ResBlock     max|default - nvidia| = {diff2:.3e}")
    assert diff2 < 1e-2, f"ResBlock diverged: {diff2}"

print("=" * 70)
print("ALL DISPATCH CHECKS PASSED")
print("=" * 70)
