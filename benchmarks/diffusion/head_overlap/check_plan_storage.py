# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU resource-count checks; multi-rank lifetime safety has a GPU stress test."""

import importlib
import sys
from collections import namedtuple
from types import ModuleType, SimpleNamespace


def main():
    device = namedtuple("Device", "type index")("cuda", 0)
    allocations = []

    class Tensor:
        dtype = "bf16"

        def __init__(self, shape):
            self.shape, self.device = tuple(shape), device

        def stride(self):
            _, sequence, heads, dim = self.shape
            return sequence * heads * dim, heads * dim, dim, 1

    class Buffer:
        def __init__(self, shape, dtype, target, group):
            self.tensor = Tensor(shape)
            allocations.append(self)

    class Plan:
        def __init__(self, *args, output=None, **kwargs):
            assert output is not None, "Prepared plans must not allocate another output"
            self.output = output

        def bind(self, inputs, joint, kernel, metadata, *, checked_layout):
            assert checked_layout

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_current_stream_capturing=lambda: False)
    torch.accelerator = SimpleNamespace(synchronize=lambda: None)
    sys.modules["torch"] = torch
    sys.modules["torch.distributed"] = ModuleType("torch.distributed")
    sys.modules["prefilled_probe"] = SimpleNamespace(PrefilledPipeline=object)
    sys.modules["overlap_probe"] = SimpleNamespace(SymmetricBuffer=Buffer)
    native = importlib.import_module("native_prefilled")
    native.NativePipeline = Plan
    inputs = tuple(Tensor((1, 12, 8, 4)) for _ in range(3))
    group = SimpleNamespace(group_name="storage-check")
    first = []
    for widths in ((4,), (1, 3), (1, 2, 1)):
        plan = native.prepare_plan(inputs, None, group, widths, None, None, None, "first")
        first.append(plan.output)
    assert len(allocations) == 1 and all(buffer is first[0] for buffer in first)
    assert len(native._PLANS) == 3
    second = native.prepare_plan(inputs, None, group, (4,), None, None, None, "second").output
    assert len(allocations) == 2 and second is not first[0]
    try:
        native.prepare_plan(inputs, None, group, (4,), None, None, None, "missing", allocate=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Sampling must reject unprepared output owners")
    assert len(allocations) == 2
    native.clear()
    assert not native._PLANS and not native._SLOT_ARENA
    print("PASS one output per owner across partitions, distinct owners, no trial allocation and bounded cleanup")


if __name__ == "__main__":
    main()
