# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fast scheduled GPU API checks, outside model inference or policy learning."""

import torch
from execution_counters import install
from lazy_head_policy import LazyHeadPolicy


def main():
    counters = install()
    policy = LazyHeadPolicy(None)
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
