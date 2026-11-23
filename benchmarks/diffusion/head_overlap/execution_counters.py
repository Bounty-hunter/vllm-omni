# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in test counters; no events, forwards or synchronization are added."""

import functools
import json
import os
import traceback

_COUNTS = None
_CORE_PHASE = None


def observe_core(function):
    """Instrument only test runs; normal execution keeps the original callable."""
    if _COUNTS is None:
        return function

    @functools.wraps(function)
    def observed(policy, *args, **kwargs):
        global _CORE_PHASE
        previous = _CORE_PHASE
        _CORE_PHASE = policy.phase
        try:
            return function(policy, *args, **kwargs)
        finally:
            _CORE_PHASE = previous

    return observed


def install():
    import torch
    import torch.distributed as dist

    counts = dict(
        accelerator_sync=0,
        cuda_sync=0,
        tensor_item=0,
        all_gather_object=0,
        all_reduce=0,
        event_records=0,
        timed_event_records=0,
        sampling_host_syncs=0,
        captured_timer_records=0,
        event_queries=0,
        core_cpu_items=0,
        core_device_items=0,
    )
    global _COUNTS
    _COUNTS = counts
    stack_keys = set()
    stacks = os.environ.get("HCMS_COUNTER_STACKS") == "1"

    def wrap(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def counted(*args, **kwargs):
            counts[label] += 1
            counts["sampling_host_syncs"] += int(
                label != "event_queries"
                and _CORE_PHASE in ("sample", "settle", "frozen")
                and (label != "tensor_item" or args[0].device.type != "cpu")
            )
            if label == "tensor_item" and _CORE_PHASE is not None:
                device = args[0].device.type
                counts["core_cpu_items" if device == "cpu" else "core_device_items"] += 1
                key = (_CORE_PHASE, device)
                if stacks and key not in stack_keys:
                    stack_keys.add(key)
                    print(
                        "HCMS_COUNTER_STACK "
                        + json.dumps(dict(phase=_CORE_PHASE, device=device, stack=traceback.format_stack(limit=12))),
                        flush=True,
                    )
            return original(*args, **kwargs)

        setattr(owner, name, counted)

    wrap(torch.accelerator, "synchronize", "accelerator_sync")
    wrap(torch.cuda, "synchronize", "cuda_sync")
    wrap(torch.Tensor, "item", "tensor_item")
    wrap(dist, "all_gather_object", "all_gather_object")
    wrap(dist, "all_reduce", "all_reduce")
    original_record = torch.cuda.Event.record

    @functools.wraps(original_record)
    def record(event, *args, **kwargs):
        counts["event_records"] += 1
        counts["timed_event_records"] += int(getattr(event, "_hcms_lazy_timer", False))
        counts["captured_timer_records"] += int(_CORE_PHASE == "frozen" and getattr(event, "_hcms_lazy_timer", False))
        return original_record(event, *args, **kwargs)

    torch.cuda.Event.record = record
    wrap(torch.cuda.Event, "query", "event_queries")
    return counts
