# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in test counters; no events, forwards or synchronization are added."""

import functools


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
    )

    def wrap(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def counted(*args, **kwargs):
            counts[label] += 1
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
        counts["timed_event_records"] += int(event.enable_timing)
        return original_record(event, *args, **kwargs)

    torch.cuda.Event.record = record
    return counts
