# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in captured CE/attention boundaries; no native profiler or normal hooks.

Both arms retain their copy order, bytes, kernels and fences. Equal external
markers bracket each Q/K/V peer batch, whole attention and peer return. Read
timestamps only after the request's existing completion point.
"""

import ctypes
import functools
from contextlib import contextmanager
from itertools import groupby

import torch
from lazy_qkv_policy import LazyQKVPolicy

NAMES = tuple(f"{kind}{edge}" for kind in ("Q", "K", "V", "A", "R") for edge in ("0", "1"))


def prepare_events(events, plan):
    events.phase_events = {name: torch.Event(device=plan.output.tensor.device, enable_timing=True) for name in NAMES}
    for event in events.phase_events.values():
        event.record()
    local = {id(operation) for operations in plan.local_forward for operation in operations}
    remote = {field: [] for field in range(3)}
    for operation, source, _ in plan.bindings:
        if id(operation) not in local:
            remote[source % 3].append(operation)
    plan.phase_field_ids = {id(operation): field for field, operations in remote.items() for operation in operations}
    if hasattr(plan, "copy_batches"):
        for operations in remote.values():
            plan.copy_batches[tuple(map(id, operations))] = (ctypes.POINTER(ctypes.c_size_t) * len(operations))(
                *operations
            )


@contextmanager
def instrument(events, plan):
    compute = torch.cuda.current_stream()
    batch, kernel, reverse = plan.emit_batch_on, plan.kernel, plan.submit_reverse

    def mark(name, stream):
        events.runtime.CUDART_CHECK(events.record_external(events.phase_events[name].event_id, stream.cuda_stream, 1))

    def copies(operations, stream):
        # Local staging keeps its existing single batch. Only remote fields
        # are bracketed, excluding ready waits, local copies and peer fences.
        if not operations or id(operations[0]) not in plan.phase_field_ids:
            return batch(operations, stream)
        for field, selected in groupby(operations, key=lambda operation: plan.phase_field_ids[id(operation)]):
            selected = list(selected)
            label = "QKV"[field]
            mark(label + "0", stream)
            batch(selected, stream)
            mark(label + "1", stream)

    def attention(*args):
        mark("A0", compute)
        output = kernel(*args)
        mark("A1", compute)
        return output

    def returned(*args):
        mark("R0", plan.stream)
        reverse(*args)
        mark("R1", plan.stream)

    plan.emit_batch_on, plan.kernel, plan.submit_reverse = copies, attention, returned
    try:
        yield
    finally:
        plan.emit_batch_on, plan.kernel, plan.submit_reverse = batch, kernel, reverse


def install():
    """Install only in a separately labeled all-core diagnostic process."""
    prepare = LazyQKVPolicy.prepare
    execute = LazyQKVPolicy.execute
    offsets = LazyQKVPolicy.diagnostic_offsets_us

    @functools.wraps(prepare)
    def prepared(policy, **kwargs):
        if policy.core_events is None or not policy.all_core_spans:
            raise ValueError("Phase diagnostics require all-core external event diagnostics")
        prepare(policy, **kwargs)
        for entry in policy.entries.values():
            for choice in range(len(entry.choices)):
                plan = entry.whole if choice == 0 else entry.staged
                for events in policy.core_events[entry.signature, choice]:
                    prepare_events(events, plan)

    @functools.wraps(execute)
    def captured(policy, *args, **kwargs):
        if policy.phase != "frozen" or not torch.cuda.is_current_stream_capturing():
            return execute(policy, *args, **kwargs)
        if len(policy.entries) != 1:
            raise ValueError("The phase diagnostic qualifies one fixed decode signature")
        entry = next(iter(policy.entries.values()))
        index = sum(plan["calls"] for plan in policy.stats.get("capture_plans", {}).values())
        events = policy.core_events[entry.signature, entry.selected][index]
        plan = entry.whole if entry.selected == 0 else entry.staged
        with instrument(events, plan):
            return execute(policy, *args, **kwargs)

    @functools.wraps(offsets)
    def measured(policy):
        rows = offsets(policy)
        selected = [
            events
            for entry in policy.entries.values()
            for events in policy.core_events[entry.signature, entry.selected]
        ]
        for row, events in zip(rows, selected, strict=True):
            origin = events.events["begin"]
            row["phase_offsets_us"] = {
                name: origin.elapsed_time(event) * 1000 for name, event in events.phase_events.items()
            }
        return rows

    LazyQKVPolicy.prepare = prepared
    LazyQKVPolicy.execute = captured
    LazyQKVPolicy.diagnostic_offsets_us = measured
