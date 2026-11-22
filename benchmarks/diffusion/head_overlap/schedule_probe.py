# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Narrow CE scheduling ablations; original buffers, bytes, and peer fences."""

import ctypes

import torch
from prefilled_probe import PrefilledPipeline

ORIGINAL_CALL = PrefilledPipeline.__call__
VARIANTS = {
    "old": (False, False, False),
    "stage-local": (True, False, False),
    "stage-serial": (True, False, False),
    "stage-gated": (True, False, False),
    "early-peer": (False, True, False),
    "stage-early": (True, True, False),
    "stage-early-defer-local": (True, True, True),
    "overlap-return": (True, False, False),
    "overlap-staging": (True, False, False),
}


class CoreEvents:
    """Opt-in timing of one captured layer, read at request completion only."""

    def __init__(self, chunks, device, runtime):
        self.runtime = runtime
        self.record_external = runtime.lib.cudaEventRecordWithFlags
        self.record_external.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        self.record_external.restype = ctypes.c_int
        self.events = {
            name: torch.Event(device=device, enable_timing=True)
            for name in (
                "begin",
                "end",
                *(f"{kind}{g}{edge}" for kind in "FAR" for g in range(chunks) for edge in "01"),
                *(f"F{g}c" for g in range(chunks)),
            )
        }
        # Realize handles at the candidate preparation boundary, outside trials.
        for event in self.events.values():
            event.record()
        self.recorded = False

    def record(self, name, stream):
        # External record nodes preserve timing handles across graph replay.
        # Ordinary captured event dependencies do not expose replay timestamps.
        self.runtime.CUDART_CHECK(self.record_external(self.events[name].event_id, stream.cuda_stream, 1))

    def run(self, plan, variant, compute):
        originals = plan.submit_forward, plan.kernel, plan.submit_reverse, plan.emit_batch_on
        group = 0
        forward_group = None

        def forward(index):
            nonlocal forward_group
            self.record(f"F{index}0", plan.stream)
            forward_group = index
            try:
                originals[0](index)
            finally:
                forward_group = None
            self.record(f"F{index}1", plan.stream)

        def copies(operations, stream):
            originals[3](operations, stream)
            if forward_group is not None:
                # Same copies and fence as normal execution. The marker splits
                # outgoing CE completion from the incoming-ready peer fence.
                self.record(f"F{forward_group}c", stream)

        def attention(*args):
            nonlocal group
            self.record(f"A{group}0", compute)
            output = originals[1](*args)
            self.record(f"A{group}1", compute)
            group += 1
            return output

        def reverse(index, output):
            self.record(f"R{index}0", plan.stream)
            originals[2](index, output)
            self.record(f"R{index}1", plan.stream)

        plan.submit_forward, plan.kernel, plan.submit_reverse = forward, attention, reverse
        plan.emit_batch_on = copies
        try:
            self.record("begin", compute)
            output = run(plan, variant)
            self.record("end", compute)
            self.recorded = True
            return output
        finally:
            plan.submit_forward, plan.kernel, plan.submit_reverse, plan.emit_batch_on = originals

    def offsets_us(self):
        if not self.recorded:
            return None
        if not self.events["end"].query():
            raise RuntimeError("Core diagnostics must follow the preceding request's normal output completion")
        offsets = {name: self.events["begin"].elapsed_time(event) * 1000 for name, event in self.events.items()}
        if offsets["end"] <= 0 or any(not 0 <= value <= offsets["end"] for value in offsets.values()):
            raise RuntimeError(f"Invalid replay event offsets: {offsets}")
        return offsets


def run(plan, variant):
    if variant == "old" or plan.chunks == 1:
        return ORIGINAL_CALL(plan)
    staged, early, deferred = VARIANTS[variant]
    overlap = plan.overlap and variant != "stage-serial"
    comm_input = variant == "overlap-staging"
    comm_output = variant in ("overlap-return", "overlap-staging")
    if comm_output and not overlap:
        raise ValueError("Staging overlap requires an overlapping plan")
    compute = torch.cuda.current_stream()

    def prefill(groups, stream=compute):
        operations = [operation for group in groups for operation in plan.local_forward[group]]
        if hasattr(plan, "emit_batch_on"):
            plan.emit_batch_on(operations, stream)
        else:
            for operation in operations:
                plan.emit_on(operation, stream)

    def return_local(group, output, stream=compute):
        for operation in plan.reverse_operations(group, output, plan.rank):
            plan.emit_on(operation, stream)

    # The preceding final peer fence protects all old input readers. Local
    # writes to F1 are disjoint from incoming F0 writes and may run during F0.
    # enter still follows the previous output clone, so F0-ready continues to
    # acknowledge every rank's old copy-out before any new reverse write.
    prefill((0,) if staged else range(plan.chunks))
    plan.enter.record(compute)
    with torch.cuda.stream(plan.stream):
        plan.stream.wait_event(plan.enter)
        plan.submit_forward(0)
    if staged and not comm_input:
        prefill(range(1, plan.chunks))
    sources = []
    for group in range(plan.chunks):
        compute.wait_event(plan.ready[group])
        if variant == "stage-gated" and group + 1 < plan.chunks:
            plan.admitted[group].record(compute)
        if early and overlap and group + 1 < plan.chunks:
            with torch.cuda.stream(plan.stream):
                plan.submit_forward(group + 1)
        output = plan.kernel(*plan.parts[group], None)
        sources.append(output)
        plan.produced[group].record(compute)
        with torch.cuda.stream(plan.stream):
            if overlap:
                if not early and group + 1 < plan.chunks:
                    if variant == "stage-gated":
                        # Host enqueue order is not a cross-stream dependency
                        # during graph replay. Wait for compute's admission of
                        # this attention, never for its completion. This keeps
                        # F(g+1) from running ahead during A(g-1)/local staging.
                        plan.stream.wait_event(plan.admitted[group])
                    if comm_input:
                        # Distinct input buffers allow this local staging to
                        # run alongside the current attention. Its consumer
                        # still waits for the group's original ready event.
                        prefill((group + 1,), plan.stream)
                    plan.submit_forward(group + 1)
                if group:
                    plan.stream.wait_event(plan.produced[group - 1])
                    plan.submit_reverse(group - 1, sources[group - 1])
                    if comm_output:
                        return_local(group - 1, sources[group - 1], plan.stream)
            else:
                plan.stream.wait_event(plan.produced[group])
                plan.submit_reverse(group, output)
                if group + 1 < plan.chunks:
                    plan.submit_forward(group + 1)
        if not deferred and (not comm_output or group == plan.chunks - 1):
            # The last local return stays on compute, overlapping the final
            # peer return. Earlier returns can overlap the next attention.
            return_local(group, output)
    if deferred:
        # No consumer reads the assembled output until return. These local
        # copies stay on compute before its final wait and before copy-out.
        for group, output in enumerate(sources):
            return_local(group, output)
    with torch.cuda.stream(plan.stream):
        if overlap:
            plan.stream.wait_event(plan.produced[-1])
            plan.submit_reverse(plan.chunks - 1, sources[-1])
        plan.output.handle.barrier()
        plan.done.record(plan.stream)
    compute.wait_event(plan.done)
    return plan.output.tensor
