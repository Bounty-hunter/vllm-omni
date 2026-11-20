# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Narrow CE scheduling ablations; original buffers, bytes, and peer fences."""

import torch
from prefilled_probe import PrefilledPipeline

ORIGINAL_CALL = PrefilledPipeline.__call__
VARIANTS = {
    "old": (False, False, False),
    "stage-local": (True, False, False),
    "early-peer": (False, True, False),
    "stage-early": (True, True, False),
    "stage-early-defer-local": (True, True, True),
}


def run(plan, variant):
    if variant == "old" or plan.chunks == 1:
        return ORIGINAL_CALL(plan)
    staged, early, deferred = VARIANTS[variant]
    compute = torch.cuda.current_stream()

    def prefill(groups):
        operations = [operation for group in groups for operation in plan.local_forward[group]]
        if hasattr(plan, "emit_batch_on"):
            plan.emit_batch_on(operations, compute)
        else:
            for operation in operations:
                plan.emit_on(operation, compute)

    # The preceding final peer fence protects all old input readers. Local
    # writes to F1 are disjoint from incoming F0 writes and may run during F0.
    # enter still follows the previous output clone, so F0-ready continues to
    # acknowledge every rank's old copy-out before any new reverse write.
    prefill((0,) if staged else range(plan.chunks))
    plan.enter.record(compute)
    with torch.cuda.stream(plan.stream):
        plan.stream.wait_event(plan.enter)
        plan.submit_forward(0)
    if staged:
        prefill(range(1, plan.chunks))
    sources = []
    for group in range(plan.chunks):
        compute.wait_event(plan.ready[group])
        if early and plan.overlap and group + 1 < plan.chunks:
            with torch.cuda.stream(plan.stream):
                plan.submit_forward(group + 1)
        output = plan.kernel(*plan.parts[group], None)
        sources.append(output)
        plan.produced[group].record(compute)
        with torch.cuda.stream(plan.stream):
            if plan.overlap:
                if not early and group + 1 < plan.chunks:
                    plan.submit_forward(group + 1)
                if group:
                    plan.stream.wait_event(plan.produced[group - 1])
                    plan.submit_reverse(group - 1, sources[group - 1])
            else:
                plan.stream.wait_event(plan.produced[group])
                plan.submit_reverse(group, output)
                if group + 1 < plan.chunks:
                    plan.submit_forward(group + 1)
        if not deferred:
            for operation in plan.reverse_operations(group, output, plan.rank):
                plan.emit_on(operation, compute)
    if deferred:
        # No consumer reads the assembled output until return. These local
        # copies stay on compute before its final wait and before copy-out.
        for group, output in enumerate(sources):
            for operation in plan.reverse_operations(group, output, plan.rank):
                plan.emit_on(operation, compute)
    with torch.cuda.stream(plan.stream):
        if plan.overlap:
            plan.stream.wait_event(plan.produced[-1])
            plan.submit_reverse(plan.chunks - 1, sources[-1])
        plan.output.handle.barrier()
        plan.done.record(plan.stream)
    compute.wait_event(plan.done)
    return plan.output.tensor
