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
    "overlap-return": (True, False, False),
    "overlap-staging": (True, False, False),
}


def run(plan, variant):
    if variant == "old" or plan.chunks == 1:
        return ORIGINAL_CALL(plan)
    staged, early, deferred = VARIANTS[variant]
    comm_input = variant == "overlap-staging"
    comm_output = variant in ("overlap-return", "overlap-staging")
    if comm_output and not plan.overlap:
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
        if early and plan.overlap and group + 1 < plan.chunks:
            with torch.cuda.stream(plan.stream):
                plan.submit_forward(group + 1)
        output = plan.kernel(*plan.parts[group], None)
        sources.append(output)
        plan.produced[group].record(compute)
        with torch.cuda.stream(plan.stream):
            if plan.overlap:
                if not early and group + 1 < plan.chunks:
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
        if plan.overlap:
            plan.stream.wait_event(plan.produced[-1])
            plan.submit_reverse(plan.chunks - 1, sources[-1])
        plan.output.handle.barrier()
        plan.done.record(plan.stream)
    compute.wait_event(plan.done)
    return plan.output.tensor
