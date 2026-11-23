# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit diagnostics for allocation and real eager sampling submission.

Only installed by the worker when requested. Profiles existing preparation and
the second real eager call, never executes a probe or synchronizes a device.
Native request profiling and Python cProfile are separate diagnostic cohorts.
"""

import cProfile
import functools
import json
import os
from contextlib import nullcontext
from pathlib import Path


def install(packed_forward, learner_class, request_state):
    import torch
    import torch.distributed as dist

    directory = os.environ.get("HCMS_PREFIX_CPU_PROFILE_DIR")
    markers = os.environ.get("HCMS_PREFIX_PHASE_MARKERS") == "1"
    root = None if directory is None else Path(directory)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    original_prepare = learner_class.prepare
    original_forward = packed_forward.forward
    original_run = learner_class.run

    def observed(kind, function, *args, **kwargs):
        state = request_state.get()
        label = state["label"]
        profiler = cProfile.Profile() if root is not None else None
        scope = torch.profiler.record_function(f"HCMS/prefix/{kind}/{label}") if markers else nullcontext()
        success = False
        with scope:
            if profiler is not None:
                profiler.enable()
            try:
                result = function(*args, **kwargs)
                success = True
                return result
            finally:
                if profiler is not None:
                    profiler.disable()
                    stem = root / f"{label}-{kind}-rank{dist.get_rank()}"
                    profiler.dump_stats(str(stem) + ".pstats")
                    Path(str(stem) + ".metadata.json").write_text(
                        json.dumps(dict(success=success, rank=dist.get_rank(), label=label, kind=kind)) + "\n"
                    )

    @functools.wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        return observed("prepare", original_prepare, self, *args, **kwargs)

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        if self.learner is not None and self.learner.phase == "sample":
            return observed("sample", original_forward, self, *args, **kwargs)
        return original_forward(self, *args, **kwargs)

    @functools.wraps(original_run)
    def run(self, *args, **kwargs):
        if markers and self.phase == "sample":
            from lazy_head_policy import candidate_index

            # This diagnostic workload has one real execution signature.
            # Refuse ambiguous labels rather than inspecting device data.
            if len(self.entries) != 1:
                raise ValueError("Prefix core markers require exactly one execution signature")
            entry = next(iter(self.entries.values()))
            index = entry.index
            choice = candidate_index(index, len(entry.choices)) if index < len(entry.events) else entry.selected
            heads = "+".join(map(str, entry.choices[choice]))
            with torch.profiler.record_function(f"HCMS/prefix/core{entry.sample_calls}/heads-{heads}"):
                return original_run(self, *args, **kwargs)
        return original_run(self, *args, **kwargs)

    learner_class.prepare = prepare
    packed_forward.forward = forward
    if markers:
        learner_class.run = run
