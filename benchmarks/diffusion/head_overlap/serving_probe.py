# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark-only serving bridge preserving native compile and graph settings.

Baseline installation is a no-op. The enabled arm samples actual component
calls and freezes after normal request output completion. No graph, probe
forward, compute stream or device wait is added.
"""

import functools
import hashlib
import importlib
import json
import os
from contextvars import ContextVar
from pathlib import Path

_ACTIVE = ContextVar("serving_head_policy", default=None)


def record(**values):
    import torch.distributed as dist

    destination = Path(os.environ["HCMS_MATRIX_AUDIT"])
    destination.mkdir(parents=True, exist_ok=True)
    row = dict(rank=dist.get_rank(), **values)
    with (destination / f"rank{dist.get_rank()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row, default=str) + "\n")


def install(pipeline):
    """Install only through serving-probe.patch in an isolated checkout."""
    chunks = pipeline.od_config.parallel_config.ulysses_head_chunks
    if chunks == 1:
        return
    if chunks != 2:
        raise ValueError("The serving experiment enables head_chunks=2")
    if pipeline.od_config.diffusion_use_graph:
        raise ValueError("This serving adapter requires uncaptured DiT execution")

    # The native utilities also import the attention owner. Load the patched
    # owner first so its native bridge cannot observe a partial native module.
    head = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    import native_prefilled as native
    from lazy_head_policy import LazyHeadPolicy, metadata_spec
    from prefilled_probe import PrefilledPipeline
    from schedule_probe import run as scheduled

    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group, get_world_group

    original_can_overlap = head.can_overlap
    original_native = native.run
    policies = {}
    request_number = 0
    request_key = None

    def can_overlap(q, k, v, md, world, chunks):
        policy = _ACTIVE.get()
        if policy is None or q.shape != k.shape or k.shape != v.shape or q.shape[0] != 1:
            return False
        if str(q.dtype) != "torch.bfloat16" or not original_can_overlap(q, k, v, md, world, chunks):
            return False
        # Unseen layouts use the original strategy after resource preparation.
        if policy.phase != "discover":
            signature = policy.signature(q, k, v, md, get_sp_group().ulysses_group)
            return signature in policy.entries
        # A new layout must fall back before entering the CE route when its
        # bounded caches are full. Sampling/frozen calls never allocate.
        return len(native._PLANS) < native._PLAN_LIMIT and len(native._SLOT_ARENA) < 128

    def native_run(q, k, v, md, pg, chunks, kernel, **kwargs):
        policy = _ACTIVE.get()
        if policy is None:
            return original_native(q, k, v, md, pg, chunks, kernel, **kwargs)
        return policy.run(q, k, v, md, pg, kernel, streams=kwargs["streams"])

    head.can_overlap = can_overlap
    head._run_native_prefilled = native.run = native_run
    PrefilledPipeline.__call__ = lambda plan: scheduled(plan, "stage-local")

    def wrap_component(original, component):
        @functools.wraps(original)
        def forward(*args, **kwargs):
            signature, supported = metadata_spec((args, kwargs))
            if request_key is None or not supported:
                return original(*args, **kwargs)
            key = request_key, component, signature
            policy = policies.get(key)
            if policy is None:
                policy = policies[key] = LazyHeadPolicy(
                    get_world_group().cpu_group, candidate_family="three-then-two", schedule="stage-local"
                )
                policy.begin()
            phase = policy.phase
            token = _ACTIVE.set(policy)
            try:
                result = original(*args, **kwargs)
            finally:
                _ACTIVE.reset(token)
            if phase == "discover":
                # First real component call discovers layer layouts; candidate
                # preparation completes before the second real call samples.
                policy.prepare(warmup_passes=1, sample_limit=32)
            elif phase == "sample":
                policy.phase = "settle"
            return result

        return forward

    for name in pipeline._dit_modules:
        module = getattr(pipeline, name, None)
        if module is not None:
            module.forward = wrap_component(module.forward, name)

    original = pipeline.forward

    @functools.wraps(original)
    def forward(request):
        nonlocal request_number, request_key
        if request.is_dummy_run():
            return original(request)
        # Read only after the previous serving request's output completion.
        # query() is nonblocking and never adds a device wait.
        for key, policy in policies.items():
            if policy.phase == "settle":
                if any(not end.query() for entry in policy.entries.values() for _, _, end, _, _ in entry.samples):
                    raise RuntimeError("Policy finish preceded normal output completion")
                policy.finish()
                record(
                    kind="policy",
                    key=hashlib.sha256(repr(key).encode()).hexdigest(),
                    stats=policy.stats,
                    reports=policy.reports,
                )
        sampling = request.sampling_params
        request_key = tuple(
            getattr(sampling, field, None) for field in ("height", "width", "num_frames", "num_inference_steps")
        )
        before = {key: dict(policy.stats) for key, policy in policies.items()} if request_number < 3 else {}
        try:
            result = original(request)
            if request_number < 3:
                record(
                    kind="request",
                    number=request_number,
                    geometry=request_key,
                    policies=[
                        dict(
                            key=hashlib.sha256(repr(key).encode()).hexdigest(),
                            phase=policy.phase,
                            stats=policy.stats,
                            previous_stats=before.get(key, {}),
                        )
                        for key, policy in policies.items()
                    ],
                    plans=len(native._PLANS),
                    graph_capture_added=False,
                    extra_forwards=0,
                )
            request_number += 1
            return result
        finally:
            request_key = None

    pipeline.forward = forward
