# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark-only serving bridge preserving native compile and graph settings.

Baseline installation is a no-op. The enabled arm samples actual component
calls and freezes between calls once all ranks' samples have completed.
No graph, probe forward, compute stream or device wait is added.
"""

import functools
import hashlib
import importlib
import json
import os
from contextvars import ContextVar
from copy import copy
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
    from vllm_omni.diffusion.envs import VLLM_OMNI_PACKED_QKV

    original_can_overlap = head.can_overlap
    original_native = native.run
    policies = {}
    request_number = 0
    request_key = None

    def can_overlap(q, k, v, md, world, chunks):
        policy = _ACTIVE.get()
        if policy is None or policy.phase == "native" or q.shape != k.shape or k.shape != v.shape or q.shape[0] != 1:
            return False
        if str(q.dtype) != "torch.bfloat16" or not original_can_overlap(q, k, v, md, world, chunks):
            return False
        if policy.phase == "discover" and q.shape[3] == 128:
            joint_q = 0 if md is None or md.joint_query is None else md.joint_query.shape[1]
            tiles = (q.shape[1] * world + joint_q + 127) // 128
            if (q.shape[2] // world // 2) * tiles < 0.8 * policy.sms:
                # The same conservative proxy cannot propose any split. Avoid
                # allocating a CE experiment merely to retain whole attention.
                # The current ordinary core completes; later layers/calls use
                # the prepared native strategies, with no probe or GPU wait.
                policy.phase = "native"
                policy.stats["native_reason"] = "query-tile proxy rejected all split candidates"
                policy._native_fallback()
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

    def finish_policy(key, policy, *, ready_only=False):
        if not policy.finish(ready_only=ready_only):
            return False
        policy.stats["serving_route"] = (
            "native-nccl"
            if all(len(entry.choices[entry.selected]) == 1 for entry in policy.entries.values())
            else "ce-head-overlap"
        )
        record(
            kind="policy",
            key=hashlib.sha256(repr(key).encode()).hexdigest(),
            stats=policy.stats,
            reports=policy.reports,
        )
        return True

    def wrap_component(original, component, module):
        # Prepare ordinary NCCL strategies once. Switching them between actual
        # component calls restores the original compiled attention path; a
        # frozen whole decision must not retain the experimental CE core.
        strategies = []
        for attention in module.modules():
            strategy = getattr(attention, "parallel_strategy", None)
            if getattr(strategy, "name", None) == "ulysses" and strategy.head_chunks > 1:
                baseline = copy(strategy)
                baseline.head_chunks = 1
                baseline._ulysses_a2a_permute = False
                baseline._packed_qkv = VLLM_OMNI_PACKED_QKV
                strategies.append((attention, strategy, baseline))
        native_mode = False

        def select_native(enabled):
            nonlocal native_mode
            if enabled != native_mode:
                for attention, overlap, baseline in strategies:
                    attention.parallel_strategy = baseline if enabled else overlap
                native_mode = enabled

        @functools.wraps(original)
        def forward(*args, **kwargs):
            signature, supported = metadata_spec((args, kwargs))
            if request_key is None or not supported:
                select_native(True)
                return original(*args, **kwargs)
            key = request_key, component, signature
            policy = policies.get(key)
            if policy is None:
                policy = policies[key] = LazyHeadPolicy(
                    get_world_group().cpu_group, candidate_family="three-then-two", schedule="stage-local"
                )
                policy.begin()
                policy._native_fallback = lambda: select_native(True)
            phase = policy.phase
            use_native = phase == "native" or (
                phase == "frozen" and all(len(entry.choices[entry.selected]) == 1 for entry in policy.entries.values())
            )
            select_native(use_native)
            if use_native:
                return original(*args, **kwargs)
            token = _ACTIVE.set(policy)
            try:
                result = original(*args, **kwargs)
            finally:
                _ACTIVE.reset(token)
            if phase == "discover":
                # First real component call discovers layer layouts; candidate
                # preparation completes before the second real call samples.
                if any(len(entry.choices) > 1 for entry in policy.entries.values()):
                    policy.prepare(warmup_passes=1, sample_limit=32)
                else:
                    # Insufficient predicted parallel work requires no split
                    # allocation, timer, trial or control collective.
                    policy.phase = "native"
                    policy.stats["serving_route"] = "native-nccl"
                    policy.reports = [
                        dict(
                            signature=entry.signature,
                            choices=entry.choices,
                            selected=entry.choices[0],
                            reason="no qualified split candidate",
                        )
                        for entry in policy.entries.values()
                    ]
                    record(
                        kind="policy",
                        key=hashlib.sha256(repr(key).encode()).hexdigest(),
                        stats=policy.stats,
                        reports=policy.reports,
                    )
                    for entry in policy.entries.values():
                        entry.inputs, entry.joint, entry.metadata, entry.kernel = (), None, None, None
            elif phase == "sample":
                policy.phase = "settle"
            if policy.phase == "settle":
                finish_policy(key, policy, ready_only=True)
            return result

        return forward

    for name in pipeline._dit_modules:
        module = getattr(pipeline, name, None)
        if module is not None:
            module.forward = wrap_component(module.forward, name, module)

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
                finish_policy(key, policy)
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
