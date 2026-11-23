# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Temporary same-thread profiler around two complete warmed worker requests."""

import functools
import json
import os
import sys
import threading
from pathlib import Path

import torch
import torch.distributed as dist

_ACTIVE = False
_STAGE_CALLS = {}


def annotate(name):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if not _ACTIVE:
                return function(*args, **kwargs)
            index = _STAGE_CALLS.get(name, 0)
            _STAGE_CALLS[name] = index + 1
            with torch.profiler.record_function(f"HCMS/{name}/call{index}"):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def install_stages(pipeline):
    """Annotate existing component calls; no per-layer work on graph replay."""
    for name in getattr(pipeline, "_dit_modules", ()):
        module = getattr(pipeline, name)
        module.forward = annotate("dit/" + name)(module.forward)
    vae = getattr(pipeline, "vae", None)
    for name in ("encode", "decode"):
        function = getattr(vae, name, None)
        if callable(function):
            setattr(vae, name, annotate("vae/" + name)(function))


def annotate_attention(function):
    @functools.wraps(function)
    def wrapped(q, k, v, md, pg, chunks, kernel, **kwargs):
        if not _ACTIVE:
            return function(q, k, v, md, pg, chunks, kernel, **kwargs)
        owner = getattr(kernel, "__self__", None)
        prefix = getattr(owner, "prefix", "unknown")
        with torch.profiler.record_function(f"HCMS/layer/{prefix}"):
            return function(q, k, v, md, pg, chunks, kernel, **kwargs)

    return wrapped


def _plan_annotations(enabled):
    module = sys.modules.get("native_prefilled")
    if module is not None:
        for _, plan in module._PLANS.values():
            plan.profile = enabled


def instrument_request(function):
    """Keep CUPTI active between requests so only entry needs alignment."""
    seen = 0
    profiler = None
    owner_thread = None
    records = []
    directory = Path(os.environ["HCMS_FULL_TRACE_DIR"])
    variant = os.environ["HCMS_FULL_VARIANT"]
    workload = os.environ.get("HCMS_FULL_WORKLOAD", "diffusion")
    warmups, requests = int(os.environ.get("HCMS_FULL_WARMUPS", "2")), 2
    if warmups not in (0, 2):
        raise ValueError("Full-request diagnostics support either startup or two warmed requests")

    def finish(success):
        nonlocal profiler
        global _ACTIVE
        _ACTIVE = False
        _plan_annotations(False)
        active, profiler = profiler, None
        active.stop()
        rank = dist.get_rank()
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{workload}-{variant}-two-requests-rank{rank}"
        active.export_chrome_trace(str(directory / f"{stem}.trace.json"))
        active.export_stacks(str(directory / f"{stem}.stacks.txt"), "self_cpu_time_total")
        metadata = dict(
            success=success,
            variant=variant,
            rank=rank,
            thread_id=owner_thread,
            warmups=warmups,
            requests=records,
            scope=(
                "DiffusionModelRunner._execute_request_list, including text encoder, "
                "native denoise schedule, VAE and output preparation; not client E2E"
            ),
            settings=dict(
                cpu=True,
                cuda=True,
                with_stack=True,
                record_shapes=False,
                profile_memory=False,
            ),
            boundary_sync=(
                "One pre-session peer alignment outside request markers; "
                "completion synchronize labelled inside each request"
            ),
            pipeline_stage_timer_enabled=False,
            latency_is_diagnostic=True,
        )
        (directory / f"{stem}.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"FULL_REQUEST_TRACE {directory / stem} success={success}", flush=True)

    @functools.wraps(function)
    def wrapped(self, reqs, *args, **kwargs):
        nonlocal seen, profiler, owner_thread
        global _ACTIVE
        if not reqs or any(req.is_dummy_run() for req in reqs):
            return function(self, reqs, *args, **kwargs)
        seen += 1
        if seen == 1:
            install_stages(self.pipeline)
        if seen <= warmups or seen > warmups + requests:
            return function(self, reqs, *args, **kwargs)
        if seen == warmups + 1:
            owner_thread = threading.get_ident()
            torch.accelerator.synchronize()
            dist.barrier()
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                experimental_config=torch.profiler._ExperimentalConfig(verbose=True),
            )
            profiler.start()
            # This alignment is outside the request. It prevents profiler setup
            # skew from appearing as a model's first input-ready fence wait.
            with torch.profiler.record_function("HCMS/profiler_entry_alignment"):
                dist.barrier()
                torch.accelerator.synchronize()
            _ACTIVE = True
            _plan_annotations(True)
        if threading.get_ident() != owner_thread:
            raise RuntimeError("Profiler requests must execute on the owning worker thread")
        index = seen - warmups - 1
        _STAGE_CALLS.clear()
        records.append(dict(index=index, request_ids=[req.request_id for req in reqs]))
        try:
            with torch.profiler.record_function(f"HCMS/{variant}/request{index}"):
                result = function(self, reqs, *args, **kwargs)
                with torch.profiler.record_function("HCMS/profiler_request_completion_wait"):
                    torch.accelerator.synchronize()
            records[-1]["stage_calls"] = dict(_STAGE_CALLS)
            profiler.step()
        except BaseException:
            finish(False)
            raise
        if seen == warmups + requests:
            finish(True)
        return result

    return wrapped
