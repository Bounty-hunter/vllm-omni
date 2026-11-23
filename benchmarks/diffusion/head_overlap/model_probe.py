# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit offline probe: fixed-case graphs and audited direct CE execution."""

import functools
import hashlib
import importlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from vllm_omni.diffusion.cuda_graph import GraphedForward, clone_static, kwargs_signature
from vllm_omni.diffusion.distributed.parallel_state import get_world_group

_ACTIVE = ContextVar("head_matrix_request", default=None)


class StableGraph(GraphedForward):
    """Keep warmup and capture on the CE buffers' owning compute stream."""

    @contextmanager
    def _warmup_stream(self):
        yield

    def _capture_graph(self, body):
        graph = torch.cuda.CUDAGraph()
        learner = getattr(self.module, "learner", None)
        # Both controls use the same preparation. This replaces the original
        # graph-context entry sync and avoids hiding learning cost behind
        # different garbage collection / allocator-cache cleanup paths.
        torch.accelerator.synchronize()
        if learner is not None:
            learner.finish()
        graph.capture_begin(pool=self._pool)
        try:
            events = getattr(learner, "component_events", None)
            outputs = body() if events is None else events.measure(body, torch.cuda.current_stream())
        finally:
            graph.capture_end()
        if self._pool is None:
            self._pool = graph.pool()
        return graph, outputs

    def _capture(self, key, *args, **kwargs):
        entry = super()._capture(key, *args, **kwargs)
        learner = getattr(self.module, "learner", None)
        if learner is not None and entry is not None:
            self._entries.pop(key)
            self._entries[(kwargs_signature(learner.cache_key), key[1])] = entry
        return entry


class PackedForward(nn.Module):
    def __init__(self, forward, learner=None):
        super().__init__()
        self.body = forward
        self.learner = learner
        self.forward_calls = 0
        self.prefix_steps = 0

    def forward(self, packed_args, packed_kwargs):
        self.forward_calls += 1
        return self.body(*packed_args, **packed_kwargs)


def record(**values):
    destination = Path(os.environ["HCMS_MATRIX_AUDIT"])
    destination.mkdir(parents=True, exist_ok=True)
    row = dict(rank=dist.get_rank(), pid=os.getpid(), **values)
    with (destination / f"rank{dist.get_rank()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
    print("HEAD_MATRIX_AUDIT " + json.dumps(row, default=str), flush=True)


def assert_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected, strict=True):
            assert_equal(item, reference)
    else:
        assert actual == expected


def graph_call(original, graphs, component, *args, **kwargs):
    state = _ACTIVE.get()
    if state is None or state["mode"] == "eager":
        return original(*args, **kwargs)
    if state["real_prefix"]:
        return real_prefix_call(original, graphs, component, *args, **kwargs)
    graph_key = state.get("graph_variant", state["variant"])
    graph = graphs.get(graph_key)
    reference = None
    if graph is None:
        state["learner"] = None
        learner = None
        if state["autotune"]:
            from lazy_head_policy import LazyHeadPolicy

            learner = LazyHeadPolicy(
                get_world_group().cpu_group,
                candidate_partition=state["candidate_partition"],
                candidate_family=state["candidate_family"],
                schedule=state["schedule"],
            )
            learner.begin()
            state["learner"] = learner
        reference = original(*clone_static(args), **clone_static(kwargs))
        if learner is not None:
            # Reference discovers; the existing sync probe and two warmups
            # provide three passes of real sampling without another forward.
            learner.prepare(warmup_passes=3)
        graph = graphs[graph_key] = StableGraph(
            PackedForward(original, learner),
            device=torch.cuda.current_stream().device,
            max_entries=8,
            warmup_iters=2,
            name=f"head-matrix/{component}",
            group=get_world_group().device_group,
        )
        graph.output_type = type(reference)
    captures_before, calls_before = graph.captures, state["direct_calls"]
    learner = graph.module.learner
    if learner is not None and learner.schedule != state["schedule"]:
        raise ValueError("A communication schedule must remain fixed within its graph case")
    if learner is not None and learner.candidate_family != state["candidate_family"]:
        raise ValueError("A candidate family must remain fixed within its graph case")
    requested_candidate = state["candidate_partition"]
    if learner is not None and learner.candidate_partition != (
        None if requested_candidate is None else tuple(requested_candidate)
    ):
        raise ValueError("An experimental candidate must remain fixed within its graph case")
    body_calls_before = graph.module.forward_calls
    if learner is not None and learner.phase == "frozen":
        state["learner"] = learner
    with torch.cuda.nvtx.range(f"DiT/{component}/graph"):
        result = graph(
            variant=learner.cache_key if learner is not None and learner.phase == "frozen" else None,
            packed_args=args,
            packed_kwargs=kwargs,
        )
    state["eager_forwards"][component] = (
        state["eager_forwards"].get(component, 0)
        + graph.module.forward_calls
        - body_calls_before
        + int(reference is not None)
    )
    if isinstance(result, dict) and graph.output_type is not dict:
        result = graph.output_type(**result)
    if not graph.enabled or not graph.num_graphs:
        raise RuntimeError(f"Requested graph was not captured: {component}")
    if graph.captures > captures_before and state["direct_calls"] == calls_before:
        record(
            kind="capture_rejected",
            component=component,
            variant=state["variant"],
            direct_calls=state["direct_calls"],
            fallbacks=state["fallbacks"],
            qkv_shapes=sorted(state["shapes"]),
            graph_stats=graph.stats(),
        )
        raise RuntimeError(f"The graph did not capture direct CE attention: {component}")
    if reference is not None:
        assert_equal(result, reference)
    if graph.captures > captures_before:
        record(
            kind="capture",
            component=component,
            variant=state["variant"],
            direct_calls=state["direct_calls"] - calls_before,
            qkv_shapes=sorted(state["shapes"]),
            graph_stats=graph.stats(),
        )
    return result


def real_prefix_call(original, graphs, component, *args, **kwargs):
    """Use two actual denoising calls, returning both outputs to the pipeline.

    The first discovers layouts. Only a bounded prefix of the second samples.
    They replace existing warmup opportunities. The original sync-debug probe,
    capture boundary and graph validation remain; none supplies timing samples.
    """
    from lazy_head_policy import LazyHeadPolicy

    state = _ACTIVE.get()
    graph_key = state.get("graph_variant", state["variant"])
    graph = graphs.get(graph_key)
    if graph is None:
        learner = LazyHeadPolicy(
            get_world_group().cpu_group,
            enabled=state["autotune"],
            candidate_partition=state["candidate_partition"],
            candidate_family=state["candidate_family"],
            schedule=state["schedule"],
        )
        learner.begin()
        graph = graphs[graph_key] = StableGraph(
            PackedForward(original, learner),
            device=torch.cuda.current_stream().device,
            max_entries=8,
            warmup_iters=0,
            name=f"head-matrix/{component}/real-prefix",
            group=get_world_group().device_group,
        )
        if state.get("capture_without_probes"):
            graph._probe_pending = False
    learner = graph.module.learner
    if learner.schedule != state["schedule"]:
        raise ValueError("A communication schedule must remain fixed within its graph case")
    if learner.candidate_family != state["candidate_family"]:
        raise ValueError("A candidate family must remain fixed within its graph case")
    requested_candidate = state["candidate_partition"]
    if learner.candidate_partition != (None if requested_candidate is None else tuple(requested_candidate)):
        raise ValueError("An experimental candidate must remain fixed within its graph case")
    state["learner"] = learner
    before = graph.module.forward_calls
    if graph.module.prefix_steps < 2:
        if graph.module.prefix_steps == 1:
            # All layouts came from step one; allocation is complete before
            # the first trial in step two, without a dummy model invocation.
            learner.prepare(sample_limit=32 if learner.candidate_family == "three-then-two" else 24)
        result = graph.module(args, kwargs)
        graph.module.prefix_steps += 1
        state["real_eager_steps"][component] = state["real_eager_steps"].get(component, 0) + 1
        graph.output_type = type(result)
        if graph.module.prefix_steps == 2:
            learner.phase = "settle"
            if state.get("capture_without_probes"):
                packed = dict(packed_args=args, packed_kwargs=kwargs)
                graph._capture((kwargs_signature(None), kwargs_signature(packed)), packed)
            else:
                validation = graph(variant=None, packed_args=args, packed_kwargs=kwargs)
                if isinstance(validation, dict) and graph.output_type is not dict:
                    validation = graph.output_type(**validation)
                assert_equal(validation, result)
            if not graph.enabled or graph.captures != 1:
                raise RuntimeError("Real-prefix graph capture failed")
            record(
                kind="capture",
                component=component,
                variant=state["variant"],
                sample_location="real denoising step two prefix",
                real_eager_steps=2,
                graph_stats=graph.stats(),
            )
        # In particular, do not substitute the validation replay's output:
        # these are the actual eager denoising results, not discarded probes.
    else:
        result = graph(variant=learner.cache_key, packed_args=args, packed_kwargs=kwargs)
        if isinstance(result, dict) and graph.output_type is not dict:
            result = graph.output_type(**result)
    state["eager_forwards"][component] = state["eager_forwards"].get(component, 0) + graph.module.forward_calls - before
    return result


def transformer_blocks(owner, hidden_states, encoder_hidden_states, *, hidden_first=False, **kwargs):
    def body(hidden_states, encoder_hidden_states, **kwargs):
        for block in owner.transformer_blocks:
            result = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, **kwargs)
            if hidden_first:
                hidden_states, encoder_hidden_states = result
            else:
                encoder_hidden_states, hidden_states = result
        return (hidden_states, encoder_hidden_states) if hidden_first else (encoder_hidden_states, hidden_states)

    state = _ACTIVE.get()
    if hidden_first and state is not None:
        from lazy_head_policy import metadata_spec

        # CFG branches may trim to different text lengths. They need separate
        # observations and graphs even when their video tensors match.
        signature, _ = metadata_spec((hidden_states, encoder_hidden_states, kwargs))
        key = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
        state["graph_variant"] = (state["variant"], key)
    try:
        return graph_call(
            body, owner._head_matrix_graphs, "transformer/blocks", hidden_states, encoder_hidden_states, **kwargs
        )
    finally:
        if hidden_first and state is not None:
            state.pop("graph_variant")


def qwen21_blocks(owner, hidden_states, *, block_caches, cache_branch, cache_write_len, **kwargs):
    """Prefill stays original; learn from the first two real decode calls."""

    def body(hidden_states, block_caches, **kwargs):
        for index, block in enumerate(owner.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                kv_cache=None if block_caches is None else block_caches[index],
                cache_branch=cache_branch,
                cache_write_len=cache_write_len,
                **kwargs,
            )
        return hidden_states

    state = _ACTIVE.get()
    if state is None:
        return body(hidden_states, block_caches, **kwargs)
    if cache_write_len is not None or block_caches is None:
        state["whole_prefix"] = True
        state["original_prefill_blocks"] += len(owner.transformer_blocks)
        try:
            return body(hidden_states, block_caches, **kwargs)
        finally:
            state["whole_prefix"] = False
    state["graph_variant"] = (state["variant"], cache_branch)
    state["capture_without_probes"] = True
    try:
        return graph_call(
            body,
            owner._head_matrix_graphs,
            f"transformer/blocks/{cache_branch}",
            hidden_states,
            block_caches=block_caches,
            **kwargs,
        )
    finally:
        state.pop("graph_variant")
        state.pop("capture_without_probes")


def install(pipeline):
    # Import the patched attention owner before the native probe, whose
    # microbenchmark utilities also reference that owner during import.
    head = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    import native_prefilled as native
    from prefilled_probe import PrefilledPipeline
    from schedule_probe import run as scheduled

    original_can_overlap = head.can_overlap

    def can_overlap(*args, **kwargs):
        state = _ACTIVE.get()
        if state is not None and (state.get("whole_prefix") or state.get("reference_whole")):
            return False
        return original_can_overlap(*args, **kwargs)

    head.can_overlap = can_overlap

    counters = {}
    if os.environ.get("HCMS_LAZY_COUNTERS") == "1":
        from execution_counters import install as install_counters

        counters = install_counters()
    original_native = native.run

    def scheduled_call(plan):
        state = _ACTIVE.get()
        return scheduled(plan, "old" if state is None else state["schedule"])

    PrefilledPipeline.__call__ = scheduled_call

    @functools.wraps(original_native)
    def native_run(q, k, v, md, pg, chunks, kernel, **kwargs):
        state = _ACTIVE.get()
        if state is None:
            return original_native(q, k, v, md, pg, chunks, kernel, **kwargs)
        fallback = kwargs["fallback"]

        def counted_fallback(*args, **options):
            state["fallbacks"] += 1
            return fallback(*args, **options)

        kwargs["fallback"] = counted_fallback
        kwargs["head_partition"] = state["head_partition"]
        shape = (
            tuple(q.shape),
            tuple(q.stride()),
            str(q.dtype),
            0 if md is None or md.joint_query is None else md.joint_query.shape[1],
        )
        state["shapes"].add(shape)
        before = state["fallbacks"]
        slot = (state["variant"], "borrowed") if state["borrow"] else None
        with native.output_slot(slot):
            if state.get("learner") is not None:
                result = state["learner"].run(q, k, v, md, pg, kernel, streams=kwargs["streams"])
            else:
                result = original_native(q, k, v, md, pg, chunks, kernel, **kwargs)
        state["direct_calls"] += int(before == state["fallbacks"])
        return result

    native.run = native_run
    head._run_native_prefilled = native_run
    components = {}
    for name in getattr(pipeline, "_dit_modules", ("transformer", "transformer_2")):
        module = getattr(pipeline, name, None)
        if module is None:
            continue
        original = module.forward
        graphs = {}
        components[name] = graphs
        from vllm_omni.diffusion.attention.layer import Attention

        for attention_name, attention in module.named_modules():
            if not isinstance(attention, Attention) or attention.skip_sequence_parallel:
                continue

            registration = []

            def capability_hook(
                attention, args, component=name, attention_name=attention_name, registration=registration
            ):
                if _ACTIVE.get() is None or torch.cuda.is_current_stream_capturing():
                    return
                active = attention._get_active_parallel_strategy()
                metadata = args[3] if len(args) > 3 else None
                _, backend, _ = attention.effective_attention()
                record(
                    kind="attention_capability",
                    component=component,
                    attention=attention_name,
                    configured_strategy=type(attention.parallel_strategy).__name__,
                    active_strategy=type(active).__name__,
                    head_chunks=getattr(active, "head_chunks", None),
                    use_sync=getattr(active, "_use_sync", None),
                    backend=backend.get_name(),
                    qkv=[list(value.shape) for value in args[:3]],
                    dtype=str(args[0].dtype),
                    metadata_keys=[] if metadata is None else list(metadata.extra),
                )
                registration[0].remove()

            registration.append(attention.register_forward_pre_hook(capability_hook))
            break

        if getattr(module, "_head_matrix_block_graph", False):
            module._head_matrix_graphs = graphs
            continue

        def make_forward(original, graphs, component):
            @functools.wraps(original)
            def forward(*args, **kwargs):
                return graph_call(original, graphs, component, *args, **kwargs)

            return forward

        module.forward = make_forward(original, graphs, name)
    if not components:
        raise RuntimeError("The probe found no declared DiT components")
    record(
        kind="model",
        pipeline=type(pipeline).__name__,
        fp8_modules={
            name: sum(
                isinstance(weight := getattr(module, "weight", None), torch.Tensor)
                and weight.dtype == torch.float8_e4m3fn
                for module in getattr(pipeline, name).modules()
            )
            for name in components
        },
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        core_event_diagnostic=os.environ.get("HCMS_CORE_EVENTS") == "1",
        sample_sigmas=getattr(pipeline, "sample_sigmas", None),
    )
    original_forward = pipeline.forward
    compute = torch.cuda.Stream()
    previous_case = None
    pending_core_events = None
    core_events_enabled = os.environ.get("HCMS_CORE_EVENTS") == "1"

    @functools.wraps(original_forward)
    def forward(request):
        nonlocal previous_case, pending_core_events
        caller = torch.cuda.current_stream()
        compute.wait_stream(caller)
        token = None
        previous_chunks = native._SELECTED_CHUNKS
        counters_before = dict(counters)
        try:
            with torch.cuda.stream(compute):
                if request.is_dummy_run():
                    return original_forward(request)
                if pending_core_events is not None:
                    label, learners = pending_core_events
                    # The benchmark submits the next request only after the
                    # preceding decoded output reached CPU. Never add a wait.
                    record(
                        kind="core_events",
                        label=label,
                        components={name: learner.diagnostic_offsets_us() for name, learner in learners.items()},
                    )
                    pending_core_events = None
                extra = request.sampling_params.extra_args or {}
                case = extra["case"]
                if case != previous_case:
                    torch.accelerator.synchronize()
                    for graphs in components.values():
                        for graph in graphs.values():
                            graph.clear()
                        graphs.clear()
                    native.clear()
                    previous_case = case
                schedule = extra["schedule"]
                state = dict(
                    variant=extra["variant"],
                    mode=extra["graph_mode"],
                    schedule=schedule.removesuffix("-shared-slots"),
                    borrow=schedule.endswith("-slots"),
                    head_partition=extra.get("head_partition"),
                    direct_calls=0,
                    fallbacks=0,
                    shapes=set(),
                    autotune=bool(extra.get("autotune", False)),
                    candidate_partition=extra.get("candidate_partition"),
                    candidate_family=extra.get("candidate_family"),
                    real_prefix=bool(extra.get("real_prefix", False)),
                    eager_forwards={},
                    real_eager_steps={},
                    original_prefill_blocks=0,
                    reference_whole=bool(extra.get("reference_whole", False)),
                )
                token = _ACTIVE.set(state)
                native._SELECTED_CHUNKS = str(extra["chunks"])
                if extra.get("trace_start"):
                    torch.accelerator.synchronize()
                    dist.barrier()
                    if dist.get_rank() == 0:
                        torch.cuda.profiler.start()
                    dist.barrier()
                with torch.cuda.nvtx.range(f"request/{extra['label']}"):
                    result = original_forward(request)
                if extra.get("trace_stop"):
                    torch.accelerator.synchronize()
                    dist.barrier()
                    if dist.get_rank() == 0:
                        torch.cuda.profiler.stop()
                    dist.barrier()
                if state["fallbacks"]:
                    raise RuntimeError(f"The direct CE comparison fell back {state['fallbacks']} times")
                selected_graphs = {
                    name + (f"/{variant[1]}" if isinstance(variant, tuple) else ""): graph
                    for name, graphs in components.items()
                    for variant, graph in graphs.items()
                    if (variant[0] if isinstance(variant, tuple) else variant) == state["variant"]
                }
                stats = {name: graph.stats() for name, graph in selected_graphs.items()}
                learners = (
                    {}
                    if not core_events_enabled
                    else {
                        name: graph.module.learner
                        for name, graph in selected_graphs.items()
                        if graph.module.learner is not None and graph.module.learner.core_events is not None
                    }
                )
                if learners:
                    pending_core_events = extra["label"], learners
                if not stats and not state["direct_calls"] and not state["reference_whole"]:
                    raise RuntimeError("The requested head-overlap route was not executed")
                if extra.get("audit"):
                    record(
                        kind="request",
                        label=extra["label"],
                        variant=state["variant"],
                        mode=state["mode"],
                        direct_python_calls=state["direct_calls"],
                        fallbacks=state["fallbacks"],
                        requested_head_partition=state["head_partition"],
                        resident_head_partitions=sorted({plan.head_sizes for _, plan in native._PLANS.values()}),
                        qkv_shapes=sorted(state["shapes"]),
                        graph_stats=stats,
                        plans=len(native._PLANS),
                        output_slots=len(native._SLOT_ARENA),
                        output_slot_bytes=sum(slot.tensor.nbytes for slot in native._SLOT_ARENA.values()),
                        execution_counters={key: value - counters_before[key] for key, value in counters.items()},
                        eager_forwards=state["eager_forwards"],
                        real_eager_steps=state["real_eager_steps"],
                        original_prefill_blocks=state["original_prefill_blocks"],
                        reference_whole=state["reference_whole"],
                        lazy_policies={
                            name: dict(stats=graph.module.learner.stats, reports=graph.module.learner.reports)
                            for name, graph in selected_graphs.items()
                            if graph.module.learner is not None
                        },
                    )
                return result
        finally:
            native._SELECTED_CHUNKS = previous_chunks
            if token is not None:
                _ACTIVE.reset(token)
            caller.wait_stream(compute)

    pipeline.forward = forward
