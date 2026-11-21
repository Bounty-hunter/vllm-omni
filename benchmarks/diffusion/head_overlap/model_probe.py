# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit offline probe: fixed-case graphs and audited direct CE execution."""

import functools
import importlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from vllm_omni.diffusion.cuda_graph import GraphedForward, clone_static
from vllm_omni.diffusion.distributed.parallel_state import get_world_group

_ACTIVE = ContextVar("head_matrix_request", default=None)


class StableGraph(GraphedForward):
    """Keep warmup and capture on the CE buffers' owning compute stream."""

    @contextmanager
    def _warmup_stream(self):
        yield

    def _capture_graph(self, body):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool, stream=torch.cuda.current_stream(self.device)):
            outputs = body()
        if self._pool is None:
            self._pool = graph.pool()
        return graph, outputs


class PackedForward(nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.body = forward

    def forward(self, packed_args, packed_kwargs):
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
    graph = graphs.get(state["variant"])
    reference = None
    if graph is None:
        reference = original(*clone_static(args), **clone_static(kwargs))
        graph = graphs[state["variant"]] = StableGraph(
            PackedForward(original),
            device=torch.cuda.current_stream().device,
            max_entries=8,
            warmup_iters=2,
            name=f"head-matrix/{component}",
            group=get_world_group().device_group,
        )
        graph.output_type = type(reference)
    captures_before, calls_before = graph.captures, state["direct_calls"]
    with torch.cuda.nvtx.range(f"DiT/{component}/graph"):
        result = graph(packed_args=args, packed_kwargs=kwargs)
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


def transformer_blocks(owner, hidden_states, encoder_hidden_states, **kwargs):
    def body(hidden_states, encoder_hidden_states, **kwargs):
        for block in owner.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, **kwargs
            )
        return encoder_hidden_states, hidden_states

    return graph_call(
        body, owner._head_matrix_graphs, "transformer/blocks", hidden_states, encoder_hidden_states, **kwargs
    )


def install(pipeline):
    # Import the patched attention owner before the native probe, whose
    # microbenchmark utilities also reference that owner during import.
    head = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    import native_prefilled as native
    from prefilled_probe import PrefilledPipeline
    from schedule_probe import run as scheduled

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
    )
    original_forward = pipeline.forward
    compute = torch.cuda.Stream()
    previous_case = None

    @functools.wraps(original_forward)
    def forward(request):
        nonlocal previous_case
        caller = torch.cuda.current_stream()
        compute.wait_stream(caller)
        token = None
        previous_chunks = native._SELECTED_CHUNKS
        try:
            with torch.cuda.stream(compute):
                if request.is_dummy_run():
                    return original_forward(request)
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
                stats = {
                    name: graph.stats()
                    for name, graphs in components.items()
                    for variant, graph in graphs.items()
                    if variant == state["variant"]
                }
                if not stats and not state["direct_calls"]:
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
                    )
                return result
        finally:
            native._SELECTED_CHUNKS = previous_chunks
            if token is not None:
                _ACTIVE.reset(token)
            caller.wait_stream(compute)

    pipeline.forward = forward
