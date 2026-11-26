# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit offline probe: fixed-case graphs and audited direct CE execution."""

import functools
import json
import os
from contextvars import ContextVar
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cuda_graph import GraphedForward, kwargs_signature
from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.diffusion.forward_context import get_ulysses_mode

from . import ce_plan
from .head_policy import HeadPolicy, metadata_spec, tensor_spec
from .query_tail import eligible

_ACTIVE = ContextVar("head_matrix_request", default=None)


class StableGraph(GraphedForward):
    def _capture_graph(self, body):
        # Freeze at capture's existing synchronization boundary.
        torch.accelerator.synchronize()
        self.module.policy.freeze()
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin(pool=self._pool)
        try:
            outputs = body()
        finally:
            graph.capture_end()
        if self._pool is None:
            self._pool = graph.pool()
        return graph, outputs


class Component(nn.Module):
    def __init__(self, body, policy):
        super().__init__()
        self.body, self.policy = body, policy
        self.prefix_steps = 0

    def forward(self, packed_args, packed_kwargs):
        return self.body(*packed_args, **packed_kwargs)


def component_call(graphs, original, *args, **kwargs):
    state = _ACTIVE.get()
    if state is None:
        return original(*args, **kwargs)
    packed = dict(packed_args=args, packed_kwargs=kwargs)
    key = (state["variant"], kwargs_signature(packed))
    graph = graphs.get(key)
    if graph is None:
        policy = HeadPolicy(get_world_group().cpu_group, adaptive=state["variant"] == "auto")
        graph = graphs[key] = StableGraph(
            Component(original, policy),
            device=torch.cuda.current_stream().device,
            warmup_iters=0,
            max_entries=1,
            name="head-overlap",
            group=get_world_group().device_group,
        )
        graph._probe_pending = False
    module = graph.module
    policy = module.policy
    state["policy"] = policy
    if module.prefix_steps < 2:
        if module.prefix_steps == 1:
            policy.prepare()
        result = module(args, kwargs)
        module.prefix_steps += 1
        graph.output_type = type(result)
        if module.prefix_steps == 2:
            # Only capture records another Python body. No probe or validation replay.
            entry = graph._capture((kwargs_signature(None), key[1]), packed)
            if entry is None:
                raise RuntimeError("Could not capture the head-overlap experiment")
            graph._entries[(kwargs_signature(policy.cache_key), key[1])] = graph._entries.pop(
                (kwargs_signature(None), key[1])
            )
        return result
    result = graph(variant=policy.cache_key, **packed)
    if isinstance(result, dict) and graph.output_type is not dict:
        return graph.output_type(**result)
    return result


def install_attention(layer, streams):
    # Install on the benchmark instance only. A zero return uses native
    # pre_attention / post_attention, including its packed QKV transport.
    def chunks(strategy, q, k, v, md, paged):
        state = _ACTIVE.get()
        if state is None or state["variant"] == "native" or state.get("prefill"):
            return 0
        policy = state.get("policy")
        if (
            policy is None
            or policy.native
            or not policy.supported
            or paged
            or layer.use_ring
            or not getattr(strategy, "_ulysses_a2a_permute", False)
            or strategy._use_sync
            or strategy._scatter_idx != 2
            or strategy._gather_idx != 1
            or strategy._sp_group.ring_world_size != 1
            or get_ulysses_mode(default="strict") != "strict"
        ):
            return 0
        world = strategy._sp_group.ulysses_world_size
        if (
            world < 2
            or q.ndim != 4
            or q.shape[0] != 1
            or q.shape[1] == 0
            or q.shape[2] % world
            or q.shape != k.shape
            or q.shape != v.shape
            or q.dtype != torch.bfloat16
            or not q.is_cuda
            or any(
                x.dtype != q.dtype
                or x.device != q.device
                or x.stride(3) != 1
                or x.stride(2) != x.shape[3]
                or x.stride(1) < x.shape[2] * x.shape[3]
                for x in (q, k, v)
            )
        ):
            return 0
        md = layer._with_kv_cache_dtype(md)
        if not eligible(layer._run_local_attention, md, q.shape[1] * world):
            return 0
        if md is not None:
            joint = (md.joint_query, md.joint_key, md.joint_value)
            if any(x is not None for x in joint) and (
                any(
                    x is None
                    or x.ndim != 4
                    or x.shape[0] != 1
                    or x.shape[2:] != q.shape[2:]
                    or x.dtype != q.dtype
                    or x.device != q.device
                    or x.stride(3) != 1
                    or x.stride(2) != q.shape[3]
                    for x in joint
                )
                or joint[1].shape != joint[2].shape
                or md.joint_strategy not in ("front", "rear")
            ):
                return 0
        if policy.adaptive and policy.phase == "discover":
            # Exercise the native fallback on one real layer of each layout,
            # before sampling or capture can switch back to it.
            layout = (world, tuple(tensor_spec(x) for x in (q, k, v)), metadata_spec(md)[0])
            seen = state["native_layouts"]
            if layout not in seen:
                seen[layout] = False
            elif not seen[layout]:
                seen[layout] = True
                return 0
        return 1

    def run(strategy, q, k, v, md, chunks):
        pg = strategy._ulysses_pg
        key = (q.device, pg.group_name)
        if key not in streams:
            streams[key] = torch.cuda.Stream(device=q.device)
        return _ACTIVE.get()["policy"].run(
            q, k, v, layer._with_kv_cache_dtype(md), pg, layer._run_local_attention, streams[key]
        )

    layer._ulysses_ce_overlap_chunks = chunks
    layer._forward_ulysses_ce_overlap = run


def wrap_component(module, graphs):
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Transformer2DModel

    if isinstance(module, QwenImage21Transformer2DModel):
        original = module._run_transformer_blocks

        @functools.wraps(original)
        def blocks(hidden_states, block_caches, **kwargs):
            state = _ACTIVE.get()
            if block_caches is None or kwargs.get("cache_write_len") is not None:
                if state is None:
                    return original(hidden_states, block_caches, **kwargs)
                state["prefill"] = True
                try:
                    return original(hidden_states, block_caches, **kwargs)
                finally:
                    state["prefill"] = False
            return component_call(graphs, original, hidden_states, block_caches, **kwargs)

        module._run_transformer_blocks = blocks
    else:
        module.forward = functools.partial(component_call, graphs, module.forward)


class HeadOverlapWorker:
    """Worker extension selected explicitly by the offline benchmark."""

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        pipeline = self.model_runner.pipeline
        graphs, streams = {}, {}
        for name in getattr(pipeline, "_dit_modules", ("transformer", "transformer_2")):
            module = getattr(pipeline, name, None)
            if module is None:
                continue
            for layer in module.modules():
                if (
                    isinstance(layer, Attention)
                    and not layer.skip_sequence_parallel
                    and not layer._explicit_cross_contract
                ):
                    install_attention(layer, streams)
            graphs[name] = {}
            wrap_component(module, graphs[name])
        if not graphs:
            raise ValueError("Expected a diffusion transformer")
        original = pipeline.forward
        compute = torch.cuda.Stream()
        previous_case = None

        @functools.wraps(original)
        def forward(request):
            nonlocal previous_case
            caller = torch.cuda.current_stream()
            compute.wait_stream(caller)
            try:
                with torch.cuda.stream(compute):
                    if request.is_dummy_run():
                        return original(request)
                    extra = request.sampling_params.extra_args
                    case, variant = extra["case"], extra["variant"]
                    if case != previous_case:
                        torch.accelerator.synchronize()
                        for components in graphs.values():
                            for graph in components.values():
                                graph.clear()
                            components.clear()
                        ce_plan.clear()
                        previous_case = case
                    token = _ACTIVE.set(dict(variant=variant, native_layouts={}))
                    try:
                        result = original(request)
                        if extra.get("audit"):
                            reports = [
                                dict(
                                    component=name,
                                    captures=graph.captures,
                                    prefix_steps=graph.module.prefix_steps,
                                    native=graph.module.policy.native,
                                    choices=graph.module.policy.reports,
                                    core_calls=graph.module.policy.core_calls,
                                    events=graph.module.policy.event_records,
                                    control_collectives=graph.module.policy.control_collectives,
                                )
                                for name, components in graphs.items()
                                for (arm, _), graph in components.items()
                                if arm == variant
                            ]
                            record(case=case, variant=variant, label=extra["label"], components=reports)
                        return result
                    finally:
                        _ACTIVE.reset(token)
            finally:
                caller.wait_stream(compute)

        pipeline.forward = forward


def record(**values):
    destination = Path(os.environ["HCMS_MATRIX_AUDIT"])
    destination.mkdir(parents=True, exist_ok=True)
    row = dict(rank=dist.get_rank(), pid=os.getpid(), **values)
    with (destination / f"rank{dist.get_rank()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
    print("HEAD_MATRIX_AUDIT " + json.dumps(row, default=str), flush=True)
