# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Automatic serving-policy correctness and paired native-baseline acceptance.

Inputs contain layouts only. No candidate, selected partition or expected
winner is prescribed. Fixed-partition scheduling diagnostics remain separate.
All synchronizations and extra timing markers below belong to the standalone
test, outside actual cores; they are not installed in serving execution.
"""

import argparse
import copy
import importlib
import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from auto_policy_cases import TYPICAL_CASES, WAN_CASES
from auto_policy_statistics import acceptance


def make_inputs(case, world, rank):
    from vllm_omni.diffusion.attention.backends.abstract import (
        AttentionMetadata,
        PackedPaddingMetadata,
        QueryRange,
        VideoTokenLayout,
    )

    torch.manual_seed(10448 + rank)
    shape = (1, case.tokens // world, case.heads, case.dim)
    inputs = tuple(torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    if case.fused_v:
        projection = torch.randn((1, shape[1], 3, case.heads, case.dim), device="cuda", dtype=torch.bfloat16)
        inputs = inputs[:2] + (projection[:, :, 2],)
    metadata = AttentionMetadata()
    if case.joint_kv:
        joint = tuple(
            torch.randn((1, length, case.heads, case.dim), device="cuda", dtype=torch.bfloat16)
            for length in (case.joint_q, case.joint_kv, case.joint_kv)
        )
        for tensor in joint:
            if tensor.numel():
                dist.broadcast(tensor, 0)
        metadata.joint_query, metadata.joint_key, metadata.joint_value = joint
    if case.padding:
        valid = case.tokens - case.padding
        boundaries = torch.tensor([0, valid, case.tokens], device="cuda", dtype=torch.int32)
        metadata.extra = dict(
            cu_seqlens_q=boundaries,
            cu_seqlens_k=boundaries,
            max_seqlen_q=valid,
            max_seqlen_k=valid,
        )
        # Publish the producer's host-known lengths, as real H3 does. Keeping
        # only opaque CUDA boundaries would test a different fallback path.
        metadata.packed_padding = PackedPaddingMetadata(valid, valid, boundaries[:2], boundaries[:2])
    if case.piecewise:
        metadata.full_attn_spans = [[(case.joint_kv, case.joint_kv + case.tokens)]]
        metadata.query_ranges = None if case.joint_q else (QueryRange(0, case.tokens, case.joint_kv),)
        metadata.attn_mask = torch.ones((1, 1, 1, case.tokens + case.joint_kv), device="cuda", dtype=torch.bool)
    if case.video_grid:
        assert math.prod(case.video_grid) == case.tokens
        metadata.extra["vsa_dit_seq_shape"] = case.video_grid
        metadata.video_layout = VideoTokenLayout(prefix_len=0, latent_grid=case.video_grid, used_len=case.tokens)
    return inputs, metadata


def oracle(inputs, metadata, core, world, rank):
    gathered = []
    for tensor in inputs:
        peers = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(peers, tensor.contiguous())
        gathered.append(torch.cat(peers, dim=1))
    joint = (metadata.joint_query, metadata.joint_key, metadata.joint_value)
    if joint[0] is not None:
        gathered = [torch.cat((prefix, tensor), dim=1) for prefix, tensor in zip(joint, gathered)]
    local = copy.copy(metadata)
    local.joint_query = local.joint_key = local.joint_value = None
    full = core.kernel(*gathered, local)
    prefix = 0 if joint[0] is None else joint[0].shape[1]
    length = inputs[0].shape[1]
    start = prefix + rank * length
    return torch.cat((full[:, :prefix], full[:, start : start + length]), dim=1)


def make_pipeline(case, inputs, metadata, group, layers, *, enabled):
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention

    head = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parallel_strategy = UlyssesParallelAttention(
                group,
                2,
                1,
                False,
                ulysses_a2a_permute=enabled,
                head_chunks=2 if enabled else 1,
            )
            self.impl = FlashAttentionImpl(case.heads // group.ulysses_world_size, case.dim, case.dim**-0.5)
            self.num_heads, self.head_size = case.heads, case.dim
            self.routes = dict(native=0, overlap=0, ordinary_ce=0)
            self.calls = 0
            self.kernel_heads = 0

        def kernel(self, q, k, v, md):
            self.kernel_heads += q.shape[2]
            return self.impl.forward_cuda(q, k, v, md)

        def forward(self, q, k, v, md):
            self.calls += 1
            strategy = self.parallel_strategy
            md = copy.copy(md)
            if strategy.head_chunks > 1 and head.can_overlap(q, k, v, md, group.ulysses_world_size, 2):
                self.routes["overlap"] += 1
                return head.head_overlap(q, k, v, md, group.ulysses_group, 2, self.kernel)
            self.routes["ordinary_ce" if strategy._ulysses_a2a_permute else "native"] += 1
            q, k, v, md, context = strategy.pre_attention(q, k, v, md)
            return strategy.post_attention(self.kernel(q, k, v, md), context)

    class Component(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Core()
            self.calls = 0

        @staticmethod
        def attention_query_token_bound(args, kwargs):
            q, _, _, md = args
            return q.shape[1] * group.ulysses_world_size + (0 if md.joint_query is None else md.joint_query.shape[1])

        def forward(self, q, k, v, md):
            self.calls += 1
            for _ in range(layers):
                result = self.attention(q, k, v, md)
            return result

    class Pipeline:
        _dit_modules = ("transformer",)

        def __init__(self):
            self.transformer = Component()
            self.od_config = SimpleNamespace(
                parallel_config=SimpleNamespace(ulysses_head_chunks=2 if enabled else 1),
                diffusion_use_graph=False,
            )
            self.requests = 0

        def forward(self, request):
            self.requests += 1
            return self.transformer(*inputs, metadata)

    return Pipeline()


@contextmanager
def installed_policy(pipeline):
    import lazy_head_policy
    import native_prefilled
    import serving_probe
    from prefilled_probe import PrefilledPipeline

    from vllm_omni.diffusion.distributed.a2a_permute import clear_a2a_permute_workspaces

    head = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    if not hasattr(head, "_run_native_prefilled"):
        raise RuntimeError("Apply native-probe.patch before running the automatic serving-policy test")
    saved = (
        head.can_overlap,
        head._run_native_prefilled,
        native_prefilled.run,
        PrefilledPipeline.__call__,
        lazy_head_policy.LazyHeadPolicy,
        serving_probe.record,
    )
    instances, audit = [], []

    class ObservedPolicy(lazy_head_policy.LazyHeadPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    lazy_head_policy.LazyHeadPolicy = ObservedPolicy
    serving_probe.record = lambda **row: audit.append(row)
    try:
        serving_probe.install(pipeline)
        yield instances, audit
    finally:
        torch.accelerator.synchronize()
        clear_a2a_permute_workspaces()
        (
            head.can_overlap,
            head._run_native_prefilled,
            native_prefilled.run,
            PrefilledPipeline.__call__,
            lazy_head_policy.LazyHeadPolicy,
            serving_probe.record,
        ) = saved


def completed_request(pipeline, request):
    start = time.perf_counter()
    output = pipeline.forward(request)
    torch.accelerator.synchronize()
    return output, (time.perf_counter() - start) * 1000


@contextmanager
def no_core_host_waits():
    from serving_probe import _ACTIVE

    saved = []
    for owner, name in (
        (torch.accelerator, "synchronize"),
        (torch.cuda, "synchronize"),
        (torch.cuda.Event, "synchronize"),
        (torch.Tensor, "item"),
        (dist, "all_gather_object"),
    ):
        original = getattr(owner, name)

        def checked(*args, _original=original, _name=name, **kwargs):
            policy = _ACTIVE.get()
            if policy is not None and policy.phase in ("sample", "settle", "frozen"):
                if _name != "item" or args[0].device.type != "cpu":
                    raise AssertionError(f"Blocking operation inside an actual sampled/frozen core: {_name}")
            return _original(*args, **kwargs)

        saved.append((owner, name, original))
        setattr(owner, name, checked)
    try:
        yield
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)


def timed_window(pipeline, request, requests, layers, control, events):
    dist.barrier(group=control)
    begin, end = events
    start = time.perf_counter()
    begin.record()
    for _ in range(requests):
        pipeline.forward(request)
    end.record()
    end.synchronize()
    wall = (time.perf_counter() - start) * 1e6 / (requests * layers)
    return dict(wall_us=wall, device_us=begin.elapsed_time(end) * 1000 / (requests * layers))


def check_case(case, options, rank, world, control, group):
    print(f"CASE_BEGIN rank={rank} {case.name}", flush=True)
    inputs, metadata = make_inputs(case, world, rank)
    native = make_pipeline(case, inputs, metadata, group, options.layers, enabled=False)
    automatic = make_pipeline(case, inputs, metadata, group, options.layers, enabled=True)
    request = SimpleNamespace(is_dummy_run=lambda: False, sampling_params=SimpleNamespace(num_inference_steps=1))
    expected = oracle(inputs, metadata, native.transformer.attention, world, rank)
    oracle_heads = native.transformer.attention.kernel_heads
    native_output, first_native_ms = completed_request(native, request)
    torch.testing.assert_close(native_output, expected, atol=2e-2, rtol=1e-2)
    with installed_policy(automatic) as (policies, audit):
        learning_ms = []
        # Each is an ordinary declared component request. Its only attention
        # calls are its layers; the learner cannot manufacture probe forwards.
        with no_core_host_waits():
            for _ in range(3):
                output, elapsed = completed_request(automatic, request)
                learning_ms.append(elapsed)
                torch.testing.assert_close(output, expected, atol=2e-2, rtol=1e-2)
        assert all(policy.phase in ("native", "frozen") for policy in policies), "Learning did not settle"
        decision = [dict(phase=policy.phase, reports=policy.reports) for policy in policies]
        peers = [None] * world
        dist.all_gather_object(peers, decision, group=control)

        def selected(rows):
            return [(row["phase"], [report["selected"] for report in row["reports"]]) for row in rows]

        assert all(selected(peer) == selected(decision) for peer in peers), "Ranks selected different partitions"
        learning_work = {}
        for name, pipeline, reference_work in (("native", native, oracle_heads), ("automatic", automatic, 0)):
            core = pipeline.transformer.attention
            actual = core.kernel_heads - reference_work
            expected_work = core.calls * case.heads // world
            assert actual == expected_work, "Learning executed extra attention work"
            learning_work[name] = dict(observed_head_work=actual, declared_head_work=expected_work)
            # Remove the test-only work counter before the 1% performance gate.
            # Plans bind the same backend callable on their next ordinary call.
            core.kernel = core.impl.forward_cuda
        frozen = [dict(policy.stats) for policy in policies]
        routes_before = dict(automatic.transformer.attention.routes)
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for event in events:
            event.record()
        events[-1].synchronize()
        calibration = timed_window(native, request, 1, options.layers, control, events)
        calibration_peers = [None] * world
        dist.all_gather_object(calibration_peers, calibration, group=control)
        request_ms = max(row["wall_us"] for row in calibration_peers) * options.layers / 1000
        requests = max(1, min(16, math.ceil(options.window_ms / request_ms)))
        windows = []
        for block in range(options.blocks):
            order = (
                ("native", "automatic", "automatic", "native")
                if block % 2 == 0
                else ("automatic", "native", "native", "automatic")
            )
            for arm in order:
                pipeline = native if arm == "native" else automatic
                windows.append(
                    dict(
                        block=block,
                        arm=arm,
                        **timed_window(
                            pipeline,
                            request,
                            requests,
                            options.layers,
                            control,
                            events,
                        ),
                    )
                )
        for policy, before in zip(policies, frozen):
            for key in ("event_records", "preparation_calls", "control_collectives"):
                assert policy.stats[key] == before[key], f"Steady-state probe: {key}"
        for pipeline in (native, automatic):
            assert pipeline.transformer.calls == pipeline.requests
            assert pipeline.transformer.attention.calls == options.layers * pipeline.requests
        core = automatic.transformer.attention
        routes = {key: value - routes_before[key] for key, value in core.routes.items()}
        assert routes["ordinary_ce"] == 0, "Steady fallback retained an ordinary experimental CE route"
        if core.parallel_strategy.head_chunks == 1:
            assert routes["overlap"] == 0 and routes["native"] > 0
            assert not core.parallel_strategy._ulysses_a2a_permute
        ranks = [None] * world
        dist.all_gather_object(ranks, windows, group=control)
        pairs, controls, device_pairs = [], [], []
        for block in range(options.blocks):
            native_samples, auto_samples = [], []
            for index in range(block * 4, block * 4 + 4):
                value = dict(
                    wall_us=max(peer[index]["wall_us"] for peer in ranks),
                    device_us=max(peer[index]["device_us"] for peer in ranks),
                )
                (native_samples if windows[index]["arm"] == "native" else auto_samples).append(value)
            pairs.append(
                tuple(statistics.mean(row["wall_us"] for row in rows) for rows in (native_samples, auto_samples))
            )
            device_pairs.append(
                tuple(statistics.mean(row["device_us"] for row in rows) for rows in (native_samples, auto_samples))
            )
            controls.append(
                tuple(row["wall_us"] for row in (native_samples if block % 2 == 0 else native_samples[::-1]))
            )
        verdict = acceptance(pairs, controls, limit=options.limit_pct / 100)
        return dict(
            case=asdict(case),
            world=world,
            qkv_shapes=[list(tensor.shape) for tensor in inputs],
            qkv_strides=[list(tensor.stride()) for tensor in inputs],
            decision=decision,
            route="native-nccl" if core.parallel_strategy.head_chunks == 1 else "ce-head-overlap",
            steady_routes=routes,
            rank_consistent=True,
            extra_forward_calls=0,
            learning_head_work=learning_work,
            independent_reference_calls=1,
            native_requests=native.requests,
            automatic_requests=automatic.requests,
            layers_per_request=options.layers,
            first_native_request_ms=first_native_ms,
            automatic_initial_requests_ms=learning_ms,
            first_shape_scope="Same-process initialization order; not an unbiased cold E2E comparison",
            policy_stats=frozen,
            frozen_probe_growth=dict(timers=0, preparation=0, control=0),
            sampled_core_host_waits=0,
            audit=audit,
            requests_per_window=requests,
            windows=windows,
            rank_windows=ranks,
            wall_acceptance=verdict,
            device_pairs_us=device_pairs,
            measurement_scope=(
                "Complete eager attention core and serving dispatch, native NCCL vs frozen automatic; no profiler"
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--suite", choices=("typical", "wan"), default="typical")
    parser.add_argument("--case", action="append", help="Select input cases only; never a head partition")
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--window-ms", type=float, default=100)
    parser.add_argument("--limit-pct", type=float, default=1.0)
    options = parser.parse_args()
    if options.layers < 32 or options.blocks < 6 or options.window_ms <= 0 or not 0 <= options.limit_pct < 100:
        parser.error("Need >=32 layers, >=6 blocks, positive window and a limit in [0,100)")
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    assert world == options.world
    control = dist.new_group(backend="gloo")
    group = SimpleNamespace(
        ulysses_group=dist.group.WORLD,
        ulysses_world_size=world,
        ulysses_rank=rank,
        ring_world_size=1,
        allgather_world_size=1,
    )
    from vllm_omni.diffusion.distributed import parallel_state

    parallel_state.get_sp_group = lambda: group
    parallel_state.get_world_group = lambda: SimpleNamespace(cpu_group=control)
    # Load the patched owner first: its native bridge must not see a partially
    # imported native_prefilled module.
    importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    cases = TYPICAL_CASES if options.suite == "typical" else WAN_CASES
    if options.case:
        unknown = set(options.case) - {case.name for case in cases}
        if unknown:
            parser.error(f"Unknown input cases: {sorted(unknown)}")
        cases = tuple(case for case in cases if case.name in options.case)
    options.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for case in cases:
            assert case.tokens % world == 0 and case.heads % world == 0
            row = check_case(case, options, rank, world, control, group)
            rows.append(row)
            (options.out / f"automatic-rank{rank}.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(f"CASE_END rank={rank} {case.name} {row['wall_acceptance']['status']} {row['route']}", flush=True)
    dist.barrier(group=control)
    dist.destroy_process_group()
    statuses = {row["wall_acceptance"]["status"] for row in rows}
    status = "regressed" if "regressed" in statuses else "inconclusive" if "inconclusive" in statuses else "pass"
    summary = dict(
        status=status,
        cases=len(rows),
        limit_pct=options.limit_pct,
        worst_latency_increase_pct=max(row["wall_acceptance"]["latency_increase_pct"] for row in rows),
        worst_ci95_upper_increase_pct=100 * max(row["wall_acceptance"]["ratio_ci95"][1] - 1 for row in rows),
        scope="Per-input steady complete-core mean; not a bound on every individual request",
    )
    (options.out / f"automatic-summary-rank{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(2 if "regressed" in statuses else 3 if "inconclusive" in statuses else 0)


if __name__ == "__main__":
    main()
