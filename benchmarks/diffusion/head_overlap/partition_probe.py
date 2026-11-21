# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unequal CE head groups: global reference, graph residency, and timing."""

import argparse
import copy
import importlib
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

PARTITIONS = {
    "whole": (1,),
    "equal": (1, 1),
    "ratio12": (1, 2),
    "ratio13": (1, 3),
    "ratio21": (2, 1),
    "ratio31": (3, 1),
    "heads57": (5, 7),
    "heads75": (7, 5),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    pg = dist.group.WORLD
    control = dist.new_group(backend="gloo")
    world = dist.get_world_size(pg)
    importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    import native_prefilled as native
    from native_prefilled import NativePipeline, partition_heads
    from overlap_probe import FlashAttentionImpl, UlyssesParallelAttention
    from schedule_probe import run as scheduled

    assert partition_heads(12, 2, (1, 2)) == (4, 8)
    assert partition_heads(12, 2, (1, 3)) == (3, 9)
    for heads, chunks, weights in (
        (14, 2, (1, 3)),
        (12, 2, (0, 1)),
        (12, 2, (1,)),
        (12, 2, (1.0, 2)),
        (12, 0, None),
        (12, 2, (True, 1)),
    ):
        try:
            partition_heads(heads, chunks, weights)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid head partition was accepted")
    group = SimpleNamespace(
        ulysses_group=pg, ulysses_world_size=world, ulysses_rank=rank, ring_world_size=1, allgather_world_size=1
    )
    strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    streams = {}
    impl = FlashAttentionImpl(12, 128, 128**-0.5)
    records = []
    entries = []
    heads = world * 12

    def kernel(q, k, v, metadata):
        return impl.forward_cuda(q, k, v, metadata)

    def failure(*args, **kwargs):
        raise AssertionError("Partition test fell back")

    with torch.inference_mode():
        # 8 partitions x 2 layouts fill the bounded 16-plan cache. Every graph
        # remains resident while another partition/layout rebinds its inputs.
        for total, joint_length in ((2112, 0), (2176, 17)):
            torch.manual_seed(10448 + rank)
            q, k = (torch.randn(1, total // world, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))
            projected = torch.randn(1, total // world, 3, heads, 128, device="cuda", dtype=q.dtype)
            inputs = (q, k, projected[:, :, 2])
            torch.manual_seed(12345)
            joint = tuple(torch.randn(1, joint_length, heads, 128, device="cuda", dtype=q.dtype) for _ in range(3))
            md = (
                None
                if not joint_length
                else AttentionMetadata(
                    joint_query=joint[0], joint_key=joint[1], joint_value=joint[2], joint_strategy="front"
                )
            )
            packed_q, packed_k, packed_v, packed_md, context = strategy.pre_attention(*inputs, copy.copy(md))
            expected = strategy.post_attention(kernel(packed_q, packed_k, packed_v, packed_md), context).clone()
            for name, weights in PARTITIONS.items():
                native._SELECTED_CHUNKS = str(len(weights))
                selected = None if len(weights) == 1 else weights

                def body(inputs=inputs, md=md, selected=selected, name=name, total=total):
                    with native.output_slot((total, name)):
                        return native.run(
                            *inputs, md, pg, 2, kernel, fallback=failure, streams=streams, head_partition=selected
                        )

                # Native's per-request chunk override is a Python capture-time
                # choice. Replaying the graph must never re-read mutable state.
                from prefilled_probe import PrefilledPipeline

                PrefilledPipeline.__call__ = lambda plan: scheduled(plan, "stage-local")
                for _ in range(3):
                    actual = body()
                torch.accelerator.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = body()
                entries.append((graph, output, expected))
        assert len(native._PLANS) == 16
        for iteration in range(8):
            for index, (graph, output, expected) in enumerate(entries if iteration % 2 else entries[::-1]):
                graph.replay()
                if rank == (iteration + index) % world:
                    torch.cuda._sleep(1_000_000)
                consumed = output.clone()
                torch.accelerator.synchronize()
                torch.testing.assert_close(consumed, expected, rtol=0, atol=0)
        print(
            f"PARTITION_QA rank={rank} world={world}: 16 resident graphs x 8 delayed-consumer replays exact", flush=True
        )
        entries.clear()
        del graph, output
        native.clear()

        if args.timing:
            comm = torch.cuda.Stream()
            for total in (1950, 3510, 6630, 7920):
                torch.manual_seed(10448 + rank)
                q, k = (
                    torch.randn(1, total // world, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
                )
                projected = torch.randn(1, total // world, 3, heads, 128, device="cuda", dtype=q.dtype)
                inputs = (q, k, projected[:, :, 2])
                packed_q, packed_k, packed_v, packed_md, context = strategy.pre_attention(*inputs, None)
                expected = strategy.post_attention(kernel(packed_q, packed_k, packed_v, packed_md), context).clone()
                graphs, plans = {}, []
                for name, weights in PARTITIONS.items():
                    sizes = partition_heads(12, len(weights), weights)
                    plan = NativePipeline(inputs, None, pg, len(weights), comm, kernel, None, head_sizes=sizes)
                    plans.append(plan)

                    def body(plan=plan, name=name):
                        value = scheduled(plan, "stage-local")
                        return value.clone() if name == "whole" else value

                    for _ in range(6):
                        body()
                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = body()
                    graph.replay()
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    graphs[name] = graph, output
                samples = {name: [] for name in graphs}
                for iteration in range(30):
                    names = list(graphs)
                    offset = iteration % len(names)
                    names = names[offset:] + names[:offset]
                    if iteration % 2:
                        names.reverse()
                    for name in names:
                        dist.barrier(group=control)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                        for _ in range(16):
                            graphs[name][0].replay()
                        end.record(stream)
                        end.synchronize()
                        peers = [None] * world
                        dist.all_gather_object(peers, start.elapsed_time(end) * 1000 / 16, group=control)
                        samples[name].append(max(peers))
                row = dict(
                    tokens=total,
                    world=world,
                    exact=True,
                    partitions=PARTITIONS,
                    median_us={name: statistics.median(values) for name, values in samples.items()},
                    samples_us=samples,
                )
                records.append(row)
                print("PARTITION_TIMING " + json.dumps({k: v for k, v in row.items() if k != "samples_us"}), flush=True)
                torch.accelerator.synchronize()
                graphs.clear()
                for plan in plans:
                    plan.kernel = plan.attention_kernel = None
                plans.clear()
                del graph, output
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"rank{rank}.json").write_text(json.dumps(dict(rank=rank, world=world, records=records), indent=2))
    dist.destroy_process_group(control)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
