# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unprofiled controls for communication budget and head-splitting overhead."""

import argparse
import copy
import hashlib
import importlib
import json
import os
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.utils.fa import dense_flash_attn_func_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--inner", type=int, default=16)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    pg = dist.group.WORLD
    control = dist.new_group(backend="gloo")
    world = dist.get_world_size()
    importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    from native_prefilled import NativePipeline, partition_heads
    from overlap_probe import FlashAttentionImpl, UlyssesParallelAttention
    from schedule_probe import run as scheduled

    group = SimpleNamespace(
        ulysses_group=pg,
        ulysses_world_size=world,
        ulysses_rank=rank,
        ring_world_size=1,
        allgather_world_size=1,
    )
    strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    comm = torch.cuda.Stream()
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(
        rank=rank,
        world=world,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        code_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        attention_function=str(dense_flash_attn_func_for(torch.bfloat16)),
        rounds=args.rounds,
        inner=args.inner,
        records=[],
    )
    config = json.loads(args.cases.read_text())
    with torch.inference_mode():
        for case in config:
            total, heads, joint_length, padding = (case[key] for key in ("tokens", "heads", "joint", "padding"))
            assert total % world == 0 and heads % (2 * world) == 0
            torch.manual_seed(10448 + rank)
            local = total // world
            inputs = tuple(torch.randn(1, local, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            if case.get("fused_v", False):
                projected = torch.randn(1, local, 3, heads, 128, device="cuda", dtype=torch.bfloat16)
                # Compare source layouts at identical Q/K/V values. This copy
                # is setup only, outside every captured/timed control.
                projected[:, :, 2].copy_(inputs[2])
                inputs = inputs[:2] + (projected[:, :, 2],)
            joint = None
            metadata = None
            joint_strategy = case.get("joint_strategy", "front")
            if joint_length:
                joint = tuple(
                    torch.randn(1, joint_length, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
                )
                for value in joint:
                    dist.broadcast(value, src=0)
                if case.get("fused_joint_v", False):
                    projected_joint = torch.empty(1, joint_length, 3, heads, 128, device="cuda", dtype=torch.bfloat16)
                    projected_joint[:, :, 2].copy_(joint[2])
                    joint = joint[:2] + (projected_joint[:, :, 2],)
                if "joint_query_length" in case:
                    joint = (joint[0][:, : case["joint_query_length"]], *joint[1:])
                metadata = AttentionMetadata(
                    joint_query=joint[0],
                    joint_key=joint[1],
                    joint_value=joint[2],
                    joint_strategy=joint_strategy,
                )
            if padding:
                assert joint is None
                bounds = torch.tensor([0, total - padding, total], dtype=torch.int32, device="cuda")
                metadata = AttentionMetadata(
                    extra=dict(
                        cu_seqlens_q=bounds,
                        cu_seqlens_k=bounds,
                        max_seqlen_q=total - padding,
                        max_seqlen_k=total - padding,
                    )
                )
            impl = FlashAttentionImpl(heads // world, 128, 128**-0.5)
            q, k, v, packed_md, context = strategy.pre_attention(*inputs, copy.copy(metadata))
            expected = strategy.post_attention(impl.forward_cuda(q, k, v, packed_md), context).clone()
            packed = tuple(value.clone() for value in (q, k, v))
            chunks = len(case["head_partition"]) if case.get("head_partition") else 2
            head_sizes = partition_heads(heads // world, chunks, case.get("head_partition"))
            parts, offset = [], 0
            for width in head_sizes:
                parts.append(tuple(value[:, :, offset : offset + width].contiguous() for value in packed))
                offset += width

            def kernel(q, k, v, unused_md):
                return impl.forward_cuda(q, k, v, packed_md)

            placement = dict(joint_strategy=joint_strategy)
            whole = NativePipeline(inputs, joint, pg, 1, comm, kernel, packed_md, **placement)
            split = NativePipeline(
                inputs, joint, pg, chunks, comm, kernel, packed_md, head_sizes=head_sizes, **placement
            )
            serial_split = NativePipeline(
                inputs, joint, pg, chunks, comm, kernel, packed_md, head_sizes=head_sizes, **placement
            )
            serial_split.overlap = False
            transport = NativePipeline(inputs, joint, pg, 1, comm, lambda q, k, v, md: q, packed_md, **placement)
            split_transport = NativePipeline(
                inputs, joint, pg, chunks, comm, lambda q, k, v, md: q, packed_md, head_sizes=head_sizes, **placement
            )
            split_transport.overlap = False

            def attention_whole():
                return kernel(*packed, None)

            def attention_split():
                return tuple(kernel(*part, None) for part in parts)

            def peer_forward(plan=transport, group=0):
                plan.enter.record(stream)
                with torch.cuda.stream(comm):
                    comm.wait_event(plan.enter)
                    plan.submit_forward(group)
                stream.wait_event(plan.ready[group])

            def peer_reverse(plan=transport, group=0):
                plan.enter.record(stream)
                with torch.cuda.stream(comm):
                    comm.wait_event(plan.enter)
                    plan.submit_reverse(group, plan.parts[group][0])
                    plan.output.handle.barrier()
                    plan.done.record(comm)
                stream.wait_event(plan.done)

            functions = {
                "attention_whole": attention_whole,
                "attention_split_serial": attention_split,
                "attention_first": lambda: kernel(*parts[0], None),
                "attention_last": lambda: kernel(*parts[-1], None),
                "peer_forward": peer_forward,
                "peer_reverse": peer_reverse,
                "peer_forward_first": lambda: peer_forward(split_transport, 0),
                "peer_forward_last": lambda: peer_forward(split_transport, chunks - 1),
                "peer_reverse_first": lambda: peer_reverse(split_transport, 0),
                "peer_reverse_last": lambda: peer_reverse(split_transport, chunks - 1),
                "transport_whole_owned": lambda: scheduled(transport, "old").clone(),
                "transport_whole_borrowed": lambda: scheduled(transport, "old"),
                "transport_split_serial_borrowed": lambda: scheduled(split_transport, "stage-local"),
                "core_whole_owned": lambda: scheduled(whole, "old").clone(),
                "core_whole_borrowed": lambda: scheduled(whole, "old"),
                "core_split_serial_owned": lambda: scheduled(serial_split, "stage-local").clone(),
                "core_split_serial_borrowed": lambda: scheduled(serial_split, "stage-local"),
                "core_overlap_owned": lambda: scheduled(split, "stage-local").clone(),
                "core_overlap_borrowed": lambda: scheduled(split, "stage-local"),
            }
            for variant in case.get("staging_variants", ()):
                if variant not in ("overlap-return", "overlap-staging"):
                    raise ValueError(f"Unknown staging ablation: {variant}")
                functions[f"core_{variant}_borrowed"] = lambda variant=variant: scheduled(split, variant)
            if case.get("staging_variants"):

                def local_forward(plan=whole, group=0):
                    for operation in plan.local_forward[group]:
                        plan.emit_on(operation, stream)

                def local_reverse(plan=whole, group=0):
                    for operation in plan.reverse_operations(group, plan.parts[group][0], rank):
                        plan.emit_on(operation, stream)

                functions.update(
                    local_forward=local_forward,
                    local_reverse=local_reverse,
                    local_forward_last=lambda: local_forward(split, chunks - 1),
                    local_reverse_first=lambda: local_reverse(split, 0),
                )
            graphs = {}
            for name, function in functions.items():
                dist.barrier(group=control)
                for _ in range(6):
                    function()
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = function()
                graph.replay()
                torch.accelerator.synchronize()
                if name.startswith("core_"):
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                if name.startswith("transport_"):
                    identity = inputs[0]
                    if joint is not None:
                        identity = torch.cat(
                            (joint[0], identity) if joint_strategy == "front" else (identity, joint[0]), dim=1
                        )
                    torch.testing.assert_close(output, identity, rtol=0, atol=0)
                graphs[name] = (graph, output)
            torch.testing.assert_close(
                torch.cat(graphs["attention_split_serial"][1], dim=2),
                graphs["attention_whole"][1],
                rtol=0,
                atol=0,
            )
            local_samples = {name: [] for name in functions}
            max_samples = {name: [] for name in functions}
            for iteration in range(args.rounds):
                names = list(functions)
                # Rotate as well as reverse, so controls do not always follow
                # the same power/memory state left by their predecessor.
                offset = iteration % len(names)
                names = names[offset:] + names[:offset]
                if (iteration // len(names)) % 2:
                    names.reverse()
                for name in names:
                    dist.barrier(group=control)
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record(stream)
                    for _ in range(args.inner):
                        graphs[name][0].replay()
                    end.record(stream)
                    end.synchronize()
                    value = begin.elapsed_time(end) * 1000 / args.inner
                    local_samples[name].append(value)
                    peers = [None] * world
                    dist.all_gather_object(peers, value, group=control)
                    max_samples[name].append(max(peers))
            medians = {name: statistics.median(samples) for name, samples in max_samples.items()}
            peer = medians["peer_forward"] + medians["peer_reverse"]
            attention = medians["attention_whole"]
            tax = medians["attention_split_serial"] - attention
            first_fraction = head_sizes[0] / sum(head_sizes)
            hideable = (
                min(medians["attention_first"], medians["peer_forward"] * (1 - first_fraction))
                + min(medians["attention_last"], medians["peer_reverse"] * first_fraction)
                if chunks == 2
                else None
            )
            saving = medians["core_whole_owned"] - medians["core_overlap_borrowed"]
            row = dict(
                case=case,
                exact=True,
                full_heads=heads // world,
                chunk_heads=head_sizes[0] if len(set(head_sizes)) == 1 else None,
                head_sizes=head_sizes,
                attention_calls=chunks,
                forward_copy_requests=sum(len(ops) for ops in split.forward_ops),
                reverse_copy_requests=chunks * world * (1 + int(split.joint_len > 0)),
                forward_peer_fences=chunks,
                final_peer_fences=1,
                qkv_shape=list(inputs[0].shape),
                qkv_strides=[list(x.stride()) for x in inputs],
                joint_shapes=None if joint is None else [list(x.shape) for x in joint],
                joint_strides=None if joint is None else [list(x.stride()) for x in joint],
                median_max_rank_us=medians,
                samples_max_rank_us=max_samples,
                samples_local_us=local_samples,
                peer_fraction_of_attention_plus_peer=peer / (attention + peer),
                split_attention_tax_us=tax,
                ideal_hideable_peer_us=hideable,
                ideal_hideable_minus_split_tax_us=None if hideable is None else hideable - tax,
                hideable_using_measured_chunks_us=(
                    min(medians["attention_first"], medians["peer_forward_last"])
                    + min(medians["attention_last"], medians["peer_reverse_first"])
                    if chunks == 2
                    else None
                ),
                split_transport_tax_us=medians["transport_split_serial_borrowed"] - medians["transport_whole_borrowed"],
                measured_core_saving_us=saving,
                measured_core_owned_saving_us=medians["core_whole_owned"] - medians["core_overlap_owned"],
                measured_core_borrowed_saving_us=medians["core_whole_borrowed"] - medians["core_overlap_borrowed"],
                measured_saving_vs_split_serial_us=medians["core_split_serial_borrowed"]
                - medians["core_overlap_borrowed"],
                # Image-token traffic only; front-joint output traffic is not
                # included in this auxiliary byte estimate or used by timing.
                peer_outgoing_image_bytes_per_layer=int(4 * inputs[0].nbytes * (world - 1) / world),
                caveats=[
                    "Isolated peer costs include stream dependencies and required peer fences",
                    "transport control also includes local staging and owned output clone",
                    "Micro fractions are not directly measured full-request communication shares",
                    "Unequal hideable estimate scales whole peer times by head widths; "
                    "fixed chunk costs and interference are excluded",
                    "All core variants retain every required peer fence; no profiler is started",
                    "Each isolated reverse-chunk timing includes its own completion fence; "
                    "the real pipeline uses one final reverse completion fence",
                ],
            )
            result["records"].append(row)
            (args.out / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
            if rank == 0:
                print(
                    "CORE_BUDGET " + json.dumps({k: v for k, v in row.items() if not k.startswith("samples")}),
                    flush=True,
                )
            torch.accelerator.synchronize()
            graphs.clear()
            del graph, output
            functions.clear()
            for plan in (whole, split, serial_split, transport, split_transport):
                plan.kernel = plan.attention_kernel = None
            whole = split = serial_split = transport = split_transport = None
            from vllm_omni.diffusion.distributed.a2a_permute import (
                clear_a2a_permute_workspaces,
            )

            clear_a2a_permute_workspaces()
    dist.destroy_process_group(control)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
