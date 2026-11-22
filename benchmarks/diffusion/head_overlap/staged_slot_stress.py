# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Asymmetric V producers and old-output consumers on a reused staged slot."""

import argparse
import copy
import os
import weakref
from types import SimpleNamespace

import torch
import torch.distributed as dist
from staged_attention import StagedAttention

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-fields", choices=("qk", "kv"), default="qk")
    parser.add_argument("--fieldwise-ready", action="store_true")
    args = parser.parse_args()
    early_fields = (0, 1) if args.early_fields == "qk" else (1, 2)
    groups = (
        ((0,), (1,), (2,))
        if args.fieldwise_ready
        else (early_fields, tuple(i for i in range(3) if i not in early_fields))
    )
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    pg = dist.group.WORLD
    control = dist.new_group(backend="gloo")
    world = dist.get_world_size()
    group = SimpleNamespace(
        ulysses_group=pg, ulysses_world_size=world, ulysses_rank=rank, ring_world_size=1, allgather_world_size=1
    )
    strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
    compute, comm = torch.cuda.Stream(), torch.cuda.Stream()
    compute.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(compute)
    impl = FlashAttentionImpl(32 // world, 128, 128**-0.5)
    with torch.inference_mode():
        for total, prefix in ((1056, 0), (4128, 4055)):
            torch.manual_seed(10448 + rank)
            frames = [
                tuple(torch.randn(1, total // world, 32, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
                for _ in range(2)
            ]
            joints = [None, None]
            if prefix:
                for i in range(2):
                    joints[i] = (
                        torch.empty(1, 0, 32, 128, device="cuda", dtype=torch.bfloat16),
                        torch.randn(1, prefix, 32, 128, device="cuda", dtype=torch.bfloat16),
                        torch.randn(1, prefix, 32, 128, device="cuda", dtype=torch.bfloat16),
                    )
                    for value in joints[i][1:]:
                        dist.broadcast(value, src=0)
            expected = []
            for frame, joint in zip(frames, joints, strict=True):
                metadata = (
                    None
                    if joint is None
                    else AttentionMetadata(joint_query=joint[0], joint_key=joint[1], joint_value=joint[2])
                )
                q, k, v, md, context = strategy.pre_attention(*frame, copy.copy(metadata))
                expected.append(strategy.post_attention(impl.forward_cuda(q, k, v, md), context).clone())
            prototypes = tuple(value.clone() for value in frames[0])
            prototype_joint = None if joints[0] is None else tuple(value.clone() for value in joints[0])
            references = [
                weakref.ref(value) for value in prototypes + (() if prototype_joint is None else prototype_joint)
            ]
            plan = StagedAttention(prototypes, prototype_joint, pg, comm, impl.forward_cuda, None, field_groups=groups)
            del prototypes, prototype_joint
            assert all(reference() is None for reference in references), "Plan retained preparation-only tensors"

            def body():
                saved = []
                for i in range(6):
                    frame = frames[i % 2]

                    def produce(group, last):
                        if last and rank != i % world:
                            torch.cuda._sleep(2_000_000)
                        # Ephemeral producers exercise source lifetime as well
                        # as delayed consumers; these copies are stress-only.
                        return tuple(frame[field].clone() for field in group)

                    producers = tuple(
                        (lambda group=group, last=j == len(groups) - 1: produce(group, last))
                        for j, group in enumerate(groups)
                    )
                    pooled = plan.run_staged(producers, joints[i % 2])
                    if rank == i % world:
                        torch.cuda._sleep(2_000_000)
                    saved.append(pooled.clone())
                return saved

            for _ in range(2):
                actual = body()
                torch.accelerator.synchronize()
                for i, value in enumerate(actual):
                    torch.testing.assert_close(value, expected[i % 2], atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=compute):
                outputs = body()
            for _ in range(6):
                dist.barrier(group=control)
                graph.replay()
                torch.accelerator.synchronize()
                for i, value in enumerate(outputs):
                    torch.testing.assert_close(value, expected[i % 2], atol=0, rtol=0)
            print(
                f"QKV_SLOT_STRESS rank={rank} tokens={total} prefix={prefix} early={args.early_fields} "
                "eager/graph exact "
                "under delayed remaining producers and delayed old-slot consumers; prototypes released",
                flush=True,
            )
            del graph, outputs, actual
            plan.kernel = plan.attention_kernel = None
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
