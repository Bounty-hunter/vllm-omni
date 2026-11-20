# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Retain and alternate dense/joint CE layouts across live CUDA graphs."""

import copy
import importlib
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    pg = dist.group.WORLD
    world = dist.get_world_size(pg)
    if world not in (2, 4):
        raise ValueError("The layout stress qualifies SP2 and SP4")
    head = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    import native_prefilled as native
    from overlap_probe import FlashAttentionImpl, UlyssesParallelAttention
    from prefilled_probe import PrefilledPipeline
    from schedule_probe import run as scheduled

    PrefilledPipeline.__call__ = lambda plan: scheduled(plan, "stage-local")
    group = SimpleNamespace(
        ulysses_group=pg, ulysses_world_size=world, ulysses_rank=rank, ring_world_size=1, allgather_world_size=1
    )
    strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    impl = FlashAttentionImpl(56 // world, 128, 128**-0.5)

    def kernel(q, k, v, metadata):
        return impl.forward_cuda(q, k, v, metadata)

    entries = []
    with torch.inference_mode():
        for total, joint_length in ((2112, 0), (2112, 17), (4128, 0)):
            torch.manual_seed(10448 + rank)
            inputs = tuple(
                torch.randn(1, total // world, 56, 128, dtype=torch.bfloat16, device="cuda") for _ in range(3)
            )
            torch.manual_seed(12345)
            joint = tuple(torch.randn(1, joint_length, 56, 128, dtype=torch.bfloat16, device="cuda") for _ in range(3))
            metadata = (
                None
                if not joint_length
                else AttentionMetadata(
                    joint_query=joint[0], joint_key=joint[1], joint_value=joint[2], joint_strategy="front"
                )
            )
            if total == 2112 and not joint_length:
                metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (2, 33, 32), "preserve_vsa_all_blocks": True})
                assert head.can_overlap(*inputs, metadata, world, 2)
                assert not head.can_overlap(*inputs, AttentionMetadata(extra={"unknown_sparse_plan": True}), world, 2)
            q, k, v, kernel_md, context = strategy.pre_attention(*inputs, copy.copy(metadata))
            reference = strategy.post_attention(kernel(q, k, v, kernel_md), context).clone()

            def body(inputs=inputs, metadata=metadata):
                with native.output_slot("layout-stress"):
                    return head.head_overlap(*inputs, metadata, pg, 2, kernel).clone()

            for _ in range(2):
                actual = body()
            torch.accelerator.synchronize()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = body()
            entries.append((graph, output, reference))
        assert len(native._PLANS) == 3
        for iteration in range(8):
            order = entries if iteration % 2 else entries[::-1]
            for graph, output, reference in order:
                if rank == iteration % 2:
                    torch.cuda._sleep(1_000_000)
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
        print(f"LAYOUT_GRAPH_STRESS rank={rank} 3 resident layouts x 8 alternating replays bitwise exact", flush=True)
        entries.clear()
        del graph, output
        native.clear()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
