# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit-slot consumers under asymmetric delays, eager and CUDA Graph."""

import argparse
import importlib
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", choices=("original", "overlap-return", "overlap-staging"), default="original")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    pg = dist.group.WORLD
    control = dist.new_group(backend="gloo")
    head = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    import native_prefilled as native
    from overlap_probe import FlashAttentionImpl, UlyssesParallelAttention

    if args.schedule != "original":
        from schedule_probe import run as scheduled

        native.NativePipeline.__call__ = lambda plan: scheduled(plan, args.schedule)

    group = SimpleNamespace(
        ulysses_group=pg, ulysses_world_size=2, ulysses_rank=rank, ring_world_size=1, allgather_world_size=1
    )
    strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    impl = FlashAttentionImpl(28, 128, 128**-0.5)

    def kernel(q, k, v, metadata):
        return impl.forward_cuda(q, k, v, metadata)

    with torch.inference_mode():
        for total in (1056, 4128):
            torch.manual_seed(10448 + rank)
            frames = [
                tuple(torch.randn(1, total // 2, 56, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
                for _ in range(2)
            ]
            expected = []
            for frame in frames:
                q, k, v, metadata, context = strategy.pre_attention(*frame, None)
                expected.append(strategy.post_attention(kernel(q, k, v, metadata), context).clone())
            # Normal ownership must still hold across later calls.
            normal = [head.head_overlap(*frame, None, pg, 2, kernel) for frame in frames]
            assert native._PLANS, "Apply native-probe.patch before running slot stress"
            torch.accelerator.synchronize()
            for actual, reference in zip(normal, expected):
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)

            def body(count):
                saved = []
                for index in range(count):
                    with native.output_slot(("stress", index % 2)):
                        pooled = head.head_overlap(*frames[index % 2], None, pg, 2, kernel)
                    if rank == index % 2:
                        torch.cuda._sleep(2_000_000)
                    saved.append(pooled.clone())
                return saved

            for _ in range(2):
                actual = body(4)
            torch.accelerator.synchronize()
            for index, value in enumerate(actual):
                torch.testing.assert_close(value, expected[index % 2], atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = body(4)
            for repetition in range(4):
                dist.barrier(group=control)
                graph.replay()
                torch.accelerator.synchronize()
                for index, value in enumerate(captured):
                    torch.testing.assert_close(value, expected[index % 2], atol=0, rtol=0)
            for value, reference in zip(normal, expected):
                torch.testing.assert_close(value, reference, atol=0, rtol=0)
            print(
                f"SLOT_STRESS rank={rank} schedule={args.schedule} tokens={total} "
                "normal-owned and delayed-slot consumers bitwise exact",
                flush=True,
            )
            del graph, captured, actual, normal
            native.clear()
    from vllm_omni.diffusion.distributed.a2a_permute import clear_a2a_permute_workspaces

    clear_a2a_permute_workspaces()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
