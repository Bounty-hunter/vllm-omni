# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Delay an output consumer while peers enqueue the following invocation."""

import importlib
import os
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    head = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    import native_prefilled

    from vllm_omni.diffusion.distributed.a2a_permute import clear_a2a_permute_workspaces

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    torch.manual_seed(432 + rank)
    impl = FlashAttentionImpl(8, 128, 128**-0.5)

    def kernel(q, k, v, md):
        return impl.forward_cuda(q, k, v, md)

    with torch.inference_mode():
        frames = [
            tuple(torch.randn(1, 1024, 16, 128, dtype=torch.bfloat16, device="cuda") for _ in range(3))
            for _ in range(2)
        ]
        expected = []
        for frame in frames:
            expected.append(head.head_overlap(*frame, None, dist.group.WORLD, 2, kernel))
            torch.accelerator.synchronize()
        plan = next(plan for _, plan in native_prefilled._PLANS.values())
        snapshots = []
        for iteration in range(16):
            frame = frames[iteration % 2]
            plan.bind(frame, None, kernel, None)
            pooled = plan()
            # In both cases, a peer can start next-call input writes while this
            # rank still has not consumed its previous final output buffer.
            if rank == 1 and iteration % 2 == 0:
                torch.cuda._sleep(15_000_000)
            if rank == 0 and iteration % 3 == 1:
                time.sleep(0.01)
            snapshots.append(pooled.clone())
        torch.accelerator.synchronize()
        for iteration, snapshot in enumerate(snapshots):
            torch.testing.assert_close(snapshot, expected[iteration % 2], atol=0, rtol=0)
        print(
            f"FENCE_STRESS rank={rank} 16 delayed-consumer/rebinding outputs bitwise exact",
            flush=True,
        )
    clear_a2a_permute_workspaces()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
