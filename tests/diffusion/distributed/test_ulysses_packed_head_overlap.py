# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real packed FlashAttention with pure head partitions on two/four ranks."""

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.parallel.head_overlap.head_policy import HeadPolicy, Trial, tensor_spec
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "H100"}, num_cards=4)]


def _worker(rank, init_method, world):
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy
    from vllm_omni.diffusion.distributed import a2a_permute

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(data_parallel_size=1, sequence_parallel_size=world, ulysses_degree=world)
    try:
        group = get_sp_group()
        native = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
        policy = HeadPolicy(group.cpu_group, 132)
        heads, dim = 56, 128
        local_heads = heads // world
        widths = tuple((h, local_heads - h) for h in range(1, local_heads // 2 + 1))
        impl = FlashAttentionImpl(num_heads=local_heads, head_size=dim, softmax_scale=dim**-0.5, causal=False)
        impl.fa_deterministic = True
        assert impl.stable_packed_head_partitions, "This regression requires vLLM FA3"
        checks, exact = 0, 0
        with torch.inference_mode():
            for rows in (128, 257, 1032, 2064):
                total, valid = rows * world, rows * world - 17
                cu = torch.tensor([0, valid, total], dtype=torch.int32, device=device)
                md = AttentionMetadata(
                    packed_padding=PackedPaddingMetadata(valid, valid, cu[:2], cu[:2]),
                    extra=dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=valid, max_seqlen_k=valid),
                )
                torch.manual_seed(101 + rank)
                packed = torch.randn(1, rows, 3, heads, dim, device=device, dtype=torch.bfloat16)
                inputs = packed.unbind(2)
                key = (world, tuple(map(tensor_spec, inputs)), (), None, None, dim**-0.5, False, (valid, valid))
                plans, _, reconnect = policy._prepare(
                    key, Trial(widths[0], alternatives=widths[1:]), device, group.ulysses_group
                )
                if reconnect:
                    policy.workspace.connect()
                for plan in plans:
                    plan.handle, plan.peer_base = policy.workspace.handle, policy.workspace.peer_base

                def kernel(q, k, v):
                    return impl.forward_cuda(q, k, v, md)

                qq, kk, vv, _, ctx = native.pre_attention(*inputs, md)
                reference = native.post_attention(kernel(qq, kk, vv), ctx).clone()
                for plan in (*plans, *reversed(plans)):
                    actual = plan.run(*inputs, kernel).clone()
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    exact += int(torch.equal(actual, reference))
                    checks += 1
                # Padding is a separate document, never a valid key for real rows.
                local_valid = max(0, min(rows, valid - rank * rows))
                packed[:, local_valid:] *= 10
                changed = plans[0].run(*packed.unbind(2), kernel)
                torch.testing.assert_close(changed[:, :local_valid], reference[:, :local_valid], rtol=0, atol=0)
                dist.barrier(group=group.cpu_group)
        print(f"packed overlap world={world} rank={rank}: {checks} comparisons, {exact} bit-exact", flush=True)
    finally:
        head_policy.clear()
        a2a_permute.clear_a2a_permute_workspaces()
        destroy_distributed_env()


@pytest.mark.parametrize("world", [2, 4])
def test_packed_flash_attention_head_partitions(tmp_path, world):
    if torch.accelerator.device_count() < world:
        pytest.skip(f"Requires {world} scheduler-allocated GPUs")
    torch.multiprocessing.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", world), nprocs=world)
