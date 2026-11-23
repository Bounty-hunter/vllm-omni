# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent global attention reference, repeated requests and shape changes."""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.local_model, *hardware_marks(res={"cuda": "H100"}, num_cards=2)]


def _worker(rank, path):
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_overlap
    from vllm_omni.diffusion.distributed.a2a_permute import clear_a2a_permute_workspaces
    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.set_device(torch.device("cuda", rank))
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=path)
    torch.manual_seed(31 + rank)

    def kernel(q, k, v, md):
        mask = None if md is None else md.attn_mask
        if mask is not None and mask.ndim == 2:
            mask = mask[:, None, None, :]
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, enable_gqa=True
        ).transpose(1, 2)

    def gather(x):
        peers = [torch.empty_like(x) for _ in range(2)]
        dist.all_gather(peers, x.contiguous())
        return torch.cat(peers, dim=1)

    with torch.inference_mode():
        checks = 0
        for seq, batch in ((19, 1), (47, 2), (11, 1), (19, 1)):
            for heads, kv_heads in ((16, 16), (32, 8)):
                for side, joint_len in (("front", 0), ("front", 7), ("rear", 5)):
                    q = torch.randn(batch, seq, heads, 32, device="cuda", dtype=torch.bfloat16)
                    k, v = [torch.randn(batch, seq, kv_heads, 32, device="cuda", dtype=q.dtype) for _ in range(2)]
                    full_q, full_k, full_v = map(gather, (q, k, v))
                    joint = [
                        torch.randn(batch, joint_len, h, 32, device="cuda", dtype=q.dtype)
                        for h in (heads, kv_heads, kv_heads)
                    ]
                    for tensor in joint:
                        if joint_len:
                            dist.broadcast(tensor, 0)
                    # Exercise one-sided masks and shared metadata reuse.
                    joint_mask = torch.ones(batch, joint_len, dtype=torch.bool, device="cuda")
                    if joint_len:
                        joint_mask[:, -1] = False
                    md = AttentionMetadata(
                        joint_query=joint[0],
                        joint_key=joint[1],
                        joint_value=joint[2],
                        joint_strategy=side,
                        joint_attn_mask=joint_mask if joint_len else None,
                    )
                    if joint_len:
                        full_q, full_k, full_v = [
                            torch.cat((j, x) if side == "front" else (x, j), dim=1)
                            for x, j in zip((full_q, full_k, full_v), joint)
                        ]
                    image_mask = torch.ones(batch, 2 * seq, dtype=torch.bool, device="cuda")
                    full_mask = torch.cat(
                        (joint_mask, image_mask) if side == "front" else (image_mask, joint_mask), dim=1
                    )
                    expected = kernel(full_q, full_k, full_v, AttentionMetadata(attn_mask=full_mask))
                    if joint_len and side == "front":
                        expected = torch.cat(
                            (
                                expected[:, :joint_len],
                                expected[:, joint_len + rank * seq : joint_len + (rank + 1) * seq],
                            ),
                            dim=1,
                        )
                    elif joint_len:
                        expected = torch.cat(
                            (expected[:, rank * seq : (rank + 1) * seq], expected[:, -joint_len:]), dim=1
                        )
                    else:
                        expected = expected[:, rank * seq : (rank + 1) * seq]
                    for chunks in (2, 4, 2):
                        actual = head_overlap(q, k, v, md, dist.group.WORLD, chunks, kernel)
                        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
                        assert md.joint_query is joint[0]
                        checks += 1
        print(f"rank={rank}: {checks} independent attention/layout/reuse checks passed", flush=True)
        with torch.cuda.stream(torch.cuda.Stream()):
            with pytest.raises(RuntimeError, match="single compute stream"):
                head_overlap(q, k, v, md, dist.group.WORLD, 2, kernel)
    clear_a2a_permute_workspaces()
    dist.destroy_process_group()


def test_head_overlap_global_reference(tmp_path):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Requires two scheduler-allocated NVLink GPUs")
    torch.multiprocessing.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)
