# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental CUDA CE head pipeline. All peer fences remain enabled.

Each group exchanges a contiguous global head block; rank r computes its r-th
slice within that block. GQA groups stay intact and the reverse exchange
restores global head order. A shared communication stream owns a separate
grow-only CE workspace, so fallback calls on the compute stream stay valid.
Each head group retains its own CE output slots. The next call's producer
fence protects reuse, avoiding allocator-delayed frees across every DiT layer.
"""

from collections.abc import Callable
from dataclasses import replace

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.distributed.a2a_permute import _exchange_ce

_COMM_STREAMS: dict[tuple[torch.device, str], tuple[torch.cuda.Stream, int]] = {}
logger = init_logger(__name__)


def can_overlap(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    md: AttentionMetadata | None,
    world: int,
    chunks: int,
) -> bool:
    if q.device.type != "cuda" or chunks < 2 or world < 2:
        return False
    if q.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2]:
        return False
    if any(x.requires_grad for x in (q, k, v)) or k.shape[2] == 0:
        return False
    if q.shape[2] % (world * chunks) or k.shape[2] % (world * chunks) or q.shape[2] % k.shape[2]:
        return False
    if q.shape[2] // (world * chunks) < 4 or q.shape[1] * world < 2048:
        return False
    if md is None:
        return True
    if md.cross_query_mask is not None:
        return False
    # Dense kernels ignore geometry, but sparse backends must keep their plan.
    if set(md.extra) - {
        "attention_mask_mode",
        "kv_cache_dtype",
        "valid_kv_length",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "npu_attn_varlen",
        "laser_input_scale",
        "vsa_h3_sparsity",
        "vsa_h3_tile_size",
        "vsa_h3_ref_policy",
        "vsa_h3_ref_keep_rate",
        "vsa_dit_seq_shape",
        "preserve_vsa_all_blocks",
    }:
        return False
    if md.extra.get("kv_cache_dtype") not in (None, "auto", "float"):
        return False
    if md.attn_mask is not None and md.attn_mask.ndim == 4 and md.attn_mask.shape[1] != 1:
        return False
    if md.joint_attn_mask is not None and md.joint_attn_mask.ndim != 2:
        return False
    joint = (md.joint_query, md.joint_key, md.joint_value)
    if any(x is not None for x in joint):
        if not all(x is not None for x in joint) or md.joint_strategy not in ("front", "rear"):
            return False
        jq, jk, jv = joint
        if jq.shape[2] != q.shape[2] or jk.shape[2] != k.shape[2] or jk.shape != jv.shape:
            return False
    return True


@torch.compiler.disable
def head_overlap(
    q,
    k,
    v,
    md: AttentionMetadata | None,
    pg,
    chunks: int,
    kernel: Callable,
    *,
    prefetch_before_attention: bool = False,
):
    """Pipeline each attention group with the next input exchange.

    CUDA event waits order the producer, transport and consumers without a
    device synchronize. The CE transport's existing peer barriers protect
    each exchange and its output copy before workspace reuse.
    """
    world, rank = dist.get_world_size(pg), dist.get_rank(pg)
    logger.info_once(
        "HCMS schedule=interleaved-pooled-v3 U=%d chunks=%d Q=%s KV=%s",
        world,
        chunks,
        tuple(q.shape),
        tuple(k.shape),
    )
    device, group = q.device, pg.group_name
    compute = torch.cuda.current_stream(device)
    stream_key = (device, group)
    if stream_key not in _COMM_STREAMS:
        _COMM_STREAMS[stream_key] = (torch.cuda.Stream(device=device), compute.cuda_stream)
    comm, producer_stream = _COMM_STREAMS[stream_key]
    if producer_stream != compute.cuda_stream:
        raise RuntimeError("head overlap buffers require a single compute stream per Ulysses group")
    comm.wait_stream(compute)

    def submit_forward(g, kernel_ready=None):
        with torch.cuda.stream(comm):
            if kernel_ready is not None:
                comm.wait_event(kernel_ready)
            inputs = tuple(x.chunk(chunks, dim=2)[g] for x in (q, k, v))
            packed = _exchange_ce(inputs, group, world, workspace_slot="head_overlap", output_slot=g)
            parts = tuple(
                part.view(x.shape[0], x.shape[1] * world, x.shape[2] // world, x.shape[3])
                for part, x in zip(packed.split([x.numel() for x in inputs]), inputs)
            )
            done = torch.cuda.Event()
            done.record(comm)
        return parts, done

    pending = submit_forward(0)

    joint_len = 0 if md is None or md.joint_query is None else md.joint_query.shape[1]
    joint_kv_len = 0 if md is None or md.joint_key is None else md.joint_key.shape[1]
    kernel_md = md
    if md is not None:
        mask = md.attn_mask
        if joint_kv_len and (mask is None or mask.ndim == 2):
            joint_mask = md.joint_attn_mask
            if mask is not None or joint_mask is not None:
                if mask is None:
                    mask = torch.ones(q.shape[0], k.shape[1] * world, dtype=torch.bool, device=device)
                if joint_mask is None:
                    joint_mask = torch.ones(q.shape[0], joint_kv_len, dtype=torch.bool, device=device)
                mask = torch.cat((joint_mask, mask) if md.joint_strategy == "front" else (mask, joint_mask), dim=1)
        kernel_md = replace(
            md, attn_mask=mask, joint_query=None, joint_key=None, joint_value=None, extra=dict(md.extra)
        )
    reverse = []
    # Keep compute-owned source storage until the final waits are enqueued.
    # Future compute writes then follow completion of every transport read,
    # without record_stream delaying each temporary's free across many layers.
    output_sources = []
    joint_outputs = []
    for g in range(chunks):
        parts, ready = pending
        compute.wait_event(ready)
        if joint_kv_len:
            joint_parts = tuple(
                x.chunk(chunks * world, dim=2)[g * world + rank] for x in (md.joint_query, md.joint_key, md.joint_value)
            )
            parts = tuple(
                torch.cat((j, x) if md.joint_strategy == "front" else (x, j), dim=1) for x, j in zip(parts, joint_parts)
            )
        # Do not let the next exchange race ahead of joint-input packing and
        # finish before the attention even starts. This event covers only
        # preparation, leaving the kernel itself independent from transport.
        kernel_ready = torch.cuda.Event()
        kernel_ready.record(compute)
        if prefetch_before_attention and g + 1 < chunks:
            pending = submit_forward(g + 1, kernel_ready)
        out = kernel(*parts, kernel_md)
        if joint_len:
            if md.joint_strategy == "front":
                joint_out, out = out[:, :joint_len], out[:, joint_len:]
            else:
                out, joint_out = out[:, :-joint_len], out[:, -joint_len:]
            joint_outputs.append(joint_out)
        produced = torch.cuda.Event()
        produced.record(compute)
        output_sources.append(out)
        # Submit the next input AFTER launching this kernel, but before the
        # reverse's wait on its output. All ranks use the same deterministic
        # F0, F1, R0, F2, R1, ... order. Queuing every input first can complete
        # all communication before Python even launches the first attention.
        if not prefetch_before_attention and g + 1 < chunks:
            pending = submit_forward(g + 1, kernel_ready)
        with torch.cuda.stream(comm):
            comm.wait_event(produced)
            packed = _exchange_ce((out,), group, world, reverse=True, workspace_slot="head_overlap", output_slot=g)
            done = torch.cuda.Event()
            done.record(comm)
            reverse.append((packed.view(q.shape[0], q.shape[1], q.shape[2] // chunks, q.shape[3]), done))
    for _, done in reverse:
        compute.wait_event(done)
    image = torch.cat([x for x, _ in reverse], dim=2)
    if not joint_len:
        return image
    # One head all-gather for all groups, preserving group/rank/head order.
    local_joint = torch.stack(joint_outputs, dim=2).contiguous()
    gathered = [torch.empty_like(local_joint) for _ in range(world)]
    dist.all_gather(gathered, local_joint, group=pg)
    joint = torch.stack(gathered, dim=3).flatten(2, 4)
    return torch.cat((joint, image) if md.joint_strategy == "front" else (image, joint), dim=1)
