#!/usr/bin/env python3
"""Minimal USP head-chunk all-to-all/attention overlap sketch.

Run with four NPU ranks, for example:

  torchrun --nproc_per_node 4 usp2_head_overlap_example.py \
    --prefix-len 10572 --volatile-len 9218 --q-heads 32 --kv-heads 4 --chunks 4

The script models step>0: prefix KV is already cached after Ulysses all-to-all,
while only the volatile image/gen Q/K/V participates in the per-step all-to-all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from native_attention_bench_utils import load_native_flash_attn


class DeviceOps:
    def __init__(self, device_type: str) -> None:
        if device_type == "npu":
            if not hasattr(torch, "npu"):
                raise RuntimeError("torch.npu is not available; install torch_npu or use --device-type cuda")
            self.module = torch.npu
            self.default_backend = "hccl"
        elif device_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda is not available; use --device-type npu on Ascend")
            self.module = torch.cuda
            self.default_backend = "nccl"
        else:
            raise ValueError(f"unsupported device_type={device_type!r}")
        self.device_type = device_type

    def set_device(self, local_rank: int) -> None:
        self.module.set_device(local_rank)

    def device(self, local_rank: int) -> torch.device:
        return torch.device(f"{self.device_type}:{local_rank}")

    def current_stream(self) -> Any:
        return self.module.current_stream()

    def stream(self, stream: Any) -> Any:
        return self.module.stream(stream)

    def Stream(self, *, priority: int | None = None) -> Any:
        if priority is None:
            return self.module.Stream()
        return self.module.Stream(priority=priority)

    def Event(self, *, enable_timing: bool = False) -> Any:
        return self.module.Event(enable_timing=enable_timing)

    def synchronize(self) -> None:
        self.module.synchronize()


def resolve_device_type(requested: str) -> str:
    if requested != "auto":
        return requested
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither torch.npu nor torch.cuda is available")


def init_dist(args: argparse.Namespace) -> tuple[int, int, torch.device, DeviceOps]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ops = DeviceOps(resolve_device_type(args.device_type))
    ops.set_device(local_rank)
    backend = ops.default_backend if args.backend == "auto" else args.backend
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), ops.device(local_rank), ops


def balanced_lengths(total_len: int, world: int) -> list[int]:
    base = total_len // world
    rem = total_len % world
    return [base + (1 if rank < rem else 0) for rank in range(world)]


def ulysses_a2a_qkv_chunk(
    x: torch.Tensor,
    *,
    seq_lens: list[int],
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, int]:
    world = dist.get_world_size(group)
    bsz, s_local, heads, head_dim = x.shape
    padded_heads = ((heads + world - 1) // world) * world
    if padded_heads != heads:
        x = F.pad(x, (0, 0, 0, padded_heads - heads))
    heads_per_rank = padded_heads // world

    send = (
        x.reshape(bsz, s_local, world, heads_per_rank, head_dim)
        .permute(2, 1, 0, 3, 4)
        .contiguous()
        .flatten(0, 1)
    )
    out = torch.empty(
        (sum(seq_lens), bsz, heads_per_rank, head_dim),
        device=x.device,
        dtype=x.dtype,
    )
    dist.all_to_all_single(
        out,
        send,
        output_split_sizes=seq_lens,
        input_split_sizes=[s_local] * world,
        group=group,
    )
    return out.permute(1, 0, 2, 3).contiguous(), heads


def ulysses_a2a_o_chunk(
    x: torch.Tensor,
    *,
    seq_lens: list[int],
    local_seq_len: int,
    orig_heads: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world = dist.get_world_size(group)
    bsz, s_global, heads_per_rank, head_dim = x.shape
    send = x.permute(1, 0, 2, 3).contiguous()
    out = torch.empty(
        (world * local_seq_len, bsz, heads_per_rank, head_dim),
        device=x.device,
        dtype=x.dtype,
    )
    dist.all_to_all_single(
        out,
        send,
        output_split_sizes=[local_seq_len] * world,
        input_split_sizes=seq_lens,
        group=group,
    )
    out = (
        out.reshape(world, local_seq_len, bsz, heads_per_rank, head_dim)
        .permute(2, 1, 0, 3, 4)
        .contiguous()
        .reshape(bsz, local_seq_len, world * heads_per_rank, head_dim)
    )
    return out[:, :, :orig_heads, :].contiguous()


def repeat_kv_to_query_heads(k_or_v: torch.Tensor, q_heads: int) -> torch.Tensor:
    if q_heads % k_or_v.shape[2] != 0:
        raise ValueError(f"q heads {q_heads} must be divisible by kv heads {k_or_v.shape[2]}")
    return k_or_v.repeat_interleave(q_heads // k_or_v.shape[2], dim=2)


_FLASH_ATTN_CANDIDATES = load_native_flash_attn()
_FLASH_ATTN_NAME = _FLASH_ATTN_CANDIDATES[0][0] if _FLASH_ATTN_CANDIDATES else "none"
_FLASH_ATTN_FN = _FLASH_ATTN_CANDIDATES[0][1] if _FLASH_ATTN_CANDIDATES else None


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if _FLASH_ATTN_FN is None:
        raise RuntimeError("flash_attn not found; install flash-attn on CUDA")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"q heads {q.shape[2]} must be divisible by kv heads {k.shape[2]}")
    out = _FLASH_ATTN_FN(q, k, v)
    if isinstance(out, tuple):
        out = out[0]
    return out.contiguous()


def split_rank_major_head_chunks(
    x: torch.Tensor,
    *,
    chunks: int,
    world: int,
) -> list[torch.Tensor]:
    """Split heads so every chunk contains one slice for every Ulysses rank.

    Full Ulysses all-to-all expects rank-major head layout:
    [rank0_local_heads, rank1_local_heads, ...].
    Chunk i keeps that layout but narrows each rank's local head block.
    """
    heads = int(x.shape[2])
    if heads % world != 0:
        raise ValueError(f"heads={heads} must be divisible by world={world}")
    local_heads = heads // world
    if local_heads % chunks != 0:
        raise ValueError(
            f"local_heads={local_heads} must be divisible by chunks={chunks}"
        )
    chunk_heads = local_heads // chunks
    result: list[torch.Tensor] = []
    for chunk_idx in range(chunks):
        pieces = []
        for rank_idx in range(world):
            start = rank_idx * local_heads + chunk_idx * chunk_heads
            pieces.append(x[:, :, start:start + chunk_heads, :])
        result.append(torch.cat(pieces, dim=2).contiguous())
    return result


def merge_rank_major_head_chunks(chunks: list[torch.Tensor], *, world: int) -> torch.Tensor:
    if not chunks:
        raise ValueError("expected at least one output chunk")
    chunk_heads_total = int(chunks[0].shape[2])
    if chunk_heads_total % world != 0:
        raise ValueError(f"chunk heads {chunk_heads_total} must be divisible by world={world}")
    chunk_heads_per_rank = chunk_heads_total // world
    rank_pieces = []
    for rank_idx in range(world):
        rank_pieces.append(
            torch.cat(
                [
                    chunk[:, :, rank_idx * chunk_heads_per_rank:(rank_idx + 1) * chunk_heads_per_rank, :]
                    for chunk in chunks
                ],
                dim=2,
            )
        )
    return torch.cat(rank_pieces, dim=2).contiguous()


def add_timing(
    timings: dict[str, list[tuple[Any, Any]]],
    name: str,
    start: Any,
    end: Any,
) -> None:
    timings.setdefault(name, []).append((start, end))


def elapsed_timing_ms(timings: dict[str, list[tuple[Any, Any]]]) -> dict[str, float]:
    return {
        name: sum(start.elapsed_time(end) for start, end in events)
        for name, events in timings.items()
    }


def record_stream_safe(tensor: torch.Tensor, stream: Any) -> None:
    try:
        tensor.record_stream(stream)
    except Exception:
        pass


def run_true_baseline(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    seq_lens: list[int],
    local_seq_len: int,
    group: dist.ProcessGroup,
    ops: DeviceOps,
    sync: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not sync:
        k_a2a, _ = ulysses_a2a_qkv_chunk(k, seq_lens=seq_lens, group=group)
        v_a2a, _ = ulysses_a2a_qkv_chunk(v, seq_lens=seq_lens, group=group)
        full_k = torch.cat((prefix_k, k_a2a), dim=1).contiguous()
        full_v = torch.cat((prefix_v, v_a2a), dim=1).contiguous()
        q_a2a, orig_heads = ulysses_a2a_qkv_chunk(q, seq_lens=seq_lens, group=group)
        out = attention(q_a2a, full_k, full_v)
        result = ulysses_a2a_o_chunk(
            out,
            seq_lens=seq_lens,
            local_seq_len=local_seq_len,
            orig_heads=orig_heads,
            group=group,
        )
        return result.contiguous(), {}

    timings: dict[str, list[tuple[Any, Any]]] = {}
    start = ops.Event(enable_timing=True)
    end = ops.Event(enable_timing=True)
    start.record(ops.current_stream())
    k_a2a, _ = ulysses_a2a_qkv_chunk(k, seq_lens=seq_lens, group=group)
    v_a2a, _ = ulysses_a2a_qkv_chunk(v, seq_lens=seq_lens, group=group)
    end.record(ops.current_stream())
    add_timing(timings, "kv_a2a_ms", start, end)

    start = ops.Event(enable_timing=True)
    end = ops.Event(enable_timing=True)
    start.record(ops.current_stream())
    full_k = torch.cat((prefix_k, k_a2a), dim=1).contiguous()
    full_v = torch.cat((prefix_v, v_a2a), dim=1).contiguous()
    end.record(ops.current_stream())
    add_timing(timings, "prefix_cat_ms", start, end)

    start = ops.Event(enable_timing=True)
    end = ops.Event(enable_timing=True)
    start.record(ops.current_stream())
    q_a2a, orig_heads = ulysses_a2a_qkv_chunk(q, seq_lens=seq_lens, group=group)
    end.record(ops.current_stream())
    add_timing(timings, "q_a2a_ms", start, end)

    attn_start = ops.Event(enable_timing=True)
    attn_end = ops.Event(enable_timing=True)
    attn_start.record(ops.current_stream())
    out = attention(q_a2a, full_k, full_v)
    attn_end.record(ops.current_stream())
    add_timing(timings, "attention_ms", attn_start, attn_end)

    start = ops.Event(enable_timing=True)
    end = ops.Event(enable_timing=True)
    start.record(ops.current_stream())
    result = ulysses_a2a_o_chunk(
        out,
        seq_lens=seq_lens,
        local_seq_len=local_seq_len,
        orig_heads=orig_heads,
        group=group,
    )
    end.record(ops.current_stream())
    add_timing(timings, "o_a2a_ms", start, end)

    start = ops.Event(enable_timing=True)
    end = ops.Event(enable_timing=True)
    start.record(ops.current_stream())
    result = result.contiguous()
    end.record(ops.current_stream())
    add_timing(timings, "merge_ms", start, end)

    ops.synchronize()
    return result, elapsed_timing_ms(timings)


def run_chunked_sequential(
    *,
    q_chunks: list[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    seq_lens: list[int],
    local_seq_len: int,
    group: dist.ProcessGroup,
    ops: DeviceOps,
    sync: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not sync:
        k_a2a, _ = ulysses_a2a_qkv_chunk(k, seq_lens=seq_lens, group=group)
        v_a2a, _ = ulysses_a2a_qkv_chunk(v, seq_lens=seq_lens, group=group)
        full_k = torch.cat((prefix_k, k_a2a), dim=1).contiguous()
        full_v = torch.cat((prefix_v, v_a2a), dim=1).contiguous()
        outputs = []
        for q_chunk in q_chunks:
            q_a2a, orig_heads = ulysses_a2a_qkv_chunk(q_chunk, seq_lens=seq_lens, group=group)
            out = attention(q_a2a, full_k, full_v)
            outputs.append(
                ulysses_a2a_o_chunk(
                    out,
                    seq_lens=seq_lens,
                    local_seq_len=local_seq_len,
                    orig_heads=orig_heads,
                    group=group,
                )
            )
        return merge_rank_major_head_chunks(outputs, world=dist.get_world_size(group)), {}

    timings: dict[str, list[tuple[Any, Any]]] = {}
    start = ops.Event(enable_timing=True)
    end = ops.Event(enable_timing=True)
    start.record(ops.current_stream())
    k_a2a, _ = ulysses_a2a_qkv_chunk(k, seq_lens=seq_lens, group=group)
    v_a2a, _ = ulysses_a2a_qkv_chunk(v, seq_lens=seq_lens, group=group)
    end.record(ops.current_stream())
    add_timing(timings, "kv_a2a_ms", start, end)

    start = ops.Event(enable_timing=True)
    end = ops.Event(enable_timing=True)
    start.record(ops.current_stream())
    full_k = torch.cat((prefix_k, k_a2a), dim=1).contiguous()
    full_v = torch.cat((prefix_v, v_a2a), dim=1).contiguous()
    end.record(ops.current_stream())
    add_timing(timings, "prefix_cat_ms", start, end)

    outputs = []
    for q_chunk in q_chunks:
        start = ops.Event(enable_timing=True)
        end = ops.Event(enable_timing=True)
        start.record(ops.current_stream())
        q_a2a, orig_heads = ulysses_a2a_qkv_chunk(q_chunk, seq_lens=seq_lens, group=group)
        end.record(ops.current_stream())
        add_timing(timings, "q_a2a_ms", start, end)

        attn_start = ops.Event(enable_timing=True)
        attn_end = ops.Event(enable_timing=True)
        attn_start.record(ops.current_stream())
        out = attention(q_a2a, full_k, full_v)
        attn_end.record(ops.current_stream())
        add_timing(timings, "attention_ms", attn_start, attn_end)

        start = ops.Event(enable_timing=True)
        end = ops.Event(enable_timing=True)
        start.record(ops.current_stream())
        outputs.append(
            ulysses_a2a_o_chunk(
                out,
                seq_lens=seq_lens,
                local_seq_len=local_seq_len,
                orig_heads=orig_heads,
                group=group,
            )
        )
        end.record(ops.current_stream())
        add_timing(timings, "o_a2a_ms", start, end)

    start = ops.Event(enable_timing=True)
    end = ops.Event(enable_timing=True)
    start.record(ops.current_stream())
    result = merge_rank_major_head_chunks(outputs, world=dist.get_world_size(group))
    end.record(ops.current_stream())
    add_timing(timings, "merge_ms", start, end)

    ops.synchronize()
    return result, elapsed_timing_ms(timings)


def run_overlap(
    *,
    q_chunks: list[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    seq_lens: list[int],
    local_seq_len: int,
    group: dist.ProcessGroup,
    ops: DeviceOps,
    comm_stream: Any,
    sync: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    compute_stream = ops.current_stream()
    outputs: list[torch.Tensor] = []
    output_events: list[Any] = []

    if not sync:
        with ops.stream(comm_stream):
            k_a2a, _ = ulysses_a2a_qkv_chunk(k, seq_lens=seq_lens, group=group)
            v_a2a, _ = ulysses_a2a_qkv_chunk(v, seq_lens=seq_lens, group=group)
            full_k = torch.cat((prefix_k, k_a2a), dim=1).contiguous()
            full_v = torch.cat((prefix_v, v_a2a), dim=1).contiguous()
            kv_ready = ops.Event()
            kv_ready.record(comm_stream)

        def launch_q_a2a(idx: int) -> tuple[torch.Tensor, int, Any]:
            with ops.stream(comm_stream):
                q_a2a, orig_heads = ulysses_a2a_qkv_chunk(
                    q_chunks[idx],
                    seq_lens=seq_lens,
                    group=group,
                )
                event = ops.Event()
                event.record(comm_stream)
            return q_a2a, orig_heads, event

        pending = launch_q_a2a(0)
        compute_stream.wait_event(kv_ready)
        for idx in range(len(q_chunks)):
            q_a2a, orig_heads, q_ready = pending
            if idx + 1 < len(q_chunks):
                next_pending = launch_q_a2a(idx + 1)
            else:
                next_pending = None

            compute_stream.wait_event(q_ready)
            record_stream_safe(q_a2a, compute_stream)
            record_stream_safe(full_k, compute_stream)
            record_stream_safe(full_v, compute_stream)
            out = attention(q_a2a, full_k, full_v)
            attn_done = ops.Event()
            attn_done.record(compute_stream)

            with ops.stream(comm_stream):
                comm_stream.wait_event(attn_done)
                record_stream_safe(out, comm_stream)
                local_out = ulysses_a2a_o_chunk(
                    out,
                    seq_lens=seq_lens,
                    local_seq_len=local_seq_len,
                    orig_heads=orig_heads,
                    group=group,
                )
                done = ops.Event()
                done.record(comm_stream)
            outputs.append(local_out)
            output_events.append(done)

            if next_pending is not None:
                pending = next_pending

        for event in output_events:
            compute_stream.wait_event(event)
        return merge_rank_major_head_chunks(outputs, world=dist.get_world_size(group)), {}

    timings: dict[str, list[tuple[Any, Any]]] = {}

    with ops.stream(comm_stream):
        start = ops.Event(enable_timing=True)
        end = ops.Event(enable_timing=True)
        start.record(comm_stream)
        k_a2a, _ = ulysses_a2a_qkv_chunk(k, seq_lens=seq_lens, group=group)
        v_a2a, _ = ulysses_a2a_qkv_chunk(v, seq_lens=seq_lens, group=group)
        full_k = torch.cat((prefix_k, k_a2a), dim=1).contiguous()
        full_v = torch.cat((prefix_v, v_a2a), dim=1).contiguous()
        end.record(comm_stream)
        add_timing(timings, "kv_a2a_cat_ms", start, end)
        kv_ready = ops.Event()
        kv_ready.record(comm_stream)

    def launch_q_a2a(idx: int) -> tuple[torch.Tensor, int, Any]:
        with ops.stream(comm_stream):
            start = ops.Event(enable_timing=True)
            end = ops.Event(enable_timing=True)
            start.record(comm_stream)
            q_a2a, orig_heads = ulysses_a2a_qkv_chunk(
                q_chunks[idx],
                seq_lens=seq_lens,
                group=group,
            )
            end.record(comm_stream)
            add_timing(timings, "q_a2a_ms", start, end)
            event = ops.Event()
            event.record(comm_stream)
        return q_a2a, orig_heads, event

    pending = launch_q_a2a(0)
    compute_stream.wait_event(kv_ready)
    for idx in range(len(q_chunks)):
        q_a2a, orig_heads, q_ready = pending
        if idx + 1 < len(q_chunks):
            next_pending = launch_q_a2a(idx + 1)
        else:
            next_pending = None

        compute_stream.wait_event(q_ready)
        attn_start = ops.Event(enable_timing=True)
        attn_end = ops.Event(enable_timing=True)
        record_stream_safe(q_a2a, compute_stream)
        record_stream_safe(full_k, compute_stream)
        record_stream_safe(full_v, compute_stream)
        attn_start.record(compute_stream)
        out = attention(q_a2a, full_k, full_v)
        attn_end.record(compute_stream)
        add_timing(timings, "attention_ms", attn_start, attn_end)

        with ops.stream(comm_stream):
            comm_stream.wait_event(attn_end)
            start = ops.Event(enable_timing=True)
            end = ops.Event(enable_timing=True)
            start.record(comm_stream)
            record_stream_safe(out, comm_stream)
            local_out = ulysses_a2a_o_chunk(
                out,
                seq_lens=seq_lens,
                local_seq_len=local_seq_len,
                orig_heads=orig_heads,
                group=group,
            )
            end.record(comm_stream)
            add_timing(timings, "o_a2a_ms", start, end)
            done = ops.Event()
            done.record(comm_stream)
        outputs.append(local_out)
        output_events.append(done)

        if next_pending is not None:
            pending = next_pending

    for event in output_events:
        compute_stream.wait_event(event)
    start = ops.Event(enable_timing=True)
    end = ops.Event(enable_timing=True)
    start.record(compute_stream)
    result = merge_rank_major_head_chunks(outputs, world=dist.get_world_size(group))
    end.record(compute_stream)
    add_timing(timings, "merge_ms", start, end)
    ops.synchronize()
    return result, elapsed_timing_ms(timings)


def time_fn(
    fn,
    *,
    warmups: int,
    iters: int,
    ops: DeviceOps,
    per_iter_sync: bool,
) -> tuple[torch.Tensor, float, float, dict[str, float]]:
    output = None
    for _ in range(warmups):
        output, _ = fn(sync=True)
    ops.synchronize()

    if per_iter_sync:
        timing_sums: dict[str, float] = {}
        start = time.perf_counter()
        for _ in range(iters):
            output, timings = fn(sync=True)
            for name, value in timings.items():
                timing_sums[name] = timing_sums.get(name, 0.0) + value
        ops.synchronize()
        total_ms = (time.perf_counter() - start) * 1000
        breakdown = {name: value / iters for name, value in timing_sums.items()}
    else:
        start_evt = ops.Event(enable_timing=True)
        end_evt = ops.Event(enable_timing=True)
        start_evt.record(ops.current_stream())
        for _ in range(iters):
            output, _ = fn(sync=False)  # enqueue iters back-to-back, no device sync
        end_evt.record(ops.current_stream())
        ops.synchronize()  # sync once after all iters finish
        total_ms = start_evt.elapsed_time(end_evt)
        _, breakdown = fn(sync=True)
        ops.synchronize()

    assert output is not None
    avg_ms = total_ms / iters
    return output, total_ms, avg_ms, breakdown


ProfileMode = Literal["baseline", "chunked", "overlap"]


def _cuda_time_ms(evt: Any) -> float:
    for attr in ("cuda_time_total", "device_time_total", "self_cuda_time_total", "self_device_time_total"):
        cuda_us = getattr(evt, attr, None)
        if cuda_us:
            return float(cuda_us) / 1000.0
    return 0.0


def _bucket_cuda_ops(prof: torch.profiler.profile) -> dict[str, float]:
    buckets: dict[str, float] = {}
    for evt in prof.key_averages():
        cuda_ms = _cuda_time_ms(evt)
        if cuda_ms <= 0:
            continue
        name = str(evt.key).lower()
        if "nccl" in name or "all_to_all" in name or "alltoall" in name:
            label = "nccl/all_to_all"
        elif "flash" in name or "attention" in name or "fmha" in name or "scaled_dot_product" in name:
            label = "attention"
        elif "memcpy" in name or "copy" in name or "contiguous" in name or "clone" in name:
            label = "memcpy/layout"
        elif "event" in name or "stream" in name:
            label = "cuda_stream/event"
        else:
            label = "other"
        buckets[label] = buckets.get(label, 0.0) + cuda_ms
    return buckets


def _torch_profiler_experimental_config() -> Any | None:
    # PyTorch 2.x: with_stack / export_stacks need ExperimentalConfig(verbose=True).
    try:
        return torch._C._profiler._ExperimentalConfig(verbose=True)
    except Exception:
        return None


def _build_profiler_kwargs(*, with_stack: bool, record_shapes: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "record_shapes": record_shapes,
        "with_stack": with_stack,
    }
    if with_stack:
        kwargs["with_modules"] = True
        experimental_config = _torch_profiler_experimental_config()
        if experimental_config is not None:
            kwargs["experimental_config"] = experimental_config
    return kwargs


def _trace_events(trace_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("traceEvents", []))
    return list(payload)


def _format_trace_python_stacks(trace_path: Path, *, row_limit: int = 30) -> str:
    from collections import defaultdict

    events = _trace_events(trace_path)
    py_nodes: dict[int, dict[str, Any]] = {}
    for ev in events:
        if ev.get("cat") != "python_function":
            continue
        args = ev.get("args") or {}
        py_id = args.get("Python id")
        if py_id is None:
            continue
        py_nodes[int(py_id)] = {
            "name": str(ev.get("name", "")),
            "parent": args.get("Python parent id"),
            "dur_us": float(ev.get("dur") or 0.0),
            "ts": float(ev.get("ts") or 0.0),
            "tid": ev.get("tid"),
        }

    def stack_frames(py_id: int | None) -> tuple[str, ...]:
        frames: list[str] = []
        cur = py_id
        seen: set[int] = set()
        while cur is not None and cur not in seen:
            seen.add(int(cur))
            node = py_nodes.get(int(cur))
            if node is None:
                break
            frames.append(node["name"])
            parent = node["parent"]
            cur = int(parent) if parent is not None else None
        return tuple(reversed(frames))

    interest_tokens = (
        "verify_usp2",
        "ulysses",
        "attention",
        "run_",
        "overlap",
        "chunk",
        "comm",
        "flash_attn",
        "native_attention",
    )
    noise_tokens = (
        "torch/profiler",
        "autograd/profiler",
        "_cuda_synchronize",
        "cuda/__init__.py",
    )

    def is_noise_stack(stack: tuple[str, ...]) -> bool:
        return any(any(token in frame for token in noise_tokens) for frame in stack)

    stack_self_us: dict[tuple[str, ...], float] = defaultdict(float)
    for py_id, node in py_nodes.items():
        dur = node["dur_us"]
        if dur < 5.0:
            continue
        stack = stack_frames(py_id)
        if is_noise_stack(stack):
            continue
        if not any(any(token in frame for token in interest_tokens) for frame in stack):
            continue
        stack_self_us[stack] += dur

    lines: list[str] = ["# python_function stacks (chrome trace)"]
    if stack_self_us:
        ranked = sorted(stack_self_us.items(), key=lambda item: item[1], reverse=True)
        for stack, total_us in ranked[:row_limit]:
            lines.append(f"--- self={total_us / 1000.0:.3f}ms ---")
            for frame in stack[-14:]:
                lines.append(f"  {frame}")
            lines.append("")
    else:
        lines.append("(no matching python_function stacks)")

    py_by_tid: dict[Any, list[tuple[float, float, int, float]]] = defaultdict(list)
    for py_id, node in py_nodes.items():
        py_by_tid[node["tid"]].append(
            (node["ts"], node["ts"] + node["dur_us"], py_id, node["dur_us"])
        )

    def enclosing_py_id(tid: Any, ts: float, end_ts: float) -> int | None:
        best: int | None = None
        best_dur = float("inf")
        for py_ts, py_end, py_id, py_dur in py_by_tid.get(tid, []):
            if py_ts <= ts and end_ts <= py_end and py_dur < best_dur:
                best = py_id
                best_dur = py_dur
        return best

    hot_cats = {"cpu_op", "user_annotation"}
    hot_name_tokens = (
        "nccl",
        "FlashAttn",
        "flash_attn",
        "fa3_fwd",
        "all_to_all",
        "record_param_comms",
    )
    op_stack_us: dict[tuple[str, tuple[str, ...]], float] = defaultdict(float)
    for ev in events:
        if ev.get("cat") not in hot_cats:
            continue
        name = str(ev.get("name", ""))
        if not any(token in name for token in hot_name_tokens):
            continue
        dur = float(ev.get("dur") or 0.0)
        if dur < 5.0:
            continue
        ts = float(ev.get("ts") or 0.0)
        py_id = enclosing_py_id(ev.get("tid"), ts, ts + dur)
        if py_id is None:
            continue
        op_stack_us[(name, stack_frames(py_id))] += dur

    if op_stack_us:
        lines.append("# cpu_op / annotation -> enclosing python stack")
        ranked_ops = sorted(op_stack_us.items(), key=lambda item: item[1], reverse=True)
        for (op_name, stack), total_us in ranked_ops[:row_limit]:
            lines.append(f"--- op={op_name} cpu={total_us / 1000.0:.3f}ms ---")
            for frame in stack[-12:]:
                lines.append(f"  {frame}")
            lines.append("")

    kernel_stack_us: dict[tuple[str, tuple[str, ...]], float] = defaultdict(float)
    for ev in events:
        if ev.get("cat") != "kernel":
            continue
        name = str(ev.get("name", ""))
        if "nccl" not in name and "flash" not in name.lower() and "fa_" not in name.lower():
            continue
        dur = float(ev.get("dur") or 0.0)
        if dur < 1.0:
            continue
        ts = float(ev.get("ts") or 0.0)
        # GPU kernels share the CPU launch thread tid in chrome trace flows; scan all CPU tids.
        best_py: int | None = None
        best_dur = float("inf")
        for tid, ranges in py_by_tid.items():
            py_id = enclosing_py_id(tid, ts, ts + dur)
            if py_id is None:
                continue
            py_dur = py_nodes[py_id]["dur_us"]
            if py_dur < best_dur:
                best_py = py_id
                best_dur = py_dur
        if best_py is not None:
            kernel_stack_us[(name[:120], stack_frames(best_py))] += dur

    if kernel_stack_us:
        lines.append("# kernel -> approximate enclosing python stack")
        ranked_kernels = sorted(kernel_stack_us.items(), key=lambda item: item[1], reverse=True)
        for (kernel_name, stack), total_us in ranked_kernels[:row_limit]:
            lines.append(f"--- kernel={kernel_name} gpu={total_us / 1000.0:.3f}ms ---")
            for frame in stack[-10:]:
                lines.append(f"  {frame}")
            lines.append("")

    return "\n".join(lines)


def _format_profile_stacks(prof: torch.profiler.profile, *, row_limit: int = 30) -> str:
    from collections import defaultdict

    grouped: dict[tuple[str, tuple[str, ...]], list[float]] = defaultdict(list)
    for evt in prof.events():
        stack = getattr(evt, "stack", None)
        if not stack:
            continue
        cuda_us = float(
            getattr(evt, "cuda_time_total", 0)
            or getattr(evt, "device_time_total", 0)
            or 0
        )
        cpu_us = float(getattr(evt, "cpu_time_total", 0) or 0)
        weight = cuda_us if cuda_us > 0 else cpu_us
        if weight <= 0:
            continue
        frames = tuple(str(frame) for frame in stack)
        grouped[(str(evt.name), frames)].append(weight)

    if not grouped:
        return ""

    ranked = sorted(
        grouped.items(),
        key=lambda item: sum(item[1]),
        reverse=True,
    )
    lines: list[str] = ["# profiler event stacks"]
    for (name, frames), weight_us_list in ranked[:row_limit]:
        weight_ms = sum(weight_us_list) / 1000.0
        lines.append(f"--- time={weight_ms:.3f}ms calls={len(weight_us_list)} ---")
        lines.append(f"op: {name}")
        for frame in frames[:16]:
            lines.append(f"  {frame}")
        lines.append("")
    return "\n".join(lines)


def _read_export_stacks_top(path: Path, *, line_limit: int = 40) -> str:
    if not path.exists():
        return ""
    rows: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if " " in line:
            stack, count_text = line.rsplit(" ", 1)
            try:
                count = int(count_text)
            except ValueError:
                stack, count = line, 0
        else:
            stack, count = line, 0
        rows.append((count, stack.replace(";", "\n    ")))
    rows.sort(key=lambda item: item[0], reverse=True)
    out: list[str] = []
    for count, stack in rows[:line_limit]:
        out.append(f"count={count}")
        out.append(f"    {stack}")
        out.append("")
    return "\n".join(out)


def profile_mode(
    mode: ProfileMode,
    *,
    q: torch.Tensor,
    q_chunks: list[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    seq_lens: list[int],
    local_seq_len: int,
    group: dist.ProcessGroup,
    ops: DeviceOps,
    profile_dir: Path,
    warmups: int,
    iters: int,
    per_iter_sync: bool,
    comm_stream: Any,
) -> None:
    rank = dist.get_rank()

    def run_once(*, sync: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
        if mode == "baseline":
            return run_true_baseline(
                q=q,
                k=k,
                v=v,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                seq_lens=seq_lens,
                local_seq_len=local_seq_len,
                group=group,
                ops=ops,
                sync=sync,
            )
        if mode == "chunked":
            return run_chunked_sequential(
                q_chunks=q_chunks,
                k=k,
                v=v,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                seq_lens=seq_lens,
                local_seq_len=local_seq_len,
                group=group,
                ops=ops,
                sync=sync,
            )
        return run_overlap(
            q_chunks=q_chunks,
            k=k,
            v=v,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            seq_lens=seq_lens,
            local_seq_len=local_seq_len,
            group=group,
            ops=ops,
            comm_stream=comm_stream,
            sync=sync,
        )

    for _ in range(max(1, warmups)):
        run_once(sync=True)
    ops.synchronize()
    dist.barrier()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if ops.device_type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def profile_loop() -> torch.Tensor:
        output = None
        if per_iter_sync:
            for _ in range(iters):
                output, _ = run_once(sync=True)
        else:
            for _ in range(iters):
                output, _ = run_once(sync=False)
        return output  # type: ignore[return-value]

    # Discard one profiler cycle to avoid NCCL cold-start inflation on the captured trace.
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ):
        profile_loop()
    ops.synchronize()
    dist.barrier()

    wall_total_ms = None
    start_evt = None
    end_evt = None
    if not per_iter_sync and ops.device_type == "cuda":
        start_evt = ops.Event(enable_timing=True)
        end_evt = ops.Event(enable_timing=True)

    with torch.profiler.profile(
        activities=activities,
        **_build_profiler_kwargs(with_stack=True, record_shapes=True),
    ) as prof:
        if start_evt is not None and end_evt is not None:
            start_evt.record(ops.current_stream())
        output = profile_loop()
        if start_evt is not None and end_evt is not None:
            end_evt.record(ops.current_stream())
    ops.synchronize()
    dist.barrier()
    if start_evt is not None and end_evt is not None:
        wall_total_ms = start_evt.elapsed_time(end_evt)

    trace_path = profile_dir / f"trace_rank{rank}_{mode}.json"
    summary_path = profile_dir / f"summary_rank{rank}_{mode}.txt"
    stack_path = profile_dir / f"summary_rank{rank}_{mode}_stack.txt"
    stacks_cuda_path = profile_dir / f"stacks_rank{rank}_{mode}_cuda.txt"
    stacks_cpu_path = profile_dir / f"stacks_rank{rank}_{mode}_cpu.txt"
    buckets_path = profile_dir / f"buckets_rank{rank}_{mode}.json"

    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    stack_table = prof.key_averages(group_by_stack_n=8).table(
        sort_by="self_cuda_time_total",
        row_limit=40,
    )
    summary_text = table
    if stack_table.strip():
        summary_text += "\n\n=== group_by_stack_n (with_stack) ===\n\n" + stack_table
    summary_path.write_text(summary_text, encoding="utf-8")

    try:
        prof.export_stacks(str(stacks_cuda_path), metric="self_cuda_time_total")
    except Exception:
        stacks_cuda_path.unlink(missing_ok=True)
    if stacks_cuda_path.exists() and stacks_cuda_path.stat().st_size == 0:
        stacks_cuda_path.unlink(missing_ok=True)
    try:
        prof.export_stacks(str(stacks_cpu_path), metric="self_cpu_time_total")
    except Exception:
        stacks_cpu_path.unlink(missing_ok=True)
    if stacks_cpu_path.exists() and stacks_cpu_path.stat().st_size == 0:
        stacks_cpu_path.unlink(missing_ok=True)

    stack_sections: list[str] = []
    if stack_table.strip():
        stack_sections.append("# torch.profiler key_averages(group_by_stack_n=8)\n" + stack_table)
    evt_stacks = _format_profile_stacks(prof)
    if evt_stacks.strip():
        stack_sections.append(evt_stacks)
    if stacks_cuda_path.exists():
        stack_sections.append("# export_stacks (self_cuda_time_total)\n" + _read_export_stacks_top(stacks_cuda_path))
    if stacks_cpu_path.exists():
        stack_sections.append("# export_stacks (self_cpu_time_total)\n" + _read_export_stacks_top(stacks_cpu_path))
    if not stack_sections:
        stack_sections.append(_format_trace_python_stacks(trace_path))
    stack_text = "\n\n".join(section for section in stack_sections if section.strip())
    stack_path.write_text(stack_text, encoding="utf-8")
    total_buckets = _bucket_cuda_ops(prof)
    buckets_payload = {
        "profile_iters": iters,
        "per_iter_sync": per_iter_sync,
        "wall_total_ms": wall_total_ms,
        "wall_avg_ms": (wall_total_ms / iters) if wall_total_ms is not None else None,
        "cuda_total_ms": total_buckets,
        "cuda_avg_ms_per_iter": {
            name: value / iters for name, value in total_buckets.items()
        },
    }
    buckets_path.write_text(json.dumps(buckets_payload, indent=2), encoding="utf-8")

    if rank == 0:
        print(f"\n=== profiler ({mode}) rank={rank} iters={iters} per_iter_sync={per_iter_sync} ===")
        if wall_total_ms is not None:
            print(f"wall_total_ms={wall_total_ms:.3f} wall_avg_ms={wall_total_ms / iters:.3f}")
        print(table)
        print(f"\n=== profiler stacks ({mode}) rank={rank} ===")
        print(stack_text)
        print(f"Saved trace={trace_path}")
        print(
            "Trace stacks: chrome trace event args['Call stack'] on "
            "cpu_op / user_annotation / python_function (open in Perfetto, click event -> Args)"
        )
        print(f"Saved summary={summary_path}")
        print(f"Saved stacks={stack_path}")
        if stacks_cuda_path.exists():
            print(f"Saved stacks_cuda={stacks_cuda_path}")
        if stacks_cpu_path.exists():
            print(f"Saved stacks_cpu={stacks_cpu_path}")
        print(f"Saved buckets={buckets_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefix-len", type=int, default=10572)
    parser.add_argument("--volatile-len", type=int, default=9218)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument(
        "--per-iter-sync",
        action="store_true",
        help="Synchronize after every iteration (default: enqueue iters back-to-back, sync once).",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-type", choices=("auto", "cuda", "npu"), default="auto")
    parser.add_argument("--backend", choices=("auto", "nccl", "hccl"), default="auto")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture torch profiler traces for baseline, chunked, and overlap.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="runtime/usp_head_overlap_profile",
        help="Output directory for chrome traces and CUDA summaries.",
    )
    parser.add_argument(
        "--profile-modes",
        nargs="+",
        choices=["baseline", "chunked", "overlap"],
        default=["chunked", "overlap", "baseline"],
        help="Modes to profile (default: chunked overlap baseline to avoid NCCL cold-start on baseline).",
    )
    parser.add_argument(
        "--profile-warmups",
        type=int,
        default=5,
        help="Extra warmups per profile mode before capturing the trace.",
    )
    args = parser.parse_args()

    rank, world, device, ops = init_dist(args)
    if ops.device_type == "cuda" and _FLASH_ATTN_FN is None:
        raise RuntimeError("flash_attn not found; install flash-attn on CUDA")
    if args.q_heads % world != 0:
        raise ValueError(f"q_heads={args.q_heads} must be divisible by world={world}")
    if args.kv_heads % world != 0:
        raise ValueError(f"kv_heads={args.kv_heads} must be divisible by world={world}")
    if (args.q_heads // world) % args.chunks != 0:
        raise ValueError(
            f"q_heads/world={args.q_heads // world} must be divisible by chunks={args.chunks}"
        )

    seq_lens = balanced_lengths(args.volatile_len, world)
    local_volatile_len = seq_lens[rank]
    kv_heads_per_rank = args.kv_heads // world
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    torch.manual_seed(1234 + rank)
    q = torch.randn(args.batch, local_volatile_len, args.q_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, local_volatile_len, args.kv_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    prefix_k = torch.randn(args.batch, args.prefix_len, kv_heads_per_rank, args.head_dim, device=device, dtype=dtype)
    prefix_v = torch.randn_like(prefix_k)
    q_chunks = split_rank_major_head_chunks(q, chunks=args.chunks, world=world)
    overlap_comm_stream = ops.Stream(priority=-1)
    if rank == 0 and ops.device_type == "cuda":
        lo, hi = torch.cuda.Stream.priority_range()
        print(f"comm_stream priority=-1 (range=({lo}, {hi}))")

    baseline_out, baseline_total_ms, baseline_ms, baseline_timings = time_fn(
        lambda sync=True: run_true_baseline(
            q=q,
            k=k,
            v=v,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            seq_lens=seq_lens,
            local_seq_len=local_volatile_len,
            group=dist.group.WORLD,
            ops=ops,
            sync=sync,
        ),
        warmups=args.warmups,
        iters=args.iters,
        ops=ops,
        per_iter_sync=args.per_iter_sync,
    )
    chunked_out, chunked_total_ms, chunked_ms, chunked_timings = time_fn(
        lambda sync=True: run_chunked_sequential(
            q_chunks=q_chunks,
            k=k,
            v=v,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            seq_lens=seq_lens,
            local_seq_len=local_volatile_len,
            group=dist.group.WORLD,
            ops=ops,
            sync=sync,
        ),
        warmups=args.warmups,
        iters=args.iters,
        ops=ops,
        per_iter_sync=args.per_iter_sync,
    )
    overlap_out, overlap_total_ms, overlap_ms, overlap_timings = time_fn(
        lambda sync=True: run_overlap(
            q_chunks=q_chunks,
            k=k,
            v=v,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            seq_lens=seq_lens,
            local_seq_len=local_volatile_len,
            group=dist.group.WORLD,
            ops=ops,
            comm_stream=overlap_comm_stream,
            sync=sync,
        ),
        warmups=args.warmups,
        iters=args.iters,
        ops=ops,
        per_iter_sync=args.per_iter_sync,
    )
    chunked_diff = float((baseline_out - chunked_out).abs().max().item())
    overlap_diff = float((baseline_out - overlap_out).abs().max().item())
    chunked_speedup = baseline_ms / chunked_ms if chunked_ms > 0 else float("inf")
    overlap_speedup = baseline_ms / overlap_ms if overlap_ms > 0 else float("inf")

    if rank == 0:
        print(
            "USP head-overlap sketch: "
            f"world={world} prefix_len={args.prefix_len} "
            f"volatile_lens={seq_lens} volatile_global={sum(seq_lens)} "
            f"kv_seq_global={args.prefix_len + sum(seq_lens)} "
            f"q_heads={args.q_heads} kv_heads={args.kv_heads} "
            f"head_dim={args.head_dim} chunks={args.chunks} dtype={args.dtype} "
            f"device_type={ops.device_type} backend={dist.get_backend()} "
            f"attn={_FLASH_ATTN_NAME} iters={args.iters} "
            f"per_iter_sync={args.per_iter_sync} "
            f"out_shape={tuple(overlap_out.shape)} "
            f"chunked_diff={chunked_diff:.6f} overlap_diff={overlap_diff:.6f}"
        )
        print("| mode       | total_ms | avg_ms | attention_ms | speedup |")
        print("| ---------- | -------: | -----: | -----------: | ------: |")
        print(
            f"| baseline   | {baseline_total_ms:.3f} | {baseline_ms:.3f} | "
            f"{baseline_timings.get('attention_ms', 0.0):.3f} |   1.000 |"
        )
        print(
            f"| chunked    | {chunked_total_ms:.3f} | {chunked_ms:.3f} | "
            f"{chunked_timings.get('attention_ms', 0.0):.3f} | {chunked_speedup:.3f} |"
        )
        print(
            f"| overlap    | {overlap_total_ms:.3f} | {overlap_ms:.3f} | "
            f"{overlap_timings.get('attention_ms', 0.0):.3f} | {overlap_speedup:.3f} |"
        )
        print("\nBaseline breakdown")
        print("| component     | avg_ms |")
        print("| ------------- | -----: |")
        for name in (
            "kv_a2a_ms",
            "prefix_cat_ms",
            "q_a2a_ms",
            "attention_ms",
            "o_a2a_ms",
            "merge_ms",
        ):
            print(f"| {name.ljust(13)} | {baseline_timings.get(name, 0.0):.3f} |")
        print("\nChunked sequential breakdown")
        print("| component     | avg_ms |")
        print("| ------------- | -----: |")
        for name in (
            "kv_a2a_ms",
            "prefix_cat_ms",
            "q_a2a_ms",
            "attention_ms",
            "o_a2a_ms",
            "merge_ms",
        ):
            print(f"| {name.ljust(13)} | {chunked_timings.get(name, 0.0):.3f} |")
        print("\nOverlap breakdown")
        print("| component     | avg_ms |")
        print("| ------------- | -----: |")
        for name in (
            "kv_a2a_cat_ms",
            "q_a2a_ms",
            "attention_ms",
            "o_a2a_ms",
            "merge_ms",
        ):
            print(f"| {name.ljust(13)} | {overlap_timings.get(name, 0.0):.3f} |")

    if args.profile:
        profile_dir = Path(args.profile_dir)
        if rank == 0:
            profile_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        if rank == 0:
            print(
                f"\n=== profiling chunks={args.chunks} iters={args.iters} "
                f"per_iter_sync={args.per_iter_sync} "
                f"profile_dir={profile_dir.resolve()} ==="
            )
        for _ in range(max(3, args.profile_warmups)):
            run_overlap(
                q_chunks=q_chunks,
                k=k,
                v=v,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                seq_lens=seq_lens,
                local_seq_len=local_volatile_len,
                group=dist.group.WORLD,
                ops=ops,
                comm_stream=overlap_comm_stream,
            )
            run_chunked_sequential(
                q_chunks=q_chunks,
                k=k,
                v=v,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                seq_lens=seq_lens,
                local_seq_len=local_volatile_len,
                group=dist.group.WORLD,
                ops=ops,
            )
            run_true_baseline(
                q=q,
                k=k,
                v=v,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                seq_lens=seq_lens,
                local_seq_len=local_volatile_len,
                group=dist.group.WORLD,
                ops=ops,
            )
        ops.synchronize()
        dist.barrier()
        for mode in args.profile_modes:
            profile_mode(
                mode,  # type: ignore[arg-type]
                q=q,
                q_chunks=q_chunks,
                k=k,
                v=v,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                seq_lens=seq_lens,
                local_seq_len=local_volatile_len,
                group=dist.group.WORLD,
                ops=ops,
                profile_dir=profile_dir,
                warmups=args.profile_warmups,
                iters=args.iters,
                per_iter_sync=args.per_iter_sync,
                comm_stream=overlap_comm_stream,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
