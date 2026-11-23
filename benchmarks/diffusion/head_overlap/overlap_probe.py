# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Controlled CE/attention experiments; not a production transport API.

The direct transport uses fixed, caller-owned symmetric buffers. All peer
completion/reuse fences are retained. Outputs alias the next invocation.
"""

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import os
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils.fa import dense_flash_attn_func_for
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.a2a_permute import _ce_runtime, _exchange_ce

head_module = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")


class SymmetricBuffer:
    def __init__(self, shape, dtype, device, group):
        stride = torch.empty(shape, device="meta").stride()
        self.tensor = symm._SymmetricMemory.empty_strided_p2p(shape, stride, dtype, device)
        self.handle = symm.rendezvous(self.tensor, group.group_name)
        assert self.tensor.data_ptr() == self.handle.buffer_ptrs[dist.get_rank(group)]


class DirectPipeline:
    """Push directly into FA input and assembled output, with no clone/cat.

    Every rank follows reuse -> F0 -> F1 -> R0 -> R1 -> completion.
    Distinct forward buffers allow F1 to write while attention consumes F0.
    Reverse writes populate disjoint head slices of one final output buffer.
    A call returns only after all incoming reverse writes have completed.
    """

    def __init__(
        self,
        inputs,
        joint,
        group,
        chunks,
        kernel,
        *,
        overlap=True,
        head_sizes=None,
        joint_strategy="front",
        output=None,
    ):
        q = inputs[0]
        batch, local_seq, heads, dim = q.shape
        assert batch == 1 and heads % dist.get_world_size(group) == 0
        self.inputs, self.joint, self.kernel = inputs, joint, kernel
        self.overlap = overlap
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.chunks, self.local_seq, self.dim = chunks, local_seq, dim
        self.joint_len = 0 if joint is None else joint[0].shape[1]
        self.joint_kv_len = 0 if joint is None else joint[1].shape[1]
        if joint_strategy not in ("front", "rear"):
            raise ValueError("The direct plan requires front or rear joint placement")
        self.joint_strategy = joint_strategy
        if joint is not None and not (
            joint[1].shape == joint[2].shape
            and joint[0].shape[:1] + joint[0].shape[2:] == joint[1].shape[:1] + joint[1].shape[2:]
        ):
            raise ValueError("Joint Q/K/V must share batch, heads and dimension; K/V must share length")
        self.sequence_offset = self.joint_len if joint_strategy == "front" else 0
        self.joint_offset = 0 if joint_strategy == "front" else local_seq * self.world
        self.output_joint_offset = 0 if joint_strategy == "front" else local_seq
        if head_sizes is None:
            if heads % (self.world * chunks):
                raise ValueError("Uniform groups must divide the per-rank head count")
            head_sizes = (heads // (self.world * chunks),) * chunks
        self.head_sizes = tuple(head_sizes)
        if (
            len(self.head_sizes) != chunks
            or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in self.head_sizes)
            or sum(self.head_sizes) * self.world != heads
        ):
            raise ValueError("Positive group head counts must cover every per-rank head exactly once")
        # Global head blocks include every rank's slice of each group. Build
        # offsets once; forward and reverse must use the same permutation.
        self.head_offsets = tuple(self.world * sum(self.head_sizes[:g]) for g in range(chunks))
        self.stream = torch.cuda.Stream()
        self.runtime = _ce_runtime()
        sequence = local_seq * self.world + max(self.joint_len, self.joint_kv_len)
        output_shape = (1, self.joint_len + local_seq, heads, dim)
        if output is not None and (
            tuple(output.tensor.shape) != output_shape
            or output.tensor.dtype != q.dtype
            or output.tensor.device != q.device
        ):
            raise ValueError("A supplied symmetric output must match the complete output layout")
        self.forward = [SymmetricBuffer((3, sequence, size, dim), q.dtype, q.device, group) for size in self.head_sizes]
        self.output = SymmetricBuffer(output_shape, q.dtype, q.device, group) if output is None else output
        self.ready = [torch.cuda.Event() for _ in range(chunks)]
        self.produced = [torch.cuda.Event() for _ in range(chunks)]
        self.enter, self.done = torch.cuda.Event(), torch.cuda.Event()
        self.profile = False
        self.outgoing_bytes = 0
        # Cache all fixed QKV transfer geometry and input views outside the call.
        self.forward_ops = [self._forward_geometry(g) for g in range(chunks)]
        self.parts = [
            tuple(
                x.tensor[i, : local_seq * self.world + (self.joint_len if i == 0 else self.joint_kv_len)].unsqueeze(0)
                for i in range(3)
            )
            for x in self.forward
        ]

    def region(self, name):
        return torch.profiler.record_function(name) if self.profile else nullcontext()

    def emit(self, args):
        self.runtime.CUDART_CHECK(self.runtime.lib.cudaMemcpy2DAsync(*args, self.stream.cuda_stream))

    def geometry(self, dest, dpitch, source, spitch, width, rows):
        return (dest, dpitch, source, spitch, width, rows, 4)

    def _forward_geometry(self, g):
        operations = []
        head_bytes = self.dim * self.inputs[0].element_size()
        width = self.head_sizes[g] * head_bytes
        shared = self.forward[g]
        section_bytes = shared.tensor[0].nbytes
        for index, x in enumerate(self.inputs):
            joint_length = self.joint_len if index == 0 else self.joint_kv_len
            sequence_offset = joint_length if self.joint_strategy == "front" else 0
            source_pitch = x.stride(1) * x.element_size()
            for peer in range(self.world):
                dest = (
                    shared.handle.buffer_ptrs[peer]
                    + index * section_bytes
                    + (sequence_offset + self.rank * self.local_seq) * width
                )
                source = x.data_ptr() + (self.head_offsets[g] + peer * self.head_sizes[g]) * head_bytes
                operations.append(self.geometry(dest, width, source, source_pitch, width, self.local_seq))
                if peer != self.rank:
                    self.outgoing_bytes += width * self.local_seq
            if self.joint is not None and joint_length:
                j = self.joint[index]
                source = j.data_ptr() + (self.head_offsets[g] + self.rank * self.head_sizes[g]) * head_bytes
                dest = shared.handle.buffer_ptrs[self.rank] + index * section_bytes + self.joint_offset * width
                operations.append(
                    self.geometry(
                        dest,
                        width,
                        source,
                        j.stride(1) * j.element_size(),
                        width,
                        joint_length,
                    )
                )
        return operations

    def submit_forward(self, g):
        with self.region(f"direct/F{g}/CE-and-peer-ready"):
            for operation in self.forward_ops[g]:
                self.emit(operation)
            # Current stream owns every outgoing CE copy. This fence also waits
            # for incoming peer writes; a local CUDA event cannot replace it.
            self.forward[g].handle.barrier()
            self.ready[g].record(self.stream)

    def submit_reverse(self, g, out):
        head_bytes = self.dim * out.element_size()
        width = self.head_sizes[g] * head_bytes
        head_offset = (self.head_offsets[g] + self.rank * self.head_sizes[g]) * head_bytes
        source_pitch = out.stride(1) * out.element_size()
        dest_pitch = self.output.tensor.stride(1) * out.element_size()
        with self.region(f"direct/R{g}/CE-direct-output"):
            for peer in range(self.world):
                dest = self.output.handle.buffer_ptrs[peer] + self.sequence_offset * dest_pitch + head_offset
                source = out.data_ptr() + (self.sequence_offset + peer * self.local_seq) * source_pitch
                self.emit(self.geometry(dest, dest_pitch, source, source_pitch, width, self.local_seq))
                if self.joint_len:
                    dest = self.output.handle.buffer_ptrs[peer] + self.output_joint_offset * dest_pitch + head_offset
                    self.emit(
                        self.geometry(
                            dest,
                            dest_pitch,
                            out.data_ptr() + self.joint_offset * source_pitch,
                            source_pitch,
                            width,
                            self.joint_len,
                        )
                    )

    def __call__(self):
        compute = torch.cuda.current_stream()
        self.enter.record(compute)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.enter)
            with self.region("direct/reuse-peer-fence"):
                self.output.handle.barrier()
            self.submit_forward(0)
        sources = []
        for g in range(self.chunks):
            compute.wait_event(self.ready[g])
            # Issue the next CE batch before the FA CPU call. No dependency on
            # this attention's output; its input buffer is separate from ours.
            if self.overlap and g + 1 < self.chunks:
                with torch.cuda.stream(self.stream):
                    self.submit_forward(g + 1)
            with self.region(f"direct/A{g}/attention"):
                out = self.kernel(*self.parts[g], None)
            sources.append(out)
            self.produced[g].record(compute)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.produced[g])
                self.submit_reverse(g, out)
                if not self.overlap and g + 1 < self.chunks:
                    self.submit_forward(g + 1)
        with torch.cuda.stream(self.stream):
            with self.region("direct/output-peer-ready"):
                self.output.handle.barrier()
            self.done.record(self.stream)
        compute.wait_event(self.done)
        return self.output.tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="4096:32:512,16384:32:512,4128:56:0:16,16416:56:0:16")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--inner", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    pg = dist.group.WORLD
    control = dist.new_group(backend="gloo")
    world = dist.get_world_size()
    assert world == 2
    _ce_runtime()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    group = SimpleNamespace(
        ulysses_group=pg,
        ulysses_world_size=world,
        ulysses_rank=rank,
        ring_world_size=1,
        allgather_world_size=1,
    )
    strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
    rows = []
    metadata = dict(
        args=vars(args),
        rank=rank,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    for name in ("attention/parallel/head_overlap.py", "distributed/a2a_permute.py"):
        path = Path("vllm_omni/diffusion") / name
        metadata[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def checkpoint():
        (outdir / f"results-rank{rank}.json").write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2))

    with torch.inference_mode():
        for case in args.cases.split(","):
            fields = list(map(int, case.split(":")))
            seq, heads, joint_len = fields[:3]
            padding = fields[3] if len(fields) > 3 else 0
            label = f"s{seq}-h{heads}-j{joint_len}"
            if padding:
                label += f"-p{padding}"
            print(f"CASE_BEGIN rank={rank} {label}", flush=True)
            torch.manual_seed(10448 + rank)
            inputs = tuple(
                torch.randn(1, seq // world, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
            )
            if padding:
                # H3 normalizes Q/K out of place; V remains a fused-QKV view.
                projected = torch.randn(1, seq // world, 3, heads, 128, device="cuda", dtype=torch.bfloat16)
                inputs = inputs[:2] + (projected[:, :, 2],)
            joint = None
            md = None
            if joint_len:
                joint = tuple(
                    torch.randn(1, joint_len, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
                )
                for x in joint:
                    dist.broadcast(x, src=0)
                md = AttentionMetadata(joint_query=joint[0], joint_key=joint[1], joint_value=joint[2])
            if padding:
                assert not joint_len and 0 < padding < seq
                boundaries = torch.tensor([0, seq - padding, seq], dtype=torch.int32, device="cuda")
                md = AttentionMetadata(
                    extra=dict(
                        cu_seqlens_q=boundaries,
                        cu_seqlens_k=boundaries,
                        max_seqlen_q=seq - padding,
                        max_seqlen_k=seq - padding,
                    )
                )
            impl = FlashAttentionImpl(heads // world, 128, 128**-0.5)

            def kernel(q, k, v, metadata):
                return impl.forward_cuda(q, k, v, metadata)

            def baseline():
                q, k, v, packed_md, context = strategy.pre_attention(*inputs, copy.copy(md))
                return strategy.post_attention(kernel(q, k, v, packed_md), context)

            reference = baseline().clone()
            packed_q, packed_k, packed_v, packed_md, _ = strategy.pre_attention(*inputs, copy.copy(md))
            packed = tuple(x.clone() for x in (packed_q, packed_k, packed_v))
            prepacked = [tuple(x.chunk(2, dim=2)[g].contiguous() for x in packed) for g in range(2)]

            def full():
                return kernel(*packed, packed_md)

            def split():
                return tuple(kernel(*parts, packed_md) for parts in prepacked)

            def split_concat():
                return torch.cat(split(), dim=2)

            torch.testing.assert_close(split_concat(), full(), atol=0, rtol=0)

            def direct_kernel(q, k, v, unused_metadata):
                return kernel(q, k, v, packed_md)

            direct1 = DirectPipeline(inputs, joint, pg, 1, direct_kernel)
            direct2 = DirectPipeline(inputs, joint, pg, 2, direct_kernel)
            direct2_serial = DirectPipeline(inputs, joint, pg, 2, direct_kernel, overlap=False)
            for direct in (direct1, direct2, direct2_serial):
                for _ in range(3):
                    actual = direct()
                torch.accelerator.synchronize()
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            print(f"DIRECT_EXACT rank={rank} {label}", flush=True)

            def current():
                return head_module.head_overlap(*inputs, md, pg, 2, kernel)

            def current_early():
                return head_module.head_overlap(*inputs, md, pg, 2, kernel, prefetch_before_attention=True)

            def transport():
                q, k, v, _, ctx = strategy.pre_attention(*inputs, copy.copy(md))
                # Communication-only workload; output is arbitrary, shape-equivalent.
                return strategy.post_attention(q, ctx)

            reverse_source = packed[0][:, joint_len:]

            def ce_forward():
                return _exchange_ce(inputs, pg.group_name, world, workspace_slot="probe_forward")

            def ce_reverse():
                return _exchange_ce(
                    (reverse_source,),
                    pg.group_name,
                    world,
                    reverse=True,
                    workspace_slot="probe_reverse",
                )

            traffic_bytes = 2 * inputs[0].nbytes  # 3 QKV remote halves + O remote half.
            traffic_source = torch.randint(0, 256, (traffic_bytes,), dtype=torch.uint8, device="cuda")
            traffic_dest = SymmetricBuffer((traffic_bytes,), torch.uint8, inputs[0].device, pg)
            traffic_stream = torch.cuda.Stream()
            traffic_start, traffic_done = torch.cuda.Event(), torch.cuda.Event()
            runtime = _ce_runtime()
            raw_args = (
                traffic_dest.handle.buffer_ptrs[1 - rank],
                traffic_bytes,
                traffic_source.data_ptr(),
                traffic_bytes,
                traffic_bytes,
                1,
                4,
            )

            def raw_ce(count=1):
                for _ in range(count):
                    runtime.CUDART_CHECK(runtime.lib.cudaMemcpy2DAsync(*raw_args, traffic_stream.cuda_stream))

            def independent(compute_fn, count=1):
                traffic_start.record(stream)
                traffic_stream.wait_event(traffic_start)
                raw_ce(count)
                result = compute_fn()
                traffic_done.record(traffic_stream)
                stream.wait_event(traffic_done)
                return result

            def traffic_only():
                return independent(lambda: None)

            variants = {
                "attention-full": full,
                "attention-split2": split,
                "attention-split2-concat": split_concat,
                "raw-ce-only": traffic_only,
                "attention-full-with-ce": lambda: independent(full),
                "attention-split2-with-ce": lambda: independent(split),
                "attention-full-with-ce-x8": lambda: independent(full, 8),
                "ce-forward": ce_forward,
                "ce-reverse": ce_reverse,
                "serial-staging-and-transport": transport,
                "baseline": baseline,
                "current": current,
                "current-early": current_early,
                "direct1": direct1,
                "direct2": direct2,
                "direct2-serial": direct2_serial,
                "direct1-owned-output": lambda: direct1().clone(),
                "direct2-owned-output": lambda: direct2().clone(),
                "direct2-serial-owned-output": lambda: direct2_serial().clone(),
            }
            for fn in (current, current_early):
                torch.testing.assert_close(fn(), reference, atol=0, rtol=0)
            print(f"CORE_EXACT rank={rank} {label}", flush=True)
            dense_fn = dense_flash_attn_func_for(torch.bfloat16)
            if not padding and dense_fn is not None and "sm_margin" in inspect.signature(dense_fn).parameters:
                # Narrow causal ablation: keep the old scheduler/transport, but
                # leave one SM free so the peer fence can become resident.
                def kernel_margin1(q, k, v, unused_metadata):
                    value = dense_fn(
                        q,
                        k,
                        v,
                        softmax_scale=128**-0.5,
                        causal=False,
                        deterministic=impl.fa_deterministic,
                        sm_margin=1,
                    )
                    return FlashAttentionImpl._unwrap_flash_output(value)

                def current_margin1():
                    return head_module.head_overlap(*inputs, md, pg, 2, kernel_margin1)

                pure_actual = kernel_margin1(*packed, packed_md)
                pure_expected = full()
                actual = current_margin1()
                diagnostic = dict(
                    case=label,
                    pure_exact=torch.equal(pure_actual, pure_expected),
                    pipeline_exact=torch.equal(actual, reference),
                    pure_max_abs=(pure_actual - pure_expected).abs().max().item(),
                    pipeline_max_abs=(actual - reference).abs().max().item(),
                )
                peers = [None] * world
                dist.all_gather_object(peers, diagnostic, group=control)
                metadata.setdefault("sm_margin_diagnostics", []).append(diagnostic)
                checkpoint()
                print("SM_MARGIN_CHECK " + json.dumps(diagnostic), flush=True)
                if all(x["pure_exact"] and x["pipeline_exact"] for x in peers):
                    variants["attention-full-margin1"] = lambda: kernel_margin1(*packed, packed_md)
                    variants["current-margin1"] = current_margin1
                else:
                    print(
                        "REJECTED sm_margin=1: output mismatch; excluded from valid performance variants",
                        flush=True,
                    )
            for fn in variants.values():
                for _ in range(5):
                    actual = fn()
            torch.accelerator.synchronize()
            torch.testing.assert_close(current(), reference, atol=0, rtol=0)
            torch.accelerator.synchronize()
            graphs = {}
            for name, fn in variants.items():
                print(f"CAPTURE rank={rank} {label} {name}", flush=True)
                dist.barrier(group=control)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result = fn()
                graph.replay()
                torch.accelerator.synchronize()
                if name == "baseline" or name.startswith(("current", "direct")):
                    torch.testing.assert_close(result, reference, atol=0, rtol=0)
                graphs[name] = (graph, result)
            for mode in ("graph", "eager"):
                samples = {name: [] for name in variants}
                walls = {name: [] for name in variants}
                for repetition in range(args.repeats):
                    order = list(variants)
                    if repetition % 2:
                        order.reverse()
                    for name in order:
                        fn = graphs[name][0].replay if mode == "graph" else variants[name]
                        dist.barrier(group=control)
                        torch.accelerator.synchronize()
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        wall_start = time.perf_counter()
                        start.record(stream)
                        for _ in range(args.inner):
                            fn()
                        end.record(stream)
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) / args.inner)
                        walls[name].append((time.perf_counter() - wall_start) * 1000 / args.inner)
                for name in variants:
                    row = dict(
                        case=label,
                        mode=mode,
                        variant=name,
                        samples_ms=samples[name],
                        wall_samples_ms=walls[name],
                        median_ms=statistics.median(samples[name]),
                        min_ms=min(samples[name]),
                        max_ms=max(samples[name]),
                        traffic_bytes=traffic_bytes,
                        exact=True,
                    )
                    rows.append(row)
                    if rank == 0:
                        print("RESULT " + json.dumps(row), flush=True)
                checkpoint()
            if args.trace:
                # Expensive profiling is separated from all latency measurements.
                selected = [
                    "attention-full",
                    "attention-split2",
                    "attention-full-with-ce",
                    "attention-full-with-ce-x8",
                    "ce-forward",
                    "ce-reverse",
                    "serial-staging-and-transport",
                    "baseline",
                    "current",
                    "current-early",
                    "direct1",
                    "direct2",
                    "direct2-serial",
                    "direct2-owned-output",
                ]
                selected += [name for name in ("attention-full-margin1", "current-margin1") if name in variants]
                direct1.profile = direct2.profile = direct2_serial.profile = True
                for name in selected:
                    fn = variants[name]
                    for _ in range(3):
                        fn()
                    torch.accelerator.synchronize()
                    dist.barrier(group=control)
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        experimental_config=torch.profiler._ExperimentalConfig(verbose=True),
                    ) as prof:
                        for iteration in range(2):
                            with torch.profiler.record_function(f"{label}/{name}/iteration{iteration}"):
                                fn()
                            torch.accelerator.synchronize()
                    prof.export_chrome_trace(str(outdir / f"{label}-{name}-rank{rank}.trace.json"))
                    prof.export_stacks(
                        str(outdir / f"{label}-{name}-rank{rank}.stacks.txt"),
                        "self_cpu_time_total",
                    )
                for name in (
                    "baseline",
                    "current",
                    "current-early",
                    "direct1",
                    "direct2",
                    "direct2-serial",
                    "direct2-owned-output",
                    "current-margin1",
                ):
                    if name not in variants:
                        continue
                    replay = graphs[name][0].replay
                    for _ in range(3):
                        replay()
                    torch.accelerator.synchronize()
                    dist.barrier(group=control)
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        experimental_config=torch.profiler._ExperimentalConfig(verbose=True),
                    ) as prof:
                        for iteration in range(2):
                            with torch.profiler.record_function(f"{label}/{name}-graph/iteration{iteration}"):
                                replay()
                            torch.accelerator.synchronize()
                    prof.export_chrome_trace(str(outdir / f"{label}-{name}-graph-rank{rank}.trace.json"))
                    prof.export_stacks(
                        str(outdir / f"{label}-{name}-graph-rank{rank}.stacks.txt"),
                        "self_cpu_time_total",
                    )
                direct1.profile = direct2.profile = direct2_serial.profile = False
            # Repeat after changing source values: checks the reuse protocol,
            # avoiding a false pass from stale symmetric memory.
            for x in inputs:
                x.add_(0.125)
            if joint is not None:
                for x in joint:
                    x.mul_(0.9)
            reference = baseline().clone()
            for fn in (current, current_early, direct1, direct2, direct2_serial):
                torch.testing.assert_close(fn(), reference, atol=0, rtol=0)
            torch.accelerator.synchronize()
            dist.barrier(group=control)
            print(
                f"CORRECT rank={rank} case={label} exact including mutated inputs",
                flush=True,
            )
            del graphs
    checkpoint()
    dist.barrier(group=control)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
