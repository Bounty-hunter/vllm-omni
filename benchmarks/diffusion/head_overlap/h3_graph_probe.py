# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-canvas, lossless CUDA Graph experiment; never installed in production."""

import contextlib
import functools
import importlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

MODE = "original"
TRACE = False
ACTIVE = False
NATIVE_CALLS = 0
CAPTURE_CALLS = 0
SCHEDULE = "old"
GRAPH_KEY = None
HEAD_PARTITION = None
TIMING = None


def record(kind, **values):
    row = dict(kind=kind, rank=dist.get_rank(), **values)
    path = Path(os.environ["HCMS_GRAPH_OUTPUT"])
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"audit-rank{dist.get_rank()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
    print("HCMS_GRAPH_AUDIT " + json.dumps(row, default=str), flush=True)


def region(name):
    return torch.cuda.nvtx.range(name) if TRACE else contextlib.nullcontext()


def eager_blocks(model, hidden, kwargs):
    import native_prefilled

    for index, block in enumerate(model.blocks):
        slot_key = (GRAPH_KEY, 0 if "-shared-" in SCHEDULE else index) if SCHEDULE.endswith("-slots") else None
        with native_prefilled.output_slot(slot_key):
            with region(f"DiT/block{index}"):
                hidden = block(hidden, **kwargs)
    return hidden


class BlockGraph:
    def __init__(self, model, hidden, kwargs):
        self.scalar = {key: value for key, value in kwargs.items() if not isinstance(value, torch.Tensor)}
        self.static = {key: value.clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)}
        self.hidden = hidden.clone()
        self.signature = self.layout(hidden, kwargs)
        self.stream = torch.cuda.current_stream()
        self.graph = torch.cuda.CUDAGraph()
        self.replays = 0
        # Initialize all kernels, events, symmetric allocations, and peer plans
        # before capture. The same compute stream owns eager and graph paths.
        for _ in range(2):
            reference = eager_blocks(model, self.hidden, self.kwargs())
        reference = reference.clone()
        torch.accelerator.synchronize()
        before = CAPTURE_CALLS
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.output = eager_blocks(model, self.hidden, self.kwargs())
        captured = CAPTURE_CALLS - before
        if captured != len(model.blocks):
            raise AssertionError(f"Native CE capture calls: {captured}; expected {len(model.blocks)}")
        self.graph.replay()
        torch.accelerator.synchronize()
        if not torch.equal(reference, self.output):
            raise AssertionError("Captured block stack differs from eager at identical inputs")
        record(
            "graph_created",
            blocks=len(model.blocks),
            native_capture_calls=captured,
            graph_key=GRAPH_KEY,
            static={key: [list(value.shape), str(value.dtype)] for key, value in self.static.items()},
            hidden=list(hidden.shape),
            stream=self.stream.cuda_stream,
            eager_graph_bitwise_equal=True,
            allocated_mb=torch.accelerator.memory_allocated() / 2**20,
        )

    @staticmethod
    def layout(hidden, kwargs):
        def spec(value):
            if isinstance(value, torch.Tensor):
                return tuple(value.shape), value.dtype, value.device
            return value

        return spec(hidden), tuple((key, spec(value)) for key, value in kwargs.items())

    def kwargs(self):
        return dict(self.scalar, **self.static)

    def __call__(self, hidden, kwargs):
        if self.layout(hidden, kwargs) != self.signature:
            raise AssertionError("Fixed-canvas graph received different tensor/scalar metadata")
        if torch.cuda.current_stream() != self.stream:
            raise AssertionError("Graph compute stream changed")
        self.hidden.copy_(hidden)
        for key, value in self.static.items():
            value.copy_(kwargs[key])
        with region("DiT/50blocks/graph-replay"):
            self.graph.replay()
        self.replays += 1
        # Callers consume on the same stream before the next replay.
        return self.output


def run_blocks(model, hidden, *, hints, control_scale, **kwargs):
    if not ACTIVE:
        for index, block in enumerate(model.blocks):
            hidden = block(hidden, **kwargs)
            if index in hints:
                hidden = hidden + hints.pop(index) * control_scale
        return hidden
    if hints or kwargs["reference_kv_state"] is not None or kwargs["reference_kv_compact"]:
        raise AssertionError("This isolated experiment supports fixed t2i only")
    if kwargs["vdn_window"] is not None:
        raise AssertionError("This experiment requires dense attention")
    if kwargs["packed_total"] <= 0 or kwargs["num_requests"] != 1:
        raise AssertionError("Expected one nonempty fixed-canvas request")
    # Dense FA3 does not consume video layout. Refuse a gate that would use it.
    if any(block.attn.to_gate_compress is not None for block in model.blocks):
        raise AssertionError("VSA gates are outside this capture contract")
    kwargs["video_layout"] = None
    if MODE != "graph":
        with region("DiT/50blocks/eager"):
            return eager_blocks(model, hidden, kwargs)
    if model.adaln_cache.max_bytes or model.adaln_cache.offload_weights:
        raise AssertionError("Graph must compute current AdaLN values from its copied embedding")
    entries = getattr(model, "_hcms_block_graphs", None)
    if entries is None:
        entries = model._hcms_block_graphs = {}
    entry = entries.get(GRAPH_KEY)
    if entry is None:
        entry = entries[GRAPH_KEY] = BlockGraph(model, hidden, kwargs)
    return entry(hidden, kwargs)


def install(pipeline_class):
    head_module = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    import native_prefilled
    from prefilled_probe import PrefilledPipeline
    from schedule_probe import run as run_schedule

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    def scheduled_call(plan):
        return run_schedule(plan, SCHEDULE.removesuffix("-slots").replace("-shared", ""))

    PrefilledPipeline.__call__ = scheduled_call

    original_native_run = native_prefilled.run

    @functools.wraps(original_native_run)
    def native_run(*args, **kwargs):
        global NATIVE_CALLS, CAPTURE_CALLS
        if not ACTIVE:
            return original_native_run(*args, **kwargs)
        capturing = torch.cuda.is_current_stream_capturing()

        def fallback(*a, **kw):
            raise AssertionError("Direct CE native path fell back in graph experiment")

        kwargs["fallback"] = fallback
        kwargs["head_partition"] = HEAD_PARTITION
        result = original_native_run(*args, **kwargs)
        NATIVE_CALLS += 1
        CAPTURE_CALLS += int(capturing)
        return result

    native_prefilled.run = native_run
    head_module._run_native_prefilled = native_run
    original_transformer = MiniMaxH3DiTModel.forward

    @functools.wraps(original_transformer)
    def transformer_forward(self, *args, **kwargs):
        stream = getattr(self, "_hcms_compute_stream", None)
        current = torch.cuda.current_stream()
        if stream is None:
            stream = self._hcms_compute_stream = torch.cuda.Stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream), region(f"DiT/step{kwargs.get('pdd_step')}/{MODE}"):
            result = original_transformer(self, *args, **kwargs)
        current.wait_stream(stream)
        return result

    MiniMaxH3DiTModel.forward = transformer_forward
    original_forward = pipeline_class.forward
    original_denoise = pipeline_class._denoise

    @functools.wraps(original_denoise)
    def denoise(*args, **kwargs):
        with region("denoise/6steps"):
            if TIMING is None:
                return original_denoise(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original_denoise(*args, **kwargs)
            end.record()
            TIMING.append((start, end))
            return result

    pipeline_class._denoise = denoise
    audited = False
    previous_canvas = None

    @functools.wraps(original_forward)
    def forward(self, request):
        global MODE, TRACE, ACTIVE, SCHEDULE, GRAPH_KEY, HEAD_PARTITION, TIMING
        nonlocal audited, previous_canvas
        if request.is_dummy_run():
            return original_forward(self, request)
        ACTIVE = True
        extra = request.sampling_params.extra_args or {}
        MODE = extra["graph_mode"]
        SCHEDULE = extra["schedule"]
        TRACE = bool(extra.get("trace", False))
        chunks = int(extra["chunks"])
        HEAD_PARTITION = tuple(extra["head_partition"]) if extra.get("head_partition") else None
        canvas = (request.sampling_params.height, request.sampling_params.width)
        if previous_canvas != canvas:
            torch.accelerator.synchronize()
            entries = getattr(self.transformer, "_hcms_block_graphs", {})
            entries.clear()
            native_prefilled.clear()
            previous_canvas = canvas
        GRAPH_KEY = (canvas, chunks, SCHEDULE, HEAD_PARTITION)
        TIMING = [] if extra.get("diagnostic") else None
        native_prefilled._SELECTED_CHUNKS = str(chunks)
        cache = self.transformer.adaln_cache
        if cache.offload_weights or cache.sidecar is not None:
            raise AssertionError("Expected resident AdaLN weights with no sidecar")
        cache.max_bytes = 256 * 1024**2 if MODE == "original" else 0
        if not cache.max_bytes and cache._entries:
            cache.clear()
        if not audited:
            fp8 = sum(
                isinstance(weight := getattr(module, "weight", None), torch.Tensor)
                and weight.dtype == torch.float8_e4m3fn
                for module in self.transformer.modules()
            )
            vae_world = dist.get_world_size(self._vae_group) if self._vae_group is not None else 1
            if (
                fp8 != int(os.environ["HCMS_EXPECTED_FP8"])
                or vae_world != int(os.environ["HCMS_EXPECTED_VAE_WORLD"])
                or self.min_generation_area is not None
            ):
                raise AssertionError(f"Wrong model configuration: FP8={fp8}, VAE={vae_world}")
            record(
                "model",
                fp8_modules=fp8,
                vae_world=vae_world,
                vae_graph=self._vae_cuda_graph,
                min_generation_area=self.min_generation_area,
                torch=torch.__version__,
                pid=os.getpid(),
                gpu=torch.cuda.get_device_name(),
            )
            audited = True
        if extra.get("trace_start"):
            torch.accelerator.synchronize()
            dist.barrier()
            if dist.get_rank() == 0:
                torch.cuda.profiler.start()
            dist.barrier()
        before = NATIVE_CALLS
        graph = getattr(self.transformer, "_hcms_block_graphs", {}).get(GRAPH_KEY)
        replays_before = graph.replays if graph else 0
        with region(f"request/{extra['label']}"):
            result = original_forward(self, request)
        calls = NATIVE_CALLS - before
        if MODE != "graph" and calls != 300:
            raise AssertionError(f"Expected 300 direct CE calls, got {calls}")
        graph = getattr(self.transformer, "_hcms_block_graphs", {}).get(GRAPH_KEY)
        if MODE == "graph" and graph is None:
            raise AssertionError("Graph mode did not create a graph")
        if MODE == "graph" and graph.replays - replays_before != 6:
            raise AssertionError("Expected exactly six block-stack graph replays per request")
        if extra.get("trace_stop"):
            torch.accelerator.synchronize()
            dist.barrier()
            if dist.get_rank() == 0:
                torch.cuda.profiler.stop()
            dist.barrier()
        if extra.get("audit"):
            active = [plan for _, plan in native_prefilled._PLANS.values()]
            record(
                "request",
                label=extra["label"],
                mode=MODE,
                chunks=chunks,
                schedule=SCHEDULE,
                native_python_calls=calls,
                graph_exists=graph is not None,
                plan_chunks=[p.chunks for p in active],
                plan_head_sizes=[p.head_sizes for p in active],
                qkv=[list(p.inputs[0].shape) for p in active],
                cache_hits=cache.hits,
                cache_misses=cache.misses,
                output_slots=len(native_prefilled._SLOT_ARENA),
                output_slot_bytes=sum(
                    slot.tensor.numel() * slot.tensor.element_size() for slot in native_prefilled._SLOT_ARENA.values()
                ),
            )
        if TIMING is not None:
            torch.accelerator.synchronize()
            record(
                "denoise_timing",
                label=extra["label"],
                graph_key=GRAPH_KEY,
                denoise_ms=[start.elapsed_time(end) for start, end in TIMING],
            )
            TIMING = None
        TRACE = False
        return result

    pipeline_class.forward = forward
