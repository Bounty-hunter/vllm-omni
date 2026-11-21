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
AUTOTUNE = False
LEARNER = None
REAL_PREFIX = False
EAGER_STEPS = 0
VIDEO_PIPELINE = False


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


def graph_entry(entries):
    if not AUTOTUNE and not REAL_PREFIX:
        return entries.get(GRAPH_KEY)
    if GRAPH_KEY in entries:
        return entries[GRAPH_KEY]
    return next(
        (
            entry
            for key, entry in entries.items()
            if len(key) == 2 and key[0] == GRAPH_KEY and entry.learner is not None
        ),
        None,
    )


class BlockGraph:
    def __init__(self, model, hidden, kwargs):
        self.scalar = {key: value for key, value in kwargs.items() if not isinstance(value, torch.Tensor)}
        self.static = {key: value.clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)}
        self.hidden = hidden.clone()
        self.signature = self.layout(hidden, kwargs)
        self.stream = torch.cuda.current_stream()
        self.graph = torch.cuda.CUDAGraph()
        self.replays = 0
        self.validations = 0
        self.prefix_steps = 0
        self.real_prefix = REAL_PREFIX
        self.learner = None
        if AUTOTUNE or REAL_PREFIX:
            from lazy_head_policy import LazyHeadPolicy

            from vllm_omni.diffusion.distributed.parallel_state import get_world_group

            self.learner = LazyHeadPolicy(get_world_group().cpu_group, enabled=AUTOTUNE)
            self.learner.begin()
        global LEARNER
        LEARNER = self.learner
        if self.real_prefix:
            return
        # Initialize all kernels, events, symmetric allocations, and peer plans
        # before capture. The same compute stream owns eager and graph paths.
        reference = eager_blocks(model, self.hidden, self.kwargs()).clone()
        if self.learner is not None:
            self.learner.prepare()
        eager_blocks(model, self.hidden, self.kwargs())
        self.capture(model, reference)

    def capture(self, model, reference):
        torch.accelerator.synchronize()
        if self.learner is not None:
            self.learner.finish()
            self.policy_key = self.learner.cache_key
        before = CAPTURE_CALLS
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.output = eager_blocks(model, self.hidden, self.kwargs())
        captured = CAPTURE_CALLS - before
        if captured != len(model.blocks):
            raise AssertionError(f"Native CE capture calls: {captured}; expected {len(model.blocks)}")
        self.graph.replay()
        self.validations += 1
        torch.accelerator.synchronize()
        if not torch.equal(reference, self.output):
            raise AssertionError("Captured block stack differs from eager at identical inputs")
        record(
            "graph_created",
            blocks=len(model.blocks),
            native_capture_calls=captured,
            graph_key=GRAPH_KEY,
            static={key: [list(value.shape), str(value.dtype)] for key, value in self.static.items()},
            hidden=list(self.hidden.shape),
            stream=self.stream.cuda_stream,
            eager_graph_bitwise_equal=True,
            real_prefix=self.real_prefix,
            real_eager_steps=self.prefix_steps,
            allocated_mb=torch.accelerator.memory_allocated() / 2**20,
            lazy_policy=None if self.learner is None else dict(stats=self.learner.stats, reports=self.learner.reports),
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

    def __call__(self, model, hidden, kwargs):
        global LEARNER, EAGER_STEPS
        LEARNER = self.learner
        actual_layout = self.layout(hidden, kwargs)
        if actual_layout != self.signature:
            expected = dict(self.signature[1], hidden=self.signature[0])
            actual = dict(actual_layout[1], hidden=actual_layout[0])
            changed = {
                key: dict(expected=value, actual=actual[key]) for key, value in expected.items() if value != actual[key]
            }
            raise AssertionError(f"Fixed-canvas graph metadata changed: {changed}")
        if torch.cuda.current_stream() != self.stream:
            raise AssertionError("Graph compute stream changed")
        self.hidden.copy_(hidden)
        for key, value in self.static.items():
            value.copy_(kwargs[key])
        if self.real_prefix and self.prefix_steps < 2:
            if self.prefix_steps == 1:
                self.learner.prepare(sample_limit=24)
            result = eager_blocks(model, self.hidden, self.kwargs())
            self.prefix_steps += 1
            EAGER_STEPS += 1
            if self.prefix_steps == 2:
                self.learner.phase = "settle"
                self.capture(model, result.clone())
            return result
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
    # Video and audio start with the same sigma (one embedding row), then
    # diverge (two rows). Use fixed capacity without another GPU operation:
    # combined_indices still selects row zero on the initial step. The eager
    # reference keeps the original one-row computation for output validation.
    if VIDEO_PIPELINE and kwargs["t_emb"].shape[0] == 1:
        kwargs["t_emb"] = kwargs["t_emb"].expand(2, -1)
    if model.adaln_cache.max_bytes or model.adaln_cache.offload_weights:
        raise AssertionError("Graph must compute current AdaLN values from its copied embedding")
    entries = getattr(model, "_hcms_block_graphs", None)
    if entries is None:
        entries = model._hcms_block_graphs = {}
    entry = graph_entry(entries)
    if entry is None:
        entry = BlockGraph(model, hidden, kwargs)
        key = (
            (GRAPH_KEY, entry.policy_key)
            if entry.learner is not None and entry.learner.phase == "frozen"
            else GRAPH_KEY
        )
        entries[key] = entry
    result = entry(model, hidden, kwargs)
    if REAL_PREFIX and entry.learner.phase == "frozen" and GRAPH_KEY in entries:
        del entries[GRAPH_KEY]
        entries[(GRAPH_KEY, entry.policy_key)] = entry
    return result


def install(pipeline_class, *, video_pipeline=False):
    global VIDEO_PIPELINE
    VIDEO_PIPELINE = video_pipeline
    diagnostic_label = None
    audio_diagnostic = os.environ.get("HCMS_AUDIO_DIAGNOSTIC")
    audio_deterministic = os.environ.get("HCMS_AUDIO_DETERMINISTIC") == "1"
    if video_pipeline and (audio_diagnostic or audio_deterministic):
        from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE

        original_audio_decode = MiniMaxH3AudioVAE.decode_latent

        @functools.wraps(original_audio_decode)
        def audio_decode(self, latent):
            # An explicit correctness experiment can constrain cuDNN's audio
            # decoder in every arm, including the original eager reference.
            context = (
                torch.backends.cudnn.flags(benchmark=False, deterministic=True)
                if audio_deterministic
                else contextlib.nullcontext()
            )
            with context:
                waveform = original_audio_decode(self, latent)
            if audio_diagnostic and diagnostic_label is not None and dist.get_rank() == 0:
                import numpy as np

                destination = Path(audio_diagnostic)
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination / f"{diagnostic_label}.npz",
                    latent=latent.detach().cpu().numpy(),
                    waveform=waveform.detach().cpu().numpy(),
                )
            return waveform

        MiniMaxH3AudioVAE.decode_latent = audio_decode
    head_module = importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    import native_prefilled
    from prefilled_probe import PrefilledPipeline
    from schedule_probe import run as run_schedule

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    counters = {}
    if os.environ.get("HCMS_LAZY_COUNTERS") == "1":
        from execution_counters import install as install_counters

        counters = install_counters()

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
        if LEARNER is not None:
            q, k, v, md, pg, _chunks, kernel = args
            result = LEARNER.run(q, k, v, md, pg, kernel, streams=kwargs["streams"])
        else:
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
    original_denoise = None if video_pipeline else pipeline_class._denoise

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

    if not video_pipeline:
        pipeline_class._denoise = denoise
    audited = False
    previous_canvas = None

    @functools.wraps(original_forward)
    def forward(self, request):
        global MODE, TRACE, ACTIVE, SCHEDULE, GRAPH_KEY, HEAD_PARTITION, TIMING
        global AUTOTUNE, LEARNER, REAL_PREFIX, EAGER_STEPS
        nonlocal audited, previous_canvas, diagnostic_label
        if request.is_dummy_run():
            return original_forward(self, request)
        counters_before = dict(counters)
        ACTIVE = True
        extra = request.sampling_params.extra_args or {}
        diagnostic_label = f"{extra['label']}-seed{request.sampling_params.seed}"
        MODE = extra["graph_mode"]
        SCHEDULE = extra["schedule"]
        TRACE = bool(extra.get("trace", False))
        chunks = int(extra["chunks"])
        HEAD_PARTITION = tuple(extra["head_partition"]) if extra.get("head_partition") else None
        AUTOTUNE = bool(extra.get("autotune", False))
        REAL_PREFIX = bool(extra.get("real_prefix", False))
        EAGER_STEPS = 0
        LEARNER = None
        canvas = (request.sampling_params.height, request.sampling_params.width)
        if video_pipeline:
            canvas = (*canvas, request.sampling_params.num_frames, extra.get("duration"), extra["case"])
        if previous_canvas != canvas:
            torch.accelerator.synchronize()
            entries = getattr(self.transformer, "_hcms_block_graphs", {})
            entries.clear()
            native_prefilled.clear()
            previous_canvas = canvas
        GRAPH_KEY = (canvas, chunks, SCHEDULE, HEAD_PARTITION)
        if REAL_PREFIX:
            GRAPH_KEY = (*GRAPH_KEY, AUTOTUNE)
        TIMING = [] if extra.get("diagnostic") else None
        if video_pipeline and TIMING is not None:
            raise ValueError("Video diagnostics must use the normal request/core counters")
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
            if video_pipeline:
                vae_world = self.video_vae.parallel_size
                vae_graph = None
                minimum_area = None
                native_vae = self.video_vae._native_parallel_state()
                if bool(native_vae["sp_enabled"]) != (vae_world > 1):
                    raise AssertionError("Video VAE native parallel state disagrees with its adapter")
            else:
                vae_world = dist.get_world_size(self._vae_group) if self._vae_group is not None else 1
                vae_graph = self._vae_cuda_graph
                minimum_area = self.min_generation_area
            if (
                (fp8 == 0 if video_pipeline else fp8 != int(os.environ["HCMS_EXPECTED_FP8"]))
                or vae_world != int(os.environ["HCMS_EXPECTED_VAE_WORLD"])
                or minimum_area is not None
            ):
                raise AssertionError(f"Wrong model configuration: FP8={fp8}, VAE={vae_world}")
            record(
                "model",
                fp8_modules=fp8,
                vae_world=vae_world,
                vae_graph=vae_graph,
                min_generation_area=minimum_area,
                pipeline=type(self).__name__,
                video_pipeline=video_pipeline,
                audio_decoder_deterministic=audio_deterministic,
                t_emb_capacity_rows=2 if video_pipeline else None,
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
        graph = graph_entry(getattr(self.transformer, "_hcms_block_graphs", {}))
        replays_before = graph.replays if graph else 0
        validations_before = graph.validations if graph else 0
        with region(f"request/{extra['label']}"):
            result = original_forward(self, request)
        calls = NATIVE_CALLS - before
        steps = int(request.sampling_params.num_inference_steps) if video_pipeline else 6
        if MODE != "graph" and calls != steps * len(self.transformer.blocks):
            raise AssertionError(f"Unexpected direct CE calls for {steps} steps: {calls}")
        graph = graph_entry(getattr(self.transformer, "_hcms_block_graphs", {}))
        if MODE == "graph" and graph is None:
            raise AssertionError("Graph mode did not create a graph")
        if MODE == "graph" and graph.replays - replays_before + EAGER_STEPS != steps:
            raise AssertionError(f"Expected exactly {steps} real denoising steps across eager and graph execution")
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
                real_prefix=REAL_PREFIX,
                real_eager_steps=EAGER_STEPS,
                denoising_graph_replays=graph.replays - replays_before if graph else 0,
                validation_replays=graph.validations - validations_before if graph else 0,
                plan_chunks=[p.chunks for p in active],
                plan_head_sizes=[p.head_sizes for p in active],
                qkv=[list(p.inputs[0].shape) for p in active],
                cache_hits=cache.hits,
                cache_misses=cache.misses,
                execution_counters={key: value - counters_before[key] for key, value in counters.items()},
                output_slots=len(native_prefilled._SLOT_ARENA),
                output_slot_bytes=sum(
                    slot.tensor.numel() * slot.tensor.element_size() for slot in native_prefilled._SLOT_ARENA.values()
                ),
                lazy_policy=None
                if graph is None or graph.learner is None
                else dict(stats=graph.learner.stats, reports=graph.learner.reports),
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
