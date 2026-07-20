#!/usr/bin/env python3
"""Profile HunyuanImage-3.0 DiT with either Ulysses or AllGather sequence
parallelism (FP8 quantization via deploy yaml).

Runs a small fixed number of requests under torch.profiler and exits. Traces
are written by the diffusion workers into ``--profiler-dir``.

Usage:
    python profile_hunyuan_ag.py \
        --mode ulysses --degree 4 \
        --profiler-dir /data/dyy/script/perf_opt/profiles_usp4 \
        --num-prompts 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_text_to_image_prompt,
    get_model_class_name,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct")
    p.add_argument(
        "--deploy-config",
        default="vllm_omni/deploy/hunyuan_image3_dit_fp8.yaml",
        help="Deploy YAML with quantization: fp8 enabled.",
    )
    p.add_argument(
        "--mode",
        choices=["ulysses", "allgather"],
        required=True,
        help="Sequence-parallel scheme to profile.",
    )
    p.add_argument("--degree", type=int, default=4, help="SP degree (ulysses or allgather).")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MoE expert parallel (default: on). Pass --no-enable-expert-parallel to disable.",
    )
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--num-inference-steps", type=int, default=8)
    p.add_argument("--num-prompts", type=int, default=2, help="Number of requests to run.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--prompt", default="a cup of coffee on the table, high quality, detailed")
    p.add_argument("--profiler-dir", required=True, help="Absolute path to save torch traces.")
    p.add_argument("--init-timeout", type=int, default=600)
    p.add_argument("--stage-init-timeout", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("HF_HUB_DISABLE_INTERACTIVE", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    profiler_dir = os.path.abspath(args.profiler_dir)
    Path(profiler_dir).mkdir(parents=True, exist_ok=True)

    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": profiler_dir,
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_stack": True,
        "torch_profiler_dump_cuda_time_total": True,
    }

    omni_kwargs: dict = {
        "model": args.model,
        "deploy_config": args.deploy_config,
        "tensor_parallel_size": args.tensor_parallel_size,
        "ulysses_degree": 1,
        "ring_degree": 1,
        "cfg_parallel_size": 1,
        "vae_patch_parallel_size": 1,
        "enable_expert_parallel": bool(args.enable_expert_parallel),
        "enable_diffusion_pipeline_profiler": True,
        "profiler_config": profiler_config,
        "init_timeout": args.init_timeout,
        "stage_init_timeout": args.stage_init_timeout,
        "mode": "text-to-image",
    }
    if args.mode == "ulysses":
        omni_kwargs["ulysses_degree"] = args.degree
    else:
        omni_kwargs["allgather_degree"] = args.degree

    print(
        f"[profile] mode={args.mode} degree={args.degree} tp={args.tensor_parallel_size} "
        f"ep={args.enable_expert_parallel} fp8=y profiler_dir={profiler_dir} "
        f"num_prompts={args.num_prompts}"
    )
    print(f"[profile] omni_kwargs={json.dumps({k: v for k, v in omni_kwargs.items() if k != 'profiler_config'}, default=str)}")

    omni = Omni(**omni_kwargs)
    model_class_name = get_model_class_name(omni)

    print("[Profiler] Starting profiling...")
    omni.start_profile()

    generation_start = time.perf_counter()

    for i in range(args.num_prompts):
        prompt_dict = build_text_to_image_prompt(
            model_class_name=model_class_name,
            prompt=args.prompt,
            negative_prompt=None,
            height=args.height,
            width=args.width,
        )
        generator = torch.Generator(device="cpu").manual_seed(args.seed + i)
        diffusion_params = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            seed=args.seed + i,
            generator=generator,
            true_cfg_scale=args.cfg_scale,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=1,
        )
        defaults = list(omni.default_sampling_params_list or [])
        sampling_params_list = [clone_sampling_params(p) for p in defaults]
        if not sampling_params_list:
            sampling_params_list = [diffusion_params]
        else:
            for idx, params in enumerate(sampling_params_list):
                if isinstance(params, OmniDiffusionSamplingParams):
                    sampling_params_list[idx] = diffusion_params

        t0 = time.perf_counter()
        outputs = omni.generate(prompt_dict, sampling_params_list=sampling_params_list)
        dt = time.perf_counter() - t0
        print(f"[profile] request {i + 1}/{args.num_prompts} done in {dt:.3f}s")

    total = time.perf_counter() - generation_start
    print(f"[profile] total generation time: {total:.3f}s")

    print("\n[Profiler] Stopping profiler and collecting results...")
    profile_results = omni.stop_profile()
    if profile_results and isinstance(profile_results, dict):
        traces = profile_results.get("traces", [])
        print("=" * 60)
        print("PROFILING RESULTS:")
        for rank, trace in enumerate(traces):
            print(f"\nRank {rank}:")
            if trace:
                print(f"  • Trace: {trace}")
        if not traces:
            print("  No traces collected.")
        print("=" * 60)
    else:
        print("[Profiler] No valid profiling data returned.")

    print(f"[profile] DONE mode={args.mode} degree={args.degree} traces under {profiler_dir}")


if __name__ == "__main__":
    main()
