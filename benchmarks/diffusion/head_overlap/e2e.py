# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunks", type=int, required=True)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--sizes", default="2048,4096")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--h3", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parallel = dict(
        ulysses_degree=a.world,
        ulysses_a2a_permute=True,
        ulysses_head_chunks=a.chunks,
        text_encoder_tp_size=a.world,
        tensor_parallel_size=1,
    )
    kwargs = dict(
        model=a.model,
        parallel_config=parallel,
        dtype="bfloat16",
        diffusion_attention_config={"default": "FLASH_ATTN"},
        enable_diffusion_pipeline_profiler=True,
        log_stats=False,
        init_timeout=900,
        stage_init_timeout=900,
    )
    if a.eager:
        kwargs["enforce_eager"] = True
    if a.h3:
        # cuDNN encoder determinism is held fixed in both A/B arms.
        kwargs["additional_config"] = {"text_encoder_cudnn_sdp": False, "vae_tile_parallel": True}
    print("CONFIG " + json.dumps(kwargs), flush=True)
    omni = Omni(**kwargs)
    records = []
    prompt = {
        "prompt": "A red bicycle leaning against a white wall, soft morning light, detailed photograph.",
        "modalities": ["image"],
    }
    try:
        for size in map(int, a.sizes.split(",")):
            for iteration in range(a.repeats + 2):
                sampling = OmniDiffusionSamplingParams(
                    height=size,
                    width=size,
                    seed=42,
                    num_inference_steps=a.steps,
                    guidance_scale=1.0 if a.h3 else 4.0,
                    true_cfg_scale=1.0 if a.h3 else 4.0,
                    num_outputs_per_prompt=1,
                )
                start = time.perf_counter()
                outputs = omni.generate(prompt, sampling_params_list=[sampling])
                latency_ms = (time.perf_counter() - start) * 1000
                images = extract_images_from_outputs(outputs)
                if not images:
                    images = [x for output in outputs for x in (output.images or [])]
                assert len(images) == 1
                image = images[0]
                if isinstance(image, np.ndarray):
                    from diffusers.utils import numpy_to_pil

                    image = numpy_to_pil(image)[0]
                array = np.asarray(image)
                digest = hashlib.sha256(array.tobytes()).hexdigest()
                row = dict(
                    size=size,
                    chunks=a.chunks,
                    iteration=iteration,
                    warmup=iteration < 2,
                    e2e_ms=latency_ms,
                    sha256=digest,
                    stage_durations=[x.stage_durations for x in outputs],
                    peak_memory_mb=[x.peak_memory_mb for x in outputs],
                    metrics=[x.metrics for x in outputs],
                )
                print("E2E " + json.dumps(row, default=str), flush=True)
                records.append(row)
                if iteration in (1, 2, a.repeats + 1):
                    image.save(out / f"s{size}-g{a.chunks}-i{iteration}.png")
                    np.save(out / f"s{size}-g{a.chunks}-i{iteration}.npy", array)
                with open(out / "results.json", "w") as f:
                    json.dump({"args": vars(a), "kwargs": kwargs, "results": records}, f, indent=2, default=str)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
