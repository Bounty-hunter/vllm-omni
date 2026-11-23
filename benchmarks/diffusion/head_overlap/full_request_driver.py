# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two warmups and two complete H3 requests, with stage timers disabled."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(
        model=args.model,
        parallel_config=dict(
            ulysses_degree=2,
            ulysses_a2a_permute=True,
            ulysses_head_chunks=2,
            text_encoder_tp_size=2,
            tensor_parallel_size=1,
        ),
        dtype="bfloat16",
        diffusion_attention_config={"default": "FLASH_ATTN"},
        enable_diffusion_pipeline_profiler=False,
        additional_config={"text_encoder_cudnn_sdp": False, "vae_tile_parallel": True},
        log_stats=False,
        init_timeout=900,
        stage_init_timeout=900,
    )
    print("FULL_REQUEST_CONFIG " + json.dumps(config), flush=True)
    records = []
    omni = Omni(**config)
    try:
        for index in range(4):
            params = OmniDiffusionSamplingParams(
                height=4096,
                width=4096,
                seed=42,
                num_inference_steps=6,
                guidance_scale=1.0,
                true_cfg_scale=1.0,
                num_outputs_per_prompt=1,
            )
            start = time.perf_counter()
            outputs = omni.generate(
                {
                    "prompt": "A red bicycle leaning against a white wall, soft morning light, detailed photograph.",
                    "modalities": ["image"],
                },
                sampling_params_list=[params],
            )
            elapsed_ms = 1000 * (time.perf_counter() - start)
            images = extract_images_from_outputs(outputs)
            if not images:
                images = [image for output in outputs for image in (output.images or [])]
            assert len(images) == 1
            array = np.asarray(images[0])
            digest = hashlib.sha256(array.tobytes()).hexdigest()
            if args.expected_sha256 is not None:
                assert digest == args.expected_sha256, digest
            row = dict(
                index=index,
                warmup=index < 2,
                diagnostic_ms=elapsed_ms,
                sha256=digest,
                shape=list(array.shape),
            )
            records.append(row)
            print("FULL_REQUEST_RESULT " + json.dumps(row), flush=True)
            (args.out / "results.json").write_text(
                json.dumps(dict(config=config, records=records), indent=2),
                encoding="utf-8",
            )
    finally:
        omni.close()


if __name__ == "__main__":
    main()
