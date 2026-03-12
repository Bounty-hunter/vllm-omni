# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import re
import time

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for DreamID-Omni (video + audio).")
    parser.add_argument("--model", required=True, help="DreamID ckpt root directory.")
    parser.add_argument("--model-type", default="dreamid-omni", help="Model type.")
    parser.add_argument("--prompt", default=None, help="Text prompt.")

    parser.add_argument("--image-path", type=str, nargs="+", help="list of image-path")
    parser.add_argument("--audio-path", type=str, nargs="+", help="list of audio-path")
    parser.add_argument("--prompt-file", type=str, default=None, help="Text prompt in json format.")

    parser.add_argument("--height", type=int, default=704, help="Video height.")
    parser.add_argument("--width", type=int, default=1024, help="Video width.")
    parser.add_argument("--num-inference-steps", type=int, default=45, help="Sampling steps.")
    parser.add_argument("--solver-name", default="unipc", help="Solver name: unipc|dpm++|euler.")
    parser.add_argument("--shift", type=float, default=5.0, help="Scheduler shift.")
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--video-negative-prompt",
        default="jitter, bad hands, blur, distortion",
        help="Negative prompt for video.",
    )
    parser.add_argument(
        "--audio-negative-prompt",
        default="robotic, muffled, echo, distorted",
        help="Negative prompt for audio.",
    )
    parser.add_argument("--output", default="dreamid_output.mp4", help="Output video path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt is None and args.prompt_file is None:
        raise ValueError("Either --prompt or --prompt-file must be provided.")

    text_prompt = args.prompt
    if args.prompt_file:
        import json

        with open(args.prompt_file) as f:
            text_prompt = json.load(f)
            text_prompt = re.sub(
                r"\[SPEAKER_TIMESTAMPS_START\].*?\[SPEAKER_TIMESTAMPS_END\]", "", text_prompt, flags=re.DOTALL
            ).strip()
            text_prompt = re.sub(
                r"\[AUDIO_DESCRIPTION_START].*?\[AUDIO_DESCRIPTION_END]", "", text_prompt, flags=re.DOTALL
            ).strip()
            text_prompt = re.sub(r"\[[A-Z_]+\]", "", text_prompt)
            text_prompt = re.sub(r"\n\s*\n", "\n", text_prompt).strip()

    prompt = {
        "prompt": text_prompt,
        "image_paths": args.image_path,
        "audio_paths": args.audio_path,
        "video_negative_prompt": args.video_negative_prompt,
        "audio_negative_prompt": args.audio_negative_prompt,
    }

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        extra_args={
            "solver_name": args.solver_name,
            "shift": args.shift,
        },
    )

    parallel_config = DiffusionParallelConfig(
        cfg_parallel_size=args.cfg_parallel_size,
    )

    omni = Omni(
        model=args.model,
        parallel_config=parallel_config,
        model_type=args.model_type,
    )
    start = time.perf_counter()
    outputs = omni.generate(prompt, sampling_params)
    elapsed = time.perf_counter() - start

    if not outputs:
        raise RuntimeError("No output returned from DreamID-Omni.")
    output = outputs[0].request_output
    generated_video = output[0].images[0][0]
    generated_audio = output[1].images[0][1]
    try:
        from vllm_omni.diffusion.models.dreamid_omni.utils.dependency_loader import ensure_dependencies

        ensure_dependencies()
        from ovi.utils.io_utils import save_video
    except Exception as e:
        raise RuntimeError(f"Failed to extract video and audio from DreamID-Omni output. Error: {e}")
    output_path = args.output
    save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
    print(f"Saved generated video to {output_path}")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
