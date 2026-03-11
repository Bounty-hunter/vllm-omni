# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import re
import time

from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for DreamID-Omni (video + audio).")
    parser.add_argument("--model", required=True, help="DreamID ckpt root directory.")
    parser.add_argument("--prompt", required=True, help="Text prompt.")

    parser.add_argument("--image-path", type=str, nargs="+", help="list of image-path")
    parser.add_argument("--audio-path", type=str, nargs="+", help="list of audio-path")
    parser.add_argument("--prompt_json_path", type=str, default=None, help="Text prompt in json format.")

    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=720, help="Video width.")
    parser.add_argument("--num-inference-steps", type=int, default=45, help="Sampling steps.")
    parser.add_argument("--solver-name", default="unipc", help="Solver name: unipc|dpm++|euler.")
    parser.add_argument("--shift", type=float, default=5.0, help="Scheduler shift.")
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
    parser.add_argument("--disable-dummy-run", action="store_true", help="Disable engine warmup dummy run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    text_prompt = args.prompt
    if args.prompt_json_path:
        import json

        with open(args.prompt_json_path) as f:
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

    start = time.perf_counter()
    engine = OmniDiffusion(
        model=args.model,
        model_class_name="DreamIDOmniPipeline",
        disable_dummy_run=True,
    )
    outputs = engine.generate(prompt, sampling_params)
    elapsed = time.perf_counter() - start

    if not outputs:
        raise RuntimeError("No output returned from DreamID-Omni.")
    output = outputs[0]
    generated_video = output.images[0] if output.images else None
    generated_audio = output.multimodal_output.get("audio") if output.multimodal_output else None
    try:
        from ovi.utils import save_video
    except Exception as e:
        raise RuntimeError(f"Failed to extract video and audio from DreamID-Omni output. Error: {e}")
    output_path = args.output
    save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
    print(f"Saved generated video to {output_path}")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
