# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import re
import time
from pathlib import Path
from typing import List, Optional

from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def print_generation_config(args: argparse.Namespace, text_prompt: str) -> None:
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Text Prompt: {text_prompt}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Solver name: {args.solver_name}")
    print(f"  Shift: {args.shift}")
    print(f"  Image size: {args.width}x{args.height}")
    print(f"  Image paths: {args.image_path if args.image_path else 'None'}")
    print(f"  Audio paths: {args.audio_path if args.audio_path else 'None'}")
    print(f"  Video negative prompt: {args.video_negative_prompt}")
    print(f"  Audio negative prompt: {args.audio_negative_prompt}")
    print(f"  Output: {args.output}")
    print(f"  Disable dummy run: {args.disable_dummy_run}")
    print(f"{'=' * 60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for DreamID-Omni (video + audio).")
    parser.add_argument("--model", required=True, help="DreamID ckpt root directory.")
    parser.add_argument("--prompt", required=True, help="Text prompt.")

    parser.add_argument("--image-path", type=str, nargs="+", help="Image file path(s) or folder path(s).")
    parser.add_argument("--audio-path", type=str, nargs="+", help="Audio file path(s) or folder path(s).")
    parser.add_argument("--prompt-json-path", type=str, default=None, help="Text prompt in json format.")

    parser.add_argument("--height", type=int, default=704, help="Video height.")
    parser.add_argument("--width", type=int, default=1024, help="Video width.")
    parser.add_argument("--num-inference-steps", type=int, default=45, help="Sampling steps.")
    parser.add_argument("--solver-name", default="unipc", help="Solver name: unipc|dpm++|euler.")
    parser.add_argument("--shift", type=float, default=5.0, help="Scheduler shift.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
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


def resolve_media_paths(
    input_paths: Optional[List[str]],
    media_type: str,
) -> List[str]:
 
    if not input_paths:
        return []

    if media_type == "image":
        valid_exts = IMAGE_EXTENSIONS
    elif media_type == "audio":
        valid_exts = AUDIO_EXTENSIONS
    else:
        raise ValueError(f"Unsupported media_type: {media_type}")

    resolved_files: List[str] = []

    for raw_path in input_paths:
        path_obj = Path(raw_path).expanduser().resolve()

        if not path_obj.exists():
            print(f"[Warning] {media_type} path does not exist: {path_obj}")
            continue

        if path_obj.is_file():
            if path_obj.suffix.lower() in valid_exts:
                resolved_files.append(str(path_obj))
            else:
                print(f"[Warning] Skip unsupported {media_type} file: {path_obj}")
        elif path_obj.is_dir():
            matched_files = sorted(
                [
                    p for p in path_obj.iterdir()
                    if p.is_file() and p.suffix.lower() in valid_exts
                ]
            )
            if not matched_files:
                print(f"[Warning] No valid {media_type} files found in directory: {path_obj}")
            resolved_files.extend([str(p.resolve()) for p in matched_files])

    unique_files = list(dict.fromkeys(resolved_files))
    return unique_files

import os
from typing import List


def normalize_paths(paths) -> List[str]:
    if paths is None:
        raise ValueError("paths is None")
    if isinstance(paths, str):
        return [os.fspath(paths)]
    if isinstance(paths, (list, tuple)):
        normalized = []
        for p in paths:
            if p is None:
                continue
            normalized.append(os.fspath(p))
        return normalized
    raise TypeError(f"Unsupported path type: {type(paths)}")

def main() -> None:
    args = parse_args()

    text_prompt = args.prompt
    if args.prompt_json_path:
        import json

        with open(args.prompt_json_path, "r", encoding="utf-8") as f:
            text_prompt = json.load(f)
            text_prompt = re.sub(
                r"\[SPEAKER_TIMESTAMPS_START\].*?\[SPEAKER_TIMESTAMPS_END\]", "", text_prompt, flags=re.DOTALL
            ).strip()
            text_prompt = re.sub(
                r"\[AUDIO_DESCRIPTION_START].*?\[AUDIO_DESCRIPTION_END\]", "", text_prompt, flags=re.DOTALL
            ).strip()
            text_prompt = re.sub(r"\[[A-Z_]+\]", "", text_prompt)
            text_prompt = re.sub(r"\n\s*\n", "\n", text_prompt).strip()

    normalized_image_paths = normalize_paths(args.image_path)
    normalized_audio_paths = normalize_paths(args.audio_path)

    resolved_image_paths = resolve_media_paths(normalized_image_paths, "image")
    resolved_audio_paths = resolve_media_paths(normalized_audio_paths, "audio")


    prompt = {
        "prompt": text_prompt,
        "image_paths": resolved_image_paths,
        "audio_paths": resolved_audio_paths,
        "video_negative_prompt": args.video_negative_prompt,
        "audio_negative_prompt": args.audio_negative_prompt,
    }

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        extra_args={
            "solver_name": args.solver_name,
            "shift": args.shift,
            "seed": args.seed,
        },
    )

    start = time.perf_counter()
    engine = OmniDiffusion(
        model=args.model,
        model_type="dreamid-omni",
        disable_dummy_run=args.disable_dummy_run,  
    )
    outputs = engine.generate(prompt, sampling_params)
    elapsed = time.perf_counter() - start

    if not outputs:
        raise RuntimeError("No output returned from DreamID-Omni.")

    output = outputs[0]
    generated_video = output.images[0][0]
    generated_audio = output.images[0][1]

    try:
        from ovi.utils.io_utils import save_video
    except Exception as e:
        raise RuntimeError(f"Failed to extract video and audio from DreamID-Omni output. Error: {e}")

    output_path = args.output
    save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
    print_generation_config(args, text_prompt)
    print(f"Saved generated video to {output_path}")
    print(f"Total time: {elapsed:.2f}s")
    


if __name__ == "__main__":
    main()
