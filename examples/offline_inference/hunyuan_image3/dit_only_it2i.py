"""
HunyuanImage-3.0-Instruct DiT-only IT2I (image-to-image) inference script.

This script runs only the DiT stage without the AR stage for image editing tasks.
It uses bot_task=None to avoid AR-specific trigger tags like <think>.

Usage:
    python dit_only_it2i.py \
        --model tencent/HunyuanImage-3.0-Instruct \
        --prompts "Change the cat to a dog" \
        --image-path input.jpg \
        --output ./results \
        --steps 50 \
        --guidance-scale 5.0
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    MAX_IMAGES_PER_REQUEST,
    build_prompt_tokens,
)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_dit.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0 DiT-only IT2I inference.")
    parser.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct", help="Model name or local path.")
    parser.add_argument("--prompts", nargs="+", required=True, help="Input text prompts.")
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Input image path(s). Comma-separated for multi-image (up to 3).",
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory to save results.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=None, help="Output image height (optional).")
    parser.add_argument("--width", type=int, default=None, help="Output image width (optional).")
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument("--deploy-config", type=str, default=None, help="Custom deploy YAML path.")
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--init-timeout", type=int, default=300, help="Initialization timeout in seconds.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument(
        "--diffusion-kv-cache-dtype",
        type=str,
        default=None,
        help="Diffusion attention KV cache dtype, for example 'fp8'.",
    )
    parser.add_argument(
        "--diffusion-kv-cache-skip-steps",
        type=str,
        default=None,
        help="Denoising step selector to keep diffusion KV cache in native dtype, for example '0,1,4-6'.",
    )
    parser.add_argument(
        "--diffusion-kv-cache-skip-layers",
        type=str,
        default=None,
        help="Transformer layer selector to keep diffusion KV cache in native dtype, for example '0-2,10'.",
    )
    parser.add_argument(
        "--additional-config",
        type=str,
        default=None,
        help=(
            "JSON object forwarded to Omni/additional_config, for example "
            '\'{"torchair_graph_config":{"enabled":true}}\'. '
        ),
    )

    return parser.parse_args()


def parse_additional_config(raw_value: str | None) -> dict | None:
    """Parse a JSON string into an additional_config mapping."""
    if raw_value is None:
        return None

    try:
        additional_config = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --additional-config JSON: {exc}") from exc

    if additional_config is None:
        return None
    if not isinstance(additional_config, dict):
        raise ValueError(f"--additional-config must decode to a JSON object, got {type(additional_config).__name__}")
    return additional_config


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    additional_config = parse_additional_config(args.additional_config)

    deploy_config = args.deploy_config or _DEFAULT_DEPLOY_CONFIG

    omni_kwargs = {
        "model": args.model,
        "deploy_config": deploy_config,
        "vae_use_tiling": args.vae_use_tiling,
        "log_stats": args.log_stats,
        "init_timeout": args.init_timeout,
        "enforce_eager": args.enforce_eager,
        "mode": "image-editing",
        "diffusion_kv_cache_dtype": args.diffusion_kv_cache_dtype,
        "diffusion_kv_cache_skip_steps": args.diffusion_kv_cache_skip_steps,
        "diffusion_kv_cache_skip_layers": args.diffusion_kv_cache_skip_layers,
    }

    if additional_config is not None:
        omni_kwargs["additional_config"] = additional_config

    omni = Omni(**omni_kwargs)

    # Load images
    image_paths = [p.strip() for p in args.image_path.split(",") if p.strip()]
    if len(image_paths) > MAX_IMAGES_PER_REQUEST:
        raise ValueError(
            f"--image-path accepts at most {MAX_IMAGES_PER_REQUEST} images for "
            f"HunyuanImage-3.0 IT2I, got {len(image_paths)}: {args.image_path}"
        )

    input_images = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")
        input_images.append(Image.open(image_path).convert("RGB"))

    if not input_images:
        raise ValueError(f"--image-path produced no usable paths: {args.image_path!r}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build prompts with bot_task=None (no AR trigger tags)
    prompts = args.prompts
    mm_image_payload = (input_images[0] if len(input_images) == 1 else input_images) if input_images else None

    formatted_prompts: list[OmniPromptType] = []
    for prompt in prompts:
        # Key: use bot_task=None to avoid <think> trigger tag
        result = build_prompt_tokens(
            user_prompt=prompt,
            tokenizer=tokenizer,
            task="it2i",
            bot_task=None,  # DiT-only mode: no AR trigger tags
            sys_type=None,  # Use default system prompt
            num_images=len(input_images),
        )

        prompt_dict: dict = {
            "prompt_token_ids": result.token_ids,
            "prompt": prompt,
            "modalities": ["image"],
            "multi_modal_data": {"image": mm_image_payload},
            "height": input_images[0].height,
            "width": input_images[0].width,
            "use_system_prompt": result.system_prompt_type,
        }
        formatted_prompts.append(prompt_dict)

    # Prepare sampling params
    params_list = list(omni.default_sampling_params_list)

    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must both be specified or both omitted.")

    for sp in params_list:
        if isinstance(sp, OmniDiffusionSamplingParams):
            sp.num_inference_steps = args.steps
            sp.guidance_scale = args.guidance_scale
            sp.guidance_scale_provided = True
            if args.seed is not None:
                sp.seed = args.seed
            if args.height is not None:
                sp.height = args.height
            if args.width is not None:
                sp.width = args.width

    print(f"\n{'=' * 60}")
    print("HunyuanImage-3.0 DiT-only IT2I Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Deploy config: {deploy_config}")
    print(f"  Num stages: {omni.num_stages}")
    print("  Bot task: None (DiT-only, no AR trigger tags)")
    print(f"  Inference steps: {args.steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")
    if args.height and args.width:
        print(f"  Output size: {args.width}x{args.height}")
    else:
        print(f"  Output size: {input_images[0].width}x{input_images[0].height} (from input)")
    print(f"  Input images: {args.image_path}")
    print(f"  diffusion_kv_cache_dtype: {args.diffusion_kv_cache_dtype}")
    print(f"  diffusion_kv_cache_skip_steps: {args.diffusion_kv_cache_skip_steps}")
    print(f"  diffusion_kv_cache_skip_layers: {args.diffusion_kv_cache_skip_layers}")
    if additional_config is not None:
        print(f"  Additional config: {additional_config}")
    print(f"  Prompts: {prompts}")
    print(f"{'=' * 60}\n")

    # Run inference
    omni_outputs = list(omni.generate(prompts=formatted_prompts, sampling_params_list=params_list))

    # Save results
    img_idx = 0
    for req_output in omni_outputs:
        images = getattr(req_output, "images", None)
        if not images:
            ro = getattr(req_output, "request_output", None)
            if ro and hasattr(ro, "images"):
                images = ro.images

        if images:
            for j, img in enumerate(images):
                save_path = os.path.join(args.output, f"output_{img_idx}_{j}.png")
                img.save(save_path)
                print(f"[Output] Saved image to {save_path}")
            img_idx += 1
        else:
            print(f"[Warning] No images generated for prompt: {prompts[img_idx]}")


if __name__ == "__main__":
    main()
