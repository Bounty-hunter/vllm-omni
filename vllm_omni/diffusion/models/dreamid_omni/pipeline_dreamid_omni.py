# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import math
import os

import librosa
import torch
import torch.distributed
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image, ImageOps
from torch import nn
from torchvision.transforms import Compose, Normalize
from tqdm import tqdm

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.dreamid_omni.utils.divisible_crop import DivisibleCrop
from vllm_omni.diffusion.models.dreamid_omni.utils.rearrange import Rearrange
from vllm_omni.diffusion.models.dreamid_omni.utils.resize import NaResize
from vllm_omni.diffusion.request import OmniDiffusionRequest

try:
    from vllm_omni.diffusion.models.dreamid_omni.utils.dependency_loader import ensure_dependency

    ensure_dependency("ovi")
    from ovi.modules.fusion import FusionModel
    from ovi.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
    from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from ovi.utils.model_loading_utils import (
        init_mmaudio_vae,
        init_text_model,
        init_wan_vae_2_2,
        load_fusion_checkpoint,
    )
except ImportError:
    raise ImportError("Failed to download and import dependency 'ovi'.")


logger = logging.getLogger(__name__)


class DreamIDOmniPipeline(nn.Module, CFGParallelMixin):
    """DreamID-Omni pipeline for vLLM-Omni."""

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config

        self.device = get_local_device()
        model = od_config.model
        ensure_dependency()
        self.target_dtype = torch.bfloat16

        # Init Models
        ## Load VAEs
        vae_model_video = init_wan_vae_2_2(model, rank=self.device)
        vae_model_video.model.requires_grad_(False).eval()
        vae_model_video.model = vae_model_video.model.bfloat16()
        self.vae_model_video = vae_model_video

        vae_model_audio = init_mmaudio_vae(model, rank=self.device)
        vae_model_audio.requires_grad_(False).eval()
        self.vae_model_audio = vae_model_audio.bfloat16()

        # Load T5 text model
        self.text_model = init_text_model(model, rank=self.device, cpu_offload=self.cpu_offload)

        # Fusion model
        ## load audio/video model config
        audio_config, video_config = self.get_audio_video_model_config()
        model = FusionModel(video_config, audio_config)

        checkpoint_path = os.path.join(
            model,
            "DreamID_Omni",
            "dreamid_omni_oneip_part1_old_1000.safetensors",
        )
        load_fusion_checkpoint(model, checkpoint_path=checkpoint_path)
        self.model = model

        # Fixed attributes, non-configurable
        self.audio_latent_channel = audio_config.get("in_dim")
        self.video_latent_channel = video_config.get("in_dim")
        self.video_latent_length = 31
        self.audio_latent_length = 157
        self.target_area = 960 * 960

        # Fixed attributes, CFG scales
        self.video_cfg_scale = 3.0
        self.video_ref_cfg_scale = 1.5
        self.audio_cfg_scale = 4.0
        self.audio_ref_cfg_scale = 2.0

        # Schedulers will be set in forward
        self.scheduler_video = None
        self.scheduler_audio = None

    def get_audio_video_model_config(self):
        import os

        BASE_DIR = os.path.dirname(__file__)
        video_config = os.path.join(BASE_DIR, "../config/video.json")
        audio_config = os.path.join(BASE_DIR, "../config/audio.json")
        with open(video_config) as f:
            video_config = json.load(f)

        with open(audio_config) as f:
            audio_config = json.load(f)
        return audio_config, video_config

    def load_image_latent_ref_ip_video(
        self,
        paths: str,
        video_frame_height_width,
    ):
        # Load size.
        patch_size = self.model.video_model.patch_size
        vae_stride = [4, 16, 16]

        def is_image_or_video_by_extension(file_path):
            image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
            audio_exts = {".wav", ".mp3", ".aac", ".flac"}

            ext = os.path.splitext(file_path)[1].lower()
            if ext in image_exts:
                return "image"
            elif ext in video_exts:
                return "video"
            elif ext in audio_exts:
                return "audio"
            else:
                return "unknown"

        # import pdb; pdb.set_trace()
        # Load image and video.
        ref_vae_latents = {
            "image": [],
            "audio": [],
        }
        video_h = video_frame_height_width[0]
        video_w = video_frame_height_width[1]
        if self.cpu_offload:
            self.vae_model_video.model = self.vae_model_video.model.to(self.device)
        ref_audio_lengths = []
        # import ipdb; ipdb.set_trace()
        for path in paths:
            if is_image_or_video_by_extension(path) == "image":
                with Image.open(path) as img:
                    img = img.convert("RGB")

                    # Calculate the required size to keep aspect ratio and fill the rest with padding.
                    img_ratio = img.width / img.height
                    target_ratio = video_w / video_h

                    if img_ratio > target_ratio:  # Image is wider than target
                        new_width = video_w
                        new_height = int(new_width / img_ratio)
                    else:  # Image is taller than target
                        new_height = video_h
                        new_width = int(new_height * img_ratio)

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # Create a new image with the target size and place the resized image in the center
                    delta_w = video_w - img.size[0]
                    delta_h = video_h - img.size[1]
                    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                    new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

                # Transform to tensor and normalize.
                image_transform = Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(
                                video_frame_height_width[0] * video_frame_height_width[1]
                            ),  # 256*448, 480*832
                            mode="area",
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                new_img = image_transform([new_img])
                new_img = new_img.transpose(0, 1)
                new_img = new_img.to(self.device)
                new_img = new_img.to(self.target_dtype)

                with torch.no_grad():
                    img_vae_latent = (
                        self.vae_model_video.wrapped_encode(new_img[:, :, None]).to(self.target_dtype).squeeze(0)
                    )
                ref_vae_latents["image"].append(img_vae_latent)

            elif is_image_or_video_by_extension(path) == "audio":
                audio_array, sr = librosa.load(path, sr=16000)
                audio_array = audio_array[int(sr * 1) : int(sr * 3)]

                audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
                # print(audio_tensor.shape)#torch.Size([1, 81280])
                audio_vae_latent = self.vae_model_audio.wrapped_encode(audio_tensor)
                # print(audio_vae_latent.shape)#torch.Size([1, 20, length])
                audio_length = audio_vae_latent.shape[2]
                ref_audio_lengths.append(audio_length)
                audio_vae_latent = audio_vae_latent.squeeze(0).transpose(0, 1)
                ref_vae_latents["audio"].append(audio_vae_latent)
            else:
                print("Unknown file type.")
        # import ipdb; ipdb.set_trace()

        ref_vae_latents["image"] = torch.cat(ref_vae_latents["image"], dim=1)

        ref_vae_latents["audio"] = torch.cat(ref_vae_latents["audio"], dim=0)

        return ref_vae_latents, ref_audio_lengths

    def get_scheduler_time_steps(self, sampling_steps, solver_name="unipc", device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps

        elif solver_name == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

        elif solver_name == "euler":
            sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
            timesteps, sampling_steps = retrieve_timesteps(
                sample_scheduler,
                sampling_steps,
                device=device,
            )

        else:
            raise NotImplementedError("Unsupported solver.")

        return sample_scheduler, timesteps

    def diffuse(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        latents_ref_image: torch.Tensor,
        latents_ref_audio: torch.Tensor,
        timesteps_video: torch.Tensor,
        timesteps_audio: torch.Tensor,
        text_embeddings_video_pos: torch.Tensor,
        text_embeddings_video_neg: torch.Tensor,
        text_embeddings_audio_pos: torch.Tensor,
        text_embeddings_audio_neg: torch.Tensor,
        max_seq_len_video: int,
        max_seq_len_audio: int,
        freqs_scaling_tensor: torch.Tensor,
        ref_ip_num: int,
        ref_audio_length: int,
        ref_audio_lengths: list,
        scheduler_video,
        scheduler_audio,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Diffusion loop with CFG parallel support for DreamID-Omni.
        """
        for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio))):
            timestep_input = torch.full((1,), t_v, device=self.device)

            model_input_video = torch.cat([video_noise[:, :-ref_ip_num], latents_ref_image], dim=1)
            model_input_video_neg = torch.cat(
                [video_noise[:, :-ref_ip_num], torch.zeros_like(latents_ref_image)], dim=1
            )

            model_input_audio = torch.cat([audio_noise[:-ref_audio_length, :], latents_ref_audio], dim=0)

            model_input_audio_neg = torch.cat(
                [audio_noise[:-ref_audio_length, :], torch.zeros_like(latents_ref_audio)], dim=0
            )
            ref_ip_lengths = [ref_ip_num]

            common_args = {
                "vid_seq_len": max_seq_len_video,
                "audio_seq_len": max_seq_len_audio,
                "freqs_scaling": freqs_scaling_tensor,
                "ref_ip_lengths": [ref_ip_lengths],
                "ref_audio_lengths": [ref_audio_lengths],
            }
            pos_args = {
                **common_args,
                "audio_context": [text_embeddings_audio_pos],
                "vid_context": [text_embeddings_video_pos],
            }
            neg_args = {
                **common_args,
                "audio_context": [text_embeddings_audio_neg],
                "vid_context": [text_embeddings_video_neg],
            }

            if get_classifier_free_guidance_world_size() > 1:
                # Enable CFG-parallel: rank0 computes positive, rank1 computes negative.
                cfg_group = get_cfg_group()
                cfg_rank = get_classifier_free_guidance_rank()

                if cfg_rank == 0:
                    pred_vid, pred_audio = self.model(
                        vid=[model_input_video], audio=[model_input_audio], t=timestep_input, **pos_args
                    )
                    pre_vid_ip_neg, _ = self.model(
                        vid=[model_input_video_neg], audio=[model_input_audio], t=timestep_input, **pos_args
                    )
                    mix_pred = pre_vid_ip_neg
                else:
                    pred_vid, pred_audio = self.model(
                        vid=[model_input_video], audio=[model_input_audio], t=timestep_input, **neg_args
                    )
                    _, pred_refaudio_neg = self.model(
                        vid=[model_input_video], audio=[model_input_audio_neg], t=timestep_input, **pos_args
                    )
                    mix_pred = pred_refaudio_neg
                pred_vid_gathered = cfg_group.all_gather(pred_vid, separate_tensors=True)
                pred_audio_gathered = cfg_group.all_gather(pred_audio, separate_tensors=True)
                mix_pred_gathered = cfg_group.all_gather(mix_pred, separate_tensors=True)
                pred_vid_pos = pred_vid_gathered[0]
                pred_vid_neg = pred_vid_gathered[1]
                pred_audio_pos = pred_audio_gathered[0]
                pred_audio_neg = pred_audio_gathered[1]
                pre_vid_ip_neg = mix_pred_gathered[0]
                pred_refaudio_neg = mix_pred_gathered[1]
            else:
                pred_vid_pos, pred_audio_pos = self.model(
                    vid=[model_input_video], audio=[model_input_audio], t=timestep_input, **pos_args
                )

                pred_vid_neg, pred_audio_neg = self.model(
                    vid=[model_input_video], audio=[model_input_audio], t=timestep_input, **neg_args
                )

                pre_vid_ip_neg, _ = self.model(
                    vid=[model_input_video_neg], audio=[model_input_audio], t=timestep_input, **pos_args
                )

                _, pred_refaudio_neg = self.model(
                    vid=[model_input_video], audio=[model_input_audio_neg], t=timestep_input, **pos_args
                )

            pred_video_guided = (
                pred_vid_neg[0]
                + self.video_cfg_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                + self.video_ref_cfg_scale * (pred_vid_pos[0] - pre_vid_ip_neg[0])
            )

            pred_audio_guided = (
                pred_audio_neg[0]
                + self.audio_cfg_scale * (pred_audio_pos[0] - pred_audio_neg[0])
                + self.audio_ref_cfg_scale * (pred_audio_pos[0] - pred_refaudio_neg[0])
            )
            video_noise = scheduler_video.step(
                pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
            )[0].squeeze(0)
            audio_noise = scheduler_audio.step(
                pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
            )[0].squeeze(0)

    def forward(
        self,
        request: OmniDiffusionRequest,
        **kwargs,
    ) -> DiffusionOutput:
        """Main forward pass for DreamID-Omni pipeline for R2AV task."""
        # Extract parameters from request
        prompt = request.prompts[0].get("prompt")
        video_negative_prompt = request.prompts[0].get("video_negative_prompt")
        audio_negative_prompt = request.prompts[0].get("audio_negative_prompt")
        image_paths = request.prompts[0].get("image_paths")
        audio_paths = request.prompts[0].get("audio_paths")

        height = request.sampling_params.height
        width = request.sampling_params.width
        num_inference_steps = request.sampling_params.num_inference_steps
        shift = request.sampling_params.extra_args.get("shift", 5.0)
        solver_name = request.sampling_params.extra_args.get("solver_name", "unipc")

        # 1. Prepare reference latents
        paths = image_paths + audio_paths

        ref_vae_latents, ref_audio_lengths = self.load_image_latent_ref_ip_video(
            paths=paths,
            video_frame_height_width=(height, width),
        )

        latents_ref_image = ref_vae_latents["image"]
        latents_ref_audio = ref_vae_latents["audio"]
        ref_ip_num = latents_ref_image.shape[1]
        ref_audio_length = latents_ref_audio.shape[0]

        # 2. scheduler
        scheduler_video, timesteps_video = self.get_scheduler_time_steps(
            sampling_steps=num_inference_steps, device=self.device, solver_name=solver_name, shift=shift
        )
        scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
            sampling_steps=num_inference_steps, device=self.device, solver_name=solver_name, shift=shift
        )

        # 3. text embedding
        text_embeddings = self.text_model([prompt, video_negative_prompt, audio_negative_prompt])
        text_embeddings = [emb.to(self.target_dtype) for emb in text_embeddings]
        text_embeddings_audio_pos = text_embeddings[0]
        text_embeddings_video_pos = text_embeddings[0]
        text_embeddings_video_neg = text_embeddings[1]
        text_embeddings_audio_neg = text_embeddings[2]

        video_latent_h, video_latent_w = height // 16, width // 16

        video_noise_len = self.video_latent_length + ref_ip_num
        audio_noise_len = self.audio_latent_length + ref_audio_length
        freqs_scaling_tensor = torch.tensor(
            self.video_latent_length / self.audio_latent_length, device=self.device, dtype=self.target_dtype
        )

        video_noise = torch.randn(
            (self.video_latent_channel, video_noise_len, video_latent_h, video_latent_w),
            device=self.device,
            dtype=self.target_dtype,
            generator=torch.Generator(device=self.device).manual_seed(42),
        )
        audio_noise = torch.randn(
            (audio_noise_len, self.audio_latent_channel),
            device=self.device,
            dtype=self.target_dtype,
            generator=torch.Generator(device=self.device).manual_seed(42),
        )

        _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]

        max_seq_len_video = (
            video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h * _patch_size_w)
        )

        max_seq_len_audio = audio_noise_len

        with torch.amp.autocast("cuda", enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
            self.diffuse(
                video_noise,
                audio_noise,
                latents_ref_image,
                latents_ref_audio,
                timesteps_video,
                timesteps_audio,
                text_embeddings_video_pos,
                text_embeddings_video_neg,
                text_embeddings_audio_pos,
                text_embeddings_audio_neg,
                max_seq_len_video,
                max_seq_len_audio,
                freqs_scaling_tensor,
                ref_ip_num,
                ref_audio_length,
                ref_audio_lengths,
                scheduler_video,
                scheduler_audio,
            )

        video_noise_for_decode = video_noise[:, :-ref_ip_num]
        audio_noise_for_decode = audio_noise[:-ref_audio_length, :]

        audio_latents_for_vae = audio_noise_for_decode.unsqueeze(0).transpose(1, 2)
        generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae).squeeze().cpu().float().numpy()

        video_latents_for_vae = video_noise_for_decode.unsqueeze(0)
        generated_video = self.vae_model_video.wrapped_decode(video_latents_for_vae).squeeze(0).cpu().float().numpy()

        return DiffusionOutput(output=(generated_video, generated_audio))
