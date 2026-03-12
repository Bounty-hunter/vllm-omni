# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import math
import os
from typing import List, Optional

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
    from vllm_omni.diffusion.models.dreamid_omni.utils.dependency_loader import ensure_dependencies

    ensure_dependencies()
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

from vllm_omni.diffusion.models.dreamid_omni.fusion import FusionModel

logger = logging.getLogger(__name__)


AUDIO_CONFIG = {
    "patch_size": [1],
    "model_type": "t2a",
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "num_heads": 24,
    "num_layers": 30,
    "in_dim": 20,
    "out_dim": 20,
    "text_len": 512,
    "window_size": [-1, -1],
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "temporal_rope_scaling_factor": 0.19676,
}

VIDEO_CONFIG = {
    "patch_size": [1, 2, 2],
    "model_type": "ti2v",
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "num_heads": 24,
    "num_layers": 30,
    "in_dim": 48,
    "out_dim": 48,
    "text_len": 512,
    "window_size": [-1, -1],
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}


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
        ensure_dependencies()
        self.target_dtype = torch.bfloat16

        # Init models
        vae_model_video = init_wan_vae_2_2(model, rank=self.device)
        vae_model_video.model.requires_grad_(False).eval()
        vae_model_video.model = vae_model_video.model.bfloat16()
        self.vae_model_video = vae_model_video

        vae_model_audio = init_mmaudio_vae(model, rank=self.device)
        vae_model_audio.requires_grad_(False).eval()
        self.vae_model_audio = vae_model_audio.bfloat16()

        self.text_model = init_text_model(model, rank=self.device)

        fusion_model = FusionModel(VIDEO_CONFIG, AUDIO_CONFIG)

        checkpoint_path = os.path.join(
            model,
            "dreamid_omni_oneip_part1_old_1000.safetensors",
        )
        load_fusion_checkpoint(fusion_model, checkpoint_path=checkpoint_path)
        self.model = fusion_model

        # Fixed attributes
        self.audio_latent_channel = AUDIO_CONFIG.get("in_dim")
        self.video_latent_channel = VIDEO_CONFIG.get("in_dim")
        self.video_latent_length = 31
        self.audio_latent_length = 157
        self.target_area = 960 * 960

        # CFG scales
        self.video_cfg_scale = 3.0
        self.video_ref_cfg_scale = 1.5
        self.audio_cfg_scale = 4.0
        self.audio_ref_cfg_scale = 2.0

        # Schedulers will be set in forward
        self.scheduler_video = None
        self.scheduler_audio = None


    def load_image_latent_ref_ip_video(
        self,
        image_paths,
        audio_paths,
        video_frame_height_width,
    ):
        patch_size = self.model.video_model.patch_size
        vae_stride = [4, 16, 16]

        ref_vae_latents = {
            "image": [],
            "audio": [],
        }
        ref_audio_lengths = []

        video_h, video_w = video_frame_height_width

        # 1) encode images from paths
        for path in image_paths:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize((video_w, video_h), Image.Resampling.LANCZOS)

            image_transform = Compose(
                [
                    NaResize(
                        resolution=math.sqrt(video_h * video_w),
                        mode="area",
                        downsample_only=True,
                    ),
                    DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                    Normalize(0.5, 0.5),
                    Rearrange("t c h w -> c t h w"),
                ]
            )

            new_img = image_transform([img])
            new_img = new_img.transpose(0, 1)
            new_img = new_img.to(self.device, dtype=self.target_dtype)

            with torch.no_grad():
                img_vae_latent = (
                    self.vae_model_video.wrapped_encode(new_img[:, :, None])
                    .to(self.target_dtype)
                    .squeeze(0)
                )

            ref_vae_latents["image"].append(img_vae_latent)

        # 2) encode audios from paths
        for path in audio_paths:
            audio_array, sr = librosa.load(path, sr=16000)
            audio_array = audio_array[int(sr * 1): int(sr * 3)]

            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                audio_vae_latent = self.vae_model_audio.wrapped_encode(audio_tensor)

            audio_length = audio_vae_latent.shape[2]
            ref_audio_lengths.append(audio_length)

            audio_vae_latent = audio_vae_latent.squeeze(0).transpose(0, 1)
            ref_vae_latents["audio"].append(audio_vae_latent)
        ref_vae_latents["image"] = torch.cat(ref_vae_latents["image"], dim=1)
        ref_vae_latents["audio"] = torch.cat(ref_vae_latents["audio"], dim=0)

        return ref_vae_latents, ref_audio_lengths

    def load_weights(self, weights):
        pass

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
        Returns the updated video_noise and audio_noise.
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
                ▪ self.video_cfg_scale * (pred_vid_pos[0] - pred_vid_neg[0])

                ▪ self.video_ref_cfg_scale * (pred_vid_pos[0] - pre_vid_ip_neg[0])

            )

            pred_audio_guided = (
                pred_audio_neg[0]
                ▪ self.audio_cfg_scale * (pred_audio_pos[0] - pred_audio_neg[0])

                ▪ self.audio_ref_cfg_scale * (pred_audio_pos[0] - pred_refaudio_neg[0])

            )
            video_noise = scheduler_video.step(
                pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
            )[0].squeeze(0)
            audio_noise = scheduler_audio.step(
                pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
            )[0].squeeze(0)

        return video_noise, audio_noise

    def forward(
        self,
        request: OmniDiffusionRequest,
        **kwargs,
    ) -> DiffusionOutput:
        """Main forward pass for DreamID-Omni pipeline for R2AV task."""
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
        seed = request.sampling_params.seed if request.sampling_params.seed is not None else 42

        if len(image_paths) == 0 and len(audio_paths) == 0:
            raise ValueError("Both image_paths and audio_paths are empty. At least one valid input is required.")

        ref_vae_latents, ref_audio_lengths = self.load_image_latent_ref_ip_video(
            image_paths=image_paths,
            audio_paths=audio_paths,
            video_frame_height_width=(height, width),
        )
        latents_ref_image = ref_vae_latents["image"]
        latents_ref_audio = ref_vae_latents["audio"]

        ref_ip_num = 0 if latents_ref_image is None else latents_ref_image.shape[1]
        ref_audio_length = 0 if latents_ref_audio is None else latents_ref_audio.shape[0]

        scheduler_video, timesteps_video = self.get_scheduler_time_steps(
            sampling_steps=num_inference_steps,
            device=self.device,
            solver_name=solver_name,
            shift=shift,
        )
        scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
            sampling_steps=num_inference_steps,
            device=self.device,
            solver_name=solver_name,
            shift=shift,
        )

        text_embeddings = self.text_model(
            [prompt, video_negative_prompt, audio_negative_prompt],
            device=self.device,
        )
        text_embeddings = [emb.to(self.target_dtype) for emb in text_embeddings]
        text_embeddings_audio_pos = text_embeddings[0]
        text_embeddings_video_pos = text_embeddings[0]
        text_embeddings_video_neg = text_embeddings[1]
        text_embeddings_audio_neg = text_embeddings[2]

        video_latent_h, video_latent_w = height // 16, width // 16

        video_noise_len = self.video_latent_length + ref_ip_num
        audio_noise_len = self.audio_latent_length + ref_audio_length
        freqs_scaling_tensor = torch.tensor(
            self.video_latent_length / self.audio_latent_length,
            device=self.device,
            dtype=self.target_dtype,
        )

        video_noise = torch.randn(
            (self.video_latent_channel, video_noise_len, video_latent_h, video_latent_w),
            device=self.device,
            dtype=self.target_dtype,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        audio_noise = torch.randn(
            (audio_noise_len, self.audio_latent_channel),
            device=self.device,
            dtype=self.target_dtype,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        if (latents_ref_image is None) or (latents_ref_audio is None):
            raise ValueError("latents_ref_image or latents_ref_audio is None: no valid reference image was loaded or encoded.")


        _patch_size_h = self.model.video_model.patch_size[1]
        _patch_size_w = self.model.video_model.patch_size[2]

        max_seq_len_video = (
            video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3]
            // (_patch_size_h * _patch_size_w)
        )
        max_seq_len_audio = audio_noise_len

        with torch.amp.autocast("cuda", enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
            video_noise, audio_noise = self.diffuse(
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

        video_noise_for_decode = video_noise[:, :-ref_ip_num] if ref_ip_num > 0 else video_noise
        audio_noise_for_decode = audio_noise[:-ref_audio_length, :] if ref_audio_length > 0 else audio_noise

        audio_latents_for_vae = audio_noise_for_decode.unsqueeze(0).transpose(1, 2)
        generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae).squeeze().cpu().float().numpy()

        video_latents_for_vae = video_noise_for_decode.unsqueeze(0)
        generated_video = self.vae_model_video.wrapped_decode(video_latents_for_vae).squeeze(0).cpu().float().numpy()

        return DiffusionOutput(output=(generated_video, generated_audio))
