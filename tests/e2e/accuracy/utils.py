import gc
import unittest

import numpy as np
import torch
from PIL import Image

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class DiffusersConsistencyTestBase(unittest.TestCase):
    # need to set these attributes in subclasses
    model_name = None

    # fixed parameters for the test
    height = 256
    width = 256
    num_inference_steps = 2
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _cached_images = None
    _cached_prompt = None

    # We can add or delete parameters as needed
    def get_diffusers_params(self):
        return dict(
            prompt=self.prompt(),
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator(self.device).manual_seed(self.seed),
        )

    def get_omni_params(self):
        prompt_params = dict(prompt=self.prompt())

        sample_params = OmniDiffusionSamplingParams(
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator(self.device).manual_seed(self.seed),
        )

        return prompt_params, sample_params

    def build_diffusers_pipeline(self):
        raise NotImplementedError

    def build_vllm_omni_pipeline(self):
        return Omni(
            model=self.model_name,
        )

    def get_prompt(self):
        return "dance monkey"

    def get_images(self):
        image = Image.new("RGB", (32, 32), color="red")
        return image

    def prompt(self):
        if self._cached_prompt is None:
            self._cached_prompt = self.get_prompt()
        return self._cached_prompt

    def images(self):
        if self._cached_images is None:
            self._cached_images = self.get_images()
        return self._cached_images

    # no need to change in most cases, but can be overridden if needed
    def run_diffusers(self):
        print("diffusers run 0")
        pipe = self.build_diffusers_pipeline()
        print("diffusers run 1")
        run_params = self.get_diffusers_params()
        print(f"diffusers run params is {run_params}")
        output = pipe(**run_params)
        print(f"diffusers output is {output}")
        output.images[0].save("diffusers_output_internal.png")
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return output

    def run_vllm_omni(self):
        pipe = self.build_vllm_omni_pipeline()
        prompt_params, sample_params = self.get_omni_params()
        output = pipe.generate(prompt_params, sample_params)
        print(f"vllm omni output is {output}")
        pipe.close()  # close the pipeline to free up resources
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return output

    def assert_image_consistent(self, generated_image, expected_image, atol=1e-3):
        gen_np = np.array(generated_image).astype(np.float32)
        exp_np = np.array(expected_image).astype(np.float32)
        assert gen_np.shape == exp_np.shape
        print(f"dyyyyy generated output is {gen_np}")
        print(f"dyyyyy expected output is {exp_np}")
        diff = np.abs(gen_np - exp_np)
        max_diff = diff.max()
        assert max_diff < atol, f"Max diff too large: {max_diff}"
