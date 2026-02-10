import torch
from PIL import Image

from tests.e2e.accuracy.utils import DiffusersConsistencyTestBase


class TestQwenImageEditPlusConsistency(DiffusersConsistencyTestBase):
    model_name = "/home/d00806799/Qwen-Image-Edit"

    def build_diffusers_pipeline(self):
        from diffusers import QwenImageEditPipeline

        pipe = QwenImageEditPipeline.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
        pipe.to(self.device)
        return pipe

    def get_images(self):
        image = Image.new("RGB", (32, 32))
        return image

    def get_diffusers_params(self):
        diffusers_params = super().get_diffusers_params()
        diffusers_params.update(
            dict(
                image=self.images(),
                negative_prompt="low quality",
            )
        )
        return diffusers_params

    def get_omni_params(self):
        prompt_params, sample_params = super().get_omni_params()
        prompt_params.update(
            dict(
                negative_prompt="low quality",
                multi_modal_data={"image": self.images()},
            )
        )
        return prompt_params, sample_params

    def test_consistency(self):
        diffusers_result = self.run_diffusers()
        diffusers_result.images[0].save("diffusers_output.png")
        omni_result = self.run_vllm_omni()
        omni_result[0].request_output[0].images[0].save("omni_output.png")
        self.assert_image_consistent(omni_result[0].request_output[0].images[0], diffusers_result.images[0], atol=1e-3)  # type: ignore
