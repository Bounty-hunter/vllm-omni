from typing import Any

import torch
from diffusers.models.autoencoders import AutoencoderKL

from vllm_omni.diffusion.distributed.autoencoders.vae_base import DistributedVaeDecode, GridSpec, TileTask


class DistributedAutoencoderKL(AutoencoderKL, DistributedVaeDecode):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model._init_distributed_decode()
        model.support_tile_parallel = True
        return model

    def tile_split_func(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        # mostly copy from AutoencoderKL
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        self.blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        self.row_limit = self.tile_sample_min_size - self.blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        tiletask_list = []
        for i in range(0, z.shape[2], overlap_size):
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tiletask_list.append(TileTask(len(tiletask_list), (i // overlap_size, j // overlap_size), tile))

        grid_spec = GridSpec(
            split_dims=(2, 3),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
        )
        return tiletask_list, grid_spec

    def tile_decode_func(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile into RGB space."""
        tile = task.tensor
        if self.config.use_post_quant_conv:
            tile = self.post_quant_conv(tile)
        decoded = self.decoder(tile)
        return decoded

    def tile_merge_func(
        self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec
    ) -> torch.Tensor:
        """Merge decoded tiles into a full image."""

        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, self.blend_extent)
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, self.blend_extent)
                result_row.append(tile[:, :, : self.row_limit, : self.row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        return dec

    def decode(self, z: torch.Tensor, return_dict: bool = True, *args: Any, **kwargs: Any):
        result = self.maybe_decode_with_distribute(z)
        if result is not None:
            if not return_dict:
                return (result,)

            from diffusers.models.autoencoders.vae import DecoderOutput

            return DecoderOutput(sample=result)

        else:
            return super().decode(z, return_dict=return_dict, *args, **kwargs)
