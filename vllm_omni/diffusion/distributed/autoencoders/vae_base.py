from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import get_dit_group

logger = init_logger(__name__)


@dataclass
class GridSpec:
    """The Grid shape split"""

    split_dims: tuple[int, ...]  # For example: (2,3,4) for (B, C, T, H, W), (2,3) for (B, C, H, W)
    grid_shape: tuple[int, ...]  # For example: (nt, nh, nw) for (B, C, T, H, W), (nh, nw) for (B, C, H, W)


@dataclass
class TileTask:
    tile_id: int
    grid_coord: tuple[int, ...]  # The coordinate of the tile in GridSpec.grid_shape
    tensor: torch.Tensor | list[torch.Tensor]  # The tile tensor


class DistributedVaeDecode(ABC):
    """
    Abstract util class for distributed patch/tile parallel VAE decoding.
    """

    def _init_distributed_decode(self):
        self.group = get_dit_group()
        self.vae_patch_parallel_size = 1
        self.support_tile_parallel = False
        self.support_patch_parallel = False

    def tile_split_func(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        """Split a latent tensor into tiles and return the list of TileTask and GridSpec."""
        pass

    def tile_decode_func(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile into RGB space."""
        pass

    def tile_merge_func(
        self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec
    ) -> torch.Tensor:
        """Merge decoded tiles into a full image."""
        pass

    def patch_split_func(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        """Split a latent tensor into tiles and return the list of TileTask and GridSpec."""
        pass

    def patch_decode_func(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile into RGB space."""
        pass

    def patch_merge_func(
        self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec
    ) -> torch.Tensor:
        """Merge decoded patches into a full image."""
        pass

    def gather_tensors(self, tensor: torch.Tensor, group: dist.ProcessGroup):
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)] if rank == 0 else None
        dist.gather(tensor, gather_list=gather_list, dst=0, group=group)
        return gather_list

    def broadcast_tensor(self, tensor: torch.Tensor, group: dist.ProcessGroup):
        dist.broadcast(tensor, src=0, group=group)
        return tensor

    def distributed_decode_flow(self, z: torch.Tensor, split_func, decode_func, merge_func):
        world_size = dist.get_world_size(self.group)
        rank = dist.get_rank(self.group)
        pp_size = min(self.vae_patch_parallel_size, world_size)

        # 1. Split into tiles
        tiletask_list, grid_spec = split_func(z)
        tid_coord_map = {task.tile_id: task.grid_coord for task in tiletask_list}

        # 2. local decode
        local_tasks = [task for task in tiletask_list if (task.tile_id + 1) % pp_size == rank]
        local_decoded = [(t.tile_id, decode_func(t)) for t in local_tasks]

        # 3. compute max_shape per rank
        local_max_shape = [0] * z.ndim
        for _, t_tensor in local_decoded:
            for i, s in enumerate(t_tensor.shape):
                local_max_shape[i] = max(local_max_shape[i], s)
        local_max_shape_tensor = torch.tensor([len(local_decoded), *local_max_shape], device=z.device)
        # 4. gather max_shapes
        max_shapes_gather = self.gather_tensors(local_max_shape_tensor, self.group)
        if rank == 0:
            global_max_shape = [0] * len(local_max_shape_tensor)
            for m in max_shapes_gather:
                for i, s in enumerate(m):
                    global_max_shape[i] = max(global_max_shape[i], int(s.item()))
        else:
            global_max_shape = [0] * len(local_max_shape_tensor)
        global_max_shape_tensor = torch.tensor(global_max_shape, device=z.device)
        self.broadcast_tensor(global_max_shape_tensor, self.group)

        # 5. prepare tile tensors
        tile_tensor = torch.zeros(global_max_shape_tensor.tolist(), device=z.device, dtype=z.dtype)
        meta_tensor = torch.full(
            (global_max_shape_tensor[0], len(grid_spec.split_dims) + 1), -1, device=z.device, dtype=torch.int64
        )
        for idx, (tid, t_tensor) in enumerate(local_decoded):
            meta_tensor[idx, 0] = tid
            for i, dim in enumerate(grid_spec.split_dims):
                meta_tensor[idx, i + 1] = t_tensor.shape[dim]
            slices = tuple(slice(0, s) for s in t_tensor.shape)
            tile_tensor[idx][slices] = t_tensor

        # 6. gather tiles & meta
        meta_gather = self.gather_tensors(meta_tensor, self.group)
        tile_gather = self.gather_tensors(tile_tensor, self.group)

        # 7. reconstruct full tensor (rank 0)
        if rank == 0:
            coord_tensor_map = {}
            for r in range(world_size):
                meta_src = meta_gather[r]
                tiles_src = tile_gather[r]
                for idx in range(meta_src.shape[0]):
                    tid = int(meta_src[idx, 0])
                    if tid < 0:
                        continue
                    slices = [slice(None)] * tiles_src[idx].ndim
                    for i, dim in enumerate(grid_spec.split_dims):
                        slices[dim] = slice(0, int(meta_src[idx, i + 1]))
                    slices = tuple(slices)
                    coord_tensor_map[tid_coord_map[tid]] = tiles_src[idx][slices]

            decoded_full = merge_func(coord_tensor_map, grid_spec)
        else:
            return torch.empty(0, device=z.device)  # Dummy return for non-zero ranks

        return decoded_full

    def _choose_decode_strategy(
        self,
        z: torch.Tensor,
    ) -> str:
        if self.vae_patch_parallel_size <= 1 or not dist.is_initialized():
            return "orig"

        if not getattr(self, "use_tiling", False):
            return "orig"

        try:
            self.group = get_dit_group()
        except Exception:
            return "orig"

        world_size = dist.get_world_size(group=self.group)
        pp_size = min(int(self.vae_patch_parallel_size), int(world_size))
        if pp_size <= 1:
            return "orig"

        tile_latent_min_size = getattr(self, "tile_latent_min_size", None)
        if tile_latent_min_size is None:
            return "tile"
        else:
            should_tile = (z.shape[-1] > tile_latent_min_size) or (z.shape[-2] > tile_latent_min_size)
            return "tile" if should_tile else "patch"

    def maybe_decode_with_distribute(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        strategy = self._choose_decode_strategy(z)
        if strategy == "tile" and not self.support_tile_parallel:
            strategy = "orig"
        elif strategy == "patch" and not self.support_patch_parallel:
            strategy = "orig"

        if strategy == "tile":
            logger.info("Using distributed tile parallel VAE decoding.")
            return self.distributed_decode_flow(
                z,
                split_func=self.tile_split_func,
                decode_func=self.tile_decode_func,
                merge_func=self.tile_merge_func,
            )
        elif strategy == "patch":
            logger.info("Using distributed patch parallel VAE decoding.")
            return self.distributed_decode_flow(
                z,
                split_func=self.patch_split_func,
                decode_func=self.patch_decode_func,
                merge_func=self.patch_merge_func,
            )
        else:
            return None
