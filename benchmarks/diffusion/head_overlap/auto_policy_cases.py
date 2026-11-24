# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Input layouts for automatic selection; no expected head partition."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CoreCase:
    name: str
    tokens: int
    heads: int
    dim: int = 128
    joint_q: int = 0
    joint_kv: int = 0
    fused_v: bool = False
    padding: int = 0
    piecewise: bool = False
    video_grid: tuple[int, int, int] | None = None
    provenance: str = "representative layout, random numerical inputs"


TYPICAL_CASES = (
    CoreCase("hunyuanimage3-1024-prefill", 4096, 32, joint_q=128, joint_kv=128, piecewise=True),
    CoreCase("hunyuanimage3-1024-decode", 4096, 32, joint_kv=128, piecewise=True),
    CoreCase("tiny-head-order", 76, 32, dim=32, joint_q=7, joint_kv=7),
    CoreCase("qwen512-decode", 1024, 32, joint_kv=256),
    CoreCase("qwen1024-decode-ref", 4096, 32, joint_kv=4352),
    CoreCase("qwen1536-decode-ref", 9216, 32, joint_kv=4352),
    CoreCase("qwen1536-prefill-joint", 9216, 32, joint_q=256, joint_kv=256),
    CoreCase("h3-short-packed", 4128, 56, fused_v=True, padding=16),
    CoreCase("h3-large-packed", 16416, 56, fused_v=True, padding=16),
    CoreCase(
        "h3-ci-video-derived",
        63232,
        56,
        fused_v=True,
        padding=16,
        provenance="planner-derived sequence, canonical packed-padding metadata; not an exact native CI observation",
    ),
    CoreCase(
        "hunyuan480p33-derived",
        14040,
        16,
        joint_q=256,
        joint_kv=256,
        provenance="derived 9*30*52 video grid and representative text",
    ),
)

WAN_CASES = (
    CoreCase(
        "wan22-ci-480x832-81f",
        32760,
        40,
        fused_v=True,
        video_grid=(21, 30, 52),
        provenance="observed native CI QKV/stride and video metadata; random numerical inputs",
    ),
    CoreCase(
        "wan22-ci-720x1280-121f",
        111600,
        40,
        fused_v=True,
        video_grid=(31, 45, 80),
        provenance="observed native CI QKV/stride and video metadata; random numerical inputs",
    ),
)
