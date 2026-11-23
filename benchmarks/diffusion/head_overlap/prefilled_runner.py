# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reuse the exact phase-1 workload and change only the direct schedule."""

import functools
import hashlib
import json
import os
from pathlib import Path

import overlap_probe
from prefilled_probe import PrefilledPipeline

overlap_probe.DirectPipeline = PrefilledPipeline
overlap_probe.head_module.head_overlap = functools.partial(
    overlap_probe.head_module.head_overlap, prefetch_before_attention=False
)
root = Path(os.environ["PREFILLED_RUN_DIR"])
root.mkdir(parents=True, exist_ok=True)
sources = [
    Path(__file__),
    Path(overlap_probe.__file__),
    Path(__file__).with_name("prefilled_probe.py"),
    Path(overlap_probe.head_module.__file__),
]
(root / f"composition-rank{os.environ['LOCAL_RANK']}.json").write_text(
    json.dumps(
        dict(
            schedule=(
                "local-input-prefill, attention-before-next-peer-input, "
                "next-attention-before-previous-peer-output, fused-reuse-fence"
            ),
            force_default_prefetch=False,
            hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        ),
        indent=2,
    )
)
overlap_probe.main()
