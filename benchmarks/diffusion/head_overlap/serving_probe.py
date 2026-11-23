# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in bridge from the existing head learner to native serving CI.

The native benchmark sends its unchanged requests. Both arms use the same
real-prefix graph owner; chunks=1 keeps original Ulysses, chunks=2 enables
the bounded three-then-two learner. No policy forwards or traces are added.
"""

import functools


def request_controls(sampling, *, enabled):
    """Describe a fixed request layout using host metadata only."""
    geometry = tuple(
        getattr(sampling, field, None)
        for field in (
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "true_cfg_scale",
            "num_outputs_per_prompt",
        )
    )
    case = "native-serving/" + repr(geometry)
    return dict(
        case=case,
        label=case,
        variant="overlap" if enabled else "baseline",
        graph_mode="graph",
        schedule="stage-local-shared-slots",
        chunks=1,
        real_prefix=True,
        real_prefix_calls=2,
        capture_without_probes=True,
        autotune=enabled,
        candidate_family="three-then-two" if enabled else None,
        reference_whole=not enabled,
    )


def install(pipeline):
    """Install only through the explicit serving benchmark integration patch."""
    from model_probe import install as install_model_probe

    chunks = pipeline.od_config.parallel_config.ulysses_head_chunks
    if chunks not in (1, 2):
        raise ValueError("The serving CI experiment toggles ulysses_head_chunks=1 or 2")
    install_model_probe(pipeline)
    original = pipeline.forward
    observed = {}

    @functools.wraps(original)
    def forward(request):
        if request.is_dummy_run():
            return original(request)
        sampling = request.sampling_params
        previous = sampling.extra_args
        controls = request_controls(sampling, enabled=chunks == 2)
        case = controls["case"]
        count = observed.get(case, 0)
        # The two native CI warmups prove capture/decisions and replay. Do not
        # serialize audit reports in the measured ten-request benchmark.
        controls["audit"] = count < 2
        controls["label"] += f"/request{count}"
        sampling.extra_args = {**(previous or {}), **controls}
        try:
            result = original(request)
            observed[case] = min(count + 1, 2)
            return result
        finally:
            sampling.extra_args = previous

    pipeline.forward = forward
