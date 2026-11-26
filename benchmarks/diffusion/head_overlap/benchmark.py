# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configured, interleaved Omni.generate comparisons; hashing is outside timing."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def arrays(outputs):
    result = []
    for output in outputs:
        for value in output.images or []:
            if isinstance(value, list):
                result.extend(np.asarray(frame) for frame in value)
            else:
                result.append(np.asarray(value))
    if not result or any(not frame.size for frame in result):
        raise AssertionError("Empty image/video output")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--case", required=True, help="Case ID in the workload JSON")
    parser.add_argument(
        "--shared-whole", action="store_true", help="Also measure unsplit attention with shared output storage"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["cases"] = [case for case in config["cases"] if case["id"] == args.case]
    if len(config["cases"]) != 1 or args.rounds < 1:
        parser.error("Choose one case from the workload and a positive number of rounds")
    if os.environ.get("VLLM_OMNI_PACKED_QKV") != "1":
        parser.error("Set VLLM_OMNI_PACKED_QKV=1 for the native baseline")
    os.environ["HCMS_MATRIX_AUDIT"] = str((args.out / "audit").resolve())
    for case in config["cases"]:
        # Reject online-only request fields before expensive model loading.
        OmniDiffusionSamplingParams(**case["sampling"])
    args.out.mkdir(parents=True, exist_ok=True)
    engine = dict(
        config["engine"],
        model=args.model,
        worker_extension_cls="benchmarks.diffusion.head_overlap.model_adapter.HeadOverlapWorker",
    )
    records = []
    destination = args.out / "results.json"

    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(engine=engine, config=config, records=records), indent=2, default=str))
        temporary.replace(destination)

    omni = Omni(**engine)
    try:
        for case in config["cases"]:
            expected = {}
            prompt = dict(config["prompt"])
            if "prompt" in case:
                prompt.update(case["prompt"])
            if "media" in case:
                media = {}
                for key, paths in case["media"].items():
                    if isinstance(paths, str):
                        media[key] = Image.open(paths).convert("RGB")
                    else:
                        media[key] = [Image.open(path).convert("RGB") for path in paths]
                prompt["multi_modal_data"] = media
            variants = ["auto", "shared_whole", "native"] if args.shared_whole else ["auto", "native"]
            runs = [("warm", name, 0, 42) for name in variants]
            for index in range(args.rounds):
                names = list(variants)
                offset = index % len(names)
                names = names[offset:] + names[:offset]
                # Reverse after a complete rotation cycle. Reversing every
                # other round pins even-sized variant sets to half the slots.
                if (index // len(names)) % 2:
                    names.reverse()
                runs += [("paired", name, index, 42) for name in names]
            runs += [("correctness", name, seed - 43, seed) for seed in (43, 44) for name in variants]
            for phase, name, index, seed in runs:
                extra = dict(
                    case=case["id"], variant=name, label=f"{case['id']}-{phase}-{name}-i{index}", audit=index == 0
                )
                sampling = dict(case["sampling"])
                extra = dict(sampling.pop("extra_args", {}), **extra)
                params = OmniDiffusionSamplingParams(**sampling, seed=seed, extra_args=extra)
                started = time.perf_counter()
                outputs = omni.generate(prompt, sampling_params_list=[params])
                elapsed_ms = (time.perf_counter() - started) * 1000
                values = arrays(outputs)
                hashes = [hashlib.sha256(value.tobytes()).hexdigest() for value in values]
                shapes = [list(value.shape) for value in values]
                dtypes = [str(value.dtype) for value in values]
                signature = (hashes, shapes, dtypes)
                if name == next(iter(variants)) and seed not in expected:
                    expected[seed] = signature
                exact_matches_reference = signature == expected[seed]
                if not exact_matches_reference:
                    np.savez_compressed(args.out / f"{case['id']}-{phase}-{name}-seed{seed}-mismatch.npz", *values)
                    raise AssertionError(f"Output mismatch for {case['id']}/{name}/seed{seed}")
                if not all(np.isfinite(value).all() for value in values):
                    raise AssertionError("Non-finite output")
                row = dict(
                    case=case["id"],
                    variant=name,
                    phase=phase,
                    index=index,
                    seed=seed,
                    e2e_ms=elapsed_ms,
                    shapes=shapes,
                    dtypes=dtypes,
                    hashes=hashes,
                    exact_matches_reference=exact_matches_reference,
                    stage_durations=[output.stage_durations for output in outputs],
                    peak_memory_mb=[output.peak_memory_mb for output in outputs],
                )
                records.append(row)
                print("MATRIX_RESULT " + json.dumps(row, default=str), flush=True)
                save()
    finally:
        omni.close()


if __name__ == "__main__":
    main()
