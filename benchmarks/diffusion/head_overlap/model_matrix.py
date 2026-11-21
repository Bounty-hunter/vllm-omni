# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configured, interleaved Omni.generate comparisons; hashing is outside timing."""

import argparse
import hashlib
import json
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
    if not result:
        raise AssertionError("The request returned no image/video output")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--save-outputs", action="store_true", help="Save the first measured output for each variant")
    parser.add_argument("--profile", action="store_true", help="Mark a separate diagnostic run for Nsight capture")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    engine = dict(config["engine"], model=args.model)
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
            variants = config["variants"]
            runs = [("warm", name, index, 42) for name in variants for index in range(args.warmups)]
            for index in range(args.rounds):
                names = list(variants)
                offset = index % len(names)
                names = names[offset:] + names[:offset]
                if index % 2:
                    names.reverse()
                runs += [("paired", name, index, 42) for name in names]
            runs += [("correctness", name, seed - 43, seed) for seed in (43, 44) for name in variants]
            paired_positions = [position for position, run in enumerate(runs) if run[0] == "paired"]
            for position, (phase, name, index, seed) in enumerate(runs):
                extra = dict(
                    variants[name],
                    case=case["id"],
                    variant=name,
                    label=f"{case['id']}-{phase}-{name}-i{index}",
                    audit=index == 0,
                    trace=args.profile,
                    trace_start=args.profile and case is config["cases"][0] and position == paired_positions[0],
                    trace_stop=args.profile and case is config["cases"][-1] and position == paired_positions[-1],
                )
                params = OmniDiffusionSamplingParams(**case["sampling"], seed=seed, extra_args=extra)
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
                if signature != expected[seed]:
                    np.savez_compressed(args.out / f"{case['id']}-{phase}-{name}-seed{seed}-mismatch.npz", *values)
                    raise AssertionError(f"Output differs from serial reference: {case['id']}/{name}/seed{seed}")
                if not all(np.isfinite(value).all() for value in values):
                    raise AssertionError("Non-finite output")
                row = dict(
                    case=case["id"],
                    variant=name,
                    phase=phase,
                    index=index,
                    seed=seed,
                    profile=args.profile,
                    e2e_ms=elapsed_ms,
                    shapes=shapes,
                    dtypes=dtypes,
                    hashes=hashes,
                    stage_durations=[output.stage_durations for output in outputs],
                    peak_memory_mb=[output.peak_memory_mb for output in outputs],
                )
                records.append(row)
                print("MATRIX_RESULT " + json.dumps(row, default=str), flush=True)
                save()
                if args.save_outputs and phase == "paired" and index == 0:
                    np.savez_compressed(args.out / f"{case['id']}-{name}.npz", *values)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
