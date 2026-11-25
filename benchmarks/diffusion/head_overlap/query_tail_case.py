# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Apply the three early-return controls to one existing model-matrix workload."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path, help="Existing matrix with the model, inputs and sampling configuration")
    parser.add_argument("case", help="Case ID from the template")
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, required=True, help="Number of complete three-arm rounds")
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")

    config = json.loads(args.template.read_text())
    cases = [case for case in config["cases"] if case["id"] == args.case]
    if len(cases) != 1:
        parser.error(f"Expected one case named {args.case!r} in {args.template}")
    policy = Path(__file__).with_name("qwen21-turbo-ref1-query-tail.json")
    arms = json.loads(policy.read_text())["variants"]
    config["variants"] = {
        name: {**arms[name], "capture_without_probes": True} for name in ("query_tail", "shared_whole", "native")
    }
    cases[0].pop("variants", None)
    config["cases"] = cases
    config["measurement"] = dict(rounds=args.rounds, warmups=1, seeds=[42, 43, 44], new_profiler_traces=False)
    for field in ("adapter_note", "ci_extra_cases"):
        config.pop(field, None)
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
