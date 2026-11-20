# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep every event category and stack for one complete request per rank."""

import argparse
import concurrent.futures
import gzip
import json
from pathlib import Path


def convert(path):
    root = path.parent.parent
    with path.open(encoding="utf-8") as source:
        raw = json.load(source)
    metadata = json.loads(path.with_name(path.name.replace(".trace.json", ".metadata.json")).read_text())
    variant, rank = metadata["variant"], metadata["rank"]
    marker = next(
        e
        for e in raw["traceEvents"]
        if e.get("cat") == "user_annotation" and e.get("name") == f"HCMS/{variant}/request0"
    )
    begin, end = marker["ts"], marker["ts"] + marker["dur"]
    output = root / "with-stacks" / f"H3-4096-{variant}-request0-rank{rank}-full6steps.trace.json.gz"
    header = {k: v for k, v in raw.items() if k != "traceEvents"}
    header["traceName"] = str(output)
    header["hcms_scope"] = dict(
        kind="complete warmed worker request",
        source_trace=str(path),
        request_index=0,
        all_event_categories_and_stacks_retained=True,
        original_timestamps_preserved=True,
        ancestor_events_clipped_to_request=True,
    )
    counts = dict(events=0, gpu_events=0, python_stack_events=0, layers=0)
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=3) as target:
        target.write(json.dumps(header, separators=(",", ":"))[:-1] + ',"traceEvents":[')
        first = True
        for event in raw["traceEvents"]:
            phase = event.get("ph")
            if phase == "M":
                selected = event
            elif "ts" not in event:
                continue
            elif phase == "X":
                finish = event["ts"] + event.get("dur", 0)
                if event["ts"] >= end or finish <= begin:
                    continue
                selected = event.copy()
                selected["ts"] = max(begin, event["ts"])
                selected["dur"] = min(end, finish) - selected["ts"]
            elif begin <= event["ts"] < end:
                selected = event
            else:
                continue
            if not first:
                target.write(",")
            target.write(json.dumps(selected, ensure_ascii=False, separators=(",", ":")))
            first = False
            counts["events"] += 1
            counts["gpu_events"] += selected.get("cat") in (
                "kernel",
                "gpu_memcpy",
                "gpu_memset",
            )
            counts["python_stack_events"] += selected.get("cat") == "python_function"
            counts["layers"] += selected.get("cat") == "user_annotation" and selected.get("name", "").startswith(
                "HCMS/layer/"
            )
        target.write("]}")
    assert counts["layers"] == 300 and counts["python_stack_events"] > 0
    return dict(
        file=output.name,
        variant=variant,
        rank=rank,
        bytes=output.stat().st_size,
        **counts,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    (args.root / "with-stacks").mkdir(exist_ok=True)
    paths = sorted((args.root / "raw").glob("*.trace.json"))
    assert len(paths) == 4
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(convert, paths):
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.root / "with-stacks-validation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
