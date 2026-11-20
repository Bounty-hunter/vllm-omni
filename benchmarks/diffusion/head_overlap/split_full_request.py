# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Export full requests and four-step diagnostic windows from untouched traces."""

import argparse
import json
from pathlib import Path

DEVICE = {"kernel", "gpu_memcpy", "gpu_memset"}
RUNTIME = {"cuda_runtime", "cuda_driver"}


def stop(event):
    return event["ts"] + event.get("dur", 0)


def merged_length(intervals):
    total, end = 0, float("-inf")
    for start, finish in sorted(intervals):
        total += max(0, finish - max(start, end))
        end = max(end, finish)
    return total


def export_scope(raw, destination, marker, steps, first_four=False):
    events = raw["traceEvents"]
    begin = steps[0]["ts"] if first_four else marker["ts"]
    cpu_end = stop(steps[3]) if first_four else stop(marker)
    calls = {e.get("args", {}).get("correlation"): e for e in events if e.get("cat") in RUNTIME and e.get("ph") == "X"}
    selected_device = [
        e
        for e in events
        if e.get("cat") in DEVICE
        and (
            begin <= calls.get(e.get("args", {}).get("correlation"), {}).get("ts", float("-inf")) < cpu_end
            if first_four
            else begin <= e["ts"] < cpu_end
        )
    ]
    assert selected_device, destination
    device_ids = {id(e) for e in selected_device}
    selected_correlations = {e.get("args", {}).get("correlation") for e in selected_device}
    end = max(cpu_end, max(stop(e) for e in selected_device))
    selected = []
    for event in events:
        phase = event.get("ph")
        if event.get("cat") == "python_function":
            continue
        if phase == "M":
            selected.append(event)
            continue
        if event.get("cat") in DEVICE:
            if id(event) in device_ids:
                selected.append(event)
            continue
        if phase in ("s", "f"):
            if begin <= event.get("ts", -1) < end and (not first_four or event.get("id") in selected_correlations):
                selected.append(event)
            continue
        if "ts" not in event:
            continue
        if phase == "X":
            if event["ts"] < cpu_end and stop(event) > begin:
                clipped = event.copy()
                clipped["ts"] = max(begin, event["ts"])
                clipped["dur"] = min(cpu_end, stop(event)) - clipped["ts"]
                if "Call stack" in clipped.get("args", {}):
                    clipped["args"] = {k: v for k, v in clipped["args"].items() if k != "Call stack"}
                selected.append(clipped)
        elif begin <= event["ts"] < cpu_end:
            selected.append(event)
    out = {k: v for k, v in raw.items() if k != "traceEvents"}
    out["traceName"] = str(destination)
    out["traceEvents"] = selected
    out["hcms_scope"] = dict(
        kind="first four denoise steps; not a complete request" if first_four else "complete warmed worker request",
        source_trace=raw.get("traceName"),
        original_timestamps_preserved=True,
        cpu_window_us=[begin, cpu_end],
        device_window_end_us=end,
        ancestor_events_clipped_to_cpu_window=True,
        view=(
            "Compact: all device events, CPU operators, runtime and annotations. "
            "Python function events and inline stack strings omitted; pristine sessions retain them."
        ),
    )
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(out, stream, ensure_ascii=False, separators=(",", ":"))
    layer_events = [
        e for e in selected if e.get("cat") == "user_annotation" and e.get("name", "").startswith("HCMS/layer/")
    ]
    expected_steps = 4 if first_four else 6
    assert len(layer_events) == 50 * expected_steps, (destination, len(layer_events))
    phases = {e.get("name") for e in selected if e.get("cat") == "user_annotation"}
    if not first_four:
        assert {
            "H3/_encode_text",
            "H3/_denoise",
            "H3/vae_decode",
            "prepare_output_for_transport",
        } <= phases, phases
    gpu_start = min(e["ts"] for e in selected_device)
    gpu_end = max(stop(e) for e in selected_device)
    busy = merged_length((e["ts"], stop(e)) for e in selected_device)
    row = dict(
        file=destination.name,
        scope=out["hcms_scope"],
        steps=expected_steps,
        layer_attention_calls=len(layer_events),
        distinct_layers=len({e["name"] for e in layer_events}),
        gpu_events=len(selected_device),
        python_stack_events=sum(e.get("cat") == "python_function" for e in selected),
        source_python_stack_events=sum(
            e.get("cat") == "python_function" and e["ts"] < cpu_end and stop(e) > begin for e in events
        ),
        phases=sorted(phases),
        window="device-event span",
        device_span_us=gpu_end - gpu_start,
        device_busy_union_us=busy,
        device_idle_us=gpu_end - gpu_start - busy,
    )
    assert row["source_python_stack_events"] > 0
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--variant", choices=("nonoverlap", "overlap"))
    args = parser.parse_args()
    root = args.root
    exported = root / "traces"
    exported.mkdir(exist_ok=True)
    rows = []
    pattern = f"*-{args.variant}-two-requests-rank*.trace.json" if args.variant else "*.trace.json"
    for path in sorted((root / "raw").glob(pattern)):
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        metadata = json.loads(
            path.with_name(path.name.replace(".trace.json", ".metadata.json")).read_text(encoding="utf-8")
        )
        assert metadata["success"]
        variant, rank = metadata["variant"], metadata["rank"]
        markers = sorted(
            (
                e
                for e in raw["traceEvents"]
                if e.get("cat") == "user_annotation" and e.get("name", "").startswith(f"HCMS/{variant}/request")
            ),
            key=lambda e: e["ts"],
        )
        assert len(markers) == 2, path
        for index, marker in enumerate(markers):
            steps = sorted(
                (
                    e
                    for e in raw["traceEvents"]
                    if e.get("cat") == "user_annotation"
                    and e.get("name", "").startswith("H3/denoise/step")
                    and marker["ts"] <= e["ts"] < stop(marker)
                ),
                key=lambda e: e["ts"],
            )
            assert [e["name"] for e in steps] == [f"H3/denoise/step{i}" for i in range(6)]
            stem = f"H3-4096-{variant}-request{index}-rank{rank}"
            for first_four in (False, True) if index == 0 else (False,):
                suffix = "first4steps" if first_four else "full6steps"
                row = export_scope(
                    raw,
                    exported / f"{stem}-{suffix}.trace.json",
                    marker,
                    steps,
                    first_four,
                )
                row.update(variant=variant, rank=rank, request_index=index)
                rows.append(row)
                print(json.dumps(row), flush=True)
        del raw
    assert len(rows) == (6 if args.variant else 12), len(rows)
    validation_name = f"trace-validation-{args.variant}.json" if args.variant else "trace-validation.json"
    (root / validation_name).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
