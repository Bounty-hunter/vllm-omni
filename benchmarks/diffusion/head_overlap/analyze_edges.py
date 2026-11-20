# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Separate CE transfer, local copies, peer waits and launch gaps per layer."""

import argparse
import bisect
import collections
import json
import statistics
from pathlib import Path


def end(event):
    return event["ts"] + event.get("dur", 0)


def summarize(values):
    ordered = sorted(values)
    return dict(
        n=len(values),
        median=statistics.median(values),
        p10=ordered[int(0.1 * (len(values) - 1))],
        p90=ordered[int(0.9 * (len(values) - 1))],
        minimum=min(values),
        maximum=max(values),
    )


def analyze(path):
    trace = json.loads(path.read_text(encoding="utf-8"))
    events = trace["traceEvents"]
    calls = {
        e.get("args", {}).get("correlation"): e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("ph") == "X"
    }
    layers = sorted(
        (e for e in events if e.get("cat") == "user_annotation" and e.get("name", "").startswith("HCMS/layer/")),
        key=lambda e: e["ts"],
    )
    starts = [e["ts"] for e in layers]
    regions = sorted(
        (e for e in events if e.get("cat") == "user_annotation" and e.get("name", "").startswith("prefilled/")),
        key=lambda e: e["ts"],
    )
    rstarts = [e["ts"] for e in regions]
    device = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    streams = collections.defaultdict(list)
    assigned = [[] for _ in layers]
    for event in device:
        streams[event["args"]["stream"]].append(event)
        call = calls.get(event.get("args", {}).get("correlation"))
        if call is None:
            continue
        index = bisect.bisect_right(starts, call["ts"]) - 1
        if index < 0 or not layers[index]["ts"] <= call["ts"] < end(layers[index]):
            continue
        region_index = bisect.bisect_right(rstarts, call["ts"]) - 1
        phase = "output-clone"
        if region_index >= 0 and call["ts"] < end(regions[region_index]):
            phase = regions[region_index]["name"]
        assigned[index].append((event, call, phase))
    for stream in streams.values():
        stream.sort(key=lambda e: e["ts"])
    rows = []
    for index, (layer, items) in enumerate(zip(layers, assigned, strict=True)):
        if not items:
            raise ValueError(layer)
        items.sort(key=lambda triple: triple[0]["ts"])
        attention = [e for e, _, _ in items if "FlashAttn" in e["name"]]
        first_fa, last_fa = (
            min(attention, key=lambda e: e["ts"]),
            max(attention, key=end),
        )
        begin = min(e["ts"] for e, _, _ in items)
        finish = max(end(e) for e, _, _ in items)
        compute = streams[first_fa["args"]["stream"]]
        previous = max((e for e in compute if end(e) <= begin), key=end, default=None)
        following = min(
            (e for e in compute if e["ts"] >= finish),
            key=lambda e: e["ts"],
            default=None,
        )
        phases = collections.defaultdict(list)
        timeline = []
        for event, call, phase in items:
            phases[phase].append(event)
            timeline.append(
                dict(
                    name=event["name"],
                    phase=phase,
                    start_us=event["ts"] - begin,
                    duration_us=event["dur"],
                    absolute_ts_us=event["ts"],
                    stream=event["args"]["stream"],
                    bytes=event.get("args", {}).get("bytes"),
                    channel_type=event.get("args", {}).get("channel_type"),
                    correlation=event.get("args", {}).get("correlation"),
                    launch_start_us=call["ts"] - begin,
                    launch_duration_us=call["dur"],
                    launch_lead_us=event["ts"] - end(call),
                )
            )
        phase_stats = {}
        for phase, values in phases.items():
            phase_stats[phase] = dict(
                span_us=max(map(end, values)) - min(e["ts"] for e in values),
                sum_us=sum(e["dur"] for e in values),
                copy_sum_us=sum(e["dur"] for e in values if e["cat"] == "gpu_memcpy"),
                p2p_sum_us=sum(e["dur"] for e in values if "PtoP" in e["name"]),
                barrier_sum_us=sum(e["dur"] for e in values if "barrier_kernel" in e["name"]),
                bytes=sum(e.get("args", {}).get("bytes", 0) for e in values),
                absolute_start_us=min(e["ts"] for e in values),
                absolute_end_us=max(map(end, values)),
            )
        f0 = phase_stats["prefilled/F0/peer-only"]
        f0_barrier = next(e for e, _, p in items if p == "prefilled/F0/peer-only" and "barrier_kernel" in e["name"])
        final_fence = next(e for e, _, p in items if p == "prefilled/output-peer-ready")
        clone = phase_stats["output-clone"]
        reverse = phase_stats[f"prefilled/R{len(attention) - 1}/peer-only"]
        row = dict(
            layer=layer["name"].split("/")[-1],
            step=index // 50,
            pre_us=first_fa["ts"] - begin,
            post_us=finish - end(last_fa),
            pre_comm_us=f0["span_us"],
            post_comm_us=end(final_fence) - reverse["absolute_start_us"],
            last_reverse_p2p_us=reverse["p2p_sum_us"],
            pre_from_previous_compute_us=None if previous is None else first_fa["ts"] - end(previous),
            post_to_next_compute_us=None if following is None else following["ts"] - end(last_fa),
            attention_sum_us=sum(e["dur"] for e in attention),
            attention_count=len(attention),
            first_attention_absolute_us=first_fa["ts"],
            last_attention_end_absolute_us=end(last_fa),
            span_us=finish - begin,
            local_prefill_us=phase_stats["prefilled/local-inputs-before-attention"]["span_us"],
            f0_p2p_us=f0["p2p_sum_us"],
            f0_barrier_us=f0["barrier_sum_us"],
            ready_to_first_attention_us=first_fa["ts"] - end(f0_barrier),
            final_barrier_us=final_fence["dur"],
            clone_us=clone["sum_us"],
            phases=phase_stats,
            timeline=timeline,
        )
        rows.append(row)
    return dict(trace=path.name, base_time_nanoseconds=trace["baseTimeNanoseconds"], rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    outputs = []
    for path in sorted((args.root / "traces").glob("*first4steps.trace.json")):
        result = analyze(path)
        outputs.append(result)
        print(result["trace"], flush=True)
        keys = (
            "pre_us",
            "post_us",
            "pre_comm_us",
            "post_comm_us",
            "pre_from_previous_compute_us",
            "post_to_next_compute_us",
            "local_prefill_us",
            "f0_p2p_us",
            "f0_barrier_us",
            "ready_to_first_attention_us",
            "final_barrier_us",
            "clone_us",
        )
        for key in keys:
            values = [r[key] for r in result["rows"] if r[key] is not None]
            print(key, json.dumps(summarize(values)), flush=True)
        target_pre, target_post = (587, 308) if "nonoverlap" in path.name else (639, 334)
        closest = sorted(
            result["rows"],
            key=lambda r: abs(r["pre_comm_us"] - target_pre) + abs(r["post_comm_us"] - target_post),
        )[:3]
        print(
            "nearest",
            [(r["step"], r["layer"], r["pre_comm_us"], r["post_comm_us"]) for r in closest],
            flush=True,
        )
    (args.root / "edge-analysis.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
