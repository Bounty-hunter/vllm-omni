# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Correlate each warmed iteration's launches with its device work."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def length(intervals):
    return sum(end - start for start, end in union(intervals))


def intersection(a, b):
    return length((max(x, u), min(y, v)) for x, y in union(a) for u, v in union(b) if max(x, u) < min(y, v))


def analyze(path):
    events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    calls = {
        e.get("args", {}).get("correlation"): e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("ph") == "X"
    }
    device = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    markers = [e for e in events if e.get("cat") == "user_annotation" and "/iteration" in e["name"]]
    python_events = [e for e in events if e.get("cat") == "python_function"]
    stack_events = [e for e in events if e.get("cat") == "cpu_op" and e.get("args", {}).get("Call stack")]
    rows = []
    for marker in markers:
        begin, end = marker["ts"], marker["ts"] + marker["dur"]
        selected = []
        for event in device:
            call = calls.get(event.get("args", {}).get("correlation"))
            if call is not None and begin <= call["ts"] < end:
                selected.append((event, call))
        if not selected:
            raise ValueError(f"No correlated device events for {path}: {marker['name']}")
        t0 = min(e["ts"] for e, _ in selected)
        t1 = max(e["ts"] + e["dur"] for e, _ in selected)
        intervals = [(e["ts"], e["ts"] + e["dur"]) for e, _ in selected]
        attention = [(e["ts"], e["ts"] + e["dur"]) for e, _ in selected if "FlashAttn" in e["name"]]
        p2p = [
            (e["ts"], e["ts"] + e["dur"]) for e, _ in selected if e.get("cat") == "gpu_memcpy" and "PtoP" in e["name"]
        ]
        aggregate = collections.defaultdict(lambda: dict(count=0, sum_us=0))
        timeline = []
        for event, call in sorted(selected, key=lambda pair: pair[0]["ts"]):
            name = event["name"]
            aggregate[name]["count"] += 1
            aggregate[name]["sum_us"] += event["dur"]
            ancestors = [
                p
                for p in python_events
                if p.get("tid") == call.get("tid") and p["ts"] <= call["ts"] and p["ts"] + p["dur"] >= call["ts"]
            ]
            ancestors.sort(key=lambda p: p["dur"], reverse=True)
            stack_ancestors = [
                p
                for p in stack_events
                if p.get("tid") == call.get("tid") and p["ts"] <= call["ts"] and p["ts"] + p["dur"] >= call["ts"]
            ]
            stack_ancestors.sort(key=lambda p: p["dur"])
            timeline.append(
                dict(
                    name=name,
                    start_us=event["ts"] - t0,
                    duration_us=event["dur"],
                    absolute_ts=event["ts"],
                    stream=event.get("args", {}).get("stream"),
                    launch_start_us=call["ts"] - t0,
                    launch_duration_us=call["dur"],
                    launch_lead_us=event["ts"] - call["ts"] - call["dur"],
                    cpu_stack=[p["name"] for p in ancestors[-12:]],
                    operator_stack=(stack_ancestors[0]["args"]["Call stack"] if stack_ancestors else None),
                )
            )
        busy = length(intervals)
        rows.append(
            dict(
                marker=marker["name"],
                window="device-event span",
                event_categories=["kernel", "gpu_memcpy", "gpu_memset"],
                span_us=t1 - t0,
                busy_union_us=busy,
                idle_us=t1 - t0 - busy,
                idle_ratio=(t1 - t0 - busy) / (t1 - t0),
                attention_sum_us=sum(y - x for x, y in attention),
                outgoing_p2p_union_us=length(p2p),
                outgoing_p2p_attention_intersection_us=intersection(attention, p2p),
                gpu_event_count=len(selected),
                aggregates=dict(aggregate),
                timeline=timeline,
            )
        )
    return dict(
        trace=path.name,
        markers=rows,
        python_stack_event_count=len(python_events),
        cpu_operator_stack_count=len(stack_events),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    root = args.directory
    summaries = [analyze(path) for path in sorted(root.glob("*.trace.json"))]
    (root / "trace-analysis.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    for summary in summaries:
        for row in summary["markers"]:
            print(
                summary["trace"],
                row["marker"].split("/")[-1],
                "span",
                round(row["span_us"], 2),
                "attention",
                round(row["attention_sum_us"], 2),
                "p2p",
                round(row["outgoing_p2p_union_us"], 2),
                "intersection",
                round(row["outgoing_p2p_attention_intersection_us"], 2),
                "events",
                row["gpu_event_count"],
                "stacks",
                summary["python_stack_event_count"],
            )
    result_files = sorted(root.glob("results-rank*.json"))
    if result_files:
        ranks = [json.loads(path.read_text()) for path in result_files]
        combined = []
        for index, row in enumerate(ranks[0]["rows"]):
            peer_rows = [rank["rows"][index] for rank in ranks]
            assert all(
                (r["case"], r["mode"], r["variant"]) == (row["case"], row["mode"], row["variant"]) for r in peer_rows
            )
            samples = [max(pair) for pair in zip(*(r["samples_ms"] for r in peer_rows))]
            wall_samples = [max(pair) for pair in zip(*(r["wall_samples_ms"] for r in peer_rows))]
            combined.append(
                dict(
                    case=row["case"],
                    mode=row["mode"],
                    variant=row["variant"],
                    rank_max_event_samples_ms=samples,
                    median_ms=statistics.median(samples),
                    min_ms=min(samples),
                    max_ms=max(samples),
                    rank_max_wall_samples_ms=wall_samples,
                    wall_median_ms=statistics.median(wall_samples),
                )
            )
        (root / "timings.json").write_text(json.dumps(combined, indent=2))
        for row in combined:
            print(
                row["case"],
                row["mode"],
                row["variant"],
                round(row["median_ms"], 5),
                "range",
                round(row["min_ms"], 5),
                round(row["max_ms"], 5),
            )


if __name__ == "__main__":
    main()
