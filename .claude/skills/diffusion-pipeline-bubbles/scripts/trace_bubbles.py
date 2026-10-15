#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize recorded per-device gaps; do not infer causes or speedup."""

import argparse
import collections
import gzip
import json
import math
import re
from pathlib import Path

DEFAULT_DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
ASCEND_DEVICE_CATEGORIES = {"npu_kernel", "npu_memcpy"}
ASCEND_COMPUTE_TYPES = {"AI_CORE", "AI_VECTOR_CORE", "AI_CPU", "MIX_AIC", "MIX_AIV", "DSA_SQE"}
ASCEND_COPY_TYPES = {"SDMA_SQE", "PCIE_DMA_SQE"}
ASCEND_CONTROL_TYPES = {
    "EVENT_WAIT", "EVENT_RECORD", "PROFILING_ENABLE", "PROFILING_DISABLE", "PLACE_HOLDER_SQE",
    # NPUGraph dispatch and cross-stream notifications. Its actual operators
    # are separately exported as AI_CORE/AI_VECTOR_CORE/etc. on graph streams.
    "MODEL_EXECUTE", "NOTIFY_WAIT", "NOTIFY_RECORD",
}
WAIT_NAME = re.compile(r"synchroniz|wait.?event|local_scalar_dense", re.I)
HOST_CATEGORIES = {"cpu_op", "user_annotation", "python_function", "cuda_runtime", "cuda_driver", "npu_runtime"}


def normalize_ascend(events):
    """Map torch_npu trace_view lanes; blank categories also contain host/derived rows."""
    process_names, process_labels, stream_names = {}, {}, {}
    for event in events:
        if event.get("ph") != "M":
            continue
        args = event.get("args") or {}
        if event.get("name") == "process_name":
            process_names[event.get("pid")] = args.get("name")
        elif event.get("name") == "process_labels":
            process_labels[event.get("pid")] = args.get("labels", "")
        elif event.get("name") == "thread_name":
            stream_names[event.get("pid"), event.get("tid")] = args.get("name", "")
    hardware = {pid for pid, name in process_names.items() if name == "Ascend Hardware"}
    if not hardware:
        raise ValueError("No Ascend Hardware metadata: cannot establish device activity")
    devices = {}
    for pid in hardware:
        match = re.fullmatch(r"NPU (\d+)", process_labels.get(pid, ""))
        if match is None:
            raise ValueError(f"Missing/unknown physical device label for hardware pid {pid}")
        devices[pid] = int(match[1])
    mapped, counts = [], collections.Counter()
    stream_types = collections.defaultdict(collections.Counter)
    for event in events:
        pid = event.get("pid")
        if event.get("ph") == "X" and pid in hardware:
            args = dict(event.get("args") or {})
            task_type = args.get("Task Type")
            counts[task_type] += 1
            if task_type in ASCEND_COMPUTE_TYPES:
                category = "npu_kernel"
            elif task_type in ASCEND_COPY_TYPES:
                category = "npu_memcpy"
            elif task_type in ASCEND_CONTROL_TYPES:
                category = "npu_control"
            else:
                raise ValueError(f"Unrecognized Ascend task type {task_type!r}; verify exporter before measuring")
            stream = args.get("Physic Stream Id")
            if stream is None or stream_names.get((pid, event.get("tid"))) != f"Stream {stream}":
                raise ValueError(f"Inconsistent physical stream metadata: pid={pid}, tid={event.get('tid')}")
            args.update(device=devices[pid], stream=stream)
            stream_types[f"device:{devices[pid]}/pid:{pid}/stream:{stream}"][task_type] += 1
            mapped.append({**event, "cat": category, "args": args})
        elif event.get("ph") == "X" and process_names.get(pid) == "CANN":
            mapped.append({**event, "cat": "npu_runtime"})
        else:
            mapped.append(event)
    return mapped, {
        "hardware_pid_to_device": devices,
        "task_type_counts": dict(counts),
        "stream_task_types": {stream: dict(types) for stream, types in sorted(stream_types.items())},
        "excluded_control_types": sorted(ASCEND_CONTROL_TYPES),
        "note": "EVENT_WAIT is dependency waiting, not useful device work; derived overlap lanes are excluded.",
    }


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def interval(event):
    try:
        start, duration = float(event["ts"]), float(event["dur"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
        return None
    return start, start + duration


def device_key(event):
    args = event.get("args") or {}
    for name in ("device", "Device Id", "device_id"):
        if args.get(name) is not None:
            return f"device:{args[name]}", False
    return f"lane_pid:{event.get('pid', 'unknown')}", True


def stream_key(event):
    args = event.get("args") or {}
    for name in ("stream", "Stream Id", "stream_id"):
        if args.get(name) is not None:
            return f"pid:{event.get('pid', 'unknown')}/stream:{args[name]}"
    return f"lane_pid:{event.get('pid', 'unknown')}/tid:{event.get('tid', 'unknown')}"


def occupancy(rows, window):
    start, end = window
    busy = union((max(a, start), min(b, end)) for a, b, _ in rows if b > start and a < end)
    gaps = []
    cursor = start
    for a, b in busy:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        gaps.append((cursor, end))
    busy_us = sum(b - a for a, b in busy)
    return {
        "window_us": [start, end],
        "window_ms": (end - start) / 1000,
        "recorded_activity_ms": busy_us / 1000,
        "no_recorded_activity_ms": (end - start - busy_us) / 1000,
        "no_recorded_activity_pct": 100 * (1 - busy_us / (end - start)) if end > start else None,
        "all_gap_count": len(gaps),
        "event_count_in_window": sum(b > start and a < end for a, b, _ in rows),
    }, gaps


def summarize(events, categories=None, window=None, min_gap_us=100.0, topn=20):
    categories = DEFAULT_DEVICE_CATEGORIES if categories is None else set(categories)
    devices = collections.defaultdict(list)
    host = []
    waits = collections.defaultdict(list)
    pid_fallback = False
    category_counts = collections.Counter()
    for event in events:
        cat = str(event.get("cat", ""))
        category_counts[cat] += 1
        span = interval(event)
        if span is None:
            continue
        if cat in categories:
            key, fallback = device_key(event)
            pid_fallback |= fallback
            devices[key].append((*span, event))
        elif cat in HOST_CATEGORIES:
            host.append((*span, event))
            if WAIT_NAME.search(str(event.get("name", ""))):
                if window is None:
                    clipped = span
                else:
                    clipped = (max(span[0], window[0]), min(span[1], window[1]))
                if clipped[1] > clipped[0]:
                    waits[(cat, str(event.get("name", "")))].append(clipped)

    warnings = [
        "Recorded activity only; absent processes/streams are not covered.",
        "Host overlaps and wait-like API names are clues, not causal attribution or savings.",
        "Stream identities/roles require exporter verification; per-stream gaps are not device idleness.",
    ]
    if pid_fallback:
        warnings.append("Some device IDs are missing: lane_pid grouping requires exporter verification.")
    if not devices:
        warnings.append("No recognized device duration events: no accelerator idle metric can be established.")
    if window is None:
        warnings.append("Per-device first-to-last-event windows omit leading/trailing request gaps.")
    reports = []
    for key, rows in sorted(devices.items()):
        rows.sort(key=lambda row: row[0])
        start, end = window if window is not None else (rows[0][0], max(row[1] for row in rows))
        metrics, gaps = occupancy(rows, (start, end))
        streams = collections.defaultdict(list)
        for row in rows:
            streams[stream_key(row[2])].append(row)
        stream_reports = []
        for stream, stream_rows in sorted(streams.items()):
            stream_metrics, _ = occupancy(stream_rows, (start, end))
            stream_reports.append({"stream": stream, **stream_metrics})
        selected = sorted((g for g in gaps if g[1] - g[0] >= min_gap_us), key=lambda g: g[1] - g[0], reverse=True)
        details = []
        for a, b in selected[:topn]:
            overlaps = sorted(
                ((x, y, e) for x, y, e in host if x < b and y > a),
                key=lambda r: (-(min(b, r[1]) - max(a, r[0])), r[1] - r[0]),
            )[:5]
            details.append(
                {
                    "start_us": a,
                    "end_us": b,
                    "duration_ms": (b - a) / 1000,
                    "overlapping_host_candidates": [
                        {
                            "name": e.get("name"),
                            "cat": e.get("cat"),
                            "pid": e.get("pid"),
                            "tid": e.get("tid"),
                            "start_us": x,
                            "duration_ms": (y - x) / 1000,
                        }
                        for x, y, e in overlaps
                    ],
                }
            )
        reports.append(
            {
                "device": key,
                **metrics,
                "streams": stream_reports,
                "selected_gap_count": len(selected),
                "selected_gap_total_ms": sum(b - a for a, b in selected) / 1000,
                "top_gaps": details,
            }
        )
    return {
        "timestamp_unit": "us",
        "duration_unit": "ms",
        "event_count": len(events),
        "category_counts": dict(category_counts),
        "device_categories": sorted(categories),
        "window_source": "explicit" if window is not None else "device_event_span",
        "warnings": warnings,
        "devices": reports,
        "host_wait_like_events": [
            {
                "cat": cat,
                "name": name,
                "count": len(spans),
                "duration_sum_ms": sum(b - a for a, b in spans) / 1000,
                "duration_union_ms": sum(b - a for a, b in union(spans)) / 1000,
                "max_ms": max(b - a for a, b in spans) / 1000,
            }
            for (cat, name), spans in sorted(waits.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exporter", choices=("chrome", "ascend"), default="chrome")
    parser.add_argument("--device-categories", help="Override activity categories after exporter normalization")
    parser.add_argument("--window-us", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--min-gap-ms", type=float, default=0.1)
    parser.add_argument("--topn", type=int, default=20)
    args = parser.parse_args()
    if not math.isfinite(args.min_gap_ms) or args.min_gap_ms < 0 or args.topn < 0:
        parser.error("min-gap-ms must be finite and nonnegative; topn must be nonnegative")
    if args.window_us and (not all(map(math.isfinite, args.window_us)) or args.window_us[1] <= args.window_us[0]):
        parser.error("window-us requires finite START < END")
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt", encoding="utf-8") as source:
        data = json.load(source)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    mapping = None
    categories = DEFAULT_DEVICE_CATEGORIES
    if args.exporter == "ascend":
        events, mapping = normalize_ascend(events)
        categories = ASCEND_DEVICE_CATEGORIES
    if args.device_categories is not None:
        categories = args.device_categories.split(",")
    report = summarize(events, categories, args.window_us, args.min_gap_ms * 1000, args.topn)
    report["exporter"] = args.exporter
    if mapping is not None:
        report["exporter_mapping"] = mapping
    report["source"] = str(args.trace.resolve())
    result = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
        print(f"Saved {len(report['devices'])} device summaries to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
