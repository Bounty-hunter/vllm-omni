# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral checks for interval accounting, independent of model hardware."""

import unittest

from trace_bubbles import ASCEND_DEVICE_CATEGORIES, normalize_ascend, summarize


def event(start, duration, device=0, stream=0, cat="kernel", name="work"):
    return {
        "ts": start,
        "dur": duration,
        "cat": cat,
        "name": name,
        "pid": 10,
        "tid": stream,
        "args": {"device": device},
    }


class TraceTests(unittest.TestCase):
    def test_ascend_excludes_waits_and_host_rows_with_blank_category(self):
        metadata = [
            {"ph": "M", "pid": 20, "name": "process_name", "args": {"name": "Ascend Hardware"}},
            {"ph": "M", "pid": 20, "name": "process_labels", "args": {"labels": "NPU 6"}},
            {"ph": "M", "pid": 20, "tid": 47, "name": "thread_name", "args": {"name": "Stream 47"}},
        ]

        def task(start, duration, kind):
            return {
                "ph": "X",
                "pid": 20,
                "tid": 47,
                "ts": str(start),
                "dur": duration,
                "args": {"Task Type": kind, "Physic Stream Id": 47},
            }

        events = metadata + [
            task(0, 10, "AI_CORE"),
            task(10, 20, "EVENT_WAIT"),
            task(10, 20, "NOTIFY_WAIT"),
            task(10, 5, "MODEL_EXECUTE"),
            task(15, 5, "NOTIFY_RECORD"),
            task(30, 5, "SDMA_SQE"),
            {"ph": "X", "pid": 30, "tid": 0, "ts": 0, "dur": 40},
        ]
        mapped, _ = normalize_ascend(events)
        report = summarize(mapped, ASCEND_DEVICE_CATEGORIES, window=(0, 40))
        self.assertEqual(len(report["devices"]), 1)
        self.assertEqual(report["devices"][0]["device"], "device:6")
        self.assertEqual(report["devices"][0]["no_recorded_activity_pct"], 62.5)
        self.assertEqual(report["devices"][0]["streams"][0]["stream"], "pid:20/stream:47")
        events[-2]["args"]["Task Type"] = "UNVERIFIED_TASK"
        with self.assertRaisesRegex(ValueError, "Unrecognized Ascend"):
            normalize_ascend(events)

    def test_ascend_requires_hardware_metadata(self):
        with self.assertRaisesRegex(ValueError, "No Ascend Hardware"):
            normalize_ascend([event(0, 10, cat="")])

    def test_other_stream_fills_gap(self):
        result = summarize([event(0, 10), event(20, 10), event(5, 20, stream=1)])
        self.assertEqual(result["devices"][0]["no_recorded_activity_ms"], 0)

    def test_devices_remain_separate(self):
        result = summarize([event(0, 10), event(20, 10), event(10, 10, device=1)], window=(0, 30))
        self.assertEqual([d["no_recorded_activity_ms"] for d in result["devices"]], [0.01, 0.02])

    def test_stream_gaps_do_not_imply_device_gaps(self):
        result = summarize([event(0, 10, stream=1), event(10, 10, stream=2)], window=(0, 20))
        device = result["devices"][0]
        self.assertEqual(device["no_recorded_activity_pct"], 0)
        self.assertEqual([s["no_recorded_activity_pct"] for s in device["streams"]], [50, 50])

    def test_streams_use_common_request_window_and_all_gaps(self):
        result = summarize([event(10, 10, stream=1), event(30, 10, stream=2)], window=(0, 50), min_gap_us=1000)
        device = result["devices"][0]
        self.assertEqual(device["selected_gap_count"], 0)
        self.assertAlmostEqual(device["no_recorded_activity_pct"], 60)
        self.assertEqual([s["window_us"] for s in device["streams"]], [[0, 50], [0, 50]])
        self.assertEqual([s["no_recorded_activity_pct"] for s in device["streams"]], [80, 80])

    def test_explicit_stream_ids_override_tid_and_keep_processes_separate(self):
        a, b, c = event(0, 10), event(10, 10, stream=9), event(20, 10)
        for row in (a, b, c):
            row["args"]["stream"] = 0
        c["pid"] = 11
        result = summarize([a, b, c], window=(0, 30))
        streams = result["devices"][0]["streams"]
        self.assertEqual(len(streams), 2)
        self.assertEqual([s["recorded_activity_ms"] for s in streams], [0.02, 0.01])

    def test_host_sync_is_not_device_work_or_savings(self):
        result = summarize([event(0, 10), event(0, 20, cat="cuda_runtime", name="cudaStreamSynchronize")])
        self.assertEqual(result["devices"][0]["no_recorded_activity_ms"], 0)
        self.assertEqual(result["host_wait_like_events"][0]["duration_sum_ms"], 0.02)

    def test_window_includes_edges(self):
        result = summarize([event(10, 10)], window=(0, 30), min_gap_us=0)
        self.assertEqual(result["devices"][0]["all_gap_count"], 2)
        self.assertEqual(result["devices"][0]["no_recorded_activity_ms"], 0.02)

    def test_nested_events_are_unioned(self):
        result = summarize([event(0, 30), event(5, 10)])
        self.assertEqual(result["devices"][0]["recorded_activity_ms"], 0.03)

    def test_repeated_small_gaps_are_counted(self):
        result = summarize([event(i * 200, 100) for i in range(5)], min_gap_us=50, topn=1)
        device = result["devices"][0]
        self.assertEqual(device["selected_gap_count"], 4)
        self.assertEqual(device["selected_gap_total_ms"], 0.4)
        self.assertEqual(len(device["top_gaps"]), 1)

    def test_cpu_only_has_no_idle_metric(self):
        result = summarize([event(0, 10, cat="cpu_op")])
        self.assertEqual(result["devices"], [])

    def test_empty_and_clipped_windows(self):
        self.assertEqual(summarize([])["devices"], [])
        result = summarize([event(0, 30)], window=(5, 10))
        self.assertEqual(result["devices"][0]["recorded_activity_ms"], 0.005)


if __name__ == "__main__":
    unittest.main()
