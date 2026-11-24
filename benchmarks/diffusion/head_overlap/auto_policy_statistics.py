# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Paired mean-latency acceptance; retain slow samples and uncertainty."""

import math
import random
import statistics


def paired_latency(pairs, *, limit=0.01, seed=10448, resamples=4000):
    if len(pairs) < 6:
        raise ValueError("At least six paired blocks are required")
    if not 0 <= limit < 1 or resamples < 100:
        raise ValueError("Invalid latency limit or resampling budget")
    if any(not math.isfinite(value) or value <= 0 for pair in pairs for value in pair):
        raise ValueError("Latency samples must be finite and positive")
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("Each block needs one native and one automatic latency")
    before, after = zip(*pairs)
    rng = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        indices = rng.choices(range(len(pairs)), k=len(pairs))
        ratios.append(sum(after[index] for index in indices) / sum(before[index] for index in indices))
    ratios.sort()
    lower = ratios[int(0.025 * resamples)]
    upper = ratios[min(resamples - 1, math.ceil(0.975 * resamples) - 1)]
    ratio = statistics.mean(after) / statistics.mean(before)
    return dict(
        native_mean_us=statistics.mean(before),
        automatic_mean_us=statistics.mean(after),
        latency_ratio=ratio,
        latency_increase_pct=100 * (ratio - 1),
        ratio_ci95=[lower, upper],
        limit_pct=100 * limit,
        status="pass" if upper <= 1 + limit else "regressed" if lower > 1 + limit else "inconclusive",
        blocks=len(pairs),
        resamples=resamples,
        pairs_us=pairs,
    )


def acceptance(pairs, native_controls, *, limit=0.01):
    result = paired_latency(pairs, limit=limit)
    control = paired_latency(native_controls, limit=limit)
    low, high = control["ratio_ci95"]
    control["resolved_within_limit"] = low >= 1 - limit and high <= 1 + limit
    result["latency_status"] = result["status"]
    result["native_aa_control"] = control
    # A noisy harness is not evidence of either successful fallback or a
    # selector regression. Do not rerun until one favorable sample passes.
    result["status"] = result["status"] if control["resolved_within_limit"] else "inconclusive"
    return result
