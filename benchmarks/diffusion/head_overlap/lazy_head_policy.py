# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Learn from existing eager calls, including bounded real-request prefixes.

One call executes one complete core and returns its real output. Preparation
never executes a probe kernel. Event results are consumed only by finish(),
after the graph owner's existing synchronization. No timers survive freezing.
"""

import hashlib
import math
import os
import statistics
import time
from dataclasses import dataclass, field, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from importlib.metadata import version

from execution_counters import observe_core


def canonical_strides(value):
    strides = list(value.stride())
    # Batch-one attention never addresses another batch. Prefix-cache clone()
    # can preserve this unused pitch whereas graph input clones canonicalize it.
    # Keep all sequence/head strides, including fused V's threefold row pitch.
    if len(value.shape) == 4 and value.shape[0] == 1:
        strides[0] = 0
    return tuple(strides)


def tensor_spec(value):
    return tuple(value.shape), canonical_strides(value), str(value.dtype), value.device.type


def metadata_spec(value):
    """Describe host semantics and tensor layouts without reading device data."""
    if hasattr(value, "shape"):
        return tensor_spec(value), True
    if isinstance(value, (int, float, bool, str, type(None))):
        return value, True
    if isinstance(value, dict):
        rows = [(key, metadata_spec(item)) for key, item in sorted(value.items())]
        return tuple((key, item[0]) for key, item in rows), all(item[1] for _, item in rows)
    if isinstance(value, (tuple, list)):
        rows = [metadata_spec(item) for item in value]
        return tuple(item[0] for item in rows), all(item[1] for item in rows)
    if is_dataclass(value):
        rows = [(item.name, metadata_spec(getattr(value, item.name))) for item in dataclass_fields(value)]
        return (type(value).__qualname__, tuple((name, item[0]) for name, item in rows)), all(
            item[1] for _, item in rows
        )
    return ("opaque", type(value).__module__, type(value).__qualname__), False


def qualified_cycles(choices):
    """Pair complete rotations, discarding a cycle if any rank/choice starves.

    Filtering each choice independently would compare different layer
    neighborhoods and reintroduce position/frequency bias.
    """
    count = min(len(rows) for rows in choices)
    discard = 2 if count >= 10 else 1
    valid = [index for index in range(discard, count) if all(row[1] for rows in choices for row in rows[index])]
    return [[max(row[0] for row in rows[index]) for index in valid] for rows in choices]


def candidates(heads, tokens, dim, world, sms, *, flops=6e14, bandwidth=None, kv_tokens=None):
    """Rank CPU cost predictions; execute at most two splits, never a sweep.

    A 128-query-tile wave model is a candidate generator, not an observed grid.
    Odd widths remain eligible; real complete-core observations decide whether
    either prediction is good enough to replace whole attention.
    """
    whole = (heads,)
    if heads < 2:
        return (whole,)
    bandwidth = bandwidth or 650e9 / world
    tiles = math.ceil(tokens / 128)
    waves = math.ceil(tiles * heads / sms)
    attention = 4 * tokens * (tokens if kv_tokens is None else kv_tokens) * heads * dim / flops
    peer_bytes = 4 * tokens * heads * dim * 2 * (world - 1) / world
    peer = peer_bytes / bandwidth + 20e-6
    forward, reverse = peer * 0.75, peer * 0.25
    ranked = []
    for first in range(1, heads // 2 + 1):
        if first * tiles < 0.8 * sms:
            continue
        a0 = attention * math.ceil(first * tiles / sms) / waves
        a1 = attention * math.ceil((heads - first) * tiles / sms) / waves
        x = first / heads
        hidden = min(a0, (1 - x) * forward) + min(a1, x * reverse)
        cost = a0 + a1 - attention - hidden + 12e-6
        ranked.append((cost, first))
    ranked.sort()
    return (whole,) + tuple((h, heads - h) for _, h in ranked[:2])


def symmetric_candidates(heads):
    """Try small ends of one then two heads within the existing prefix budget."""
    return ((heads,),) + tuple((edge, heads - 2 * edge, edge) for edge in (1, 2) if heads - 2 * edge >= edge)


def three_then_two_candidates(heads, tokens, dim, world, sms, *, flops=6e14, bandwidth=None, kv_tokens=None):
    """Propose one symmetric triple and one double; never benchmark a sweep.

    This CPU wave/stream-order proxy ranks candidates, not observed launch
    grids. Real complete-core observations retain the final veto.
    """
    whole = (heads,)
    doubles = candidates(heads, tokens, dim, world, sms, flops=flops, bandwidth=bandwidth, kv_tokens=kv_tokens)[1:]
    triples = [(edge, heads - 2 * edge, edge) for edge in range(1, heads // 3 + 1)]
    if not triples:
        return (whole,) + doubles[:1]
    tiles = math.ceil(tokens / 128)
    waves = math.ceil(tiles * heads / sms)
    attention = 4 * tokens * (tokens if kv_tokens is None else kv_tokens) * heads * dim / flops
    peer_bytes = 4 * tokens * heads * dim * 2 * (world - 1) / world
    peer = peer_bytes / (bandwidth or 650e9 / world) + 20e-6
    forward, reverse = peer * 0.75, peer * 0.25

    def elapsed(parts):
        # Communication submits next F, then previous R. Each A waits for its
        # own input and the preceding compute; this mirrors stage-local order.
        comm, compute = parts[0] / heads * forward + 6e-6, 0.0
        ready, produced = [comm], []
        for index, width in enumerate(parts):
            compute = max(compute, ready[index]) + attention * math.ceil(tiles * width / sms) / waves + 4e-6
            produced.append(compute)
            if index + 1 < len(parts):
                comm += parts[index + 1] / heads * forward + 6e-6
                ready.append(comm)
            if index:
                comm = max(comm, produced[index - 1]) + parts[index - 1] / heads * reverse
        return max(comm, produced[-1]) + parts[-1] / heads * reverse

    triple = min(triples, key=lambda parts: (elapsed(parts), parts[0]))
    return (whole, triple) + doubles[:1]


def choose(samples, *, minimum=6, relative_margin=0.015, absolute_margin_us=5.0):
    """Use position-paired observations and an explicit noise margin."""
    baseline = samples[0]
    if len(baseline) < minimum:
        return 0
    best, saving = 0, 0.0
    center = statistics.median(baseline)
    for index, values in enumerate(samples[1:], 1):
        count = min(len(baseline), len(values))
        if count < minimum:
            continue
        differences = [a - b for a, b in zip(baseline[:count], values[:count])]
        gain = statistics.median(differences)
        mad = statistics.median(abs(value - gain) for value in differences)
        margin = max(absolute_margin_us, center * relative_margin, 2 * mad)
        if gain > margin and gain > saving and sum(d > 0 for d in differences) >= math.ceil(0.8 * count):
            best, saving = index, gain
    return best


def choose_three_then_two(samples, choices, paired=None):
    """Prefer a qualified triple unless the double establishes a further gain."""
    if len(choices) < 2 or len(choices[1]) != 3:
        return choose(samples)
    three = samples[:2] if paired is None else paired["whole_vs_three"]
    if len(choices) == 2:
        return choose(three)
    # Four-way sampling interleaves a second whole control, reusing its plan.
    two_index = len(choices) - 1
    control = 2 if len(choices) == 4 else 0
    two = [samples[control], samples[two_index]] if paired is None else paired["whole_vs_two"]
    direct = [samples[1], samples[two_index]] if paired is None else paired["three_vs_two"]
    if choose(three) == 1:
        return two_index if choose(two) == 1 and choose(direct) == 1 else 1
    return two_index if choose(two) == 1 else 0


def candidate_index(index, count):
    cycle, position = divmod(index, count)
    order = list(range(count))
    offset = cycle % count
    order = order[offset:] + order[:offset]
    if (cycle // count) % 2:
        order.reverse()
    return order[position]


@dataclass
class Observation:
    signature: tuple
    choices: tuple
    inputs: tuple = ()
    joint: tuple | None = None
    metadata: object = None
    kernel: object = None
    group: object = None
    comm: object = None
    calls: int = 0
    index: int = 0
    selected: int = 0
    events: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    sample_calls: int = 0


class LazyHeadPolicy:
    """Small graph-owned state; native plan/slot ownership stays in its cache."""

    def __init__(self, group, *, enabled=True, candidate_partition=None, candidate_family=None, schedule="stage-local"):
        import torch

        from vllm_omni.diffusion.attention.backends.utils.fa import (
            dense_flash_attn_func_for,
            flash_attn_varlen_func,
            vllm_varlen_version_kwargs,
        )

        self.group, self.enabled = group, enabled
        if schedule not in (
            "stage-local",
            "stage-serial",
            "stage-defer-fence",
            "stage-gated",
            "overlap-return",
            "overlap-input",
            "overlap-staging",
        ):
            raise ValueError("Lazy learning requires an explicit local-staging schedule")
        self.schedule = schedule
        if candidate_family not in (None, "symmetric-ends", "three-then-two"):
            raise ValueError("Unknown head candidate family")
        if candidate_family is not None and candidate_partition is not None:
            raise ValueError("Choose either a candidate partition or a candidate family")
        self.candidate_family = candidate_family
        self.candidate_partition = None if candidate_partition is None else tuple(candidate_partition)
        if self.candidate_partition is not None and (
            not 2 <= len(self.candidate_partition) <= 4
            or any(not isinstance(h, int) or isinstance(h, bool) or h <= 0 for h in self.candidate_partition)
        ):
            raise ValueError("An experimental candidate needs two to four positive integer head counts")
        properties = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
        function = dense_flash_attn_func_for(torch.bfloat16)
        backend = (
            function.__module__,
            function.__qualname__,
            tuple(sorted(vllm_varlen_version_kwargs(flash_attn_varlen_func).items())),
        )
        self.identity = (
            properties.name,
            properties.major,
            properties.minor,
            properties.multi_processor_count,
            str(torch.__version__),
            version("vllm"),
            backend,
        )
        self.sms = properties.multi_processor_count
        self.entries = {}
        self.phase = "discover"
        self.flops, self.bandwidth = 6e14, None
        self.stats = dict(
            schedule=schedule,
            candidate_family=candidate_family,
            core_calls=0,
            event_records=0,
            preparation_calls=0,
            control_collectives=0,
            prepare_ms=0.0,
            finish_ms=0.0,
        )
        self.reports = []
        self.core_events = {} if os.environ.get("HCMS_CORE_EVENTS") == "1" else None
        self.all_core_spans = os.environ.get("HCMS_CORE_EVENT_SCOPE") == "all-cores"
        self.component_events = None
        if self.all_core_spans and self.core_events is None:
            raise ValueError("All-core spans require explicit core event diagnostics")
        self.force_diagnostic_split = os.environ.get("HCMS_FORCE_DIAGNOSTIC_SPLIT") == "1"
        if self.force_diagnostic_split and self.core_events is None and not os.environ.get("HCMS_FULL_TRACE_DIR"):
            raise ValueError("A forced diagnostic split requires event or full-request profiler diagnostics")

    def signature(self, q, k, v, md, pg):
        import torch.distributed as dist

        metadata, self.metadata_supported = metadata_spec(md)
        return self.identity, dist.get_world_size(pg), tuple(tensor_spec(x) for x in (q, k, v)), metadata

    def begin(self):
        self.entries.clear()
        self.phase = "discover"

    @observe_core
    def run(self, q, k, v, md, pg, kernel, *, streams):
        import torch
        import torch.distributed as dist
        from native_prefilled import prepare_plan
        from schedule_probe import run as scheduled

        compute = torch.cuda.current_stream(q.device)
        stream_key = (q.device, pg.group_name)
        if stream_key not in streams:
            streams[stream_key] = (torch.cuda.Stream(device=q.device), compute.cuda_stream)
        comm, producer = streams[stream_key]
        if producer != compute.cuda_stream:
            raise RuntimeError("Lazy policy requires the native plan's owning compute stream")
        signature = self.signature(q, k, v, md, pg)
        entry = self.entries.get(signature)
        heads = q.shape[2] // dist.get_world_size(pg)
        joint = None if md is None or md.joint_query is None else (md.joint_query, md.joint_key, md.joint_value)
        metadata = (
            None
            if md is None
            else replace(md, joint_query=None, joint_key=None, joint_value=None, extra=dict(md.extra))
        )
        if entry is None:
            if self.phase != "discover":
                raise RuntimeError(
                    f"Undiscovered core layout in {self.phase}: actual={signature[1:]}; "
                    f"known={[key[1:] for key in self.entries]}"
                )
            if len(self.entries) >= 16:
                raise RuntimeError("Graph-owned lazy policy signature budget exceeded")
            tokens = q.shape[1] * dist.get_world_size(pg) + (0 if joint is None else joint[0].shape[1])
            kv_tokens = k.shape[1] * dist.get_world_size(pg) + (0 if joint is None else joint[1].shape[1])
            packed = md is not None and "cu_seqlens_q" in md.extra
            # Keep known unsafe low-occupancy packed SplitKV calls whole.
            safe = self.metadata_supported and (
                md is None
                or (
                    md.attn_mask is None
                    and md.joint_attn_mask is None
                    and not md.full_attn_spans
                    and (not packed or md.packed_padding is not None)
                )
            )
            choices = (
                candidates(
                    heads,
                    tokens,
                    q.shape[3],
                    dist.get_world_size(pg),
                    self.sms,
                    flops=self.flops,
                    bandwidth=self.bandwidth,
                    kv_tokens=kv_tokens,
                )
                if self.enabled and safe and (not packed or tokens >= 4096)
                else ((heads,),)
            )
            if self.enabled and safe and (not packed or tokens >= 4096) and self.candidate_partition is not None:
                if sum(self.candidate_partition) != heads:
                    raise ValueError("The experimental candidate must cover the per-rank heads exactly")
                choices = ((heads,), self.candidate_partition)
            if self.enabled and safe and (not packed or tokens >= 4096) and self.candidate_family == "symmetric-ends":
                choices = symmetric_candidates(heads)
            if self.enabled and safe and (not packed or tokens >= 4096) and self.candidate_family == "three-then-two":
                choices = three_then_two_candidates(
                    heads,
                    tokens,
                    q.shape[3],
                    dist.get_world_size(pg),
                    self.sms,
                    flops=self.flops,
                    bandwidth=self.bandwidth,
                    kv_tokens=kv_tokens,
                )
            entry = self.entries[signature] = Observation(signature, choices)
        if self.phase == "discover":
            entry.inputs, entry.joint, entry.metadata, entry.kernel, entry.group, entry.comm = (
                (q, k, v),
                joint,
                metadata,
                kernel,
                pg,
                comm,
            )
        choice = entry.selected
        event_pair = None
        if self.phase == "discover":
            entry.calls += 1
            choice = 0
        elif self.phase == "sample":
            entry.sample_calls += 1
            if entry.index < len(entry.events):
                choice = candidate_index(entry.index, len(entry.choices))
                event_pair = entry.events[entry.index]
                entry.index += 1
        plan = prepare_plan(
            (q, k, v),
            joint,
            pg,
            entry.choices[choice],
            comm,
            kernel,
            metadata,
            ("lazy", signature),
            allocate=self.phase == "discover",
            joint_strategy="front" if md is None else md.joint_strategy,
        )
        self.stats["core_calls"] += 1
        if self.phase == "frozen" and torch.cuda.is_current_stream_capturing():
            key = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
            audit = self.stats.setdefault("capture_plans", {}).setdefault(
                key,
                dict(
                    chunks=plan.chunks,
                    heads=plan.head_sizes,
                    overlap_branch=plan.chunks > 1 and plan.overlap and self.schedule != "stage-serial",
                    output_pointer=plan.output.tensor.data_ptr(),
                    calls=0,
                ),
            )
            audit["calls"] += 1
        if event_pair is not None:
            event_pair[0].record()
            submitted = time.perf_counter()
        diagnostic = None
        if self.core_events is not None and self.phase == "frozen" and torch.cuda.is_current_stream_capturing():
            if self.all_core_spans:
                diagnostic = self.core_events[signature, choice][audit["calls"] - 1]
            elif audit["calls"] == min(16, entry.calls):
                diagnostic = self.core_events[signature, choice]
        result = scheduled(plan, self.schedule) if diagnostic is None else diagnostic.run(plan, self.schedule, compute)
        if event_pair is not None:
            event_pair[1].record()
            submit_us = (time.perf_counter() - submitted) * 1e6
            # A pending start proves the complete core was enqueued before
            # the GPU reached it. Otherwise eager CPU gaps can mispredict the
            # graph's performance. query() is nonblocking, never a host wait.
            queued = not event_pair[0].query()
            entry.samples.append((choice, *event_pair, queued, submit_us))
            self.stats["event_records"] += 2
        return result

    def prepare(self, *, warmup_passes=1, sample_limit=None):
        import torch
        import torch.distributed as dist
        from native_prefilled import prepare_plan

        start = time.perf_counter()
        self.stats["preparation_calls"] += 1
        if sample_limit is not None:
            if sample_limit < 1 or warmup_passes != 1:
                raise ValueError("A real request prefix must be one bounded pass")
            self.stats["sample_limit_per_signature"] = sample_limit

        def slots_for(entry):
            slots = entry.calls * warmup_passes
            return slots if sample_limit is None else min(slots, sample_limit)

        for entry in self.entries.values():
            # No extra forward is manufactured when this signature is rare.
            slots = slots_for(entry)
            if sample_limit is not None and self.candidate_family is None:
                # A short real prefix prioritizes one promising split:
                # more paired evidence and one fewer resident CE allocation.
                entry.choices = entry.choices[:2]
            if self.candidate_family == "three-then-two" and len(entry.choices) == 3:
                whole, triple, double = entry.choices
                # The previous 24-call prefix already ran eight whole cores
                # after its eight trials per candidate. Interleave those same
                # whole calls as matched controls to sustain eager launch lead.
                # Both controls reuse one plan; no new forward or GPU work.
                entry.choices = (whole, triple, whole, double) if slots >= 28 else (whole, double)
            if slots < 7 * len(entry.choices):
                entry.choices = entry.choices[:2]
            if slots < 7 * len(entry.choices):
                entry.choices = entry.choices[:1]
        header = (self.schedule, [(entry.signature, entry.choices, entry.calls) for entry in self.entries.values()])
        if any(len(entry.choices) > 1 for entry in self.entries.values()):
            headers = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(headers, header, group=self.group)
            self.stats["control_collectives"] += 1
            if any(peer != header for peer in headers):
                raise RuntimeError("Ranks disagree before candidate allocation/sampling")
        for entry in self.entries.values():
            slots = slots_for(entry)
            for widths in entry.choices:
                plan = prepare_plan(
                    entry.inputs,
                    entry.joint,
                    entry.group,
                    widths,
                    entry.comm,
                    entry.kernel,
                    entry.metadata,
                    ("lazy", entry.signature),
                    joint_strategy="front" if entry.metadata is None else entry.metadata.joint_strategy,
                )
            if self.core_events is not None:
                from schedule_probe import CoreEvents

                for choice, widths in enumerate(entry.choices):
                    if self.all_core_spans:
                        self.core_events[entry.signature, choice] = [
                            CoreEvents(0, entry.inputs[0].device, plan.runtime) for _ in range(entry.calls)
                        ]
                        if self.component_events is None:
                            self.component_events = CoreEvents(0, entry.inputs[0].device, plan.runtime)
                    else:
                        self.core_events[entry.signature, choice] = CoreEvents(
                            len(widths), entry.inputs[0].device, plan.runtime
                        )
            if len(entry.choices) > 1:
                entry.events = [
                    (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(slots)
                ]
                # Realize timer handles before measured calls; no kernels/forward.
                for pair in entry.events:
                    pair[0]._hcms_lazy_timer = pair[1]._hcms_lazy_timer = True
                    pair[0].record()
                    pair[1].record()
                self.stats["event_records"] += 2 * len(entry.events)
        self.phase = "sample"
        self.stats["prepare_ms"] += (time.perf_counter() - start) * 1000

    def finish(self):
        """Caller has already synchronized at its original capture boundary."""
        import torch.distributed as dist

        start = time.perf_counter()
        local = []
        for entry in self.entries.values():
            grouped = [[] for _ in entry.choices]
            for choice, begin, end, queued, submit_us in entry.samples:
                grouped[choice].append((begin.elapsed_time(end) * 1000, queued, submit_us))
            local.append((entry.signature, entry.choices, grouped))
        peers = [local]
        if any(len(entry.choices) > 1 for entry in self.entries.values()):
            peers = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(peers, local, group=self.group)
            self.stats["control_collectives"] += 1
        if any(
            [(signature, choices) for signature, choices, _ in peer] != [(s, c) for s, c, _ in local] for peer in peers
        ):
            raise RuntimeError("Ranks discovered different execution signatures/candidates")
        for index, entry in enumerate(self.entries.values()):
            rank_rows_by_choice = []
            raw_maximum, submit_times = [], []
            for choice in range(len(entry.choices)):
                counts = [len(peer[index][2][choice]) for peer in peers]
                if len(set(counts)) != 1:
                    raise RuntimeError("Ranks sampled different candidate schedules")
                rows = [[peer[index][2][choice][i] for peer in peers] for i in range(counts[0])]
                raw_maximum.append([max(row[0] for row in rank_rows) for rank_rows in rows])
                submit_times.append([max(row[2] for row in rank_rows) for rank_rows in rows])
                rank_rows_by_choice.append(rows)
            maximum = qualified_cycles(rank_rows_by_choice)
            paired = None
            if self.candidate_family == "three-then-two" and len(entry.choices) in (3, 4):
                # Keep complete all-rank paired rotations. A starved triple
                # must not erase otherwise valid whole/double comparisons.
                double = len(entry.choices) - 1
                control = 2 if len(entry.choices) == 4 else 0
                paired = {
                    name: qualified_cycles([rank_rows_by_choice[a], rank_rows_by_choice[b]])
                    for name, a, b in (
                        ("whole_vs_three", 0, 1),
                        ("whole_vs_two", control, double),
                        ("three_vs_two", 1, double),
                    )
                }
            forced = self.force_diagnostic_split and len(entry.choices) > 1
            entry.selected = (
                1
                if forced
                else (
                    choose_three_then_two(maximum, entry.choices, paired)
                    if self.candidate_family == "three-then-two"
                    else choose(maximum)
                )
            )
            self.reports.append(
                dict(
                    signature=entry.signature,
                    choices=entry.choices,
                    samples_max_rank_us=maximum,
                    raw_samples_max_rank_us=raw_maximum,
                    host_submission_max_rank_us=submit_times,
                    selected=entry.choices[entry.selected],
                    diagnostic_forced=forced,
                    pairing="complete rotated cycles on every rank",
                    discovery_calls=entry.calls,
                    sampling_pass_calls=entry.sample_calls,
                    timed_prefix_calls=entry.index,
                    allocated_sample_pairs=len(entry.events),
                    paired_samples_max_rank_us=paired,
                )
            )
            entry.inputs, entry.joint, entry.metadata, entry.kernel, entry.group, entry.comm = (
                (),
                None,
                None,
                None,
                None,
                None,
            )
            entry.events.clear()
            entry.samples.clear()
        self.phase = "frozen"
        key = (
            self.schedule,
            tuple((entry.signature, entry.choices[entry.selected]) for entry in self.entries.values()),
        )
        self._cache_key = hashlib.sha256(repr(key).encode()).hexdigest()
        self.stats["finish_ms"] += (time.perf_counter() - start) * 1000

    @property
    def cache_key(self):
        if self.phase != "frozen":
            raise RuntimeError("Capture requires a frozen head policy")
        return self._cache_key

    def diagnostic_offsets_us(self):
        """The caller must have completed its normal output transfer/sync."""
        if self.core_events is None:
            return None
        if self.all_core_spans:
            origin = self.component_events.events["begin"]
            component_span = self.component_events.offsets_us()["end"]
            return [
                dict(
                    heads=entry.choices[entry.selected],
                    layer=layer,
                    offsets_us=events.offsets_us(),
                    component_offset_us=origin.elapsed_time(events.events["begin"]) * 1000,
                    component_span_us=component_span,
                )
                for entry in self.entries.values()
                for layer, events in enumerate(self.core_events[entry.signature, entry.selected], 1)
            ]
        return [
            dict(
                heads=entry.choices[entry.selected],
                layer=min(16, entry.calls),
                offsets_us=self.core_events[entry.signature, entry.selected].offsets_us(),
            )
            for entry in self.entries.values()
        ]
