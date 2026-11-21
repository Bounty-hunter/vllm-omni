# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Learn only from the eager calls a lazy graph already executes.

One call executes one complete core and returns its real output. Preparation
never executes a probe kernel. Event results are consumed only by finish(),
after the graph owner's existing synchronization. No timers survive freezing.
"""

import hashlib
import math
import statistics
import time
from dataclasses import dataclass, field, is_dataclass, replace
from importlib.metadata import version

from execution_counters import observe_core


def candidates(heads, tokens, dim, world, sms, *, flops=6e14, bandwidth=None):
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
    attention = 4 * tokens * tokens * heads * dim / flops
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


class LazyHeadPolicy:
    """Small graph-owned state; native plan/slot ownership stays in its cache."""

    def __init__(self, group, *, enabled=True):
        import torch

        from vllm_omni.diffusion.attention.backends.utils.fa import (
            dense_flash_attn_func_for,
            flash_attn_varlen_func,
            vllm_varlen_version_kwargs,
        )

        self.group, self.enabled = group, enabled
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
            core_calls=0,
            event_records=0,
            preparation_calls=0,
            control_collectives=0,
            prepare_ms=0.0,
            finish_ms=0.0,
        )
        self.reports = []

    @staticmethod
    def tensor_spec(value):
        return tuple(value.shape), tuple(value.stride()), str(value.dtype), value.device.type

    def signature(self, q, k, v, md, pg):
        import torch.distributed as dist

        def spec(value):
            if hasattr(value, "shape"):
                return self.tensor_spec(value)
            if isinstance(value, (int, float, bool, str, type(None))):
                return value
            if isinstance(value, (tuple, list)):
                return tuple(spec(item) for item in value)
            if isinstance(value, dict):
                return tuple((key, spec(item)) for key, item in sorted(value.items()))
            if is_dataclass(value):
                return type(value).__qualname__, tuple(
                    (item.name, spec(getattr(value, item.name))) for item in fields(value)
                )
            raise TypeError(f"Opaque attention metadata cannot share policy statistics: {type(value)}")

        fields = () if md is None else tuple((key, spec(value)) for key, value in vars(md).items() if key != "extra")
        extra = () if md is None else tuple((key, spec(value)) for key, value in sorted(md.extra.items()))
        return self.identity, dist.get_world_size(pg), tuple(self.tensor_spec(x) for x in (q, k, v)), fields, extra

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
                raise RuntimeError("An undiscovered execution signature reached sampling/capture")
            if len(self.entries) >= 16:
                raise RuntimeError("Graph-owned lazy policy signature budget exceeded")
            tokens = q.shape[1] * dist.get_world_size(pg) + (0 if joint is None else joint[0].shape[1])
            packed = md is not None and "cu_seqlens_q" in md.extra
            # Keep known unsafe low-occupancy packed SplitKV calls whole.
            safe = md is None or (md.attn_mask is None and md.joint_attn_mask is None and not md.full_attn_spans)
            choices = (
                candidates(
                    heads,
                    tokens,
                    q.shape[3],
                    dist.get_world_size(pg),
                    self.sms,
                    flops=self.flops,
                    bandwidth=self.bandwidth,
                )
                if self.enabled and safe and (not packed or tokens >= 4096)
                else ((heads,),)
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
        elif self.phase == "sample" and entry.index < len(entry.events):
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
        )
        self.stats["core_calls"] += 1
        if event_pair is not None:
            event_pair[0].record()
        result = scheduled(plan, "stage-local")
        if event_pair is not None:
            event_pair[1].record()
            entry.samples.append((choice, *event_pair))
            self.stats["event_records"] += 2
        return result

    def prepare(self, *, warmup_passes=1):
        import torch
        import torch.distributed as dist
        from native_prefilled import prepare_plan

        start = time.perf_counter()
        self.stats["preparation_calls"] += 1
        for entry in self.entries.values():
            # No extra forward is manufactured when this signature is rare.
            slots = entry.calls * warmup_passes
            if slots < 7 * len(entry.choices):
                entry.choices = entry.choices[:2]
            if slots < 7 * len(entry.choices):
                entry.choices = entry.choices[:1]
        header = [(entry.signature, entry.choices, entry.calls) for entry in self.entries.values()]
        if any(len(entry.choices) > 1 for entry in self.entries.values()):
            headers = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(headers, header, group=self.group)
            self.stats["control_collectives"] += 1
            if any(peer != header for peer in headers):
                raise RuntimeError("Ranks disagree before candidate allocation/sampling")
        for entry in self.entries.values():
            slots = entry.calls * warmup_passes
            for widths in entry.choices:
                prepare_plan(
                    entry.inputs,
                    entry.joint,
                    entry.group,
                    widths,
                    entry.comm,
                    entry.kernel,
                    entry.metadata,
                    ("lazy", entry.signature),
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
            for choice, begin, end in entry.samples:
                grouped[choice].append(begin.elapsed_time(end) * 1000)
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
            maximum = []
            for choice in range(len(entry.choices)):
                counts = [len(peer[index][2][choice]) for peer in peers]
                if len(set(counts)) != 1:
                    raise RuntimeError("Ranks sampled different candidate schedules")
                values = [max(peer[index][2][choice][i] for peer in peers) for i in range(counts[0])]
                maximum.append(values[2 if len(values) >= 10 else 1 :])  # real calls, never extra calls
            entry.selected = choose(maximum)
            self.reports.append(
                dict(
                    signature=entry.signature,
                    choices=entry.choices,
                    samples_max_rank_us=maximum,
                    selected=entry.choices[entry.selected],
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
        key = tuple((entry.signature, entry.choices[entry.selected]) for entry in self.entries.values())
        self._cache_key = hashlib.sha256(repr(key).encode()).hexdigest()
        self.stats["finish_ms"] += (time.perf_counter() - start) * 1000

    @property
    def cache_key(self):
        if self.phase != "frozen":
            raise RuntimeError("Capture requires a frozen head policy")
        return self._cache_key
