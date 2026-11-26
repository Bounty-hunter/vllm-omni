# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Learn from existing eager calls, including bounded real-request prefixes.

One call executes one complete core and returns its real output. Preparation
never executes a probe kernel. Event results are consumed only by finish(),
at the owner's completion boundary or after nonblocking readiness checks.
No timers survive freezing.
"""

import hashlib
import math
import statistics
from dataclasses import dataclass, field, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from importlib.metadata import version

import torch
import torch.distributed as dist

from .ce_plan import prepare_plan
from .query_tail import candidate


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


def candidate_index(index):
    # Repeat the original two-choice rotation in both directions.
    cycle, position = divmod(index, 2)
    return position ^ (cycle % 2) ^ ((cycle // 2) % 2)


@dataclass
class Observation:
    signature: tuple
    choices: tuple
    tail_rows: int = 0
    discovery: tuple = ()
    calls: int = 0
    samples_seen: int = 0
    untimed: int = 0
    selected: int = 0
    events: list = field(default_factory=list)
    samples: list = field(default_factory=list)


class HeadPolicy:
    """Sample one real component pass after layout discovery and freeze before capture."""

    def __init__(self, group, adaptive):
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            dense_flash_attn_func_for,
            flash_attn_varlen_func,
            vllm_varlen_version_kwargs,
        )

        self.group = group
        self.adaptive = adaptive
        props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
        fn = dense_flash_attn_func_for(torch.bfloat16)
        self.identity = (
            props.name,
            props.major,
            props.minor,
            props.multi_processor_count,
            str(torch.__version__),
            version("vllm"),
            getattr(fn, "__module__", None),
            getattr(fn, "__qualname__", None),
            tuple(sorted(vllm_varlen_version_kwargs(flash_attn_varlen_func).items())),
        )
        self.supported = props.major == 9 and getattr(fn, "__module__", None) == "fa3_fwd_interface"
        self.sms = props.multi_processor_count
        self.entries = {}
        self.phase = "discover"
        self.cache_key = None
        self.native = False
        self.reports = []
        self.event_records = self.control_collectives = self.core_calls = 0

    def run(self, q, k, v, md, pg, kernel, comm):
        world = dist.get_world_size(pg)
        spec, understood = metadata_spec(md)
        signature = (self.identity, world, tuple(tensor_spec(x) for x in (q, k, v)), spec)
        joint = None if md is None or md.joint_query is None else (md.joint_query, md.joint_key, md.joint_value)
        metadata = None if md is None else replace(md, joint_query=None, joint_key=None, joint_value=None)
        entry = self.entries.get(signature)
        if entry is None:
            if self.phase != "discover":
                raise RuntimeError("Layer layout changed within a fixed benchmark case")
            heads = q.shape[2] // world
            choices, tail = ((heads,),), 0
            if self.adaptive and understood and self.supported:
                choices, tail = candidate(heads, q.shape[1] * world, world, self.sms)
            entry = self.entries[signature] = Observation(signature, choices, tail)
        args = ((q, k, v), joint, pg)
        opts = dict(joint_strategy="front" if md is None else md.joint_strategy, balanced_peers=self.adaptive)
        choice = entry.selected
        timer = None
        if self.phase == "discover":
            entry.discovery = (args, comm, kernel, metadata, opts)
            entry.calls += 1
        if self.phase == "sample":
            i = entry.samples_seen - entry.untimed
            entry.samples_seen += 1
            if 0 <= i < len(entry.events):
                choice, timer = candidate_index(i), entry.events[i]
        widths = entry.choices[choice]
        plan = prepare_plan(
            *args,
            widths,
            comm,
            kernel,
            metadata,
            signature,
            allocate=self.phase == "discover",
            query_tail_rows=entry.tail_rows if choice else 0,
            **opts,
        )
        self.core_calls += 1
        if timer is None:
            return plan.run()
        start, end = timer
        start.record()
        result = plan.run()
        end.record()
        # A pending start indicates that Python submitted the complete core
        # before the GPU reached it. Drop CPU-starved pairs on every rank.
        entry.samples.append((choice, start, end, not start.query()))
        self.event_records += 2
        return result

    def gather(self, value):
        peers = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(peers, value, group=self.group)
        self.control_collectives += 1
        return peers

    def prepare(self):
        from .ce_plan import _PLAN_LIMIT, _PLANS

        remaining = _PLAN_LIMIT - len(_PLANS)
        for entry in self.entries.values():
            if entry.calls < 14 or remaining <= 0:
                entry.choices = entry.choices[:1]
            remaining -= len(entry.choices) - 1
        if self.adaptive:
            header = [(e.signature, e.choices, e.calls, e.tail_rows) for e in self.entries.values()]
            if any(peer != header for peer in self.gather(header)):
                raise RuntimeError("Ranks disagree on layouts before candidate allocation")
        for entry in self.entries.values():
            args, comm, kernel, metadata, opts = entry.discovery
            entry.discovery = ()
            if len(entry.choices) == 1:
                continue
            prepare_plan(
                *args,
                entry.choices[1],
                comm,
                kernel,
                metadata,
                entry.signature,
                query_tail_rows=entry.tail_rows,
                **opts,
            )
            slots = min(24, entry.calls)
            entry.untimed = min(6, entry.calls - slots)
            entry.events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(slots)
            ]
            for start, end in entry.events:
                start.record()
                end.record()
            self.event_records += 2 * slots
        self.phase = "sample"

    def freeze(self):
        # The graph owner has completed its ordinary capture-entry sync.
        local = []
        for entry in self.entries.values():
            rows = [[] for _ in entry.choices]
            for choice, start, end, pending in entry.samples:
                rows[choice].append((start.elapsed_time(end) * 1000, pending))
            local.append(rows)
        peers = (
            self.gather(local) if self.adaptive and any(len(e.choices) > 1 for e in self.entries.values()) else [local]
        )
        for i, entry in enumerate(self.entries.values()):
            choices = []
            for j in range(len(entry.choices)):
                choices.append(list(zip(*(peer[i][j] for peer in peers), strict=True)))
            samples = qualified_cycles(choices)
            entry.selected = choose(samples)
            self.reports.append(
                dict(
                    heads=entry.choices[entry.selected],
                    tail_rows=entry.tail_rows if entry.selected else 0,
                    qualified_pairs=len(samples[0]),
                    timed_calls=len(entry.samples),
                )
            )
            entry.events.clear()
            entry.samples.clear()
        self.native = self.adaptive and all(e.selected == 0 for e in self.entries.values())
        self.phase = "frozen"
        selection = tuple(
            (e.signature, e.choices[e.selected], e.tail_rows if e.selected else 0) for e in self.entries.values()
        )
        self.cache_key = hashlib.sha256(repr(selection).encode()).hexdigest()
