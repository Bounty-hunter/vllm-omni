# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reuse the bounded prefix learner for complete projection/attention plans."""

import hashlib
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from execution_counters import observe_core
from lazy_head_policy import LazyHeadPolicy, Observation, candidate_index, metadata_spec, tensor_spec
from native_prefilled import prepare_plan
from staged_attention import StagedAttention


@dataclass
class ProjectionObservation(Observation):
    whole: object = None
    staged: object = None


class LazyQKVPolicy(LazyHeadPolicy):
    """One real producer/core per call; inherited all-rank conservative selection."""

    def __init__(self, group, *, mode):
        if mode not in ("fused", "auto"):
            raise ValueError("QKV plan must be fused or auto")
        super().__init__(group, enabled=mode == "auto")
        self.qkv_mode = mode
        self.stats["qkv_mode"] = mode
        self.stats["measurement_scope"] = "projection + QK norm/RoPE + complete attention core"

    @observe_core
    def execute(self, hidden, projector, fused, normalize, normalization, metadata, pg, kernel):
        from vllm_omni.diffusion.attention.parallel.head_overlap import _COMM_STREAMS

        joint = (
            None
            if metadata is None or metadata.joint_query is None
            else (metadata.joint_query, metadata.joint_key, metadata.joint_value)
        )
        compute = torch.cuda.current_stream(hidden.device)
        stream_key = (hidden.device, pg.group_name)
        if stream_key not in _COMM_STREAMS:
            _COMM_STREAMS[stream_key] = (torch.cuda.Stream(device=hidden.device), compute.cuda_stream)
        comm, producer = _COMM_STREAMS[stream_key]
        if producer != compute.cuda_stream:
            raise RuntimeError("QKV plan requires its owning compute stream")
        semantic, supported = metadata_spec(metadata)
        if not supported:
            raise ValueError("Opaque metadata is outside the QKV prototype's execution contract")
        signature = (
            self.identity,
            dist.get_world_size(pg),
            tensor_spec(hidden),
            type(projector.kernel).__qualname__,
            repr(projector.kernel.config),
            tuple((tuple(w.shape), tuple(w.stride()), str(w.dtype)) for w in projector.weights),
            normalization,
            semantic,
        )
        entry = self.entries.get(signature)
        fields = None
        if entry is None:
            if self.phase != "discover" or len(self.entries) >= 16:
                raise RuntimeError("QKV layout was not discovered within the existing signature budget")
            fields = fused()
            heads = fields[0].shape[2] // dist.get_world_size(pg)
            plan = prepare_plan(fields, joint, pg, (heads,), comm, kernel, None, ("qkv", signature))
            choices = ("fused", "qk_then_v") if self.enabled else ("fused",)
            entry = self.entries[signature] = ProjectionObservation(signature, choices, whole=plan)
        pair = None
        choice = entry.selected
        if self.phase == "discover":
            entry.calls += 1
            choice = 0
        elif self.phase == "sample":
            entry.sample_calls += 1
            if entry.index < len(entry.events):
                choice = candidate_index(entry.index, len(entry.choices))
                pair = entry.events[entry.index]
                entry.index += 1
        if pair is not None:
            pair[0].record()
            submitted = time.perf_counter()
        if entry.choices[choice] == "fused":
            fields = fused() if fields is None else fields
            entry.whole.bind(fields, joint, kernel, None)
            result = entry.whole()
        else:
            values, scales = projector.quantize(hidden)
            qk = projector.project(hidden, values, scales, 0)
            q_width = projector.widths[0] - projector.widths[1]
            query, key = normalize(*qk.split((q_width, projector.widths[1]), dim=-1))
            entry.staged.attention_kernel = kernel
            result = entry.staged.run_staged(
                query,
                key,
                lambda: projector.project(hidden, values, scales, 1).unflatten(-1, (query.shape[2], query.shape[3])),
                joint,
            )
        if pair is not None:
            pair[1].record()
            submit_us = (time.perf_counter() - submitted) * 1e6
            entry.samples.append((choice, *pair, not pair[0].query(), submit_us))
            self.stats["event_records"] += 2
        if self.phase == "discover":
            entry.inputs, entry.joint, entry.metadata, entry.kernel, entry.group, entry.comm = (
                fields,
                joint,
                metadata,
                kernel,
                pg,
                comm,
            )
        self.stats["core_calls"] += 1
        if self.phase == "frozen" and torch.cuda.is_current_stream_capturing():
            key = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
            plan = entry.whole if choice == 0 else entry.staged
            audit = self.stats.setdefault("capture_plans", {}).setdefault(
                key,
                dict(
                    execution_plan=entry.choices[choice],
                    heads=plan.head_sizes,
                    chunks=1,
                    overlap_branch=choice != 0,
                    output_pointer=plan.output.tensor.data_ptr(),
                    calls=0,
                ),
            )
            audit["calls"] += 1
        return result

    def prepare(self, *, warmup_passes=1, sample_limit=None):
        import native_prefilled as native

        if warmup_passes != 1 or sample_limit != 24:
            raise ValueError("QKV learning uses only the existing 24-call real request prefix")
        start = time.perf_counter()
        self.stats["preparation_calls"] += 1
        self.stats["sample_limit_per_signature"] = sample_limit
        for entry in self.entries.values():
            if min(entry.calls, sample_limit) < 14:
                entry.choices = ("fused",)
        if any(len(entry.choices) > 1 for entry in self.entries.values()):
            header = (self.qkv_mode, [(e.signature, e.choices, e.calls) for e in self.entries.values()])
            peers = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(peers, header, group=self.group)
            self.stats["control_collectives"] += 1
            if any(peer != header for peer in peers):
                raise RuntimeError("Ranks disagree on prepared QKV execution plans")
        for entry in self.entries.values():
            if len(entry.choices) == 1:
                continue
            inputs = (
                *entry.inputs[:2],
                torch.empty(entry.inputs[2].shape, device=entry.inputs[2].device, dtype=entry.inputs[2].dtype),
            )
            if len(native._PLANS) >= native._PLAN_LIMIT:
                raise RuntimeError("QKV candidates exceed the existing plan cache budget")
            entry.staged = StagedAttention(inputs, entry.joint, entry.group, entry.comm, entry.kernel, None)
            # Reuse the native graph owner's cache and release lifecycle.
            key = (inputs[0].device, entry.group.group_name, 1, ("qkv-staged", entry.signature))
            native._PLANS[key] = (entry.signature, entry.staged)
            entry.events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(min(entry.calls, sample_limit))
            ]
            for begin, end in entry.events:
                begin._hcms_lazy_timer = end._hcms_lazy_timer = True
                begin.record()
                end.record()
            self.stats["event_records"] += 2 * len(entry.events)
        self.phase = "sample"
        self.stats["prepare_ms"] += (time.perf_counter() - start) * 1000
