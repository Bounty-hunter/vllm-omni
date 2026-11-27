# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Learn an uneven head split from native denoising layers and agree once per layout."""

import math
import statistics
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from weakref import WeakSet

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.cuda_graph import kwargs_signature

from ..ulysses_ce_overlap import CEOverlapUlysses, PreparationError, prepare_local
from .head_candidates import candidates, eligible


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


def layout_family(signature):
    """A dense layer group's structure, independent of Q/KV lengths and grid size."""
    world, qkv, joint, placement, grid, scale, borrowed, packing = signature

    def family(spec):
        shape, strides, dtype, device = spec
        return (shape[0], *shape[2:]), strides, dtype, device

    return (
        world,
        tuple(map(family, qkv)),
        tuple(map(family, joint)),
        placement,
        grid is not None,
        scale,
        borrowed,
        packing is not None,
    )


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
    """Rotate and reverse complete native/candidate cycles across real layers."""
    cycle, position = divmod(index, count)
    position = (position + cycle // 2) % count
    return count - 1 - position if cycle % 2 else position


# Three complete rotations fit all eight pairs plus native into 27 real calls.
# A native lead call lets launches get ahead; require all three valid rotations
# and unanimous positive paired gains. A cold/noisy request stays native.
_ROTATIONS = 3

logger = init_logger(__name__)
_OWNERS = WeakSet()


def clear():
    for policy in list(_OWNERS):
        if policy.layouts:
            torch.accelerator.synchronize()
        policy.layouts.clear()
        policy.pending.clear()
        policy.component_layouts.clear()
        policy.routes.clear()
        policy.discovered.clear()
        policy.visits.clear()
        policy.slots.clear()
        policy.workspace = None
        policy.prepared = None
        policy.layer_group = None
        policy.probe_name = None


_PREPARING_GRAPH = ContextVar("ulysses_preparing_graph", default=False)


@contextmanager
def frozen_graph_policy(original_context):
    """Warmup and capture can only consume a previously measured decision."""
    token = _PREPARING_GRAPH.set(True)
    try:
        with original_context():
            yield
    finally:
        _PREPARING_GRAPH.reset(token)


@dataclass
class Trial:
    widths: tuple
    signature: tuple = ()
    alternatives: tuple = ()
    example: tuple = ()
    observations: dict = field(default_factory=dict)
    timers: list = field(default_factory=list)
    plan: CEOverlapUlysses | None = None
    plans: tuple = ()
    report: torch.Tensor | None = None
    work: object = None
    commit_at: int | None = None
    seen: int = -1  # Unassigned: leave spare real layers native before sampling.
    selected: bool | None = None


class HeadPolicy:
    """Owned by one DiT component and shared by its eligible Ulysses layers."""

    def __init__(self, group, sms, sample_group=None):
        _OWNERS.add(self)
        self.group = group
        self.sample_group = sample_group if sample_group is not None else group
        self.sms = sms
        self.layouts = {}
        self.pending = set()
        self.component_layouts = {}
        self.component_key = None
        self.discovery = None
        self.discovered = {}
        self.enabled_for_call = False
        self.generation = 0
        self.routes = {}
        self.visits = {}
        self.slots = {}
        self.step = 0
        self.request_scoped = True
        self.workspace = None
        self.prepared = None
        self.layer_group = None  # Validated layer names/structure, never a choice or samples.
        self.probe_name = None
        self.active = False

    def begin_request(self):
        """Forget all choices and samples; retain only reusable CE resources."""
        self.active = True
        self.layouts.clear()
        self.pending.clear()
        self.component_layouts.clear()
        self.visits.clear()
        self.routes.clear()
        self.discovered.clear()
        self.slots.clear()
        self.discovery = None
        self.probe_name = None
        self.step = 0
        self.generation = 0

    def end_request(self):
        # A rank may have submitted an asynchronous reduction before its peer.
        # Drain complete sample batches on every rank before another request;
        # never leave an unmatched collective behind. Partial batches submit none.
        self.resolve_pending(force=True)
        for trial in self.layouts.values():
            if trial.work is not None:
                trial.work.wait()
                trial.work = None
        self.active = False

    def begin_component(self, module, args, kwargs):
        # Component inputs precede sequence sharding. All SP peers enter a new
        # shape's discovery pass even if a local core is ineligible.
        if not self.active:
            self.enabled_for_call = False
            self.discovery = None
            self.routes = {}
            self.probe_name = None
            return
        self.step += 1
        self.resolve_pending(commit=True)
        self.component_key = kwargs_signature((args, kwargs))
        keys = self.component_layouts.get(self.component_key)
        frozen = _PREPARING_GRAPH.get() or torch.cuda.is_current_stream_capturing()
        discover = keys is None and not frozen and len(self.component_layouts) < 16
        self.discovered.clear()
        self.discovery = [] if discover else None
        self.probe_name = self.layer_group[1][0] if discover and self.layer_group is not None else None
        self.routes = self.visits.get(self.component_key, {})
        self.enabled_for_call = discover or any(
            self.layouts[key].selected or (not frozen and key in self.pending) for key in keys or ()
        )

        self._assign_slots(frozen)

    def _assign_slots(self, frozen=False):
        self.slots = {}
        if self.pending and not frozen:
            visits = Counter(id(trial) for trial in self.routes.values())
            lead = {}
            for trial in self.routes.values():
                if trial.selected is None and trial.seen < 0:
                    # Preparation/agreement can drain the launch queue. Use
                    # spare layers as lead, without adding calls or delaying
                    # the commit beyond the next component boundary.
                    lead[id(trial)] = max(1, visits[id(trial)] - len(trial.timers))
                    trial.seen = 0
            for name, trial in self.routes.items():
                if lead.get(id(trial), 0):
                    lead[id(trial)] -= 1
                    continue
                if trial.selected is None and trial.seen < len(trial.timers):
                    index = trial.seen
                    trial.seen += 1
                    self.slots[name] = index

    def try_run(self, strategy, layer, q, k, v, md):
        """First visit confirms a learned group; subsequent visits can sample now."""
        result = self._try_run(strategy, layer, q, k, v, md)
        if self.probe_name == strategy.head_layer_name:
            # Every peer visits this agreed layer, even if its actual metadata
            # is ineligible. Do not put the collective behind local eligibility.
            self._probe_group(strategy.head_layer_name)
        return result

    def _probe_group(self, name):
        self.probe_name = None
        signature = next((key for visit, key in self.discovery if visit == name), None)
        family, names = self.layer_group
        proposal = None
        if signature is not None and layout_family(signature) == family:
            trial = self.layouts.get(signature) or self.discovered[signature]
            if trial.selected is not False:
                proposal = signature, trial.widths, trial.alternatives, names
        peers = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(peers, proposal, group=self.group)
        if proposal is None or any(peer != proposal for peer in peers):
            return  # Whole-component discovery remains native on every rank.
        trial = self.layouts.setdefault(signature, trial)
        self.routes = self.visits[self.component_key] = {visit: trial for visit in names}
        self.component_layouts[self.component_key] = (signature,)
        self.discovery = None
        self.discovered.clear()
        self._prepare_trial(signature, trial)
        self._assign_slots()
        logger.info("Ulysses inline preparation: step=%d layers=%d Q=%s", self.step, len(names), signature[1][0][0])

    def _try_run(self, strategy, layer, q, k, v, md):
        """Return None to execute the normal attention path, including native CE."""
        name = strategy.head_layer_name
        trial = self.routes.get(name)
        if self.discovery is None and (trial is None or trial.plan is None):
            return None
        if q.ndim != 4 or q.shape[0] != 1 or not q.is_cuda or q.dtype != torch.bfloat16:
            return None
        world = strategy._sp_group.ulysses_world_size
        if (
            world < 2
            or q.shape[1] == 0
            or q.shape[2] % world
            or q.shape != k.shape
            or q.shape != v.shape
            or any(
                x.dtype != q.dtype
                or x.device != q.device
                or x.stride(3) != 1
                or x.stride(2) != x.shape[3]
                or x.stride(1) < x.shape[2] * x.shape[3]
                for x in (q, k, v)
            )
        ):
            return None
        md = layer._with_kv_cache_dtype(md)
        if not eligible(layer._run_local_attention, md, q.shape[1] * world):
            return None
        joint = ()
        if md is not None and any(x is not None for x in (md.joint_query, md.joint_key, md.joint_value)):
            joint = md.joint_query, md.joint_key, md.joint_value
            if (
                any(
                    x is None
                    or x.ndim != 4
                    or x.shape[0] != 1
                    or x.shape[2:] != q.shape[2:]
                    or x.dtype != q.dtype
                    or x.device != q.device
                    or x.stride(3) != 1
                    or x.stride(2) != q.shape[3]
                    for x in joint
                )
                or joint[1].shape != joint[2].shape
                or md.joint_strategy not in ("front", "rear")
            ):
                return None
        # Each owner is bound to one device, process group and installed BF16 FA3 backend.
        # A cached signature describes every participating Q/K/V layout, including the prefix KV.
        placement = md.joint_strategy if joint else None
        grid = md.video_layout.latent_grid if md is not None and md.video_layout else None
        packed = md.packed_padding if md is not None else None
        packing = (packed.q_length, packed.kv_length) if packed is not None else None
        if self.discovery is not None:
            signature = (
                world,
                tuple(tensor_spec(x) for x in (q, k, v)),
                tuple(tensor_spec(x) for x in joint),
                placement,
                grid,
                layer.softmax_scale,
                layer.borrow_ulysses_output,
                packing,
            )
            trial = self.layouts.get(signature) or self.discovered.get(signature)
            if trial is None and len(self.layouts) + len(self.discovered) < 16:
                heads = q.shape[2] // world
                choices = candidates(heads, q.shape[1] * world, self.sms)
                trial = self.discovered[signature] = Trial(
                    choices[0] if choices else (heads,), signature, alternatives=choices[1:]
                )
                if not choices:
                    trial.selected = False
                else:
                    trial.example = (q.device, strategy._ulysses_pg)
            if trial is not None:
                self.discovery.append((name, signature))
            return None
        # All admitted tensors have batch one, dense heads, BF16 and the same
        # device. Validate the variable geometry without rebuilding/hashing the
        # discovery signature or recursively walking the metadata every layer.
        _, input_specs, joint_specs, expected_placement, expected_grid, _, _, expected_packing = trial.signature
        if (
            q.shape != input_specs[0][0]
            or placement != expected_placement
            or grid != expected_grid
            or packing != expected_packing
            or any(x.stride(1) != spec[1][1] for x, spec in zip((q, k, v), input_specs))
            or any(x.shape != spec[0] or x.stride(1) != spec[1][1] for x, spec in zip(joint, joint_specs))
        ):
            return None

        kernel = layer._run_local_attention

        def split(plan):
            metadata = replace(md, joint_query=None, joint_key=None, joint_value=None) if joint else md
            out = plan.run(
                q, k, v, lambda a, b, c: kernel(a, b, c, metadata), joint_kv=joint[1:] if plan.prefix else ()
            )
            return out if layer.borrow_ulysses_output else out.clone()

        if trial.selected:
            return split(trial.plan)
        index = self.slots.pop(name, None)
        if index is None:
            return None
        start, end = trial.timers[index]
        choice = candidate_index(index, len(trial.plans) + 1)
        plan = trial.plans[choice - 1] if choice else None
        start.record()
        if plan is not None:
            result = split(plan)
        else:
            qq, kk, vv, native_md, ctx = strategy.pre_attention(q, k, v, md)
            result = strategy.post_attention(kernel(qq, kk, vv, native_md), ctx)
        end.record()
        trial.observations[index] = start, end, not start.query()
        return result

    def finish_component(self, module, inputs, output):
        # Hooks execute once per real component call, outside individual attention layers.
        self.enabled_for_call = False
        if _PREPARING_GRAPH.get() or torch.cuda.is_current_stream_capturing():
            return
        if self.discovery is not None:
            order, self.discovery = self.discovery, None
            # Include ordered visits: a different order would issue unmatched
            # native/split collectives even if the set of layouts agreed.
            trials = self.layouts | self.discovered
            self.discovered.clear()
            info = [(name, key, trials[key].widths, trials[key].alternatives) for name, key in order]
            peers = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(peers, info, group=self.group)
            agreed = all(peer == info for peer in peers)
            # Only uniquely identifiable layer visits get a sampling slot. All
            # ranks derive the entire next component schedule from this table.
            names = Counter(name for name, _ in order)
            visits = {name: key for name, key in order if agreed and names[name] == 1}
            self.visits[self.component_key] = {name: trials[key] for name, key in visits.items()}
            keys = self.component_layouts[self.component_key] = tuple(dict.fromkeys(visits.values()))
            # Reuse only an agreed homogeneous group. Its first real layer
            # will confirm new Q/KV geometry before any future request samples.
            if len(keys) == 1 and len(visits) == len(order) and trials[keys[0]].selected is not False:
                self.layer_group = layout_family(keys[0]), tuple(visits)
            for key in keys:
                trial = self.layouts.setdefault(key, trials[key])
                self._prepare_trial(key, trial)
            return
        for key in self.pending:
            trial = self.layouts[key]
            if trial.seen == len(trial.timers) and trial.commit_at is None:
                # Commit at the very next component, not after a native grace
                # step. All peers reach the same boundary even if CPU launches
                # have outrun the GPU; only this boundary may wait for samples.
                trial.commit_at = self.step + 1
        self.resolve_pending()

    def _prepare_trial(self, key, trial):
        if not trial.example:
            return
        device, pg = trial.example
        trial.example = ()
        started = time.perf_counter()
        try:
            plans, trial.timers, reconnect = prepare_local(lambda: self._prepare(key, trial, device, pg), self.group)
        except PreparationError as exc:
            logger.warning("Keeping native Ulysses: %s", exc)
            trial.selected = False
            return
        if reconnect:
            self.workspace.connect()
        for plan in plans:
            plan.handle = self.workspace.handle
            plan.peer_base = self.workspace.peer_base
        trial.plans = tuple(plans)
        trial.plan = plans[0]
        self.prepared = key, trial.plans, trial.timers
        logger.info(
            "Ulysses head preparation: Q=%s candidates=%s prepare_agree_connect_ms=%.3f workspace_mib=%.3f",
            key[1][0][0],
            [plan.widths for plan in plans],
            (time.perf_counter() - started) * 1000,
            self.workspace.capacity / 2**20,
        )
        self.pending.add(key)

    def _prepare(self, key, trial, device, pg):
        if self.prepared is not None and self.prepared[0] == key:
            return self.prepared[1], self.prepared[2], False
        self.prepared = None
        _, qkv, joint, placement, *_ = key
        _, rows, heads, dim = qkv[0][0]
        if self.workspace is None:
            self.workspace = CEOverlapUlysses(pg, device=device)
        pool = self.workspace
        pool._configure(
            rows,
            heads,
            dim,
            trial.widths,
            torch.bfloat16,
            prefix=joint[1][0][1] if joint else 0,
            front=placement == "front",
            defer_rendezvous=True,
        )
        plans = tuple(pool.partition_view(widths) for widths in (trial.widths, *trial.alternatives))
        for plan in plans:
            if plan.self_inputs:
                continue
            from .staging import prepare_staging

            pitches = tuple(spec[1][1] for spec in (*qkv, *(joint[1:] if plan.prefix else ())))
            try:
                plan.staging = tuple(
                    prepare_staging((*parts, *prefix), pitches, (plan.rank * sum(plan.widths) + offset) * dim)
                    for parts, prefix, offset in zip(plan.local_inputs, plan.prefix_inputs, plan.offsets, strict=True)
                )
            except torch.AcceleratorError:
                raise
            except Exception as exc:
                raise PreparationError(f"Could not compile local Ulysses staging: {exc}") from exc
        timers = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(_ROTATIONS * (len(plans) + 1))
        ]
        for event in (*pool.ev_attn, *pool.ev_in_done.values(), *pool.ev_out_done.values(), pool.ev_inputs):
            event.record()
        for start, end in timers:
            start.record()
            end.record()
        return plans, timers, getattr(pool, "connected_alloc", None) is not pool.alloc

    def resolve_pending(self, *, commit=False, force=False):
        """One asynchronous MAX per sample batch, one deterministic commit.

        Local event queries never initiate a per-step readiness collective.
        A separate CPU group isolates delayed reports from layout preparation.
        Submission follows discovery order even when event readiness differs.
        """
        for key, trial in self.layouts.items():
            if key not in self.pending or trial.commit_at is None:
                continue
            due = commit and self.step >= trial.commit_at
            if trial.report is None:
                if not (force or due) and not all(end.query() for _, end, _ in trial.observations.values()):
                    break
                started = time.perf_counter()
                if force or due:
                    # Only the fixed commit/drain boundary can wait. Never wait
                    # after a layer, and never switch just because one rank is ready.
                    for _, end, _ in trial.observations.values():
                        if not end.query():
                            end.synchronize()
                rows = []
                for index in range(len(trial.timers)):
                    observation = trial.observations.get(index)
                    if observation is None:
                        rows.append(float("inf"))
                    else:
                        begin, end, queued = observation
                        rows.append(begin.elapsed_time(end) * 1000 if queued else float("inf"))
                trial.report = torch.tensor(rows, dtype=torch.float64, device="cpu")
                trial.work = dist.all_reduce(trial.report, op=dist.ReduceOp.MAX, group=self.sample_group, async_op=True)
                logger.info(
                    "Ulysses sample submission: step=%d values=%d host_ms=%.3f",
                    self.step,
                    len(rows),
                    (time.perf_counter() - started) * 1000,
                )
            if not due:
                continue
            started = time.perf_counter()
            trial.work.wait()
            wait_ms = (time.perf_counter() - started) * 1000
            count = len(trial.plans) + 1
            samples = [[] for _ in range(count)]
            rows = trial.report.tolist()
            # Do not spend a whole rotation warming every request. Kernels and
            # launchers are prepared before sampling. Invalidate a whole
            # rotation if any choice/rank had an unqueued or missing call.
            for cycle in range(_ROTATIONS):
                values = rows[cycle * count : (cycle + 1) * count]
                if all(math.isfinite(value) for value in values):
                    for position, value in enumerate(values):
                        samples[candidate_index(cycle * count + position, count)].append(value)
            best = choose(samples, minimum=_ROTATIONS)
            trial.plan = trial.plans[best - 1] if best else None
            trial.selected = bool(best)
            trial.widths = trial.plan.widths if best else (sum(trial.widths),)
            self.generation += int(trial.selected)
            logger.info(
                "Ulysses request decision: step=%d Q=%s heads=%s samples=%d "
                "costs_us=%s commit_wait_ms=%.3f query_tail=0",
                self.step,
                key[1][0][0],
                trial.widths,
                len(samples[0]),
                [statistics.median(values) if values else None for values in samples],
                wait_ms,
            )
            trial.work = None
            trial.report = None
            trial.observations.clear()
            trial.timers = []  # Resource owner retains reusable events, not observations.
            trial.plans = ()
            self.pending.remove(key)


@contextmanager
def request_head_overlap(pipeline):
    """Request-local decisions, shared workspace lifetime owned by the component."""
    policies = getattr(pipeline, "_head_overlap_policies", ())
    for policy in policies:
        policy.begin_request()
    try:
        yield
    finally:
        for policy in policies:
            policy.end_request()


def configure_head_overlap(pipeline):
    """Attach shared state to a component's normal attention strategies at model load."""
    from vllm_omni.diffusion.attention.backends.utils.fa import dense_flash_attn_func_for
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_cuda() or torch.version.hip is not None:
        return
    props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    backend = getattr(dense_flash_attn_func_for(torch.bfloat16), "__module__", None)
    group = get_sp_group().cpu_group
    signature = (props.name, props.major, props.minor, props.multi_processor_count, backend)
    peers = [None] * dist.get_world_size(group)
    dist.all_gather_object(peers, signature, group=group)
    if any(peer != signature for peer in peers) or props.major != 9 or backend != "fa3_fwd_interface":
        logger.info("Automatic Ulysses head overlap uses native attention on this GPU/backend")
        return
    policies = []
    for name in getattr(pipeline, "_dit_modules", ("transformer", "transformer_2")):
        module = getattr(pipeline, name, None)
        if module is None:
            continue
        strategies = ((key, getattr(layer, "parallel_strategy", None)) for key, layer in module.named_modules())
        enabled = [(key, strategy) for key, strategy in strategies if getattr(strategy, "auto_head_overlap", False)]
        if not enabled:
            continue
        sample_group = dist.new_group(
            ranks=dist.get_process_group_ranks(group), backend="gloo", use_local_synchronization=True
        )
        policy = HeadPolicy(group, props.multi_processor_count, sample_group)
        policies.append(policy)
        for layer_name, strategy in enabled:
            strategy.head_layer_name = layer_name
            strategy.head_policy = policy
        module._ulysses_head_policy = policy
        module.register_forward_pre_hook(policy.begin_component, with_kwargs=True)
        module.register_forward_hook(policy.finish_component)

    pipeline._head_overlap_policies = tuple(policies)
    if policies:
        logger.info("Request-scoped head trials use eager DiT calls; decisions are relearned for every request")
