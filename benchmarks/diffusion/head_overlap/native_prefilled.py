# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native-model experiment with prewarmed graphs and bounded output ownership."""

import ctypes
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

import torch
import torch.distributed as dist
from overlap_probe import SymmetricBuffer
from prefilled_probe import PrefilledPipeline

_PLANS = {}
_SLOT_ARENA = {}
_OUTPUT_SLOT = ContextVar("hcms_output_slot", default=None)
_SELECTED_CHUNKS = os.environ.get("HCMS_NATIVE_PLAN_CHUNKS")
_PLAN_LIMIT = 16
_COPY_SUBMITTER = os.environ.get("HCMS_COPY_SUBMITTER")


@contextmanager
def output_slot(key):
    """Scope a static graph/layer's storage; ordinary calls remain owned.

    Keys must identify one fixed layout. A graph consumes each slot on the
    owning compute stream before replaying that slot. Destroy graphs before
    clear() releases their external buffers. The caller bounds graph residency.
    """
    token = _OUTPUT_SLOT.set(key)
    try:
        yield
    finally:
        _OUTPUT_SLOT.reset(token)


def layout(tensors):
    # Import at first layout preparation, after opt-in counters are installed.
    from lazy_head_policy import canonical_strides

    return tuple((x.shape, canonical_strides(x), x.dtype, x.device) for x in tensors)


def partition_heads(heads, chunks, weights=None):
    """Resolve an exact ratio once per eager/capture call; never round heads."""
    if chunks <= 0:
        raise ValueError("Head partition must contain at least one chunk")
    weights = (1,) * chunks if weights is None else tuple(weights)
    if len(weights) != chunks or any(
        not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0 for weight in weights
    ):
        raise ValueError("Head partition weights must be positive integers, one per chunk")
    denominator = sum(weights)
    if heads <= 0 or heads % denominator:
        raise ValueError(f"Head count {heads} cannot be partitioned exactly with weights {weights}")
    return tuple(heads * weight // denominator for weight in weights)


class NativePipeline(PrefilledPipeline):
    @staticmethod
    def geometry(dest, dpitch, source, spitch, width, rows):
        values = [dest, dpitch, source, spitch, width, rows, 4]
        return (ctypes.c_size_t * 7)(*values) if _COPY_SUBMITTER else values

    def __init__(self, inputs, joint, pg, chunks, comm, kernel, metadata, *, head_sizes=None, joint_strategy="front"):
        self.attention_kernel, self.metadata = kernel, metadata
        super().__init__(inputs, joint, pg, chunks, self.attend, head_sizes=head_sizes, joint_strategy=joint_strategy)
        if _COPY_SUBMITTER:
            # Descriptor pointers remain valid when bind() updates their source
            # addresses in place. Prepare every forward batch before sampling.
            self.copy_library = ctypes.CDLL(_COPY_SUBMITTER)
            self.submit_copies = self.copy_library.hcms_submit_copies
            self.submit_copies.argtypes = [
                ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            self.submit_copies.restype = ctypes.c_int
            self.memcpy_2d = ctypes.cast(self.runtime.lib.cudaMemcpy2DAsync, ctypes.c_void_p)
            groups = [
                *self.local_forward,
                *self.remote_forward,
                [op for group in self.local_forward for op in group],
                [op for group in self.local_forward[1:] for op in group],
            ]
            self.copy_batches = {
                tuple(map(id, operations)): (ctypes.POINTER(ctypes.c_size_t) * len(operations))(*operations)
                for operations in groups
            }
        self.base_output = self.output
        self.stream = comm
        tensors = inputs + (() if joint is None else joint)
        self.input_layout = layout(tensors)
        pointers = [x.data_ptr() for x in tensors]
        self.bindings = []
        for operations in self.forward_ops:
            offset = 0
            for field in range(3):
                for operation in operations[offset : offset + self.world]:
                    self.bindings.append((operation, field, operation[2] - pointers[field]))
                offset += self.world
                if joint is not None and joint[field].shape[1]:
                    operation = operations[offset]
                    self.bindings.append((operation, field + 3, operation[2] - pointers[field + 3]))
                    offset += 1
            assert offset == len(operations)

    def emit_batch_on(self, operations, stream):
        if not _COPY_SUBMITTER:
            for operation in operations:
                self.emit_on(operation, stream)
            return
        batch = self.copy_batches[tuple(map(id, operations))]
        self.runtime.CUDART_CHECK(self.submit_copies(batch, len(batch), stream.cuda_stream, self.memcpy_2d))

    def attend(self, q, k, v, unused_metadata):
        out = self.attention_kernel(q, k, v, self.metadata)
        if out.stride(3) != 1 or out.stride(2) != out.shape[3]:
            out = out.contiguous()
        return out

    def bind(self, inputs, joint, kernel, metadata):
        tensors = inputs + (() if joint is None else joint)
        if layout(tensors) != self.input_layout:
            raise ValueError("Cannot bind a different layout to a fixed CE plan")
        pointers = [x.data_ptr() for x in tensors]
        for operation, field, offset in self.bindings:
            operation[2] = pointers[field] + offset
        self.inputs, self.joint = inputs, joint
        self.attention_kernel, self.metadata = kernel, metadata


def prepare_plan(
    inputs, joint, pg, head_sizes, comm, kernel, metadata, slot_key, *, allocate=True, joint_strategy="front"
):
    """Bind an existing plan, or allocate it only at a shape-preparation boundary.

    This does not execute attention or communication. Callers own stream ordering
    and must prepare every candidate before sampling or capture begins.
    """
    q = inputs[0]
    tensors = inputs + (() if joint is None else joint)
    signature = (tuple(head_sizes), layout(tensors), joint_strategy)
    key = (q.device, pg.group_name, len(head_sizes), signature)
    cached = _PLANS.get(key)
    if cached is None:
        if not allocate or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CE plan was not prepared before sampling/capture")
        if len(_PLANS) >= _PLAN_LIMIT:
            raise RuntimeError("Destroy captured graphs and clear CE plans before adding more layouts")
        plan = NativePipeline(
            inputs,
            joint,
            pg,
            len(head_sizes),
            comm,
            kernel,
            metadata,
            head_sizes=head_sizes,
            joint_strategy=joint_strategy,
        )
        _PLANS[key] = (signature, plan)
    else:
        plan = cached[1]
        plan.bind(inputs, joint, kernel, metadata)
    arena_key = (key, slot_key)
    slot = _SLOT_ARENA.get(arena_key)
    if slot is None:
        if not allocate or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CE output slot was not prepared before sampling/capture")
        if len(_SLOT_ARENA) >= 128:
            raise RuntimeError("The fixed-canvas experiment exceeded its output-slot budget")
        tensor = plan.base_output.tensor
        slot = _SLOT_ARENA[arena_key] = SymmetricBuffer(tuple(tensor.shape), tensor.dtype, tensor.device, pg)
    plan.output = slot
    return plan


def clear():
    if _PLANS or _SLOT_ARENA:
        torch.accelerator.synchronize()
    for _, plan in _PLANS.values():
        # Break the bound-kernel cycle, then release all symmetric allocations.
        plan.kernel = plan.attention_kernel = plan.metadata = None
    _PLANS.clear()
    _SLOT_ARENA.clear()


def run(
    q,
    k,
    v,
    md,
    pg,
    chunks,
    kernel,
    *,
    fallback,
    streams,
    prefetch_before_attention=False,
    head_partition=None,
):
    inputs = (q, k, v)
    world = dist.get_world_size(pg)
    supported = (
        q.device.type == "cuda"
        and q.dtype == torch.bfloat16
        and torch.accelerator.current_device_index() == q.device.index
        and all(x.dtype == q.dtype and x.device == q.device for x in inputs)
        and q.ndim == 4
        and q.shape == k.shape == v.shape
        and q.shape[0] == 1
        and q.shape[2] % (world * chunks) == 0
        and all(
            x.stride(3) == 1 and x.stride(2) == x.shape[3] and x.stride(1) >= x.shape[2] * x.shape[3] for x in inputs
        )
    )
    joint = None
    joint_strategy = "front"
    if md is not None:
        supported = supported and md.attn_mask is None and md.joint_attn_mask is None
        if any(x is not None for x in (md.joint_query, md.joint_key, md.joint_value)):
            joint = (md.joint_query, md.joint_key, md.joint_value)
            joint_strategy = md.joint_strategy
            supported = (
                supported
                and joint_strategy in ("front", "rear")
                and all(x is not None for x in joint)
                and joint[1].shape == joint[2].shape
                and joint[0].shape[0] == 1
                and all(x.shape[0] == 1 and x.shape[2:] == q.shape[2:] for x in joint)
                and all(x.dtype == q.dtype and x.device == q.device for x in joint)
                and all(x.stride(3) == 1 and x.stride(2) == x.shape[3] for x in joint)
            )
    if not supported:
        return fallback(
            q,
            k,
            v,
            md,
            pg,
            chunks,
            kernel,
            prefetch_before_attention=prefetch_before_attention,
        )
    # Only the experiment uses this override, to compare the same optimized
    # transport with a single group versus a two-group overlap pipeline.
    selected_chunks = chunks if _SELECTED_CHUNKS is None else int(_SELECTED_CHUNKS)
    if selected_chunks not in (1, chunks):
        raise ValueError("Native probe permits only the configured chunks or 1")
    head_sizes = partition_heads(q.shape[2] // world, selected_chunks, head_partition)
    compute = torch.cuda.current_stream(q.device)
    key = (q.device, pg.group_name)
    if key not in streams:
        streams[key] = (torch.cuda.Stream(device=q.device), compute.cuda_stream)
    comm, producer = streams[key]
    if producer != compute.cuda_stream:
        raise RuntimeError("head overlap buffers require a single compute stream")
    tensors = inputs + (() if joint is None else joint)
    signature = (head_sizes, layout(tensors), joint_strategy)
    kernel_md = (
        None if md is None else replace(md, joint_query=None, joint_key=None, joint_value=None, extra=dict(md.extra))
    )
    # Joint and concatenated attention can coexist in one captured model.
    # A resident graph retains its CE addresses, so never replace its plan
    # merely because the following layer has another layout.
    plan_key = (*key, selected_chunks, signature)
    cached = _PLANS.get(plan_key)
    if cached is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native CE plan must be warmed before capture")
        if len(_PLANS) >= _PLAN_LIMIT:
            raise RuntimeError("Destroy captured graphs and clear CE plans before adding more layouts")
        print(
            f"NATIVE_PREFILLED chunks={selected_chunks} heads={head_sizes} q={tuple(q.shape)} "
            f"packed={md is not None and 'cu_seqlens_q' in md.extra}",
            flush=True,
        )
        # Allocations are collective and occur only before capture. Retain
        # every live layout until its graph owner explicitly calls clear().
        torch.accelerator.synchronize()
        plan = NativePipeline(
            inputs,
            joint,
            pg,
            selected_chunks,
            comm,
            kernel,
            kernel_md,
            head_sizes=head_sizes,
            joint_strategy=joint_strategy,
        )
        _PLANS[plan_key] = (signature, plan)
    else:
        plan = cached[1]
        plan.bind(inputs, joint, kernel, kernel_md)
    # Unscoped attention results own their storage. The next layer/request can
    # reuse transport buffers only after this copy and its consumers on compute.
    current_slot = _OUTPUT_SLOT.get()
    if current_slot is None:
        plan.output = plan.base_output
        return plan().clone()
    # Explicit graph-body scope: the caller consumes this layer's slot on the
    # owning compute stream before the next entry fence. Normal calls above
    # still return an independent tensor. No peer fence is removed.
    slot_key = (plan_key, current_slot)
    slot = _SLOT_ARENA.get(slot_key)
    if slot is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Output slots must be allocated before graph capture")
        if len(_SLOT_ARENA) >= 128:
            raise RuntimeError("The fixed-canvas experiment exceeded its output-slot budget")
        tensor = plan.base_output.tensor
        slot = _SLOT_ARENA[slot_key] = SymmetricBuffer(tuple(tensor.shape), tensor.dtype, tensor.device, pg)
    plan.output = slot
    return plan()
