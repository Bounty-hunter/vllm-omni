# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eager native-model experiment with bounded, caller-fenced buffer ownership."""

import os
from dataclasses import replace

import torch
import torch.distributed as dist
from prefilled_probe import PrefilledPipeline

_PLANS = {}
_SELECTED_CHUNKS = os.environ.get("HCMS_NATIVE_PLAN_CHUNKS")


def layout(tensors):
    return tuple((x.shape, x.stride(), x.dtype, x.device) for x in tensors)


class NativePipeline(PrefilledPipeline):
    @staticmethod
    def geometry(dest, dpitch, source, spitch, width, rows):
        return [dest, dpitch, source, spitch, width, rows, 4]

    def __init__(self, inputs, joint, pg, chunks, comm, kernel, metadata):
        self.attention_kernel, self.metadata = kernel, metadata
        super().__init__(inputs, joint, pg, chunks, self.attend)
        self.stream = comm
        tensors = inputs + (() if joint is None else joint)
        self.input_layout = layout(tensors)
        pointers = [x.data_ptr() for x in tensors]
        operations_per_field = self.world + int(joint is not None)
        self.bindings = []
        for operations in self.forward_ops:
            for index, operation in enumerate(operations):
                field = index // operations_per_field
                if joint is not None and index % operations_per_field == self.world:
                    field += 3
                self.bindings.append((operation, field, operation[2] - pointers[field]))

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


def clear():
    if _PLANS:
        torch.accelerator.synchronize()
    for _, plan in _PLANS.values():
        # Break the bound-kernel cycle, then release all symmetric allocations.
        plan.kernel = plan.attention_kernel = plan.metadata = None
    _PLANS.clear()


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
        and not torch.cuda.is_current_stream_capturing()
    )
    joint = None
    if md is not None:
        supported = supported and md.attn_mask is None and md.joint_attn_mask is None
        if md.joint_query is not None and md.joint_query.shape[1]:
            joint = (md.joint_query, md.joint_key, md.joint_value)
            supported = (
                supported
                and md.joint_strategy == "front"
                and all(x is not None for x in joint)
                and joint[0].shape == joint[1].shape == joint[2].shape
                and joint[0].shape[0] == 1
                and joint[0].shape[2:] == q.shape[2:]
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
    compute = torch.cuda.current_stream(q.device)
    key = (q.device, pg.group_name)
    if key not in streams:
        streams[key] = (torch.cuda.Stream(device=q.device), compute.cuda_stream)
    comm, producer = streams[key]
    if producer != compute.cuda_stream:
        raise RuntimeError("head overlap buffers require a single compute stream")
    tensors = inputs + (() if joint is None else joint)
    signature = (selected_chunks, layout(tensors))
    kernel_md = (
        None if md is None else replace(md, joint_query=None, joint_key=None, joint_value=None, extra=dict(md.extra))
    )
    cached = _PLANS.get(key)
    if cached is None or cached[0] != signature:
        print(
            f"NATIVE_PREFILLED chunks={selected_chunks} q={tuple(q.shape)} "
            f"packed={md is not None and 'cu_seqlens_q' in md.extra}",
            flush=True,
        )
        # A shape change is rare and collective, as with the existing CE
        # workspace. Bound memory to one live plan per device/process group.
        torch.accelerator.synchronize()
        if cached is not None:
            cached[1].kernel = cached[1].attention_kernel = cached[1].metadata = None
        plan = NativePipeline(inputs, joint, pg, selected_chunks, comm, kernel, kernel_md)
        _PLANS[key] = (signature, plan)
    else:
        plan = cached[1]
        plan.bind(inputs, joint, kernel, kernel_md)
    # Public attention results own their storage. The next layer/request can
    # reuse transport buffers only after this copy and its consumers on compute.
    return plan().clone()
