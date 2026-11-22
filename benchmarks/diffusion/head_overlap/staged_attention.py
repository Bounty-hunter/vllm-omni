# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Publish each ready field group while producing the next one.

Model adapters supply tensors and a V producer; all stream ownership, peer
fences, slot lifetime and copy geometry remain inside the attention plan.
"""

import ctypes

import torch
from lazy_head_policy import canonical_strides
from native_prefilled import NativePipeline


class StagedAttention(NativePipeline):
    def __init__(self, inputs, joint, pg, comm, kernel, metadata, *, field_groups=((0, 1), (2,))):
        if not 2 <= len(field_groups) <= 3 or any(not group for group in field_groups):
            raise ValueError("Staged transport needs two or three nonempty field groups")
        if sorted(field for group in field_groups for field in group) != [0, 1, 2]:
            raise ValueError("Field groups must publish Q, K and V exactly once")
        super().__init__(inputs, joint, pg, 1, comm, kernel, metadata)
        self.field_groups = tuple(map(tuple, field_groups))
        self.input_ready = [self.enter] + [torch.cuda.Event() for _ in field_groups[1:]]
        # Realize dependency handles before any timed real call. This enqueues
        # no producer, attention, copy or peer fence.
        for event in (*self.input_ready, *self.ready, *self.produced, self.done):
            event.record()
        self.field_bindings = [[], [], []]
        self.field_local, self.field_remote = [[], [], []], [[], [], []]
        local_ids = set(map(id, self.local_forward[0]))
        offset = 0
        for field in range(3):
            count = self.world + int(joint is not None and joint[field].shape[1] > 0)
            operations = self.forward_ops[0][offset : offset + count]
            self.field_local[field] = [op for op in operations if id(op) in local_ids]
            self.field_remote[field] = [op for op in operations if id(op) not in local_ids]
            self.field_bindings[field] = [binding for binding in self.bindings if binding[1] % 3 == field]
            offset += count
        self.local_batches = [[op for field in group for op in self.field_local[field]] for group in self.field_groups]
        self.peer_batches = [[op for field in group for op in self.field_remote[field]] for group in self.field_groups]
        # The controls also measure the combined pre-final peer window.
        self.early_remote = [op for batch in self.peer_batches[:-1] for op in batch]
        self.late_local = self.local_batches[-1]
        if hasattr(self, "copy_batches"):
            for operations in (*self.local_batches, *self.peer_batches, self.early_remote):
                self.copy_batches[tuple(map(id, operations))] = (ctypes.POINTER(ctypes.c_size_t) * len(operations))(
                    *operations
                )
        self.source_layout = [(tuple(x.shape), canonical_strides(x), x.dtype, x.device) for x in inputs]
        # Geometry owns integer descriptors; neither the dummy V allocation nor
        # the discovering request's prefix cache is a replay input owner.
        self.inputs, self.joint = (), None

    def bind_field(self, field, tensor, joint):
        if (tuple(tensor.shape), canonical_strides(tensor), tensor.dtype, tensor.device) != self.source_layout[field]:
            raise ValueError("Staged producer changed a prepared input layout")
        for operation, source, offset in self.field_bindings[field]:
            operation[2] = (tensor if source < 3 else joint).data_ptr() + offset

    def run_staged(self, producers, joint=None):
        compute = torch.cuda.current_stream()
        sources = []
        stages = zip(self.field_groups, producers, self.local_batches, self.peer_batches, self.input_ready, strict=True)
        for fields, produce, local, peer, ready in stages:
            tensors = produce()
            sources.extend(tensors)
            for field, tensor in zip(fields, tensors, strict=True):
                self.bind_field(field, tensor, None if joint is None else joint[field])
            self.emit_batch_on(local, compute)
            ready.record(compute)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                self.emit_batch_on(peer, self.stream)
        # One original peer fence publishes all fields and acknowledges old
        # output consumers. No peer barrier is introduced between producers.
        with torch.cuda.stream(self.stream):
            self.mark_forward_ready(0)
        compute.wait_event(self.ready[0])
        output = self.kernel(*self.parts[0], None)
        self.produced[0].record(compute)
        for operation in self.reverse_operations(0, output, self.rank):
            self.emit_on(operation, compute)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.produced[0])
            self.submit_reverse(0, output)
            self.output.handle.barrier()
            self.done.record(self.stream)
        compute.wait_event(self.done)
        return self.output.tensor
