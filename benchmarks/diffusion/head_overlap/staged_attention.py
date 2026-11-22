# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Q/K-ready CE submission overlapping the deferred V producer.

Model adapters supply tensors and a V producer; all stream ownership, peer
fences, slot lifetime and copy geometry remain inside the attention plan.
"""

import ctypes

import torch
from lazy_head_policy import canonical_strides
from native_prefilled import NativePipeline


class StagedAttention(NativePipeline):
    def __init__(self, inputs, joint, pg, comm, kernel, metadata):
        super().__init__(inputs, joint, pg, 1, comm, kernel, metadata)
        self.value_ready = torch.cuda.Event()
        self.value_ready.record()
        self.field_bindings = [[], [], []]
        self.field_local, self.field_remote = [[], [], []], [[], [], []]
        offset = 0
        for field in range(3):
            count = self.world + int(joint is not None and joint[field].shape[1] > 0)
            operations = self.forward_ops[0][offset : offset + count]
            local_ids = set(map(id, self.local_forward[0]))
            self.field_local[field] = [op for op in operations if id(op) in local_ids]
            self.field_remote[field] = [op for op in operations if id(op) not in local_ids]
            self.field_bindings[field] = [binding for binding in self.bindings if binding[1] % 3 == field]
            offset += count
        self.qk_local = self.field_local[0] + self.field_local[1]
        self.qk_remote = self.field_remote[0] + self.field_remote[1]
        if hasattr(self, "copy_batches"):
            for operations in (self.qk_local, self.qk_remote, self.field_local[2], self.field_remote[2]):
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

    def run_staged(self, query, key, value_producer, joint=None):
        compute = torch.cuda.current_stream()
        self.bind_field(0, query, None if joint is None else joint[0])
        self.bind_field(1, key, None if joint is None else joint[1])
        self.emit_batch_on(self.qk_local, compute)
        self.enter.record(compute)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.enter)
            self.emit_batch_on(self.qk_remote, self.stream)
        # No QK peer barrier is needed here: attention consumes all three
        # fields only after the original single forward-ready peer fence.
        value = value_producer()
        self.bind_field(2, value, None if joint is None else joint[2])
        self.emit_batch_on(self.field_local[2], compute)
        self.value_ready.record(compute)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.value_ready)
            self.emit_batch_on(self.field_remote[2], self.stream)
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
