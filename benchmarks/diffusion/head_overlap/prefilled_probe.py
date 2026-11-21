# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Place local copies before their compute consumers; keep P2P independent."""

import torch
from overlap_probe import DirectPipeline


class PrefilledPipeline(DirectPipeline):
    """Same buffers and copies, with input-ready also acknowledging reuse.

    Local DtoD copies can be issued through the COMPUTE channel on this GPU.
    Prefill every local input before attention, and copy each local output on
    the compute stream before the next attention. Only peer CE traffic remains
    on the communication stream. Returned pooled storage has the base contract.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_forward, self.remote_forward = [], []
        for buffer, operations in zip(self.forward, self.forward_ops):
            base = buffer.handle.buffer_ptrs[self.rank]
            end = base + buffer.tensor.nbytes
            self.local_forward.append([op for op in operations if base <= op[0] < end])
            self.remote_forward.append([op for op in operations if not base <= op[0] < end])
        # An optional schedule bounds prefetch to the attention currently ready
        # on compute. Realize dependency handles at preparation, before trials.
        self.admitted = [torch.cuda.Event() for _ in range(self.chunks - 1)]
        for event in self.admitted:
            event.record()

    def emit_on(self, args, stream):
        self.runtime.CUDART_CHECK(self.runtime.lib.cudaMemcpy2DAsync(*args, stream.cuda_stream))

    def submit_forward(self, g):
        with self.region(f"prefilled/F{g}/peer-only"):
            if hasattr(self, "emit_batch_on"):
                self.emit_batch_on(self.remote_forward[g], self.stream)
            else:
                for operation in self.remote_forward[g]:
                    self.emit(operation)
            self.forward[g].handle.barrier()
            self.ready[g].record(self.stream)

    def reverse_operations(self, g, out, peer):
        head_bytes = self.dim * out.element_size()
        width = self.head_sizes[g] * head_bytes
        head_offset = (self.head_offsets[g] + self.rank * self.head_sizes[g]) * head_bytes
        spitch = out.stride(1) * out.element_size()
        dpitch = self.output.tensor.stride(1) * out.element_size()
        dest = self.output.handle.buffer_ptrs[peer] + self.sequence_offset * dpitch + head_offset
        source = out.data_ptr() + (self.sequence_offset + peer * self.local_seq) * spitch
        operations = [self.geometry(dest, dpitch, source, spitch, width, self.local_seq)]
        if self.joint_len:
            dest = self.output.handle.buffer_ptrs[peer] + self.output_joint_offset * dpitch + head_offset
            source = out.data_ptr() + self.joint_offset * spitch
            operations.append(self.geometry(dest, dpitch, source, spitch, width, self.joint_len))
        return operations

    def submit_reverse(self, g, out):
        with self.region(f"prefilled/R{g}/peer-only"):
            for peer in range(self.world):
                if peer != self.rank:
                    for operation in self.reverse_operations(g, out, peer):
                        self.emit(operation)

    def __call__(self):
        compute = torch.cuda.current_stream()
        # The previous final peer fence plus compute's final wait protect every
        # input-buffer reader. F0-ready below also acknowledges that all ranks
        # finished their previous output copy-out before any new reverse write.
        # Thus reuse needs no additional peer barrier at this entry.
        with self.region("prefilled/local-inputs-before-attention"):
            for operations in self.local_forward:
                for operation in operations:
                    self.emit_on(operation, compute)
        self.enter.record(compute)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.enter)
            self.submit_forward(0)
        sources = []
        for g in range(self.chunks):
            compute.wait_event(self.ready[g])
            with self.region(f"prefilled/A{g}/attention"):
                out = self.kernel(*self.parts[g], None)
            sources.append(out)
            self.produced[g].record(compute)
            with torch.cuda.stream(self.stream):
                if self.overlap:
                    # These are pure peer CE copies: no pre-copy SM barrier.
                    # Submit after attention so host overhead cannot cause the
                    # whole prefetch to finish before its intended compute.
                    if g + 1 < self.chunks:
                        self.submit_forward(g + 1)
                    # Likewise launch this attention before returning the
                    # previous group's output, whose source is retained below.
                    if g:
                        self.stream.wait_event(self.produced[g - 1])
                        self.submit_reverse(g - 1, sources[g - 1])
                else:
                    self.stream.wait_event(self.produced[g])
                    self.submit_reverse(g, out)
                    if g + 1 < self.chunks:
                        self.submit_forward(g + 1)
            with self.region(f"prefilled/R{g}/local-before-next-attention"):
                for operation in self.reverse_operations(g, out, self.rank):
                    self.emit_on(operation, compute)
        with torch.cuda.stream(self.stream):
            if self.overlap:
                self.stream.wait_event(self.produced[-1])
                self.submit_reverse(self.chunks - 1, sources[-1])
            with self.region("prefilled/output-peer-ready"):
                self.output.handle.barrier()
            self.done.record(self.stream)
        compute.wait_event(self.done)
        return self.output.tensor
