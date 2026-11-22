# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prepared QK/V output slices sharing fused weights and input quantization."""

import copy
from dataclasses import replace
from types import SimpleNamespace

from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel


class StagedProjection:
    """Two GEMMs, one quantizer, no weight copy or repacking in forward.

    The narrow prototype supports the existing aligned CUTLASS FP8 kernel.
    Unsupported quantization/packing stays on the original fused Linear.
    Weight reload and adapters must invalidate this prepared view explicitly.
    """

    @classmethod
    def prepare(cls, layer, q_width, kv_width, *, head_dim, early_fields=(0, 1)):
        kernel = getattr(getattr(layer, "quant_method", None), "fp8_linear", None)
        if not isinstance(kernel, CutlassFP8ScaledMMLinearKernel):
            return None
        weight, scale, input_scale, upper = kernel._get_layer_params(layer)
        if early_fields not in ((0, 1), (1, 2)):
            raise ValueError("Early projection must be contiguous QK or KV columns")
        field_widths = (q_width, kv_width, kv_width)
        groups = (early_fields, tuple(field for field in range(3) if field not in early_fields))
        widths = tuple(sum(field_widths[field] for field in group) for group in groups)
        if (
            weight.ndim != 2
            or weight.shape[1] != sum(widths)
            or weight.stride() != (1, weight.shape[0])
            or scale.numel() != 1
            or getattr(layer, "bias", None) is not None
            or any(width % 16 for width in (*widths, weight.shape[0]))
        ):
            return None
        return cls(kernel, weight, scale, input_scale, upper, widths, groups, field_widths, head_dim)

    def __init__(self, kernel, weight, scale, input_scale, upper, widths, groups, field_widths, head_dim):
        self.kernel, self.widths = kernel, tuple(widths)
        self.groups, self.field_widths = groups, field_widths
        self.head_dim = head_dim
        self.scale, self.input_scale, self.upper = scale, input_scale, upper
        self.weights = tuple(
            weight.narrow(1, sum(field_widths[: group[0]]), width) for group, width in zip(groups, widths, strict=True)
        )
        self.kernels = []
        for part in self.weights:
            config = replace(kernel.config, weight_shape=(part.shape[1], part.shape[0]))
            selected = copy.copy(kernel)
            selected.config = config
            # All dimensions are aligned: this sets kernel metadata only.
            params = dict(zip(kernel.layer_param_names, (part, scale, input_scale, upper), strict=True))
            owner = SimpleNamespace(**params)
            selected.process_weights_after_loading(owner)
            if getattr(owner, kernel.layer_param_names[0]) is not part:
                raise RuntimeError("Staged projection unexpectedly repacked a weight view")
            self.kernels.append(selected)

    def quantize(self, hidden):
        return self.kernel.quant_fp8(hidden.view(-1, hidden.shape[-1]), self.input_scale, self.upper)

    def project(self, hidden, quantized, scale, part):
        return self.kernels[part].apply_scaled_mm(
            A=quantized,
            B=self.weights[part],
            As=scale,
            Bs=self.scale,
            out_dtype=hidden.dtype if self.kernel.config.out_dtype is None else self.kernel.config.out_dtype,
            bias=None,
            output_shape=[*hidden.shape[:-1], self.widths[part]],
        )
