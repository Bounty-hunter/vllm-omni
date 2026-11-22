# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prepared QK/V output slices sharing fused weights and input quantization."""

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
    def prepare(cls, layer, q_width, kv_width):
        kernel = getattr(getattr(layer, "quant_method", None), "fp8_linear", None)
        if not isinstance(kernel, CutlassFP8ScaledMMLinearKernel):
            return None
        weight, scale, input_scale, upper = kernel._get_layer_params(layer)
        widths = (q_width + kv_width, kv_width)
        if (
            weight.ndim != 2
            or weight.shape[1] != sum(widths)
            or weight.stride() != (1, weight.shape[0])
            or scale.numel() != 1
            or getattr(layer, "bias", None) is not None
            or any(width % 16 for width in (*widths, weight.shape[0]))
        ):
            return None
        return cls(kernel, weight, scale, input_scale, upper, widths)

    def __init__(self, kernel, weight, scale, input_scale, upper, widths):
        self.kernel, self.widths = kernel, tuple(widths)
        self.scale, self.input_scale, self.upper = scale, input_scale, upper
        self.weights = tuple(weight.split(self.widths, dim=1))
        self.kernels = []
        for part in self.weights:
            config = replace(kernel.config, weight_shape=(part.shape[1], part.shape[0]))
            selected = type(kernel)(config, kernel.layer_param_names)
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
