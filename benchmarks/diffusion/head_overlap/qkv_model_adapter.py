# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Narrow model glue; reusable projection/attention plans own all scheduling."""

import copy
import functools

from lazy_head_policy import tensor_spec
from staged_projection import StagedProjection

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.layers.exact_bf16_qk_norm_rope import qk_rotary
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Attention


def install(pipeline):
    from model_probe import _ACTIVE, record

    prepared, unsupported = 0, 0
    modules = (module for name in pipeline._dit_modules for module in getattr(pipeline, name).modules())
    for module in modules:
        if not isinstance(module, QwenImage21Attention):
            continue
        projection = StagedProjection.prepare(
            module.to_qkv, module.num_heads * module.head_dim, module.num_kv_heads * module.head_dim
        )
        if projection is None:
            unsupported += 1
            continue
        original = module.forward

        def make_forward(module, original, projection):
            @functools.wraps(original)
            def forward(hidden_states, freqs, attn_metadata=None, **kwargs):
                state = _ACTIVE.get()
                learner = None if state is None else state.get("learner")
                branch = kwargs.get("cache_branch", "cond")
                cache = kwargs.get("kv_cache")
                eligible = (
                    state is not None
                    and state.get("qkv_mode") is not None
                    and not state.get("whole_prefix")
                    and not state["reference_whole"]
                    and learner is not None
                    and hasattr(learner, "qkv_mode")
                    and kwargs.get("sp_decode", False)
                    and kwargs.get("cache_write_len") is None
                    and cache is not None
                    and branch in cache
                    and module.prefix_kv_cache_dtype is None
                    and module.num_heads == module.num_kv_heads
                    and module.norm_q.bias is None
                    and module.norm_k.bias is None
                    and module.norm_q.eps == module.norm_k.eps
                    and (
                        attn_metadata is None
                        or (attn_metadata.attn_mask is None and attn_metadata.full_attn_spans is None)
                    )
                )
                if not eligible:
                    return original(hidden_states, freqs, attn_metadata, **kwargs)
                implementation, backend, _ = module.attn.effective_attention()
                strategy = module.attn._get_active_parallel_strategy()
                if backend.get_name() != "FLASH_ATTN" or not hasattr(strategy, "_ulysses_pg"):
                    return original(hidden_states, freqs, attn_metadata, **kwargs)
                metadata = copy.copy(attn_metadata) if attn_metadata is not None else AttentionMetadata()
                metadata.joint_query = hidden_states.new_empty(1, 0, module.num_heads, module.head_dim)
                metadata.joint_key, metadata.joint_value = cache[branch]["key"], cache[branch]["value"]
                metadata.joint_strategy = "front"

                def normalize(q, k):
                    q, k = (value.unflatten(-1, (module.num_heads, module.head_dim)) for value in (q, k))
                    pair = qk_rotary(q, k, module.norm_q.weight, module.norm_k.weight, freqs, module.norm_q.eps)
                    if pair is None:
                        q = module.norm_q(q).to(hidden_states.dtype)
                        k = module.norm_k(k).to(hidden_states.dtype)
                        return (
                            module._apply_rotary_emb(q, freqs),
                            module._apply_rotary_emb(k, freqs),
                        )
                    return pair

                def fused():
                    qkv, _ = module.to_qkv(hidden_states)
                    width = module.num_heads * module.head_dim
                    q, k, v = qkv.split(width, dim=-1)
                    return *normalize(q, k), v.unflatten(-1, (module.num_heads, module.head_dim))

                normalization = (
                    "original_qk_norm_rope",
                    tensor_spec(freqs),
                    module.norm_q.eps,
                    tensor_spec(module.norm_q.weight),
                    tensor_spec(module.norm_k.weight),
                )
                output = learner.execute(
                    hidden_states,
                    projection,
                    fused,
                    normalize,
                    normalization,
                    metadata,
                    strategy._ulysses_pg,
                    implementation.forward_cuda,
                )
                state["direct_calls"] += 1
                state["qkv_execution_calls"] += 1
                return module.to_out(output.flatten(2, 3).type_as(hidden_states))

            return forward

        module.forward = make_forward(module, original, projection)
        prepared += 1
    record(
        kind="qkv_adapters",
        prepared=prepared,
        unsupported=unsupported,
        scope="CUTLASS tensorwise FP8 MHA decode; original prefill/cache writes retained",
    )
