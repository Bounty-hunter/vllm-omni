# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight QK/V producer, transport and complete-core controls; no profiler."""

import argparse
import copy
import importlib
import json
import os
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from core_budget import CopyEvents
from model_probe import StableGraph
from native_prefilled import NativePipeline
from safetensors import safe_open
from staged_attention import StagedAttention
from staged_projection import StagedProjection
from torch import nn
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod, _fp8_scale
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym, weight_amax

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.layers.exact_bf16_qk_norm_rope import qk_rotary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 1536, 2048])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--inner", type=int, default=8)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl")
    control = dist.new_group(backend="gloo")
    pg, world = dist.group.WORLD, dist.get_world_size()
    importlib.import_module("vllm_omni.diffusion.attention.parallel.head_overlap")
    compute, comm = torch.cuda.Stream(), torch.cuda.Stream()
    compute.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(compute)
    device = torch.device("cuda", rank)
    torch.set_default_dtype(torch.bfloat16)
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.model / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def tensor(name):
        with safe_open(args.model / "transformer" / index[name], framework="pt", device="cpu") as source:
            return source.get_tensor(name).to(device)

    result = dict(
        code_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        rank=rank,
        world=world,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        model=str(args.model),
        rounds=args.rounds,
        inner=args.inner,
        records=[],
        scope="Independent synthetic activations and cached prefix, actual checkpoint weights. "
        "One complete attention/core, borrowed outputs on all arms. No model E2E claim.",
    )
    group = SimpleNamespace(
        ulysses_group=pg, ulysses_world_size=world, ulysses_rank=rank, ring_world_size=1, allgather_world_size=1
    )
    strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
    with torch.inference_mode():
        for number in args.layers:
            prefix = f"transformer_blocks.{number}.attn."
            weight = torch.cat([tensor(prefix + f"to_{field}.weight") for field in "qkv"], dim=0)
            scale = _fp8_scale(weight_amax(weight).reshape(1))
            qweight, _ = ops.scaled_fp8_quant(weight, scale=scale)
            kernel = init_fp8_linear_kernel(
                activation_quant_key=Fp8PerTensorOnlineLinearMethod.activation_quant_key,
                weight_quant_key=kFp8StaticTensorSym,
                weight_shape=weight.shape,
                input_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                module_name="Fp8PerTensorOnlineLinearMethod",
            )
            linear = SimpleNamespace(
                weight=qweight.t(),
                weight_scale=scale,
                input_scale=None,
                bias=None,
                quant_method=SimpleNamespace(fp8_linear=kernel),
            )
            kernel.process_weights_after_loading(linear)
            projection = StagedProjection.prepare(linear, 4096, 4096)
            if projection is None:
                raise RuntimeError(f"Unsupported loaded projection backend: {type(kernel).__name__}")
            assert all(
                w.untyped_storage().data_ptr() == linear.weight.untyped_storage().data_ptr() for w in projection.weights
            )
            wq, wk = (tensor(prefix + f"norm_{field}.weight") for field in "qk")
            for size in args.sizes:
                total = (size // 16) ** 2
                assert total % world == 0
                local = total // world
                torch.manual_seed(42 + rank)
                hidden = torch.randn(1, local, 4096, device=device)
                frequencies = torch.polar(
                    torch.ones(local, 64, device=device, dtype=torch.float32),
                    torch.randn(local, 64, device=device, dtype=torch.float32),
                )
                joint = (
                    torch.empty(1, 0, 32, 128, device=device),
                    torch.randn(1, 4055, 32, 128, device=device),
                    torch.randn(1, 4055, 32, 128, device=device),
                )
                for item in joint[1:]:
                    dist.broadcast(item, src=0)
                metadata = AttentionMetadata(joint_query=joint[0], joint_key=joint[1], joint_value=joint[2])

                def normalize(q, k):
                    pair = qk_rotary(q.unflatten(-1, (32, 128)), k.unflatten(-1, (32, 128)), wq, wk, frequencies, 1e-6)
                    if pair is None:
                        raise RuntimeError("Expected original fused QK norm/RoPE")
                    return pair

                def fused():
                    q, k, v = kernel.apply_weights(linear, hidden).split(4096, dim=-1)
                    return *normalize(q, k), v.unflatten(-1, (32, 128))

                def qk_prepared(quantized, activation_scale):
                    q, k = projection.project(hidden, quantized, activation_scale, 0).split(4096, dim=-1)
                    return normalize(q, k)

                def split():
                    quantized, activation_scale = projection.quantize(hidden)
                    q, k = qk_prepared(quantized, activation_scale)
                    return q, k, projection.project(hidden, quantized, activation_scale, 1).unflatten(-1, (32, 128))

                inputs, split_inputs = fused(), split()
                impl = FlashAttentionImpl(32 // world, 128, 128**-0.5)
                full = NativePipeline(inputs, joint, pg, 1, comm, impl.forward_cuda, None)
                serial = NativePipeline(split_inputs, joint, pg, 1, comm, impl.forward_cuda, None)
                staged = StagedAttention(split_inputs, joint, pg, comm, impl.forward_cuda, None)

                def fused_core():
                    full.bind(fused(), joint, impl.forward_cuda, None)
                    return full()

                def serial_core():
                    serial.bind(split(), joint, impl.forward_cuda, None)
                    return serial()

                def overlap_core():
                    quantized, activation_scale = projection.quantize(hidden)
                    q, k = qk_prepared(quantized, activation_scale)
                    return staged.run_staged(
                        q,
                        k,
                        lambda: projection.project(hidden, quantized, activation_scale, 1).unflatten(-1, (32, 128)),
                        joint,
                    )

                checks = []
                for seed in (42, 43, 44):
                    torch.manual_seed(seed + rank)
                    hidden.normal_()
                    expected_fields, actual_fields = fused(), split()
                    checks.extend(
                        dict(seed=seed, field=field, exact=torch.equal(actual, expected))
                        for field, actual, expected in zip("QKV", actual_fields, expected_fields, strict=True)
                    )
                    q, k, v, md, context = strategy.pre_attention(*expected_fields, copy.copy(metadata))
                    expected = strategy.post_attention(impl.forward_cuda(q, k, v, md), context)
                    for name, function in (
                        ("fused_core", fused_core),
                        ("serial_core", serial_core),
                        ("overlap_core", overlap_core),
                    ):
                        actual = function()
                        checks.append(dict(seed=seed, field=name, exact=torch.equal(actual, expected)))
                quantized, activation_scale = projection.quantize(hidden)
                functions = dict(
                    fused_projection=lambda: kernel.apply_weights(linear, hidden),
                    v_gemm=lambda: projection.project(hidden, quantized, activation_scale, 1),
                    fused_producer=fused,
                    split_producer=split,
                    fused_core=fused_core,
                    serial_core=serial_core,
                    overlap_core=overlap_core,
                )

                # A projection-only control quantizes once and executes both GEMMs.
                def split_projection():
                    values, scales = projection.quantize(hidden)
                    return tuple(projection.project(hidden, values, scales, i) for i in range(2))

                functions["split_projection"] = split_projection
                copy_events = CopyEvents(staged.runtime, device)

                def copy_control():
                    staged.enter.record(compute)
                    with torch.cuda.stream(comm):
                        comm.wait_event(staged.enter)
                        copy_events.run(lambda: staged.emit_batch_on(staged.qk_remote, comm), comm)
                        staged.forward[0].handle.barrier()
                        staged.done.record(comm)
                    compute.wait_event(staged.done)

                functions["qk_copy_control"] = copy_control
                graphs = {}
                owner = StableGraph(nn.Identity(), device=device, max_entries=1)
                for name, function in functions.items():
                    for _ in range(3):
                        function()
                    copy_events.external = True
                    graphs[name] = owner._capture_graph(function)
                    graphs[name][0].replay()
                    copy_events.external = False
                torch.accelerator.synchronize()
                begin, end = (torch.Event(device=device, enable_timing=True) for _ in range(2))
                local_samples, copy_samples = {name: [] for name in functions}, []
                for iteration in range(args.rounds):
                    names = list(functions)
                    offset = iteration % len(names)
                    names = names[offset:] + names[:offset]
                    if (iteration // len(names)) % 2:
                        names.reverse()
                    for name in names:
                        begin.record()
                        for _ in range(args.inner):
                            graphs[name][0].replay()
                        end.record()
                        end.synchronize()
                        local_samples[name].append(begin.elapsed_time(end) * 1000 / args.inner)
                        if name == "qk_copy_control":
                            copy_samples.append(copy_events.begin.elapsed_time(copy_events.end) * 1000)
                peers = [None] * world
                dist.all_gather_object(
                    peers, dict(samples=local_samples, copy=copy_samples, checks=checks), group=control
                )
                samples = {
                    name: [max(peer["samples"][name][i] for peer in peers) for i in range(args.rounds)]
                    for name in functions
                }
                medians = {name: statistics.median(values) for name, values in samples.items()}
                copy_us = statistics.median(max(peer["copy"][i] for peer in peers) for i in range(args.rounds))
                row = dict(
                    layer=number,
                    size=size,
                    qkv_input=[1, local, 32, 128],
                    prefix=4055,
                    kernel=type(kernel).__name__,
                    weight_storage_shared=True,
                    checks_by_rank=[peer["checks"] for peer in peers],
                    all_exact=all(check["exact"] for peer in peers for check in peer["checks"]),
                    median_max_rank_us=medians,
                    samples_max_rank_us=samples,
                    qk_copy_window_us=copy_us,
                    v_overlap_bound_us=min(copy_us, medians["v_gemm"]),
                    producer_tax_us=medians["split_producer"] - medians["fused_producer"],
                    schedule_saving_us=medians["serial_core"] - medians["overlap_core"],
                    net_core_saving_us=medians["fused_core"] - medians["overlap_core"],
                )
                result["records"].append(row)
                (args.out / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
                if rank == 0:
                    print("QKV_BUDGET " + json.dumps(row), flush=True)
                graphs.clear()
                for plan in (full, serial, staged):
                    plan.kernel = plan.attention_kernel = None
                full = serial = staged = None
                del owner
    dist.destroy_process_group()
    if not all(row["all_exact"] for row in result["records"]):
        raise AssertionError("QK/V splitting or core output differs from original Ulysses")


if __name__ == "__main__":
    with set_current_vllm_config(VllmConfig()):
        main()
