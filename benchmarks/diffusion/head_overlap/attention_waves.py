# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Counterfactual head partitions, with no communication or output copies."""

import argparse
import hashlib
import inspect
import json
import math
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils.fa import dense_flash_attn_func_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    world = dist.get_world_size()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    fn = dense_flash_attn_func_for(torch.bfloat16)
    properties = torch.cuda.get_device_properties(rank)
    result = dict(
        rank=rank,
        sm_count=properties.multi_processor_count,
        attention_module=fn.__module__,
        attention_source=inspect.getsourcefile(fn),
        attention_signature=str(inspect.signature(fn)),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="FA backend only, CUDA Graph timing; no CE, fences, clones, or cats inside timed region",
        records=[],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for tokens in (1950, 3510, 6630, 7920, 12870, 16380):
            torch.manual_seed(10448 + rank)
            inputs = tuple(torch.randn(1, tokens, 12, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            impl = FlashAttentionImpl(12, 128, 128**-0.5)
            variants = {"12": (12,), "6+6": (6, 6), "4+8": (4, 8), "5+7": (5, 7)}
            entries = {}
            for name, partition in variants.items():
                groups = []
                offset = 0
                for heads in partition:
                    groups.append(tuple(value[:, :, offset : offset + heads].contiguous() for value in inputs))
                    offset += heads

                def body(groups=groups):
                    return tuple(impl.forward_cuda(*parts, None) for parts in groups)

                for _ in range(6):
                    body()
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = body()
                graph.replay()
                torch.accelerator.synchronize()
                entries[name] = graph, output, groups
                torch.testing.assert_close(torch.cat(output, dim=2), entries["12"][1][0], rtol=0, atol=0)
            samples = {name: [] for name in variants}
            for iteration in range(30):
                names = list(variants)
                offset = iteration % len(names)
                names = names[offset:] + names[:offset]
                if iteration % 2:
                    names.reverse()
                for name in names:
                    dist.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record(stream)
                    for _ in range(16):
                        entries[name][0].replay()
                    end.record(stream)
                    end.synchronize()
                    peers = [None] * world
                    dist.all_gather_object(peers, start.elapsed_time(end) * 1000 / 16)
                    samples[name].append(max(peers))
            q_tiles = math.ceil(tokens / 128)
            row = dict(
                tokens=tokens,
                heads=12,
                exact=True,
                median_us={name: statistics.median(values) for name, values in samples.items()},
                samples_us=samples,
                waves_proxy={
                    name: sum(math.ceil(q_tiles * h / properties.multi_processor_count) for h in partition)
                    for name, partition in variants.items()
                },
                waves_proxy_note=(
                    "Hypothesis: 128-query tiles with one resident CTA per SM; not a collected launch grid"
                ),
            )
            result["records"].append(row)
            (args.out / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
            if rank == 0:
                print(
                    "ATTENTION_WAVES " + json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                    flush=True,
                )
            torch.accelerator.synchronize()
            entries.clear()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
