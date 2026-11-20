# CE head-overlap experiments

This branch records the prototype for
[issue #10448](https://github.com/vllm-project/vllm-omni/issues/10448).
The library's `ulysses_head_chunks=2` route is the pooled generic prototype.
The latest direct-CE schedule is an explicit experimental probe in this
directory. `ulysses_head_chunks=1` remains the library default.

The direct probe writes into attention inputs and assembled output head slots,
puts local copies on compute, and leaves peer copies on communication. It
submits attention before the next input exchange, returns the preceding
output during the next attention, and retains three peer fences for two
groups. See [FENCE-PROOF.txt](FENCE-PROOF.txt) for reuse and lifetime ordering.

This is an experiment snapshot, with no stable service speedup accepted.
CUDA BF16 batch-one MHA with dense or front-joint inputs is the direct probe's
tested scope. Unsupported layouts use the generic fallback. Compiled
attention retains the conservative library fallback. NPU, video, hybrid
parallelism, HTTP serving and concurrent requests are not qualified.

## Files

- `overlap_probe.py`: fixed-buffer CE geometry and controlled workload.
- `prefilled_probe.py`: the latest peer-only communication schedule.
- `native_prefilled.py`: dynamic inputs, bounded plans and owned outputs.
- `prefilled_runner.py`: microbenchmark composition and source hashes.
- `stress_fences.py`: delayed consumers and alternating source pointers.
- `e2e.py`: unprofiled `Omni.generate` A/B measurements.
- `profile_full_request.py`: same-thread CPU/CUDA request capture.
- `full_request_driver.py`: two H3 warmups and two captured requests.
- `split_full_request.py`: complete requests and four-step compact views.
- `export_rich_request.py`: complete request-zero traces retaining stacks.
- `analyze_probe.py` and `analyze_edges.py`: device/runtime correlation.

The two `.patch` files contain the exact temporary integration used by the
experiments. Apply one at a time in an isolated checkout. Undo it before
switching probes. They are kept explicit to avoid installing diagnostic
instrumentation in the library's default path.

## Environment

Run from the repository root in the existing editable installation, on two
scheduler-assigned CUDA GPUs with working symmetric-memory peer access.
The recorded environment used two NVIDIA L20X GPUs with NV18 connectivity,
PyTorch `2.14.0+cu130`, vLLM `0.34.0`, BF16 and FLASH_ATTN/FA3.
Use the environment's GPU scheduler for every model, benchmark and test command.

```bash
export PROBE_DIR="$PWD/benchmarks/diffusion/head_overlap"
export PYTHONPATH="$PROBE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PREFILLED_RUN_DIR=/path/to/new/experiment-output
export MODEL_PATH=/path/to/the/H3-image-checkpoint
```

H3 image revision: `5f90473023975e6271b0ed1dd84c4e88e4bbc6d3`.
Bundled text-encoder revision: `42ed227ee7df40d41602854ae760620d6eb651fe`.
The image checkpoint has six fused PDD heads and requires its six-step
schedule. A four-step trace below is a window within a normal six-step
request, not a four-step generation.

H3-Image's VAE tile parallelism already defaults to enabled across the DiT
group. Both drivers explicitly retain `vae_tile_parallel=True` so this
condition appears in their recorded configurations. The historical runs
omitted that key and used the same enabled default. This is the image VAE's
tile distribution, not the generic `vae_patch_parallel_size` path; it uses
the existing two workers. VAE tiling and CUDA-graph settings stay identical
between overlap and serial runs.

Resolution sweeps must use the generated canvas rather than the requested
output size. H3-Image defaults to a minimum generation area of 2048²: a 1024²
request still generates on a 2048² canvas. Lowering `min_generation_area`
changes the workload and requires separate output-quality evaluation.

Online FP8 in this image pipeline is wired to the DiT linears. It does not
pass that configuration to the text encoder, and the direct overlap probe
requires BF16 attention inputs. Faster linears may increase communication's
share of request latency without changing the communication bytes or the
time available to hide them inside attention. FP8 model quality and E2E
overlap gains are not qualified by the BF16 experiments here.

## Controlled microbenchmark

Run this on the unpatched library prototype. The runner selects the direct
schedule while retaining the original serial and generic-pipeline controls.

```bash
torchrun --standalone --nproc-per-node=2 "$PROBE_DIR/prefilled_runner.py" \
  --out "$PREFILLED_RUN_DIR/micro" \
  --cases 4096:32:512,16384:32:512,4128:56:0:16,16416:56:0:16 --trace
python "$PROBE_DIR/analyze_probe.py" "$PREFILLED_RUN_DIR/micro"
```

Timing precedes expensive profiling. Default timing uses nine alternating
rounds and four invocations per sample, takes the maximum of the two ranks
for each sample, then reports the median. Graph and eager results are separate.
Compare `direct1-owned-output`, `direct2-owned-output` and
`direct2-serial-owned-output` to separate splitting from overlap. Reductions
against `baseline` also include staging and packing improvements.

## Native-model A/B and correctness

```bash
git apply --check "$PROBE_DIR/native-probe.patch"
git apply "$PROBE_DIR/native-probe.patch"
HCMS_NATIVE_PLAN_CHUNKS=2 torchrun --standalone --nproc-per-node=2 \
  "$PROBE_DIR/stress_fences.py"
python -m pytest -q -s tests/diffusion/distributed/test_head_overlap.py
```

Both arms configure `ulysses_head_chunks=2`; the experiment-only environment
override selects one actual group for A and two for B. This compares the same
optimized transfer and output-ownership implementation.

```bash
HCMS_NATIVE_PLAN_CHUNKS=1 python "$PROBE_DIR/e2e.py" --h3 --chunks 2 \
  --model "$MODEL_PATH" --sizes 2048,4096 --steps 6 --out "$PREFILLED_RUN_DIR/A1"
HCMS_NATIVE_PLAN_CHUNKS=2 python "$PROBE_DIR/e2e.py" --h3 --chunks 2 \
  --model "$MODEL_PATH" --sizes 2048,4096 --steps 6 --out "$PREFILLED_RUN_DIR/B1"
HCMS_NATIVE_PLAN_CHUNKS=2 python "$PROBE_DIR/e2e.py" --h3 --chunks 2 \
  --model "$MODEL_PATH" --sizes 2048,4096 --steps 6 --out "$PREFILLED_RUN_DIR/B2"
HCMS_NATIVE_PLAN_CHUNKS=1 python "$PROBE_DIR/e2e.py" --h3 --chunks 2 \
  --model "$MODEL_PATH" --sizes 2048,4096 --steps 6 --out "$PREFILLED_RUN_DIR/A2"
git apply --reverse "$PROBE_DIR/native-probe.patch"
```

Each arm uses two warmups and five measured requests per shape. The timer
spans `Omni.generate`; image extraction, hashing and persistence follow it.
This is engine request latency, not HTTP latency. The existing pipeline stage
timers are enabled consistently in these historical measurements.

## Complete-request profiling

Start with the native probe undone. The full-request patch includes native
integration and temporary worker/step annotations. It starts and stops CUPTI
on the owning worker thread, excludes dummy and real warmups, and keeps one
session active across two requests. Initial peer alignment is outside request
markers. Pipeline stage timers are disabled in the capture driver.

```bash
git apply --check "$PROBE_DIR/full-request-probe.patch"
git apply "$PROBE_DIR/full-request-probe.patch"
HCMS_FULL_TRACE_DIR="$PREFILLED_RUN_DIR/full/raw" \
  HCMS_FULL_VARIANT=nonoverlap HCMS_NATIVE_PLAN_CHUNKS=1 \
  python "$PROBE_DIR/full_request_driver.py" --model "$MODEL_PATH" \
  --out "$PREFILLED_RUN_DIR/full/nonoverlap"
HCMS_FULL_TRACE_DIR="$PREFILLED_RUN_DIR/full/raw" \
  HCMS_FULL_VARIANT=overlap HCMS_NATIVE_PLAN_CHUNKS=2 \
  python "$PROBE_DIR/full_request_driver.py" --model "$MODEL_PATH" \
  --out "$PREFILLED_RUN_DIR/full/overlap"
git apply --reverse "$PROBE_DIR/full-request-probe.patch"
python "$PROBE_DIR/split_full_request.py" "$PREFILLED_RUN_DIR/full"
python "$PROBE_DIR/export_rich_request.py" "$PREFILLED_RUN_DIR/full"
python "$PROBE_DIR/analyze_edges.py" "$PREFILLED_RUN_DIR/full"
```

Optionally pass `--expected-sha256` to the capture driver. The recorded RGB
hash for this prompt, checkpoint and seed at 4096 was
`fa22cfc6756228e678b80d6d8b6b20f196d53ed03de762fd2d643a7a04f9d0e8`.
Always compare hashes between the two arms.

The worker window includes request preparation, text encoding, all six
denoising steps, VAE, host image transfer and output preparation. Each complete
request checks 300 head-attention calls; the first-four-step window checks 200.
Compact exports retain device events, CUDA runtime, CPU operators and phase
markers. Rich exports retain Python functions and inline stacks. Raw sessions
remain untouched. Stack-enabled exports are GB-scale and require substantial
host memory. Profiler and export durations are diagnostic, not performance data.

## Recorded evidence

[Micro samples](evidence/micro-timings.json) and
[native A/B/B/A samples](evidence/native-summary.json) are from the frozen
2026-11-20 experiment. Source identities and publication changes are recorded
in [source-provenance.json](source-provenance.json). The published harness uses
portable accelerator equivalents for device selection and synchronization;
the compute, transfer and fence schedule is preserved.

Graph microbenchmark medians in milliseconds, with independent output storage:

| Shape | Optimized whole | Same two groups, serial | Two groups, overlap |
| --- | ---: | ---: | ---: |
| Qwen-shaped 16384 + 512 joint | 3.81598 | 3.89911 | 3.68721 |
| H3-shaped 4128, 16 padding | 0.63140 | 0.72146 | 0.62742 |
| H3-shaped 16416, 16 padding | 6.75507 | 6.79179 | 6.44428 |

The isolated CE / (attention + CE) ratio is 30.5% for the 4128-token case
and 9.3% for 16416 tokens. These are isolated-work ratios, not full-request
communication shares. At 4128 tokens, splitting adds 90.06 µs to the serial
path and overlapping reduces that split path by 94.04 µs, leaving 3.98 µs
net. At 16416 tokens those differences are 36.72, 347.51 and 310.79 µs.
Thus a higher communication fraction alone does not identify a useful
overlap workload. The eager 4128-token case regressed by 1.26 µs.

H3 4096 engine-request medians for A1/B1/B2/A2 were
5925.730/5918.882/5869.468/5906.826 ms. The forward/reverse comparisons reduced
latency by 0.116% and 0.632%, with overlapping run variation. H3 2048 and short
Qwen-shaped cases did not establish a benefit. No stable service speedup is
accepted from these results.

The recorded distributed suite passed 72 global-reference/layout checks per
rank using its declared SDPA tolerance. Delayed-consumer stress passed 16
bitwise comparisons per rank. Controlled FA3 microbenchmarks and H3 generated
RGB pixels matched exactly. `sm_margin=1` was rejected after output mismatches.
The earlier compiled Qwen output drift remains unresolved and is not uniquely
attributed to overlap.

The [two-rank edge timeline](evidence/edge-wait-step1-block38.svg) and
[correlated events](evidence/edge-wait-proof.json) show why rank-zero peer
windows can remain long: its copies finish sooner, then fences wait for the
other rank's later input or attention completion. GPU frequency, temperature
and power were not sampled concurrently, so the underlying device timing
difference is not assigned to a specific hardware cause.

AI assistance: Codex prepared the implementation, experiment scripts and
analysis. The branch is retained for investigation; it is not PR-ready
production integration.
