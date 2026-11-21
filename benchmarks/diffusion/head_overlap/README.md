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

This is an experiment snapshot. The unprofiled model matrix is recorded in
[MATRIX-RESULTS.txt](MATRIX-RESULTS.txt), with samples and source identities in
[model-matrix-summary.json](evidence/model-matrix-summary.json). H3-Image
FP8 SP4 at 3072 and 4096 reduced request latency by 4.38% and 3.01%,
respectively, in a 30-pair confirmation. These are explicit offline graph
experiments; no default-service speedup is accepted.
CUDA BF16 batch-one MHA with dense or front-joint inputs is the direct probe's
tested scope. Unsupported layouts use the generic fallback. Compiled
attention retains the conservative library fallback. The matrix also exercises
Wan2.2 TI2V-5B video with FP8 DiT, SP2 and VAE2, but establishes no stable gain
for those four cases. NPU, hybrid parallelism, HTTP serving and concurrent
requests are not qualified.

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

The `.patch` files contain explicit temporary integration used by the
experiments. The model matrix composes `native-probe.patch` with the appropriate
graph adapter; the historical full-request patch is a separate integration.
Apply them only in an isolated checkout and undo them before switching probes.
They are kept explicit to avoid installing diagnostic
instrumentation in the library's default path.

## Unprofiled model matrix

`model_matrix.py` alternates complete requests after both paths warm in the
same process, checks output hashes at three seeds, and writes raw samples
after the request timer. The default run does not start a profiler.
Frozen configs include `h3-sp4-confirm.json`, `h3-bf16-matrix.json`,
`qwen-edit-matrix.json`, `flux-bf16-confirm.json`, and `wan-matrix.json`.

For example, in an isolated checkout with four scheduled CUDA devices:

```bash
export PROBE_DIR="$PWD/benchmarks/diffusion/head_overlap"
export PYTHONPATH="$PROBE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HCMS_GRAPH_OUTPUT="$RUN_DIR/audit"
export HCMS_EXPECTED_FP8=260 HCMS_EXPECTED_VAE_WORLD=4
git apply "$PROBE_DIR/native-probe.patch"
git apply "$PROBE_DIR/h3-graph-probe.patch"
python "$PROBE_DIR/model_matrix.py" --model "$H3_IMAGE_MODEL" \
  --config "$PROBE_DIR/h3-sp4-confirm.json" \
  --out "$RUN_DIR/results" --rounds 30 --warmups 5
git apply --reverse "$PROBE_DIR/h3-graph-probe.patch"
git apply --reverse "$PROBE_DIR/native-probe.patch"
```

For the other models, compose `native-probe.patch` with `model-probe.patch`
instead and set `HCMS_MATRIX_AUDIT` to the output audit directory. Qwen also
requires `qwen-block-graph.patch`; it captures the original block loop while
keeping conditioning preparation outside the graph. Undo patches in reverse
order. Wan uses flat deployment parallel arguments because its registered
pipeline does not consume the generic nested `parallel_config` override.
Use a task-local `TRITON_CACHE_DIR` on local storage when the shared cache
reports stale file handles. This changes compilation storage, not the kernels.

## Communication budget and splitting cost

[BUDGET-ANALYSIS.txt](BUDGET-ANALYSIS.txt) separates isolated peer transport,
attention-only splitting, and complete core execution. Wan's 6630-token
case spends 74.48 us extra in sequential 6+6-head attention, matching its
74.53 us half-peer budget. H3 SP4 has a much larger communication budget
relative to the request and a smaller splitting penalty.

`attention_waves.py` compares whole 12 heads with 6+6, 4+8 and 5+7 using
the same FA3 backend, with no communication in the timed region. At 3510
tokens, 4+8 reduces 164.89 to 126.46 us; at 6630, 5+7 reduces 490.58 to
408.75 us. All outputs match exactly. These controls qualify attention
partitions. The subsequent unequal CE implementation and complete-request
comparisons are recorded in [PARTITION-RESULTS.txt](PARTITION-RESULTS.txt),
with samples in [head-partition-summary.json](evidence/head-partition-summary.json).

```bash
torchrun --standalone --nproc-per-node=2 "$PROBE_DIR/core_budget.py" \
  --cases "$PROBE_DIR/evidence/core-budget-sp2-cases.json" --out "$RUN_DIR/sp2"
torchrun --standalone --nproc-per-node=4 "$PROBE_DIR/core_budget.py" \
  --cases "$PROBE_DIR/evidence/core-budget-sp4-cases.json" --out "$RUN_DIR/sp4"
torchrun --standalone --nproc-per-node=2 "$PROBE_DIR/attention_waves.py" \
  --out "$RUN_DIR/partitions"
```

These are CUDA-event performance controls with no profiler. Raw samples,
denominators and source metadata are in
[core-budget-summary.json](evidence/core-budget-summary.json).
Request communication fractions derived from these controls are estimates,
not newly measured request profiles. The previous Qwen matrix requested
`true_cfg_scale=4` without a negative prompt, so it actually ran a single CFG
branch; the old timing samples stand, and the configuration description is corrected.

## Unequal CE head groups

The explicit native probe accepts `head_partition` as exact positive integer
weights. Wan SP2 has 12 heads per rank: `[1, 2]` gives 4+8 and `[1, 3]` gives
3+9. The primary even alternatives are 6+6, 4+8 and 2+10; the odd 3+9 remains
an explicitly labeled control. H3 SP4 has 14 heads per rank, so its even
alternatives are 6+8 (`[3, 4]`) and 4+10 (`[2, 5]`), with 7+7 as the equal
control. An inexact ratio raises instead of silently rounding.

Widths and prefix offsets are built before capture. Forward and reverse use
the same head permutation, and resident plan keys include the actual widths.
There are no additional copies or fences relative to equal two-way splitting.
The total communication bytes are unchanged. An even head count is a useful
candidate constraint; kernel execution waves and exposed pipeline edges still
depend on the shape. It is not an unconditional faster-kernel guarantee.

`partition_probe.py` exercises 16 resident graphs against a global reference,
including dense and front-joint layouts and delayed consumers. The frozen
`wan-partition-matrix.json` and `h3-partition-matrix.json` screen full requests;
the corresponding `*-partition-confirm.json` files freeze selected cases for
30-round confirmation with both whole and equal controls. Use the same patch
composition as the model matrix above. No profiler is enabled. Source SHAs,
per-rank route audits and bitwise seed comparisons accompany the results.

## Staging and graph-private output experiment

`schedule_probe.py` supplies explicit scheduling ablations over the same
prefilled buffers, bytes and peer fences. `stage-local` starts F0 after its own
local prefill, then places the remaining local prefill under F0 communication.
The old schedule is retained as a control. Early peer submission and deferred
local output copies remain separate ablations, not defaults.

`native_prefilled.py` permits capture only after the fixed plan is warm and on
its owning compute stream. It caches chunk counts separately for same-process
comparisons. Its optional `output_slot(key)` context gives a static graph/layer
a private symmetric output slot, allocated before capture. Reverse CE writes
then target this slot directly. Ordinary calls retain independent output clones.
Keys must identify one fixed layout. Consume graph-private values on the owning
stream before replaying the slot; destroy graphs before clearing their external
buffers. Slot residency is bounded. See the fence argument for these lifetimes.

The 2026-11-20 H3-Image confirmation used true 2048², six PDD steps, SP2, FP8 DiT
linears, BF16 FA3/CE, VAE tile parallel 2 and a graph of the 50 DiT blocks. AdaLN
exact memoization was disabled consistently. After warming all five graphs,
30 requests per variant were interleaved in alternating order in one process:

| Graph configuration | Median `Omni.generate` latency |
| --- | ---: |
| One chunk, serial | 1033.917 ms |
| Two chunks, original overlap | 1038.044 ms |
| Two chunks, staged local prefill | 1026.799 ms |
| Two chunks, private output slots | 1025.175 ms |
| Two chunks, staging and output slots | 1021.551 ms |

The combined probe reduced request latency by 1.20% versus one chunk and 1.59%
versus the original overlap in this run. A separate forward/reverse block run
also favored the combined probe, with different absolute timings; this is a
small workload-dependent gain. Seeds 42/43/44 were RGB-bitwise equal. The graph
capture executed 50 native calls and each request replayed six block graphs.
The two active 50-slot variants use about 2.76 GiB of symmetric output storage
per GPU at this canvas; selecting only one needs about 1.38 GiB.

A second interleaved confirmation compared a single borrowed output slot for
the complete graph with the per-layer slots. Each layer consumes its borrowed
attention result on compute before entering the next attention. The next F0
peer-ready fence therefore acknowledges those consumers on every rank before
any reverse write can reuse the slot. This narrower lifetime permits all 50
layers to share one fixed-layout slot; no borrowed attention result may escape
to a consumer after its next reuse. Normal calls continue to clone.

| Second confirmation, 30 requests each | Median `Omni.generate` latency |
| --- | ---: |
| One chunk, serial | 1029.147 ms |
| Two chunks, original overlap | 1019.619 ms |
| Two chunks, staging and per-layer slots | 1017.590 ms |
| Two chunks, staging and one graph slot | 1016.727 ms |

The shared-slot configuration was faster than serial in 30/30 paired rounds
and reduced its median latency by 1.21%. Its gain over original overlap was
only 0.28%, with 17/30 faster paired rounds. Thus the additional latency gain
over original overlap is not consistent between these confirmations. Treat
the repeatable roughly 1.2% result as a comparison to serial, rather than a
universal gain over an already-overlapping route. Seeds 42/43/44 remained
RGB-bitwise equal. One slot needs 29,589,504 bytes, or 28.2 MiB/GPU, at 2048².
Asymmetric delayed consumers also passed with a single reused slot at both
1056 and 4128 tokens, in eager and graph modes.

A separate FA3-only graph control, with no CE, peer fence or local copy, used
the schedule microbenchmark's packed layout `[0, N-16, N]`. At 4128 tokens,
28 heads and head dimension 128, one attention took about 384–390 us, while
two sequential 14-head attentions took about 443–451 us. Their outputs were
bitwise equal. This isolates a substantial splitting tax even without
communication; it does not attribute all model-trace FA3 growth to contention.
Profiler activity sums remain diagnostics, separate from request latency.

After applying `native-probe.patch`, validate normal output ownership and
graph-private slots with asymmetric consumers on the two allocated GPUs:

```bash
torchrun --standalone --nproc-per-node=2 "$PROBE_DIR/graph_slot_stress.py"
```

This remains an explicitly integrated fixed-canvas experiment. It does not
enable H3 DiT graphs or the direct transport in the default service path.
CUDA 13.0's 3D batch-copy API was rejected because capture is unsupported in the
tested runtime; no dependency change or silent eager fallback is included.

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

## Reproducible model matrix

The fixed H3 block-graph integration and interleaved request driver are now
included in the branch. Use an isolated checkout, applying `native-probe.patch`
followed by `h3-graph-probe.patch`. The latter explicitly installs the diagnostic
H3 adapter; undo both patches after the run. Set `PROBE_DIR` on `PYTHONPATH`
before starting workers. `MODEL_PATH` must resolve the complete image checkpoint
and its encoder dependencies.

```bash
git apply "$PROBE_DIR/native-probe.patch"
git apply "$PROBE_DIR/h3-graph-probe.patch"
export HCMS_EXPECTED_FP8=260
export HCMS_EXPECTED_VAE_WORLD=2
export HCMS_GRAPH_OUTPUT=/path/to/fresh/audit
python "$PROBE_DIR/model_matrix.py" --model "$MODEL_PATH" \
  --config "$PROBE_DIR/h3-matrix.json" --out /path/to/fresh/results
```

`h3-matrix.json` declares 1024², 2048², 1440×2560, 3072² and 4096² cases, with
SP2, FP8 DiT and VAE tile parallelism. These additional shapes are a test matrix,
not a support or speedup claim. The driver warms both variants, alternates their
order each round, checks repeat outputs and seeds 42/43/44 bitwise, and records
every unprofiled request. Media and sampling parameters are configuration data;
hashing and artifact persistence are outside the `Omni.generate` timer.

This adapter disables torch.compile and AdaLN memoization consistently while
capturing only the DiT block body. The fixed-layout graph contract and peer
fences remain the same as the recorded 2K experiment. The worker audit checks
actual quantized module count, VAE degree, input geometry and six graph replays.

For other declared DiT components, `model-probe.patch` installs the explicit
offline `model_probe.py` adapter after model loading. Apply it after
`native-probe.patch`, without the H3 block patch. Set `HCMS_MATRIX_AUDIT` to a
fresh directory and use `model_matrix.py` with the Qwen-edit, Wan or Flux
configuration. The configurations are exploratory cases, not validated model
support claims. They keep both comparison arms on the same uncompiled kernels
and capture the model forward on its fixed CE compute stream. A failed graph,
a missing direct-CE route or a native fallback is an error, not a faster result.
Each new graph records actual direct calls and QKV geometry in the audit.

Qwen editing additionally uses `qwen-block-graph.patch`: its original timestep
and conditioning-index preparation remains outside capture, while the shared
probe captures the unchanged block loop. This preserves its conditioning and
avoids capturing synchronous constant construction. The generic adapter also
preserves structured model-output types when the graph returns cloned tensors.
The explicit model patch permits positive sequences below the library's
2048-token heuristic so short matrix controls exercise the requested route.

The CE plan cache now retains up to 16 distinct layouts. This permits joint
and concatenated attention to coexist without invalidating addresses retained
by another graph. Graph owners must release their graphs before `clear()`;
there is no automatic eviction of potentially live layouts. Normal unscoped
attention still returns owned output storage. `layout_graph_stress.py` retains
three dense/joint layouts and alternates them under asymmetric delays.

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
