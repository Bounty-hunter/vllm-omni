# Qwen-Image-2.1: choosing the right path and counting unit

This example was inspected at source commit
`e461c00e0e149cdf22fecf82860664e7088c60c7`. Re-resolve locations and conditions
before applying it to another revision. None of the candidates below is a
validated optimization merely because its source behavior is known.

After examining the complete trace, the user clarified that the below-1% goal
concerns gaps *between the first and last useful device events of a request*.
Retain whole-request occupancy as context, but exclude the request head, host
output tail and time between requests from the internal target. Count
communication as useful device activity. Require zero numerical change,
including per-step floating-point outputs/latents and floating-point decoder
output, not only final image pixels. The first patch below was intermediate;
the later frozen third candidate met the clarified internal goal in each of
three matched profiled samples. The frozen third candidate also included an
output-path experiment, which was subsequently removed; still later VAE-constant
and RoPE-index experiments were removed too. The cleaned working tree is not
byte-identical to that candidate, so its exact numerical and performance results
require a separate matched rerun before being attributed to the cleaned tree.

## Minimal diagnostic workload

Use one T2I request, 1024 x 1024, eight denoising steps, seed 42 and CFG scale 1.
Keep dtype, attention backend, weights and compilation fixed. Warm up requests
before collecting repeated E2E samples, and collect a separate worker trace.
Shorten the number of steps while retaining the prefill-to-decode transition;
do not equate prefill with a warmed decode iteration.

Request-mode and step-mode answer different questions. Request-mode executes the
loop within the pipeline; step-mode additionally exposes per-step KV assembly,
worker dispatch and scheduling. Compare their paths without calling their E2E
difference the benefit of any one proposed fix.

Use passive ranges around `transformer.forward`, `try_decode`,
`_dynamic_lora_wrappers_present`, `_assemble_kv_cache`, `refresh_prefix`,
`sequence_prepare`, RoPE, token metadata, prompt extraction and VAE decoding.
Install labels only after collecting the uninstrumented baseline. Record graph
entries/capture state and actual backend; graph enabled in config does not prove
replay was used. Keep `enable_diffusion_pipeline_profiler=False`: its stage
timers insert synchronization before and after wrapped functions.

At this source revision, `decode_graph.py::_check_static_eligibility` requires
CUDA parameters. On NPU, decode falls back to eager execution even with graph
decode enabled. Recheck the call path: the pre-replay mask guard and repeated
LoRA scan need not execute there, while eager token-layout/RoPE work can recur.
The diagnostic method transfers across platforms; measured costs and gains do
not. Keep NPU and GPU results separate and record the platform-specific backend.

## Candidate inventory

| Candidate | Source anchor under `vllm_omni/diffusion/` | Required distinction |
| --- | --- | --- |
| GPU mask consumed by Python before graph replay | `models/qwen_image_21/decode_graph.py`, `try_decode` | Boolean indexing and `bool(...all())` can produce separate waits at the same expression. Batch 1 still reaches this guard. |
| Repeated traversal looking for LoRA wrappers | `models/qwen_image_21/decode_graph.py`, `_dynamic_lora_wrappers_present` | CPU work; cache/invalidate the property before considering threading. Measure whether device work hides it. |
| KV assembly and prefix copies | `models/qwen_image_21/pipeline_qwen_image_21.py`, `_assemble_kv_cache`; `decode_graph.py`, `refresh_prefix` | Step-mode creates fresh `cat` outputs even with one state, defeating the identity shortcut. Check both host submission and device copies. |
| RoPE and token-layout metadata | `models/qwen_image_21/qwen_image_21_transformer.py`, `sequence_prepare`, `build_token_metadata` | These build device tensors through CPU submission and control; some indexing/scalar decisions can synchronize. Inspect tensor placement and the actual eager/fallback path. Successful graph decode returns before them, so do not charge them to every replay. |
| Layout/padding validation | Transformer batch-layout guard; graph `register_prefill` | The layout reduction requires batch > 1. Registration checks require a non-None validity mask. Count actual branches. |
| Prompt/VLM output extraction | `pipeline_qwen_image_21.py`, `_extract_masked_hidden`, `_downsample_image_pad_tokens` | Once-per-request costs; image-run scalar/list reads require image conditioning. T2I does not cover those IT2I paths. |
| Output transfer and packing | `worker/diffusion_worker.py`, `_return_result`; `ipc.py`, tensor packing | Determine output type and owner thread. A background completion wait need not block compute; PIL output may avoid IPC tensor transfers. |
| Observer-introduced barriers | `profiler/diffusion_pipeline_profiler.py`, `profiler` | Disabled in the performance baseline and the overlap trace. |

These are eight finding groups, not eight synchronization calls or eight
removable bubbles. A useful measured result states, for example, the number of
`try_decode` ranges, runtime synchronization calls inside those ranges, their
overlap with recorded device work, and exposed gaps. Never add nested
`aten::item`, `_local_scalar_dense` and runtime wait durations as separate savings.

## Falsifiable follow-ups

- Guard hypothesis: verify a host read occurs before each graph launch. An
  authorized request-scoped validation experiment must preserve invalidation for
  layout/padding changes and remove those occurrences while retaining output
  correctness. A lower sync count without lower unprofiled E2E is insufficient.
- CPU traversal hypothesis: identify gaps overlapping the scan and verify the
  next submission depends on completing it. If device work fully hides the scan,
  removing it may save CPU without reducing latency.
- KV hypothesis: compare actual `cat` and `copy_` occurrences and buffer identity
  behavior. An authorized reuse experiment must cover batch composition changes,
  mixed prefix lengths, cache dtype and request ownership, not only singleton T2I.
- Metadata hypothesis: first prove the eager/fallback path is active. Optimizing
  metadata bypassed by graph replay cannot improve its repeated decode path.

Eight-step E2E results apply to that workload. A full-length estimate requires
separating fixed prompt/VAE costs from repeated decode costs and explicitly
stating extrapolation assumptions; measured gains still require matched A/B runs.

The eventual acceptance result must include bubble percentages, milliseconds
and percentage-point reduction on the *user-declared window*, alongside the
per-device union and E2E latency. Keep whole-request occupancy separately when
the target is internal device work. Use the same eight-step workload and
boundary semantics on both sides. Static candidates alone do not supply either
side of that measured comparison.

## NPU practice: observed baseline, 2026-10-14

At the source revision above, the eight-step workload ran on Ascend 910B1,
BF16, FLASH_ATTN, no Cache-DiT, and eager decode (zero captured graphs).
Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post4, vLLM 0.31.0+empty,
diffusers 0.40.0 and transformers 5.14.1 were used. Each mode had two warmup
requests, five unprofiled requests, and three separate traced requests.
The request boundary was `Omni.generate` through completed image extraction,
excluding image saving, startup and warmup.

| Baseline mode | Physical NPU | Unprofiled median | Compute stream bubble median (range) | Host stream-sync calls per traced request |
| --- | --- | --- | --- | --- |
| Request | 6 | 3494.09 ms | 6.19% (5.89–6.58%) | 205 |
| Step | 0 | 3497.89 ms | 5.60% (5.52–5.66%) | 207 |

These are two unchanged-code baselines, not a before/after optimization. Each
trace recorded useful compute/copy work only on stream 47, so its occupancy and
the per-device union agree. Stream 46 contained dependency waits; stream 48
contained profiler control. The Ascend adapter excludes those control tasks.
The activity contract includes communication alongside compute and transfers.
These single-card runs had no cross-card communication, so this clarification
does not change the recorded ratios.
Six clock anchors supported an affine mapping with conservative boundary bounds
of 0.343 ms and 0.310 ms respectively. Sample request windows were about 3.54–3.57 s.

Observed lessons for prioritization:

- Each request executed eight RoPE calls and sixteen token-metadata calls.
  RoPE ranges contained 48 host stream-sync calls and overlapped about 33–35 ms
  of device inactivity. Token metadata contained another 48 such calls and
  overlapped about 9–11 ms. These are candidates for request-local invariant
  reuse, not proven savings. RoPE, mask, image IDs and token metadata are not
  simply CPU calculations: their tensor operations run on the device, while
  Python prepares/submits them and selected host reads introduce synchronization.
- Sixteen mask-expansion ranges consumed about 1.75–1.80 s of host wall time,
  but overlapped only about 2–3 ms of device inactivity: most wait time coincided
  with useful queued device work. The 880 scalar-read operator events per request
  are not 880 device synchronizations; some act on host tensors.
- In one representative step request, eight KV-assembly calls consumed 29.14 ms
  on the host but overlapped only 0.086 ms of device inactivity. A static redundant
  `cat` finding does not establish a large bubble on this workload.
- Text-encoder ranges overlapped roughly 78–97 ms of device inactivity. This
  motivates a separate host-submission analysis; one request's text encoder
  produces inputs required by its denoiser, so simply making it async cannot hide
  it behind that same request's dependent computation.

All five baseline and three diagnostic outputs in both modes had identical pixel
SHA256. The diagnostic capture added about 1–2% to E2E versus the unprofiled
baseline. Both processes saved complete result records and parsed traces, then
aborted on shutdown with heap corruption (exit 134). The profiler also warned
about stopping in RECORD state. The captured six anchors, eight transformer calls
per request, output copy and stable device-event counts were checked; retain the
shutdown/export warnings as limitations, not a successful process-exit claim.

Local practice artifacts are under the workspace's
`outputs/qwen21-bubbles-npu-20261014-1120/run10/request/` and `run11/step/`:
`result.json`, `trace_view.json.gz`, `bubbles.json`, and `attribution.json`.
They are not bundled with this skill. No model change or optimization A/B was
performed in this baseline exercise; after-values remain unmeasured.

## Validated optimization: request-local sequence metadata, 2026-10-14

After the user authorized optimization, reuse the expanded image-slot mask,
RoPE frequencies, image IDs and target-token mask within each request. The cache
is owned by the generation loop or single-request step state, with separate CFG
entries. Its signature includes image geometry, mask shape and device. Padding
length changes rebuild it; dynamic multi-request step batches build fresh
metadata. Keep the cached tensors unsharded and preserve the existing SP output
boundary. The caller must retain immutable image-slot contents within a request.
Do not put this cache on the singleton pipeline or reuse it across requests.

The measured patch changed `cfg_parallel.py`, `pipeline_qwen_image_21.py` and
`qwen_image_21_transformer.py` at the baseline revision above. Full-index Git
patch SHA256: `9aa903426d586086c723024cb679105250bd44e688c569a3f4d944c869ef89d3`.
Local and remote full-index patches matched. Changes remained uncommitted.

Use the same eight-step 1024x1024 workload, BF16, FLASH_ATTN, no Cache-DiT and
eager decode. All A/B runs used physical Ascend 910B1 device 0 in one reservation.
Request mode ran A/B/B/A: ten unprofiled and six profiled samples per variant.
Step mode ran A/B: five unprofiled and three profiled samples per variant.
Each process warmed up with two requests. Request boundaries and activity
accounting match the measurement contract, including communication as activity.

| Mode / active stream role | Before bubble ms / request ms | Before bubble % | After bubble ms / request ms | After bubble % | Reduction |
| --- | --- | --- | --- | --- | --- |
| Request / compute+transfer | 220.91 / 3560.52 | 6.20% | 167.89 / 3474.58 | 4.83% | 1.37 pp |
| Step / compute+transfer | 197.42 / 3540.60 | 5.58% | 163.29 / 3467.56 | 4.71% | 0.87 pp |

Each reported column is the median of its individual request samples; the ratio
of two column medians need not equal the median of per-request ratios exactly.
Each trace had one active compute/transfer stream (ID 45 or 47), so its result
equals the per-device activity union. Control-only streams remain excluded.
Request bubble sample ranges were 5.70–6.56% before and 4.50–5.36% after;
Step ranges were 5.49–5.66% and 4.71–4.82%. Conservative clock-boundary uncertainty
was below 0.395 ms in every run.

| Mode | Unprofiled E2E median before | After | Latency reduction | Runtime stream-sync calls/request |
| --- | --- | --- | --- | --- |
| Request | 3504.99 ms | 3432.28 ms | 2.07% | 205 → 74 |
| Step | 3500.01 ms | 3436.35 ms | 1.82% | 207 → 76 |

RoPE calls fell from 8 to 1 per request, token-metadata calls from 16 to 1, and
mask expansion from 16 to 1. In one matched diagnostic pair, the transformer
host ranges' overlap with device inactivity fell from 63.11 ms to 20.41 ms.
This is supporting trace evidence; nested label overlaps are not additive savings.
The request's activity time also fell because redundant device work was removed.
Use the measured E2E result rather than treating the bubble reduction alone as
the full performance gain.

All baseline, candidate and diagnostic outputs were bitwise equal. Additional
eight-step requests changed the prompt, resolution to 768x768 and CFG to 4, then
returned to the original request; their outputs also matched across all six
processes. Request-mode peak allocated memory remained 39,810,947,072 bytes;
Step mode changed from 39,810,589,184 to 39,812,737,536 bytes (about 2.05 MiB more).
All 59 related
test assertions passed, including metadata reuse/CFG/request isolation/padding
checks, and all 13 trace-analyzer tests passed.

Retain the environment limitation: the unit-test process and all six benchmark
processes aborted during interpreter shutdown with exit 134 after saving complete
records. Profiler export warnings remain. Check six anchors, eight transformer
calls and stable per-request device-event counts (Request 22,853 → 22,413;
Step 23,437 → 22,997) before accepting the traces. These are measured single-card
NPU eager results; CUDA graph and multi-card SP behavior were not runtime-tested.
The reservation was released and no personal compute process remained.

Artifacts: `outputs/qwen21-bubbles-ab-20261014/run3/summary.json`, each run's
`result.json`, `bubbles.json`, raw compressed trace and image outputs;
`bubble-before-after.png` and `request-timelines.png` provide the sample
distribution and matched request-lane excerpts. Scripts and the frozen patch are
under `work/qwen21-bubbles-ab-20261014/`. The important reusable lesson is to carry
an observed repeated-work hypothesis through implementation and matched A/B,
instead of ending the optimization task after reporting a baseline.

## Further iterations and transferable decisions, 2026-10-14

These are distinct evidence stages, all on a single Ascend 910B1 and the same
eight-step workload. The source-level inventory in the first section had eight
*groups* of possible causes; it was not eight removable synchronizations.

| Finding | Origin | What the trace or experiment changed |
| --- | --- | --- |
| Reuse RoPE, image IDs, expanded mask and token metadata within a request | Experience-led repeated-work inspection, then profiling confirmation | These are device tensor computations with CPU submission/control and some host/device waits, not simply CPU calculations. The first patch reduced eight RoPE calls to one, sixteen token-metadata calls to one and stream-sync calls from 205 to 74 in Request mode. The measured full-request gap fell 6.20% to 4.83%, with 3505 to 3432 ms profiler-free E2E. |
| Reuse padding/prompt-validity decisions | Static host-read hypothesis, refined by profiling | Build data-dependent validity information once for the request/CFG branch, preserving layout/device/shape invalidation. It was combined with text graphs in iteration 2; no isolated gain is claimed. |
| Text-encoder host submission | Profiling discovery | Text-encoder CPU ranges overlapped about 78–105 ms of recorded device inactivity. Replaying 36 text layers with an NPU graph reduced that overlap to about 3 ms in the iteration-2 diagnostic. This is an execution-path optimization, not async overlap with dependent denoising in the *same* request. |
| KV assembly/prefix copies | Experience-led code suspect, rejected by profiling for this workload | Eight Step-mode KV assemblies used about 29 ms host wall time but overlapped only 0.086 ms of device inactivity in one traced request. Avoid a KV rewrite solely to reduce this bubble. |
| Mask expansion and scalar reads | Experience-led synchronization suspect, deprioritized by profiling | Sixteen mask-expansion ranges used 1.75–1.80 s host wall time yet overlapped only 2–3 ms device inactivity. The 880 scalar-read operator events were not 880 device synchronizations; inspect tensor placement and exposed idle before counting waits. |
| Decode graph eligibility | Profiling discovery | The configured graph decode did not capture on NPU (`graph_entries=0`): its static eligibility checked CUDA parameters. Do not spend effort on graph replay guards that this run bypasses. |
| IPC/CPU lookup output path | Static hypothesis rejected by host probe | The offline path handed a BF16 NPU tensor directly to Qwen postprocessing in the orchestrator thread; no IPC tensor-pack path executed. Raw-BF16 IPC and a CPU lookup experiment could not explain this measured request. They were removed locally after the diagnostic. |
| Output layout and VAE constants | Host-trace follow-up, not validated | A strided NumPy image made PIL conversion expensive in a microexperiment; constant H2D copies and RoPE indexing were also candidates. Those experimental changes were removed from the working tree. They are outside the current request-internal optimization scope. |

The measured iteration ledger separates the two idea sources. Both rows use
full-request device-union idle and profiler-free E2E; each row has its own
matched reference run.

| Stage / origin | What changed | Bubble before → after | Reduction | E2E before → after | Reduction | A/B artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| 1 / experience-led repeated-work inspection | Request-local reuse of RoPE, image IDs, expanded mask and token metadata | 6.20% → 4.83% (220.91 → 167.89 ms) | 1.37 pp | 3504.99 → 3432.28 ms | 2.07% | `outputs/qwen21-bubbles-ab-20261014/run3/summary.json` |
| 2 / profiling-led host-submission gap | Text-layer NPU graph replay plus prompt-validity reuse | 4.58% → 2.15% (158.62 → 72.37 ms) | 2.43 pp | 3432.28 → 3384.06 ms | 1.40% | `outputs/qwen21-bubbles-iter2/perf1/`, `outputs/qwen21-bubbles-iter3/review-summary.json` |

The stage-2 trace showed the text-encoder host range overlapping roughly
78–105 ms of device inactivity; graph replay reduced that overlap to about
3 ms. Stage 2 bundled two changes, so their separate gains are unknown.
Its reference bubble percentage differs from stage 1's candidate because these
were distinct capture sessions; do not add the row reductions. Graph layers
also moved about 31 ms of useful work off the original physical stream. The
stage-2 candidate main-stream idle was about 3.06%, while device-union idle
was 2.15%. Stage 1 established final-pixel equality and related tests, not
the full per-step floating-point contract. The full raw-tensor comparison below
belongs to the frozen third candidate.

The frozen third candidate packaged the text graphs with guarded eligibility,
fallback and graph ownership, and kept prompt extraction metadata on the CPU.
Its matched **cumulative original-to-candidate** Request comparison showed
whole-request device-union idle 7.05% → 2.19% (4.86 pp less) and profiler-free
E2E 3487.6 → 3359.7 ms (3.67% less). The main physical stream's candidate
idle was 3.10%. This is not an incremental gain over iteration 2, whose
full-request union was 2.15%.

For the clarified internal window, use each request's first and last *useful*
hardware event. The matched cumulative original/candidate comparison was
**5.72% to 0.65%** median internal device-union idle (202.13 to 21.61 ms
median idle), a 5.07 percentage-point reduction.
Candidate sample percentages were 0.679%, 0.633% and 0.651%; all were below
the 1% target. A subsequent host-instrumented candidate-only diagnostic gave
0.621%, 0.588% and 0.590% (median 0.590%), with internal gap totals 20.59,
19.46 and 19.56 ms. Its largest single internal gaps were only 1.76, 1.66 and
1.58 ms. The host diagnostic is not a fourth A/B performance result.

In the first host-diagnostic request, the final recorded useful device operator
ended at trace-relative 3381.69 ms, and the `Omni.generate`-through-extraction
request ended at 3425.96 ms. The intervening 44.26 ms is a
*last-recorded-device-to-return* tail, with postprocessing, a runtime memcpy
range and result handling. The next request started 741.28 ms later because
the serial benchmark hashed/saved output and performed control work between
calls. Neither interval is an internal computation bubble. If optimizing
cross-request overlap later, submit the next request before the previous
return and verify that the next request's first device work starts before that
return; this serial trace cannot establish that behavior.

Iteration 3 Request and Step candidates each matched the original Request
baseline over four correctness requests: 73 raw floating snapshots and final
pixels, including repeated requests, 768x768 CFG 4 and a return to 1024x1024
CFG 1. The observed environment still aborted on interpreter shutdown with
exit 134, reproduced on the clean original baseline; complete result and trace
records were saved. This is a limit on clean process-exit claims, not a reason
to replace exact tensor evidence with image-only comparison.

The retained local code contains the request-owned sequence metadata cache,
prompt metadata handling, and guarded NPU text-layer graph. It is not the exact
frozen candidate used for the A/B above: the later experimental output changes
were removed after that measurement. Treat the 5.72% to 0.65% result as evidence
for the frozen candidate and the current tree as awaiting its own A/B validation.

Local evidence: `outputs/qwen21-bubbles-iter2/perf1/`,
`outputs/qwen21-bubbles-iter3/perf1/`, `validate2/`, and
`host1/c/visible-gap-audit.json` under the workspace. The host trace's
request markers were mapped from six monotonic anchors with about 0.32 ms
boundary uncertainty. Raw device events, not the synthetic review marker,
set the internal first/last boundaries.

## Clean-restart practice on one NPU, 2026-10-14

The later clean-restart experiment began again from `e461c00e0`, used one
Ascend 910B1 (physical card 1), the eight-step Request workload above, two
warmups, five unprofiled E2E requests and three independently marked traced
requests per candidate. The target was the device-activity union between the
first and last useful task of each request. Communication and copies counted
as activity; request head, output tail and time between calls did not.

| Stage and origin | Internal union bubble, median | Unprofiled E2E median | Interpretation |
| --- | --- | --- | --- |
| Original | 156.10 ms / 4.468% | 3506.60 ms | Baseline. |
| Experience: request-local mask/RoPE reuse | 129.52 ms / 3.760% | 3456.68 ms | Removed repeated eager preparation and associated host/device waits. |
| Experience, then trace: image IDs/token metadata reuse | 125.17 ms / 3.647% | 3438.74 ms | Smaller further gain; do not credit both reuse steps with the same gap. |
| Pre-graph matched reference after a CUDA-only guard | 116.01 ms / 3.388% | 3436.82 ms | NPU computation unchanged; the ratio shift from the preceding row is run variation, not a credited patch gain. |
| Profiling: NPU text-layer graph replay with request-scoped enablement | 28.69 ms / 0.863% | 3366.15 ms | Final default-on NPU candidate; three samples were 1.011%, 0.863%, 0.860%. The first exceeded 1% by 0.011 percentage points, so report the range. |

The matched pre-graph-to-final graph row reduced median internal idle by
2.525 percentage points and median E2E by 2.06%. Original-to-final was
4.468% to 0.863% internal idle and 3506.60 to 3366.15 ms E2E (4.01%).
The text graph captured 72 layer graphs and replayed 324 times in the final
eight-step run, with no capture failures. Exactness was checked separately:
four requests, including changed prompt/size/CFG and return to the original
shape, matched the original on 73 unrounded floating-point snapshots and final
pixels. The baseline and candidate precision-probe processes both wrote their
complete results before an exit-stage heap corruption (exit 134); performance
processes exited 0. Peak NPU allocation increased by about 31.5 MiB.

Two rejected experiments sharpen the selection rule. Request-invariant metadata
reuse changed the normal CUDA-graph output hash even though a synchronizing
float-snapshot probe passed, so the final product keeps CUDA on its original
preparation path. An opt-in NPU *decode* graph failed in engine dummy-run capture
with `torch_npu` allocator's `captures_underway.empty()` assertion before any
measured request; its code was removed. A text-*encoder-layer* graph uses a
different capture boundary and passed normal and exactness validation. A graph
failure at one boundary is not evidence that all graph candidates fail.

Local evidence is under `outputs/qwen21-bubbles-restart-20261014/` (per-stage
`result.json`, `internal.json`, compressed trace, and exact manifests). The
source changes remain uncommitted; recheck their SHA and working tree before
using these numbers on another revision or platform.
