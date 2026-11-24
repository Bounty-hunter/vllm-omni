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

## Wider shape coverage

[COVERAGE-RESULTS.txt](COVERAGE-RESULTS.txt) records the broader FLUX.2-klein-4B,
Qwen-Image-Edit-2511, Wan2.2-T2V-A14B and H3-Image comparison. It includes
small-canvas controls, reference-image inputs, videos, independent 30-round
confirmations, and whole-attention controls with both owned and borrowed
outputs. Raw samples, source identities and paired intervals are in
[coverage-summary.json](evidence/coverage-summary.json); compressed logs,
configs and patches are in [coverage-raw.tar.gz](evidence/coverage-raw.tar.gz).

The frozen `*-coverage-confirm.json` and `h3-policy-confirm.json` recipes
default to the original single group. Only explicitly selected cases override
that setting. Small-case A/A differences are timing noise. These recipes
qualify specific offline graph cases; they do not implement a production
shape selector or establish that every split is nonregressing.

The current Qwen and FLUX pipelines ignore `vae_patch_parallel_size=2`, as
their registry warnings confirm. Their results use SP2 with no VAE patch
parallelism. Wan-A14B configures distributed tiled VAE4, and H3 audits its
native VAE world size of four. The configured VAE degree alone does not
establish how many tiles every small frame distributes.

The `coverage-core-sp2.json` and `coverage-core-sp4.json` controls separately
time peer communication and attention splitting. The report labels request
communication shares as estimates from these isolated controls, rather than
in-request measurements. Total peer bytes remain unchanged.

H3 SP4 at true 1024² also exposes a packed FA3 auto-SplitKV precision limit.
The failed split screen and diagnosis are retained in
[coverage-diagnostics.json](evidence/coverage-diagnostics.json), with scripts
and logs in [coverage-diagnostics.tar.gz](evidence/coverage-diagnostics.tar.gz).
Setting `num_splits=1` restores exactness in the pure-attention control;
the complete model with that setting has not been qualified. The backend
default remains unchanged and the small H3 recipe keeps whole attention.

## Learning from real denoising prefixes

The opt-in `real_prefix` recipes move learning into actual request execution.
The first real call to each DiT component runs whole attention and discovers
its execution signatures. Before its second real call, all discovered
candidate buffers and events are prepared together. Only the first 24 cores
of each signature in that second call compare whole attention with one
predicted split. Every layer executes one candidate; its output feeds the
next layer. Both eager component results return to the real pipeline.
Wan-A14B's two expert components learn separately at their own first two
calls; these are not necessarily global denoising steps one and two.

The two actual calls replace the graph owner's previous warmup opportunities.
The generic adapter retains its original synchronization-debug probe with
sampling disabled, then uses its existing capture synchronization to read
events, gather maximum-rank durations and freeze the policy. H3 replaces its
two eager warmups with the two actual calls. These adapters retain one graph
validation replay. The later Qwen2.1 decode-block adapter captures directly,
without an auxiliary synchronization probe or validation replay; its following
real decode steps replay that graph. Execution budgets distinguish these paths.
The second actual eager result is returned after capture and any validation;
the validation output does not substitute for it. Later calls directly replay
the frozen graph. There is no additional probe forward, candidate graph or
per-layer synchronization. The remaining layers after the sampling window
use whole attention until the capture boundary completes the decision.

Both `serial1` and `adaptive` in `prefix-*-matrix.json` use this same delayed
capture path and shared output ownership. `serial1` disables candidate learning.
The `legacy1` smoke arm additionally checks equivalence to the previous whole
path at seeds 42/43/44. Qwen's audited first-request Python core count falls
from 300 to 240; the latter includes two actual eager calls, the original
debug probe and capture recording. H3 still records 150 Python core calls,
but now executes two actual eager steps plus four actual graph steps, and
retains its separate validation replay. Subsequent requests replay every step.
These counts must be compared between matching controls, rather than counting
capture recording as another executed model forward.

A short prefix prioritizes one promising split to increase paired evidence
and avoid allocating a second split's CE buffers. The original confidence
thresholds remain: at least six complete valid cycles, 80% positive paired
differences and the absolute/relative/noise margins described below. Rare
signatures, insufficient fully queued samples and uncertain gains stay whole.
Candidate generation remains a CPU proxy; odd widths remain eligible.
No extra calls are manufactured to fill the sample budget.

The completed 26-case matrix covers Qwen editing with one/two references,
FLUX text and reference-image shapes, H3-Image from 512 through 4096,
Wan-A14B from 9 through 65 frames, and Wan-TI2V-5B at four video shapes.
[PREFIX-RESULTS.txt](PREFIX-RESULTS.txt) records every result, including
independent 30-pair confirmations, single-block fallbacks, negative screens
and first requests in separate model processes. The earlier results below
qualify the earlier warmup-sampling method.

The independent confirmations repeat selection from the real prefix in a new
process. These cases selected the same split as the initial matrix and had
positive paired 95% bootstrap intervals in the confirmation:

| Model and output | Heads per rank | Request latency reduction | Faster pairs |
| --- | --- | --- | --- |
| H3-Image 3072² | 5+9 | 3.79% | 30/30 |
| H3-Image 4096² | 3+11 | 3.40% | 29/30 |
| Qwen-Image-Edit-2511 2048², one reference | 4+8 | 1.56% | 21/30 |
| FLUX.2-klein-4B 1024², one reference | 1+11 in single-stream blocks | 1.68% | 20/30 |
| Wan2.2-T2V-A14B 480×832, 33 frames | 3+7 in each expert | 2.83% | 29/30 |
| Wan2.2-T2V-A14B 480×832, 65 frames | 5+5 in each expert | 1.52% | 28/30 |

The 65-frame initial eight-pair screen was 2.36% slower; both observations
are retained. Qwen 1536² with one reference and FLUX 2048² with one reference
had positive median ratios but confirmation intervals crossing zero, so
they do not establish stable gains. Whole-attention fallbacks are A/A
comparisons even when their measured difference is positive. Wan-TI2V-5B
does not establish an overlap gain with this bounded selector.

The controls use FP8 DiT for Qwen, H3 and Wan, and BF16 for FLUX. Qwen and
both Wan models use eight denoising steps, FLUX four, and H3-Image six.
Attention remains BF16. Qwen/FLUX VAE2 is ignored by their registry; H3
audits native VAE4, and Wan uses the configured distributed VAE described
above. Communication shares reuse earlier isolated controls at matching
geometry and are estimates, not in-request profiles. Peer bytes remain
unchanged; the schedule reduces visible communication on the critical path.
These estimates cover attention-core peer traffic; they exclude communication
inside the text encoder and VAE.
Output geometry alone does not describe attention load: at 480×832 and
65 frames, the audited global sequence is 26,520 tokens for A14B and 6,630
for TI2V-5B. The report includes actual attention geometry for every case.

First requests include candidate preparation, timer records, trials and graph
setup. The report separates these observations from steady performance;
no additional forward does not imply no additional first-request latency.
For example, the first separate-process Qwen 1536² reference-image pair
measured 6494 to 6925 ms, with reported peak memory 56498 to 57872 MB.
The report retains both arm orders and distinguishes these observations from
pure incremental preparation cost on a shared host.
Low-confidence shapes retain whole attention. These are fixed-case offline
adapters with graph-owned statistics, without cross-expert aggregation or
a production request-shape controller.
Insufficient fully queued eager samples can miss shapes where earlier fixed
partitions improved performance. Such a fallback limits selector coverage;
it does not establish that graph execution cannot benefit from splitting.

Sources and compact results are in
[prefix-summary.json](evidence/prefix-summary.json), request records and rank
audits in [prefix-raw.tar.gz](evidence/prefix-raw.tar.gz), and full observations
and verification scripts in [prefix-analysis.tar.gz](evidence/prefix-analysis.tar.gz).
The benchmark's `warm` record labels its first complete `generate` request.
Selection uses actual denoising layers inside that request, with their outputs
feeding the pipeline; auxiliary graph probes supply no learning observations.
Packed FA3 precision must be checked for every newly exercised layout;
a numerical mismatch is a failed case, not an acceptable speedup.

## Three-group priority with a two-group fallback

The `head3-*-matrix.json` recipes explicitly enable
`candidate_family="three-then-two"`. The CPU model proposes one symmetric
three-group partition and one two-group partition, alongside whole attention.
It estimates query-tile work and stream dependencies; its wave counts are not
observed kernel work assignments. Odd head counts remain eligible. Real core
observations decide whether either proposed split qualifies.

These recipes reuse the first two actual calls to each component. Qwen2.1
keeps its original prefill and learns from actual decode calls. The second
call has a 32-layer budget per execution signature. Its first two layers run
whole attention without sampling, allowing launch lead to build after resource
preparation. The remaining 30 layers supply eight whole/two-group pairs and
seven whole/three-group pairs, with rotated/reversed local controls. This
changes no model forward count. Shorter signatures retain the existing
two-group or whole-attention fallback.

The later Qwen recipes explicitly set `real_prefix_calls=3` for every arm,
including whole attention. Call one discovers; the next two actual decode
calls provide separate whole/triple and whole/double pairs. A 32-layer
component has two initial untimed whole layers, 15 triple pairs, and 16 double
pairs. All buffers and 62 event pairs are ready before call two. Results are
read once at the capture boundary after call three. Each actual eager output
continues into the pipeline. There is one capture, followed by four replays
in a seven-decode-call first request: `3 + 4 = 7`, just as `2 + 5 = 7` before.
The extra eager call adds first-request host overhead and replaces one replay;
capture recording is not an executed model forward. Ordinary two-call recipes
retain their previous sampling order and count.

The six-valid-pair confidence gate is unchanged. A qualifying three-group
partition has priority; the two-group partition replaces it when qualified
observations establish an additional gain. Insufficient evidence retains
whole attention. Whole fallbacks are timing controls, not overlap speedups.
Early eager observations can differ from graph execution, so full-request
measurements remain necessary. These recipes do not guarantee a speedup for
every shape or a zero-cost first request.

Candidate preparation shares one symmetric output slot for the same owner and
complete layout, independent of head partition. Different explicit owners
retain separate slots. No peer-ready or output-completion fence is removed;
consumers remain ordered on the owning compute stream. `check_plan_storage.py`
checks resource counts, and `graph_slot_stress.py --mixed-partitions` checks
delayed consumers across partitions and resident graphs.

`--worker-profile` marks separate diagnostic cohorts using
`head-request-profile.patch`. Each fixed-partition profile records two complete
requests after two warmups, with CPU/CUDA activity and Python stacks. Keep
physical GPU identities, software, checkpoint, shape and output ownership
matched across variants. Report pure outgoing P2P activity separately from
barriers, local staging and copy gaps. Profile durations are not unprofiled
request-latency results. First-prefix CPU/phase instrumentation is also opt-in
and is not installed in ordinary performance runs.

## CI shapes and schedule controls

[CI-RESULTS.txt](CI-RESULTS.txt) retains all later Qwen-Image-2.1,
Wan-A14B T2V/I2V, HunyuanVideo-1.5 T2V/I2V, and H3 image/video results.
It includes common 1024 image/video shapes, CI workloads, failed attempts,
whole fallbacks, first requests and every additional comparison arm.
LTX2 could not initialize with the installed diffusers decoder; it has no
performance sample. Shared dependencies were not upgraded.
[ci-summary.json](evidence/ci-summary.json) and
[ci-evidence.zip](evidence/ci-evidence.zip) retain source revisions, raw
samples, all-rank audits, integration patches and verification scripts.

The bounded prefix can compare whole attention with one explicitly supplied
two-, three- or four-group partition. The generic recipes use
`candidate_partition`; H3 reuses `head_partition`. A small final group can
cover the preceding large group's return, but adds another attention call,
copy submissions and a forward-ready fence. The reports separate the pure
attention splitting cost from complete-core and complete-request savings.
The default CPU candidate generator still predicts two groups. These supplied
multi-group candidates are controlled experiments, without a general automatic
selector among two, three and four groups.

`candidate_family="symmetric-ends"` instead compares whole attention with
`1 + (H-2) + 1` and `2 + (H-4) + 2`, where the middle group must be at least
as large as either end. For eight heads per rank this is whole, 1+6+1 and
2+4+2. The existing first 24 real prefix cores rotate and reverse these
three choices eight times each. The family is mutually exclusive with an
explicit partition and remains opt-in; the default two-group generator is
unchanged. The original sample-confidence and all-rank qualification checks
still apply. Insufficient real-call budget reduces the candidate set rather
than adding a forward. This bounded family does not search every possible
end width for larger head counts.

The [symmetric-end results](SYMMETRIC-ENDS-RESULTS.txt) cover 297 complete
Turbo requests and nine isolated core controls. Each shape uses 30 paired
rounds with whole, the previous learner and the symmetric family:

| Canvas with one reference | Previous choice / reduction | Symmetric choice / reduction |
| --- | --- | --- |
| 1024 square | whole, A/A | whole, A/A |
| 1536 square | 3+5, 0.96% | whole, A/A |
| 2048 square | 2+6, 1.80% | 2+4+2, 1.92% |

At 2048 the extra paired saving over the old learner has a 95% interval of
-7.62 to 12.89 ms, so an additional E2E gain is unconfirmed. The new family's
42.70 ms saving is about 35.4% of the frozen barrier-excluded P2P-window
estimate; the 50% E2E target remains unmet. At 1536 all three choices execute
eight times, but zero complete cycles qualify as fully queued on every rank.
Whole fallback reflects insufficient evidence, not proof that three chunks
cannot help graph execution. Candidate preparation takes about 1.1-1.3 seconds
in these family trials, including allocation and distributed alignment.
First requests, all-rank counters, failed qualification and negative results
are retained. There are no extra forwards or steady probing operations.

The learner freezes its communication schedule with the selected partition.
The schedule participates in the rank agreement and policy cache key; changing
it within a generic graph case raises instead of replaying an incompatible
graph. `overlap-return` and `overlap-staging` remain optional ablations.
`stage-gated` additionally orders the next prefetch after compute is ready for
the current attention, without waiting for that attention to finish. Its
dependency handles are realized during preparation. All peer fences and CE
source lifetimes remain intact; see [FENCE-PROOF.txt](FENCE-PROOF.txt).
An extra dependency can remove useful lookahead. These alternatives are
measured against the same captured partition and are not default schedules.
The separate CUDA API capability check reads graph-exec instantiation flags
of one: `UseNodePriority` (eight) is absent at compute priorities zero and
minus one. It establishes no model speedup from changing stream priorities.

The optional `HCMS_COPY_SUBMITTER` library batches host submission of the
original `cudaMemcpy2DAsync` calls. It does not add a GPU kernel, change copy
bytes, or use an incompatible batch-copy CUDA API. Experiment payloads build
`copy_submitter.cpp` with `c++ -O2 -std=c++17 -shared -fPIC`, then record build
time and source/binary hashes. The environment variable points to that
task-local library; there is no automatic shared installation. Its purpose
is to reduce CPU submission gaps during the actual prefix, so that qualified
observations better represent graph execution. Its build cost and candidate
preparation are reported separately from steady request latency.

H3 video compares video pixels and FP32 audio bytes at three seeds. The
original audio decoder produced different waveforms even for identical latent
inputs in an A/A run. Accepted video comparisons explicitly set
`HCMS_AUDIO_DETERMINISTIC=1` for every arm and the eager reference. This wraps
only audio decode in cuDNN deterministic mode with benchmarking disabled.
It is a deployment difference, not a relaxed output tolerance. H3's eager
reference uses whole direct CE; the Qwen/Hy `reference_whole` arms use original
Ulysses. The reports distinguish these references.

Runtime audits verify actual FP8 module counts, and H3 verifies native VAE
world size four. Qwen2.1 uses `DistributedAutoencoderKLQwenImage21` with VAE4
configured; the earlier Qwen/FLUX VAE2 warning does not apply to this model.
Per-request active VAE tile allocation was not recorded. Historical
"communication shares" use frozen matching-layout peer controls and actual
graph core occurrences. Those controls include peer barriers and stream
dependencies: they are peer-exchange envelopes, **not pure P2P shares**.
They exclude Qwen2.1's original prefill and text-encoder/VAE communication.
The historical numerical denominator is retained with this corrected label;
it must not decide whether savings reach half of pure P2P time.

The Qwen2.1 1536-square reference-image anchor separately measures CE-only
event windows with `ci-core-qwen21-copy-window.json`. The markers surround
only outgoing copies on the communication stream. Required peer barriers
remain after the end marker, and local staging remains outside the window.
Results are read after the control's existing completion synchronization.
These windows still include marker and inter-copy scheduling overhead;
they are neither a memcpy-kernel busy union nor an in-request measurement.
Other geometries retain an unavailable pure-copy estimate until measured.

`stage-defer-fence` is an opt-in causal control. It overlaps the next group's
copies and previous group's return with current attention, but waits for the
current produced event before the next forward-ready peer barrier. It reuses
the existing events, buffers and byte counts. A longer gap can outweigh any
reduction of concurrent-fence interference; the schedule is not a default.
The matched `ci-qwen21-defer-events.json` diagnostic forces the same partition
with equal marker counts on all split arms. Only the uninstrumented
`ci-qwen21-defer-confirm.json` qualifies request performance, and its real
prefix learner may conservatively fall back to whole attention. Different
selected partitions cannot establish a pure scheduling speedup.

The later anchor uses official `Qwen/Qwen-Image-2.1-Turbo` revision
`fc13ba1986fcb905854dc8de2f52eee395f49b5b`, its saved eight-sigma schedule and
CFG one. Original prefix KV caching is retained. Every request has 224 decode
cores, plus the original prefill. Three-seed output equality also covers the
original Ulysses eager reference. The complete budget and negative controls
are in [QWEN-ANCHOR-ANALYSIS.txt](QWEN-ANCHOR-ANALYSIS.txt).

Thirty-pair Turbo results with one reference image:

| Canvas | Captured heads | Whole / adaptive ms | Latency reduction |
| --- | --- | ---: | ---: |
| 512 square | whole | 544.991 / 542.108 | A/A |
| 1024 square | whole | 887.257 / 885.453 | A/A |
| 1536 square | 3+5 | 1387.173 / 1367.366 | 1.43% |
| 2048 square | 2+6 | 2213.527 / 2177.634 | 1.62% |

An independent 30-pair 1536 confirmation reduced 1391.992 to 1373.226 ms
(1.35%). Pure-copy-window estimates cover all four Turbo layouts. At 1536,
the estimated share is 5.04% of E2E, and the measured saving is about 28% of
that budget. The target of half the pure P2P budget is not established.
Splitting identical bytes into 3+5 increases isolated serial copy windows
from about 313 to 353 microseconds. Copy latency does not scale strictly
with head width. These are window estimates, not hardware busy-time sums.

`HCMS_CORE_EVENT_SCOPE=all-cores` additionally requires `HCMS_CORE_EVENTS=1`.
This diagnostic records exactly two external events per complete core and
one pair around the component, with identical counts for whole and split.
It reads all 32 layers of the last decode replay at request completion;
normal recipes retain zero steady diagnostic events. In the four-pair
diagnostic, cores save 3.36 ms per step, while other component intervals grow
0.99 ms. The resulting 2.37 ms step saving predicts 16.6 ms across seven
decodes, versus 19.8 ms measured E2E. Last-step extrapolation, marker effects
and other request stages remain uncertainty. The outside intervals include
device gaps; they do not establish a slower FFN kernel or a hardware cause.

`overlap-input` moves only the next local input staging to communication;
local output staging keeps its original ordering. The same captured 3+5
partition reduced 1402.296 to 1377.497 ms with the original schedule, versus
1385.492 ms with input overlap. The input change and deferred-fence change
remain rejected optional controls. All buffers, bytes, required peer fences
and output ownership are preserved. Invalid multi-group staging without an
overlapping plan is rejected before device submission.

The verification checks actual captured widths, identical rank decisions,
normal prefix call counts, and no added steady API synchronization,
sampling timers or Python attention execution. It reports reused smoke graphs
separately from newly captured first shapes. There is no extra probe forward;
candidate resources, event records and trying a slower plan can still increase
first-request latency. This remains a fixed-case offline graph experiment,
without default serving or concurrent-request qualification.

## Previous warmup-sampling prototype

The opt-in `autotune` variant implements a graph-owned lazy experiment in
`lazy_head_policy.py`. It observes only shapes reached by actual requests.
The first existing reference pass discovers attention execution signatures;
the original probe and warmup passes supply real core observations. Each
attention call executes one candidate and returns its output to the model.
It adds no model forward, dummy kernel, candidate graph or shape sweep.
Models without existing pre-capture eager opportunities cannot learn this way.

A CPU query-tile/SM-wave and communication proxy proposes whole attention
plus at most two unequal splits. Odd head counts remain eligible. This model
generates candidates; its constants are not fitted online and its waves are
not measured launch grids. Real complete-core observations select among them.
Signatures include device/backend versions, SP, QKV shapes/strides/dtypes and
metadata, including joint and packed layouts. Same-signature layers share
samples within their graph owner; subsequent requests replay the frozen graph.
There is no cross-process policy persistence or cross-expert aggregation.

Candidates rotate and reverse identically on every rank. Buffers and timing
events are prepared before sampling, using the existing plan/slot caches.
Timing spans input exchange, attention, output exchange and their dependencies,
excluding projection and FFN. A nonblocking start-event readiness check rejects
cores not fully submitted before the GPU started them. A complete rotated
cycle is discarded if any candidate on any rank fails that check. After the
original synchronization boundary, one CPU-group gather combines maximum-rank
durations. Selection needs at least six valid cycles, 80% positive paired
differences, and a margin above 5 us, 1.5% of core time and twice the paired MAD.
Otherwise it freezes whole attention. A separate preparation header gather
checks rank agreement before allocating candidates. Neither is per layer.

The selected signature/partition digest enters the existing graph key. Timing
event references are then released. Capture and steady replay contain no
learning timers, event queries or exploration gathers. Existing peer fences
and pipeline operations still execute. Candidate buffers stay in the existing
bounded native caches until graph-owner cleanup, increasing memory residency.
Masked/opaque metadata remains whole; small packed varlen layouts remain whole
because of the FA3 precision limit
described above. A new signature unexpectedly reached after discovery fails
closed rather than allocating or silently changing a captured layout.

`lazy-*-cold.json` repeats the 13 workload cases in both arm orders with shape
changes invalidating graphs. Both arms use borrowed output slots. Generic
whole and adaptive arms share the same manual capture preparation. These
are shape-cache misses within one model process; they are not independent
process startups. `lazy-*-startup-{base,auto}.json` supplies the separate
single-arm processes used to measure the first request after model loading.
First-request wall time includes exploration, preparation, audit and graph
setup. No extra forward does **not** imply no extra first-request latency.

Use the same isolated-checkout patch composition as the unprofiled matrix
below, choosing the corresponding `lazy-*.json` recipe. For example, after
applying the generic and Qwen block adapters on two scheduled devices:

```bash
export HCMS_LAZY_COUNTERS=1 HCMS_MATRIX_AUDIT="$RUN_DIR/results/audit"
python "$PROBE_DIR/check_lazy_policy.py"
python "$PROBE_DIR/check_lazy_runtime.py"
python "$PROBE_DIR/model_matrix.py" --model "$QWEN_EDIT_MODEL" \
  --config "$PROBE_DIR/lazy-qwen-cold.json" --out "$RUN_DIR/results" \
  --rounds 8 --warmups 1
```

`HCMS_LAZY_COUNTERS=1` installs test-only API counters before importing the
policy. They verify actual core/body calls, timing, queries and synchronization;
they are not enabled by normal execution. The audit field `eager_forwards`
also counts Python capture recording, which must be separated from executed
eager passes and graph replays when interpreting forward counts.

Results, first-request costs, missed opportunities and evidence identities
are in [LAZY-RESULTS.txt](LAZY-RESULTS.txt). This remains an explicit offline
prototype; it does not enable automatic selection in the library default.

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

## Native serving CI experiment

### Automatic-policy acceptance

`check_auto_policy.py` is the automatic-selection acceptance suite. Its inputs
in `auto_policy_cases.py` describe QKV shape, strides, prefixes and masks only;
they do not prescribe a partition or expected winner. The ordinary serving
bridge learns from declared component calls, then freezes. Test all-rank
agreement, selected-output correctness, native NCCL restoration, exact declared
core/head work, and zero growth in frozen sampling/preparation/control counters.
Test-only guards reject host waits and per-core control gathers during sampling
or frozen execution; remove these guards and work counters before timing.

After applying the existing native/serving patches in an isolated checkout and
building the copy submitter, run inside a scheduler allocation:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/diffusion/head_overlap/check_auto_policy.py --world 4 \
  --suite typical --out /data/dyy/logs/TASK/automatic
```

Use `--suite wan --world 2` for the two observed Wan CI layouts. `--case` selects
inputs for a smoke test, never head cuts. The older fixed-partition schedules
remain independent numerical/scheduling diagnostics, not this acceptance gate.

The default performance budget is at most **1% higher steady mean latency per
input layout** than ordinary NCCL. Measure full eager cores and serving dispatch
in balanced ABBA/BAAB blocks on the same allocation. Keep every sample and
aggregate maximum-rank windows. The paired ratio's upper 95% bootstrap bound
must be at most 1.01; existing native/native windows must also resolve the 1%
budget. Report `inconclusive` when noise crosses that boundary, not a pass.
Neither a failed gate nor an inconclusive measurement may change the selector's
frozen answer or trigger retries until a favorable result appears.

Initial real calls and their costs are reported separately; the suite does not
promise a 1% bound on cold startup, individual requests, or model E2E latency.
Its same-process first-call numbers do not replace the native CI's independent
cold-cache comparison. The selector still compares whole CE and split CE;
acceptance against NCCL deliberately checks the additional transport boundary.
Packed fixtures publish canonical host-known padding lengths, as H3 does.

`serving_probe.py` connects the existing learner to ordinary serving requests,
only through `serving-probe.patch` and `native-probe.patch` in an isolated
checkout. Baseline (`--ulysses-head-chunks 1`, the default) installation is a
no-op. The enabled arm uses `--ulysses-a2a-permute --ulysses-head-chunks 2`.
CE is an explicit feature dependency: a native NCCL baseline versus this arm
measures CE plus overlap, not overlap in isolation.

Retain each native CI configuration's precision, HSDP, offload, compilation,
graph settings and workload. No manual DiT graph is installed. The first real
component call discovers layouts; the next real call samples at most 32 cores
per signature. Outputs flow normally into denoising. At existing component-call
boundaries, each rank checks event readiness without waiting and the control
plane gathers either completed timings or a pending sentinel. All ranks defer
if any samples remain pending; otherwise they freeze and use the decision in
later steps of the same request. Request completion remains a strict fallback
boundary. No probe forward, per-layer collective or device synchronize is
added. Frozen calls have no timing or readiness probes. Unknown
layouts use the original strategy. When a component has no eligible split, or
all of its frozen decisions retain whole attention, subsequent component calls
restore ordinary NCCL strategies. This also restores the original compiled
attention route instead of retaining the experimental single-group CE core.
Strategies are prepared once and switched only between component calls; new
signatures can restore overlap. Shared-output single-group CE controls remain
available during qualified sampling to distinguish transport and split gains.
The query-tile occupancy proxy filters undersized symmetric ends; it predicts
candidates and is not proof of a kernel's actual launch grid. If the proxy
cannot propose any split, serving rejects admission before allocating CE trial
buffers or timers. Initial discovery
and compilation still cost time and must be included in first-request results.
This adapter accepts uncaptured DiT paths;
it does not change an existing graph configuration to make learning possible.

Run the complete native baseline benchmark, then the enabled benchmark. Keep
the native warmup and measured request counts (including H3's one warmup and
three requests), and report arithmetic `mean_e2el_ms` only after all configured
requests complete successfully. A partial or failed run is invalid, even if it
contains latency samples. Speedup = baseline mean / overlap mean. Do not interpret fallback
result templates with zero completed requests as performance measurements.
Record the original configuration, source hashes, minimal feature dependency
delta, first-request learning and all-rank decisions alongside the results.

AI assistance: Codex prepared the implementation, experiment scripts and
analysis. The branch is retained for investigation; it is not PR-ready
production integration.
