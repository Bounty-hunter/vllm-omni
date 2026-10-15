---
name: diffusion-pipeline-bubbles
description: Optimize request-internal vLLM-Omni diffusion device bubbles caused by delayed CPU submission or synchronization on GPU, NPU, or other accelerators. Make one experience-led and at most one profiling-led optimization round, validate exact outputs, bubble occupancy and end-to-end latency, then report for the user to decide whether to continue. Also supports diagnosis-only requests.
---

# Diffusion pipeline bubble optimization

Find what prevents the next useful operation from starting. A blank device lane
is a symptom, not a root cause. Cover the host, accelerator, transport, scheduler,
and output path; use the profiler appropriate to the actual platform.
For an optimization request, carry the work through a targeted implementation,
correctness checks and matched before/after measurements. A baseline or list of
suspects is an intermediate result. If a candidate fails, use the evidence to
revise it or reject it explicitly rather than claiming a gain.

## Scope and evidence

Honor the current requested mode: code inspection, trace diagnosis, or optimization.
An explicit diagnosis-only restriction limits implementation until the user
authorizes optimization; recognize later instructions that change that scope
and continue without repeatedly asking for the same authorization.
Diagnosis may use external measurement wrappers that preserve the
computation; identify their overhead and keep them out of performance baselines.
Reuse the environment's execution skill when remote measurement is requested.
For remote jobs, start a short status loop after submission: check the task's
`exit-code`, expected result/trace files and log progress at intervals suited to
the run (typically 10–30 seconds), including while the scheduler queues it.
Stop waiting as soon as an exit code appears, then inspect the complete result;
do not sleep through a finished job or equate a tmux/SSH return with success.
If artifacts appear before `exit-code`, check that the producer has finished
writing before consuming them. On connection loss, recheck scheduler/process
state before resubmitting so the same workload does not run twice.

Read applicable AGENTS.md and freeze the source SHA plus relevant working-tree
changes. Resolve current files and actual call paths; old reports are navigation
hints, not evidence about the current checkout. Recover known workload details
from commands and artifacts before asking for missing information.

Record model/checkpoint revision, input/seed, shape/steps, dtype/backend,
device/topology, batch/concurrency, offload/cache/compile/graph state, execution
mode, and latency/throughput objective. Offline work needs runnable generation
and benchmark commands; serving additionally needs startup and client commands.
Do not require a server for an offline investigation.
When multiple accelerators are available, use one user-chosen platform for
iteration and defer other-platform regression checks until a stable candidate;
do not duplicate every experimental A/B on every platform.

Default to a target of **less than 1% request-internal device-wide bubble**
unless the user specifies another target. Define the target window as first to
last useful device event belonging to one request. Measure the union of compute,
communication and transfer activity across its device streams; report each
stream separately for diagnosis. Report the head, internal gaps, output tail,
and time between requests separately. Never charge a gap between two benchmark
calls to either request.
If the target concerns internal computation, a host-only return tail is a
different scheduling/output concern; retain it as context without silently
adding it to the target numerator. See [measurement.md](references/measurement.md).
For pure bubble elimination, preserve dtype, arithmetic/reduction semantics,
RNG consumption and model/scheduler semantics. Require exact numerical
equivalence; approximate caching, reduced precision or tolerated output drift
belong to a separately authorized optimization scope.

## Two-round optimization budget

Use experience for round 1: identify a small patch that removes or overlaps an
exposed CPU submission delay or host/device synchronization. Reusing request
invariants is relevant only when it eliminates a measured or strongly evidenced
device idle interval; reducing device computation while the device is busy is
outside this bubble target. Check tensor placement and branch reachability before
patching.
A comprehensive profiling investigation is not a prerequisite to implementing
that first candidate. Preserve a runnable original snapshot and fixed workload
so baseline measurements can still be collected after preparing the candidate.
Keep candidates small enough to attribute their effects; an unfamiliar wait
with unclear dependencies requires investigation before removal.
For metadata reuse, key by request and CFG branch and invalidate on shape,
layout, device, or mask changes. Avoid model-global caches for request-owned
values. Check the actual backend and capture/replay state: enabling a graph in
configuration does not imply that the measured path replays it.

Validate round 1 with exact correctness, profiler-free E2E and matched traces.
Use those traces to select **at most one** profiling-led patch for round 2,
targeting the largest remaining exposed internal gap with a defensible cause.
Validate it the same way. A failed or reverted candidate still consumes its
round; do not cycle through substitutes. Stop early if repeated matched samples
already meet the target. After round 2, stop even if bubble remains above 1%.
Return the best validated candidate, residual gaps and evidence to the user;
only a new user request starts another optimization round. Apply this two-round
budget separately to each requested model. Environment setup and failed job
submission are not optimization rounds, but bound retries and report persistent
execution failures.

Label each finding separately:

- `static`: code establishes a behavior and its trigger, but not its cost.
- `observed`: a valid trace shows the behavior on the frozen workload.
- `validated`: a controlled experiment supports the proposed cause/remedy.
- `inconclusive`: coverage, attribution, correctness, or repeatability is missing.

## Follow the execution chain

Map preparation -> admission -> submission -> device work -> transfer -> result
consumption. Distinguish request-level loops from externally scheduled steps,
prefill from decode, and warmed graph replay from capture/fallback. Identify the
thread/process/stream that owns each operation and the real data dependencies.

For each stalled successor, ask in order:

1. Are its required inputs ready? If not, follow the producer or transfer.
2. If ready, has the host submitted the work? If not, inspect CPU work,
   event-loop blocking, locks, IPC, queues, and scheduler decisions.
3. If submitted, what stream/event, collective, or resource dependency delays it?
4. Does the delay lie on the target request/throughput critical path? Is there
   independent ready work that could have run during it?

Prioritize by the overlap of CPU work or synchronization with *device inactivity*,
not by its wall duration or call count. A long host range can be hidden under
queued kernels. Device-side RoPE, masks, attention or repeated calculations are
active work, not bubbles; optimize them under a different performance goal unless
their host preparation delays device submission. Inspect the largest internal
gaps and the distribution of smaller gaps; when no large repeatable internal gap
remains, do not keep optimizing a visible request tail merely to lower the target.

### Branch A: CPU work delays preparation or submission

Inspect repeated tokenization, image processing, metadata/mask/RoPE construction,
Python traversal, tensor allocation, batch/KV repacking, serialization, and
postprocessing. Classify each operation before calling it CPU work: Python may
only prepare arguments and submit a device tensor operation; a data-dependent
host read may then synchronize with that device operation. Record the input and
output tensor devices, host submission interval, device kernels, and any wait.
Separate necessary work from invariants recomputed every step. For a repeated
metadata path, determine whether the exposed gap comes from host computation,
slow submission, redundant device work, or a host/device synchronization. Reuse
request-invariant results when valid; do not prescribe CPU async for device work.
For image diffusion, inspect repeated mask expansion, RoPE generation and token
metadata together. A device mask converted to a host list or used in a
data-dependent output-size operation can impose a per-step wait even when the
Python code looks like preparation. Construct immutable layout results once
per request and CFG branch, then pass them through the denoising loop; keep
step-mode and mutable-mask callers on the original path. Measure each reuse
stage separately, since the next metadata read may become the new wait.
Check whether a supposedly async request handler directly calls blocking code.
Trace the actual output type, thread and IPC path before changing output
transport or conversion. A plausible packing/serialization function may not
run on the measured path at all.

Prefer removing/reusing redundant work before moving it to another thread.
For useful independent work, propose preparation ahead of consumption or bounded
background processing. An async function alone does not parallelize CPU work;
choose I/O async, threads for suitable GIL-releasing work, or processes/native
execution as appropriate. Describe backpressure, memory cost, ownership, request
ordering, cancellation, and the exact point at which results must be consumed.

### Branch B: synchronization destroys overlap

Inspect explicit device/stream synchronization, host reads (`item`, `tolist`,
`bool`, `int`, `cpu`), data-dependent output sizes, blocking collectives, early
future/result waits, and barriers in logging/timing/output handling. Check the
tensor's device and the actual branch before declaring a host/device sync.

For each wait, identify who waits for whom, why it waits now, and the smallest
dependency actually needed. Consider keeping metadata on the host, validating
request-invariant conditions once, deferring consumption, or replacing an overly
broad wait with a precise event dependency. Do not remove required dependencies.
An event wait on a side stream is not automatically a host-blocking wait.
When graph replay moves operators onto additional streams, evaluate device-wide
activity union as well as the original physical stream. A less busy main stream
may coexist with a much smaller device-wide gap.

The two branches interact: a premature sync can expose CPU work that was formerly
hidden by device execution. Adjacent denoising steps usually depend on updated
latents; propose overlap only for genuinely independent preparation, transfers,
or requests. More overlap can worsen latency through resource contention.

## Collect and interpret evidence

Read [measurement.md](references/measurement.md) before collecting traces or
estimating gains. Use a warm, profiler-free baseline and a separate short
diagnostic capture. Inspect instrumentation for inserted synchronization.

For supported Chrome/Perfetto traces, run:

```bash
python .claude/skills/diffusion-pipeline-bubbles/scripts/trace_bubbles.py trace.json.gz --min-gap-ms 0.1 --output bubble-summary.json
```

The helper reports each stream and the union of recorded activity per device
over the same window, lists repeated small gaps and wait-like API events, and
retains timestamps for verification.
The current activity contract includes compute, communication, and transfers;
union overlapping intervals rather than adding their durations. Communication
kernel-internal waits count as active; standalone dependency-wait and control
tasks do not. Keep the same contract across before/after measurements.
Its default activity categories are PyTorch CUDA exports. For a torch_npu
`trace_view.json`, use `--exporter ascend`: it verifies hardware/stream metadata
and excludes dependency-wait tasks. Other exporters require verified mappings;
CPU traces do not establish accelerator idleness.
The helper does not infer dependencies, utilization, critical paths, or savings.
Use the native viewer/API correlations to establish causality.

Count source locations, executed occurrences, and critical-path waiting separately.
An inventory must state the search scope and conditions; do not present selected
examples as an exhaustive synchronization count. Nested APIs and multiple waits
for the same device work must not be added into a speedup estimate.

## Test a hypothesis and report

For each prioritized candidate, record:

| Field | Required content |
| --- | --- |
| Identity | Finding ID, source SHA/path/line, execution mode and trigger |
| Behavior | Host work/submission, device tensor work or wait; frequency, tensor placement and owner thread/stream |
| Causal claim | Delayed successor, necessary dependency, avoidable work/wait |
| Evidence | Static or observed status; artifact, time range, event/API correlation |
| Experiment | One variable, expected timeline change, a result that disproves it |
| Outcome | Baseline/candidate E2E, spread, correctness, memory, remaining limits |

For an optimization result, show the bubble ratio on the *declared target window*
before and after, in addition to E2E latency. Report bubble milliseconds, window
duration, percentage and percentage-point reduction for each relevant stream;
also report the per-device union across streams so work moving between streams
cannot masquerade as eliminating device idleness. If an explicit full-request
boundary exists, retain its head/output-tail breakdown for context even when
the target is internal computation. Use matched boundaries and the measurement
contract in [measurement.md](references/measurement.md).
If only baseline evidence exists, leave the after values unmeasured.

In the final report, add a chronological optimization ledger. For each accepted
iteration, identify whether the idea came from experience-led code inspection
or a specific profiling observation, what changed, and the matched A/B evidence:
device-union bubble before/after on the declared window (ms, %, percentage-point
reduction), profiler-free E2E before/after (ms, percentage reduction), exact
correctness status, and artifact paths. Report an original-to-final comparison
separately from incremental stages. If one A/B includes several changes, credit
the bundle rather than inventing per-change savings; if a stage lacks matched
measurements, mark its gain unmeasured. Do not add percentage-point reductions
from different windows or runs. See the ledger format in
[measurement.md](references/measurement.md#iteration-ledger).

Implement only within the user's authorized scope. If code changes are prohibited,
finish diagnosis with measured baseline and trace evidence where available, and
leave optimization savings unvalidated. Do not silently monkeypatch away a wait
or change a guard in a "diagnostic" run.

After an authorized change, confirm both the intended wait/work reduction and
profiler-free E2E improvement on matched inputs, warmup, placement and load.
Use repeated/interleaved A/B runs; retain individual samples and variability.
Check output correctness with the established model validation contract and
baseline repeatability. Exact-equivalence checks include unrounded floating-point
outputs at the changed computation boundary; identical final pixels alone can
hide numerical drift. Run such checks separately from performance measurements;
see [measurement.md](references/measurement.md#exact-numerical-equivalence).
For a synchronization change, also compare profiler-free output hashes across
separate baseline and candidate processes. A probe that copies intermediate
tensors to the host adds waits and can conceal a scheduling race; passing that
probe alone does not establish equivalence in the normal execution path.
For scheduling changes also check target-load throughput,
tail latency, queue growth, cancellation and peak memory as relevant.

Accept negative results: no exposed bubble, necessary dependency, local-only
improvement, or no statistically convincing E2E benefit. A graph on/off comparison
does not isolate a particular synchronization fix. A no-op model or fake tensor
can validate analysis mechanics, not model performance.

Use [diffusion-perf-opt](../diffusion-perf-opt/SKILL.md) when the diagnosis instead
points to kernel/communication throughput or broad configuration selection.

For Qwen-Image-2.1 investigations, see the conditional-path example in
[qwen-image-21.md](references/qwen-image-21.md). Its source anchors are versioned;
verify them rather than treating the example as an evergreen performance claim.
