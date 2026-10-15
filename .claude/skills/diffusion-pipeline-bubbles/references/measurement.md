# Measurement and attribution

## Baseline versus diagnosis

Save executable commands, source/checkpoint hashes, versions and workload JSON.
Time from the declared input boundary to completed output; timing only Python
submission is not a device latency measurement. Keep output saving outside or
inside that boundary consistently. Separate startup/capture from warmed requests.
Start with one representative scenario and expand only to answer a hypothesis.

For structurally repeated denoising steps, prefer a short run (for example eight
steps) for diagnosis. Keep prefill, initial graph capture and warmed decode
distinct; confirm that step-dependent cache, guidance, callbacks or offload do not
change the path before treating steps as equivalent. Report short-run E2E as that
workload's result, not a measured full-length-generation gain. Repeated requests
still matter for variability even when each request has few steps.

Run warmup until the relevant graph/compile paths are ready, then repeated
unprofiled requests. Three repetitions can reveal a large effect but do not
justify a p95 claim; use enough samples for the requested statistic and effect.
Store raw samples, median and spread in ms, and peak memory. Do not enable a
synchronizing stage timer merely because it is described as lightweight.

Collect a short separate trace after warmup. NVIDIA: Nsight Systems can connect
CUDA APIs, kernels, NVTX and host scheduling; torch.profiler can add operators and
stacks. Ascend: use the available torch_npu/Ascend profiler and verify task, stream
and host correlation fields. CPU-only: collect host execution/scheduling and
queue waits. Do not promise one export schema across these tools or platforms.
Start with minimal instrumentation; capture shapes or stacks separately if needed.
Never upgrade shared dependencies merely to make a profiler available.

In vLLM-Omni, inspect the current docs/contributing/profiling.md and worker control
path. Capture actual workers, not just the parent. Verify start/stop success and
trace completeness, including all relevant ranks for communication hypotheses.
Prefer passive labels for request/stage/step; do not use `.item()` to label a
device timestep. CPU range boundaries describe host execution, not necessarily
device completion. Cross-host timestamps require clock alignment.

## Reading a bubble

- Merge device activity intervals across all captured streams on that device.
  Do not merge different devices into a single utilization number. A partial
  process capture describes recorded activity only, not whole-device idleness.
- For the current accounting contract, count compute, communication, and transfer
  activity together, using their interval union to avoid double counting overlap.
  Retain stream roles for attribution. Communication kernels can wait internally;
  that time remains active under this contract. An optional communication-efficiency
  investigation should inspect all participating ranks separately.
- Use a bounded request/stage window. First-to-last-device-event windows omit
  preparation and final-consumption gaps; report that limitation explicitly.
- For an internal-device-work target, take the first and last **useful**
  compute/communication/copy events within each independently marked request.
  The device activity union between these anchors is the internal denominator.
  Keep the request head and final-consumption tail in separate fields. Control
  events, output saving, and inter-request intervals cannot set these anchors.
- An apparent long blank lane after the last device task may contain host output
  conversion, runtime copies that the hardware exporter does not show, and
  result delivery. Call it a *last-recorded-device-to-return tail* until those
  paths are traced; do not claim it is all removable internal device idleness.
- Inspect repeated sub-ms gaps as well as large outliers. Group by verified
  source/phase; report occurrence count and union time, not only the longest gap.
- CPU ranges overlapping a gap are candidates, not proof. Connect the producer,
  host submission, runtime correlation, device event and consumer dependency.
- Long synchronization duration includes waiting for useful queued work. Trace
  flush, graph capture/warmup, OOM recovery, profiler command-buffer pressure and
  shutdown are different from steady-state inference overhead.

## Estimating gains

### Required before/after presentation

The final optimization report must show how the fraction of the **declared
target window** occupied by device-stream bubbles changes. For whole-request
occupancy, define `W = [request_start, request_return]` using clock-aligned
markers. For internal device bubbles, define `W_internal = [first useful device
event, last useful device event]` separately for each request; report its head
(`request_start` to first event), internal gaps and tail (last event to
`request_return`); include the internal gaps in the numerator, but not the head
or tail.
Let `T = end - start` for the chosen window. State which boundary is the target,
including whether admission/queueing, host preparation, transfers and output
consumption are included. Use the same semantics before and after; the durations
themselves may differ. A worker-only capture without request markers supports
an internal worker-window metric but cannot establish head/tail or whole-request
occupancy. Never silently relabel a whole-request ratio as an internal ratio.

For each relevant device stream `s`, clip its device activity to `W` and union
overlapping intervals. Let `A_s` be the resulting active duration. Report
`bubble_ms_s = T - A_s` and `bubble_pct_s = 100 * (T - A_s) / T`. Include compute,
communication, copy and memset activity consistently, and identify communication
and transfer stream roles. Count communication kernels/device tasks as active,
including waiting internal to those kernels; exclude standalone dependency-wait
and control tasks. Freeze this accounting contract across A/B measurements.
Also union activity across streams on each device and apply the
same formula for a separate device-wide no-activity percentage. A single
stream's gap does not mean the device is idle; do not average stream percentages
to obtain the device-wide result. These are timeline occupancy metrics, not SM
utilization or proof that every gap is avoidable.

Use one isolated request for attribution, or correlate concurrent activity to
requests explicitly. State capture coverage; unrecorded processes/streams cannot
be assumed idle. Preserve startup/warmup exclusions, profiling settings and
workload across A/B captures. Map streams by their verified role because numeric
stream IDs can change between runs. Include all gaps *within the chosen window*
in the ratio; a display threshold for listing large gaps must not change its
numerator. If a benchmark hashes/saves the image between calls, exclude that
entire inter-request interval and disclose it. To assess whether request B hides
request A's output tail, submit B before A returns and separately record B's
admission and first device event, A's last device event and return. A serial
loop provides no evidence of cross-request overlap.

Present a table with these columns:

| Scope / device / stream role | Before bubble ms / window ms | Before % | After bubble ms / window ms | After % | Reduction (percentage points) |
| --- | --- | --- | --- | --- | --- |

Report `before_pct - after_pct` as a signed percentage-point change, together
with matched timeline excerpts showing where gaps changed. Split prefill,
steady decode, step boundaries and stage boundaries when that explains the
result, retaining any separately measured full-request total. Show repeated-sample variation and
profiler-free E2E latency separately. Missing after measurements stay unmeasured;
do not fill them with predicted ratios or report a decrease without evidence.

### Iteration ledger

Preserve a short row for every accepted change, including the first
experience-led patch and each later profiling-led patch. Record the concrete
observation that selected it and use the immediately preceding validated
candidate as its reference when a matched A/B exists. A final row may compare
the original baseline with the final candidate; label that as cumulative, not
as another incremental gain.

| Stage / origin | Change and evidence that selected it | Window / device union bubble before → after | Bubble reduction | Profiler-free E2E before → after | E2E reduction | Correctness / artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| Experience-led 1 | Request-invariant work reused; source inspection and later trace confirmation | ms and % on the declared window | percentage points | median ms on matched runs | percent | exact check and trace/benchmark paths |
| Profiling-led 2 | Largest exposed gap and its correlated host/device cause; targeted change | ms and % on the same window | percentage points | median ms on matched runs | percent | exact check and trace/benchmark paths |

Use `100 × (E2E_before − E2E_after) / E2E_before` for the E2E reduction.
Keep profiled occupancy and unprofiled latency as separate measurements. If a
stage combines changes, name the bundle and leave individual contributions
unknown. Different A/B sessions may have different reference values even on
the same workload; show each session's own before value and do not sum its
gains with another session's as if they were one continuous trace. If the
declared window changes, add a new column or ledger and do not combine those
bubble percentages. Include rejected or inconclusive candidates briefly so
the final narrative distinguishes tested gains from hypotheses.

The bundled analyzer provides per-stream statistics and their per-device union
over the same window. Supply an explicit `--window-us` for the whole-request
metric; its default first-to-last-device-event window can support an internal
metric only after verifying that the selected events belong to one request.
Verify stream identities, activity categories and request/clock correlation
against the native profiler before interpreting either statistic.

### Ascend export mapping

For torch_npu `trace_view.json`, use `--exporter ascend`. In the verified
torch-npu 2.10.0.post4 export, host CANN APIs, actual device tasks and derived
overlap-analysis rows all lack `cat`. Selecting the empty category as device
activity would count host work and derived `Free` intervals as busy time.
The adapter instead requires `Ascend Hardware` process metadata, its `NPU N`
physical device label, and matching `Stream N` / `Physic Stream Id` metadata.
It includes known compute and DMA task types and rejects unknown task types.
Communication implemented by those hardware task types is included; task names
such as HCCL are not exclusion criteria. Verify any new task types against the
exporter before extending the mapping. Derived `Communication` lanes are not
additional hardware activity. CUDA exports already include communication kernels
such as NCCL through the default `kernel` category.

`EVENT_WAIT` can span milliseconds while another stream executes. Exclude it
from compute/communication/copy occupancy, and retain its type/count as dependency
evidence. Exclude event-record/profiling/control tasks and derived overlap lanes
too. A control-only stream is not a compute stream with removable 100% idleness.
Review the exported stream task inventory before assigning roles.

NPUGraph exports also contain `MODEL_EXECUTE`, `NOTIFY_WAIT` and `NOTIFY_RECORD`.
Exclude these dispatch/notification tasks, counting the separately exported
compute/copy operators on graph streams. Report both the original compute stream
and graph streams with the device union; a graph moving kernels to another lane
must not be presented as a single-stream occupancy gain.

Some exports store epoch-sized microsecond timestamps as strings. Preserve
precision when subtracting a common origin before analysis. Compare multiple
clock anchors across the capture: Ascend export time and host monotonic time
may need a small affine rate correction, not just one constant offset. Report
the fit constraints and boundary uncertainty; reject inconsistent anchors or
uncertainty too large for the reported percentage-point precision.

### Savings bounds

Do not infer savings from counts or sum synchronization API durations. Even a
small scalar read may drain a long queue, most of which was necessary compute.
On a supported critical-path analysis, define B as the union of plausibly
removable exposed delays in a request of duration T. Removing only these delays
has an optimistic latency-reduction ceiling B/T and speedup ceiling T/(T-B).
State assumptions; remaining dependencies/contention may lower the saving.
With incomplete coverage or no counterfactual, report the estimate as unknown.
Never multiply a profiled idle ratio by an unprofiled E2E time to predict gains.

### Target and stopping criterion

Unless the user specifies otherwise, target below 1% bubble in the request's
first-to-last-useful-device-event window, using the union of activity across
the device's streams. Report individual streams for diagnosis so moving work
between lanes cannot manufacture a gain. A dedicated copy/control stream need
not be busy throughout the target window. This target applies to the declared
workload, not to all possible resolutions or deployment modes.

Judge the target with repeated samples and boundary uncertainty, together with
profiler-free E2E and exact-equivalence checks. Keep steps, the *declared* window,
activity categories and useful workload fixed: lengthening compute, adding dummy
work, silently excluding gaps **inside** that window or counting standalone waits
as work cannot qualify as eliminating bubbles. If the user specifies a different
window, record that change and show both metrics where measured. If necessary
dependencies or
measurement limits prevent reaching the target, report the residual and
supporting evidence rather than silently weakening the threshold. Complete no
more than one experience-led and one profiling-led optimization round per model;
after round 2 report the best validated result and let the user decide whether
to authorize further rounds, even if the target remains unmet.

### Exact numerical equivalence

For pure bubble elimination, require bitwise equality on matched inputs and
execution configuration. Preserve weights, dtype, rounding/reduction semantics,
RNG consumption, scheduler and output contract. Operator or kernel changes must
also satisfy bitwise equivalence. Reorder only independent work and preserve data
dependencies. A change in floating-point rounding is a
numerical change even if image-quality metrics look unchanged.

In a separate correctness run, first establish baseline repeatability. Compare
shape, dtype and raw tensor values at the affected stage boundaries. For a
denoising-loop change, compare each step's noise prediction and updated latents,
then the floating-point decoder output and final output. Snapshot tensors before
their buffers are reused; do not compare only rounded/clamped image pixels or
relax to an SSIM/PSNR/tolerance check. Instrumentation, copies and completion waits
needed for these comparisons must not enter the performance baseline or trace
used to claim gains.

If the baseline itself is nondeterministic, investigate and report that obstacle.
Do not call exact equivalence established or substitute a tolerance without the
user changing the acceptance contract. Changes to deterministic execution settings
require re-establishing both correctness and matched performance baselines.

Compare actual before/after performance without the profiler. For asynchronous
execution, additionally validate event ordering,
buffer lifetime and absence of cross-request contamination. An output that looks
plausible is insufficient evidence of correctness.

## Analyzer limits and checks

`trace_bubbles.py` reads duration events in microseconds from Chrome JSON/JSON.gz.
It groups by explicit device ID when present, otherwise by exported device-lane
pid (with a warning). It cannot align/merge separate trace clocks. Select activity
categories from actual export contents; never classify CPU launch/enclosing
annotations as device work. Nsight reports can instead be inspected in their
native viewer or exported with the installed version's supported tooling.

The report's host wait totals include nested/overlapping events; only per-name
statistics are supplied, with no global savings total. Candidate host overlaps
do not perform causal attribution. Missing device events yield no idle metric.

Before relying on a new parser/export mapping, verify cases where another stream
fills the gap, different devices overlap, a long host wait overlaps useful
compute, edge gaps are omitted without an explicit window, and CPU-only traces
contain no device evidence.
