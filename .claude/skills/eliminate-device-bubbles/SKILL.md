---
name: eliminate-device-bubbles
description: Diagnose and eliminate GPU/NPU execution bubbles in PyTorch workloads by correlating Python/CPU stacks, launch lead, copies, synchronization, and device traces. Use for profiler-based analysis or lossless performance work when an accelerator timeline has idle gaps or a sustained low-density region; do not use as the primary workflow for kernel optimization when the device is already continuously busy.
---

# Eliminate Device Bubbles

Keep the accelerator fed by finding why host-side work stopped staying ahead of device execution. Optimize the earliest demonstrated cause without changing model semantics.

## Use the right causal model

- In a healthy steady-state pipeline, Python/CPU enqueues device work noticeably before the device executes it. The resulting queue depth keeps the device busy.
- For a correlated device operation, define its approximate **launch lead** as `device start - host enqueue end`. The clocks and correlation data must be comparable. A comfortably positive lead means the operation was queued in advance; a lead approaching zero means the device is catching up with the host.
- A device bubble is usually the downstream symptom. Start at the bubble or sparse region, then walk backward to the earliest point where launch lead collapsed. The causal event often occurs before the visible idle interval.
- Treat a sustained run of small gaps or low device density as one starvation region. Do not require a single large blank interval.
- Do not infer causality from duration alone. Establish a chain such as `Python stack -> CPU operation/copy/wait -> reduced launch lead -> device idle or sparse execution`.

## Collect sufficient evidence

Use a PyTorch profiler trace that includes CPU/Python activity, the relevant device activity, runtime/API calls, transfers, and stacks. Enable `with_stack=True`; retain the raw trace and source locations. Add iteration or phase markers with `record_function`/NVTX when boundaries are otherwise ambiguous.

Profile representative steady-state iterations with the same input, concurrency, and profiler schedule used for comparison. Include warmup, but exclude compilation, lazy initialization, and first-use allocation from a steady-state diagnosis unless startup latency is the requested target. Enable expensive options such as shape or memory recording only when their evidence is needed, because profiling itself can perturb timing.

Prefer at least two profiled steady-state requests or iterations after warmup. This distinguishes a process-level first use from work that repeats at the start of every request and makes request-boundary effects visible. Keep per-request markers and analyze requests separately before averaging them.

Define profiling metrics before comparing variants. Within each request marker, merge overlapping kernel, memcpy, and memset intervals across the selected device or stream before computing busy time. Report at least one of these explicitly named windows:

- **Device-event span:** first selected device event start to last selected device event end. Internal idle is `span - union(device events)` and the idle ratio uses that span as its denominator.
- **Request-marker window:** request marker start to end. Its idle value also includes host-only work before the first and after the last device event, so label it separately.

Do not use the whole trace window or a two-request combined span as the denominator for a single request. Do not call either window end-to-end latency: end-to-end comes from an unprofiled request timer with its boundary stated. For every number, name the request, window, included event categories, numerator, and denominator.

Classify repeated boundary gaps from the two request traces. A gap seen only in the first measured request may be process-level first use; a gap that recurs at the start of both warmed requests is per-request work. Do not describe the latter as startup overhead merely because it occurs at step zero.

Start and stop the profiler on the same thread when the backend requires thread-affine profiler ownership. An RPC that registers profiling on one thread and starts or stops it on a worker thread can produce an invalid trace, profiler-state errors, or a Kineto/CUPTI crash. If the normal profiling entrypoint has this limitation, add narrowly scoped temporary instrumentation around the real model/pipeline forward, remove it after trace collection, and keep it out of the proposed optimization.

Exclude dummy warmup requests and early-return paths from request-scoped instrumentation unless they are the explicit target. Every path that starts a profiler must stop and export it; prefer a guarded `try/finally` when temporary instrumentation spans code that may raise or return early. Confirm trace creation, nonzero device events, the intended iteration/phase, and a successful process exit before analyzing it.

If the trace cannot correlate device work to host launches or cannot show the responsible Python stack, recollect it. Do not guess from a device-only timeline.

Before inventing a performance harness, search for a repository-native benchmark or performance case for the model. Use that workload for final acceptance when it exists. Preserve its inputs and measurement semantics; environment-only path substitutions are acceptable when documented. A custom microbenchmark can isolate a cause, but it does not replace the native case.

## Analyze from symptom to cause

1. Mark the target interval. Record its timestamps, duration, iteration/phase, and whether it is fully idle or merely low density.
2. Find the transition from dense execution to degraded execution. Inspect several device operations before the first visible gap, not only the operations adjacent to the gap.
3. Correlate those device operations with their host enqueue/runtime calls and Python stacks. Compare launch lead before and during the degradation.
4. On the CPU timeline, inspect the interval in which new device work should have been issued. Identify the earliest blocking transfer, synchronization, long CPU computation, or absence of launches that explains the lost lead.
5. Confirm the counterfactual: if that host-side event were removed, shortened, deferred, or moved earlier, would enough work have been queued to cover the device gap? If not, continue tracing backward.
6. Classify the root cause and choose the smallest lossless change. When several causes exist, address the earliest causal one first and measure changes separately.

Use a candidate ladder when a broader rewrite removes additional waits: measure the narrow cause-specific edit first, then add one mechanism at a time. A candidate that removes more synchronizations but adds enough device work to hurt the native workload must be rejected and rolled back.

## Common root causes and lossless remedies

### Device to CPU or host observation

Look for D2H copies, scalar extraction (`item()`), `.cpu()`, `.numpy()`, host-side printing/inspection of tensor values, data-dependent Python branches, and explicit synchronize calls. These operations commonly make Python wait for preceding device work and drain the launch queue.

Prefer, in order:

- keep the value and dependent logic on the device;
- defer the host read until unrelated device work has already been enqueued;
- aggregate several host reads into one transfer;
- use an asynchronous transfer and consume it only after completion, when the backend API and program structure make that safe.

If the host genuinely needs the value immediately to determine subsequent work, do not claim the wait can simply be removed. Report the dependency and consider only semantically equivalent restructuring.

When Python already knows a tensor's semantics from request metadata, propagate that fact instead of re-reading tensor contents on every layer or iteration. For example, compute sequence lengths from Python shapes once and publish whether an attention mask is `none` or `padding`; do not repeatedly evaluate a device predicate such as `if torch.any(~mask)`. Preserve the same dense or masked kernel path for each case, and keep a conservative fallback when trustworthy host metadata is unavailable.

Validate published host metadata before it can select a faster path: check batch cardinality, ranges, and any packing invariant the consumer assumes. Add new optional API parameters without changing the meaning of existing positional arguments. Treat malformed or unavailable metadata conservatively rather than silently bypassing a real mask or dependency.

### CPU to device transfer

Look for H2D copies, `.to(device)`, `.cuda()`/backend equivalents, `copy_`, or a tensor constructed on CPU and immediately transferred. A default blocking copy can stall further host issuance.

Prefer constructing deterministic tensors directly on the target device. Otherwise, consider a non-blocking copy only after verifying all of the following:

- the backend supports the intended asynchronous path;
- host memory is pinned when required for true asynchronous transfer;
- the source remains alive and unmodified until the copy completes;
- the consumer is ordered after the copy on the same stream, or an explicit cross-stream dependency is present.

Device stream ordering makes a same-stream consumer wait for the copy; it does not by itself guarantee host-buffer lifetime or cross-stream correctness. Do not apply `non_blocking=True` mechanically.

### Excessive CPU work between launches

Look for Python loops, mask/index construction, list or shape processing, preprocessing, allocation churn, and repeated invariant calculations. Useful remedies include:

- construct tensor-valued data directly on the device;
- vectorize equivalent Python loops;
- precompute or cache truly invariant values;
- hoist work out of the critical iteration;
- reduce repeated Python dispatch while preserving operation order where it affects results.

A representative pattern is building a deterministic mask on CPU and then copying it to the device. Building the same mask directly on the device can remove both the CPU construction time and the blocking transfer. Verify values, dtype, shape, broadcasting, and boundary conditions rather than assuming equivalence.

### Explicit waits and one-time work

Inspect device/stream/event synchronization, allocator waits, compilation, lazy module initialization, and first-use library setup. Remove or defer an explicit wait only when no data, lifetime, error-checking, or ordering dependency requires it. For legitimate one-time work, warm it up or hoist it only if the requested workload semantics allow that; do not present a startup-only improvement as a steady-state speedup.

## Enforce the lossless bar

The optimization must preserve observable behavior: outputs, dtype, shape, device placement expected by callers, ordering dependencies, side effects, and error behavior relevant to the program. Preserve RNG state and random-number consumption order; moving random tensor construction from CPU to device is not automatically equivalent.

Use exact comparison when bitwise identity is expected. Otherwise use the project's established numerical tolerance and explain why non-bitwise equality is acceptable. Lower precision, approximate kernels, fewer model steps, changed sampling, or altered inputs are outside this skill's default scope. If a proposal is not provably lossless, label it as an optional tradeoff and do not apply it without user authorization.

## Implement and verify

If the user requested diagnosis only, stop after the evidence-backed root cause and proposed fix. If they requested elimination, make the narrowest relevant code change and test it.

Verify both correctness and causality:

1. Run a focused functional comparison using identical inputs and deterministic seeds where applicable.
2. Re-profile the same warmed-up workload and profiler window.
3. Compare the target interval's idle time or density, launch lead, end-to-end latency, and throughput. Report profiler settings and run-to-run variation when they affect the conclusion.
4. Confirm that the bubble disappeared or shrank for the predicted reason rather than merely moving elsewhere.
5. Note residual or unavoidable waits explicitly.

Bind every reported variant to its exact source state: record the commit plus the selected diff or file/blob hashes, including untracked files. After rolling back an unsuccessful candidate, rerun the focused tests and required profiling on the exact final files; results from the superseded candidate do not validate the rollback.

Keep metric scopes separate. GPU span and idle ratio diagnose queue starvation; server forward or generation time measures model-side latency; client-observed end-to-end includes scheduling, polling, transport, and persistence. Never label a shorter profiler span as end-to-end improvement. Report all relevant scopes when coarse polling or asynchronous APIs can hide a real model-side change.

Treat profiler latency as diagnostic, not as the final performance number. For latency A/B, keep model initialization, compilation caches, and input configuration identical; prefer several requests in the same process, discard clearly warming iterations, and report the chosen steady-state window plus its median and variation. Separate-process one-shot runs are often dominated by compilation, autotuning, model loading, or scheduler noise and may reverse the apparent result. If results remain noisy or overlapping, say that the end-to-end effect is inconclusive even when the trace proves the synchronization was removed.

Use the repository-native performance runner as the final gate when one exists. If its end-to-end or primary server metric shows no repeatable benefit, report zero accepted speedup and do not keep the change solely because the trace looks better. When ordering or machine drift may dominate, repeat in reverse A/B order before deciding.

Report results in this compact structure:

- **Target interval:** timestamps/phase and bubble or sparse-region size.
- **Lead collapse:** where host issuance stopped staying ahead and the before/after lead evidence.
- **Root cause:** event names plus the responsible Python stack/source location.
- **Change:** proposed or applied edit and why it is lossless.
- **Validation:** correctness result and before/after performance evidence.
- **Residuals:** remaining bubbles, rejected candidates, uncertainty, or required follow-up trace data.
