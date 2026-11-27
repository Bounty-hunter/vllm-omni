---
title: Diffusion Parallelism
kind: module
status: draft
owners:
  - "@princepride"
  - "@yuanheng-zhao"
  - "@xuechendi"
  - "@Bounty-hunter"
primary_code_paths:
  - vllm_omni/diffusion/distributed/**
  - vllm_omni/diffusion/attention/parallel/**
related_code_paths:
  - vllm_omni/distributed/**
  - vllm_omni/config/composable_parallel/**
depends_on:
  - diffusion_runtime.md
  - ../vllm_omni_config.md
validation_paths:
  - tests/diffusion/distributed/**
  - tests/diffusion/attention/**
upstream_refs:
  - torch.distributed
last_reviewed: 2026-07-16
---

# Diffusion parallelism

Parallelism owns diffusion rank topology, process groups, tensor and sequence
sharding, collectives, and distributed execution.

## Candidate invariants

### PARALLEL-INV-001: Topology has one source of truth

**Rule:** Rank coordinates, group membership, and parallel dimensions MUST be
derived from validated configuration.

### PARALLEL-INV-002: Sharding contracts are explicit

**Rule:** Every distributed boundary MUST define tensor shape, sharded
dimension, placement, dtype, producer, and consumer.

### PARALLEL-INV-003: Collectives are symmetric

**Rule:** Process-group members MUST invoke compatible collectives in the same
logical order.

### PARALLEL-INV-004: Single-rank behavior remains valid

**Rule:** Distributed implementations SHOULD preserve a supported single-rank
path without distributed initialization.

### Adaptive head overlap

The strict Ulysses `auto` setting shares one policy across a DiT component. Each worker learns from the
component's real attention calls. New layouts and head partitions must agree across SP ranks before
allocating symmetric buffers. The communication stream moves subsequent head groups and returns completed
remote rows while the compute stream runs attention. Keep K/V in canonical sequence order when permuting
query rows. Join all peer writes before returning. A caller that consumes the result on the same stream before the next
attention can borrow the workspace. Other adaptive callers receive an owned tensor. Read trial events only after they
complete and agree at a component boundary. Graph preparation consumes frozen decisions.

The request-scoped experiment tests small-first two-group partitions (up to sixteen,
for at most thirty-two rank-local heads). A wave estimate orders candidates but never
selects the winner. After one native lead call, three rotating/reversed
native-and-candidate cycles execute in real layers (28 calls for sixteen heads).
Selection requires three valid rotations and positive paired gains in all three;
insufficient or noisy samples keep native attention. All candidates share one symmetric
allocation, peer mapping, streams and flags. Their packed views and staging
launchers are prepared before measurement. The resource owner can reuse the last
prepared geometry; it retains no winner or timing samples between requests.
Staging row counts and launch-region boundaries are runtime arguments, so a
different prompt length reuses compiled code for the same head partition and
strides. Widths stay specialized; crossing the 32-bit indexing limit selects a
separate wide-index variant. Workspace growth can still require reconnection.

An agreed, homogeneous layer group discovered during a real warmup request keeps
only its ordered layer names and layout structure. On subsequent requests its
first layer confirms the actual Q/KV signature across ranks, including changed
prefix lengths, before preparing candidates and sampling the remaining layers
in that same component call. Ineligible peers participate in this confirmation
with an empty proposal; disagreement retains whole-component native discovery.
No chosen partition or timings survive request reset. Unseen or heterogeneous
groups use the original full discovery pass. Sampling may span components when
the candidate budget exceeds the number of eligible layers.

Sample collection performs no per-layer host synchronization and no per-step
object collective. Completed events populate one CPU tensor, reduced with MAX
asynchronously on a dedicated Gloo group. Every rank commits at the same fixed
component boundary immediately following sampling, without an extra native step.
That boundary waits if events or reduction are late; this cost is not claimed to
be zero. Missing/starved observations invalidate the whole rotation. Insufficient
confidence keeps native attention. Request teardown drains submitted batches;
the next request clears all selection state, including KV-length-dependent choices.

This prototype is activated only by full-request runner execution. Stepwise calls
outside that scope remain native. Request-scoped automatic trials run eager rather
than replaying graphs containing a previous request's choice. The tested Qwen
configuration already uses eager execution. Query rows are never subdivided.

## Safe-change guide

Test topology validation, single-rank execution, affected parallel modes,
combined modes, invalid shapes, and orderly teardown.
