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

## Safe-change guide

The experimental CUDA head pipeline in `attention/parallel/head_overlap.py`
keeps stream and CE workspace ownership per device/Ulysses group. All ranks
enqueue the same `F0, F1, R0, F2, R1, ...` sequence of input and reverse
exchanges, with each next input submitted after launching the current kernel
and before waiting on its output. Per-group event
waits protect consumers, existing CE peer barriers protect workspace reuse,
and outputs return in the original global head order. The serial workspace
remains separate so a shape or metadata fallback does not change its stream
ownership. Each CE plan retains one input/output slot per head group, reused
after the next call's compute-producer fence. Compute-owned reverse inputs
stay live until all transport waits are enqueued. This bounds cross-stream
temporary retention instead of deferring frees across many DiT layers.
Group teardown synchronizes workspaces before releasing resources.

Test topology validation, single-rank execution, affected parallel modes,
combined modes, invalid shapes, and orderly teardown.
