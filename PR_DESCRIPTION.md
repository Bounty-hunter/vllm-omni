# PR: Add Weight Transfer Support for Multi-Stage RL Training

## Purpose

Enable dynamic model weight updates during inference without engine restart, which is required for RLHF / RL training loops (e.g. verl-omni) that update policy weights every training iteration.

This PR ports the upstream vLLM weight transfer architecture (backends registered in upstream main: `ipc`, `nccl`, `sparse_nccl`, plus an omni-specific `omni_ipc` for diffusers-style diffusion pipelines) to vLLM-Omni's multi-stage orchestration. The four-phase protocol (`init_weight_transfer_engine` → `start_weight_update` → `update_weights` → `finish_weight_update`), the worker engine construction (upstream factory signature, post-model-load), and the payload contracts (backend-specific, built by the upstream trainer-side engines) all match upstream so future upstream improvements flow downstream automatically.

Key changes:

- **Config propagation**: `weight_transfer_config` added as a pipeline-wide deploy field (stage-level override supported), plumbed through the diffusion structured config projection into the worker's `VllmConfig`; the `--weight-transfer-config` CLI flag is inherited from upstream `EngineArgs`
- **AR stage worker**: weight transfer inherited as-is from upstream vLLM `gpu_worker`
- **Diffusion stage worker** (`diffusion_worker.py`): full lifecycle with state machine validation; the engine is created after model load via the upstream factory signature (`create_engine(config, vllm_config, device, model)`)
- **`omni_ipc` backend** (`diffusion/worker/weight_transfer_engine.py`): IPC payloads applied in place via the pipeline's `load_weights`, skipping the layerwise-reload protocol that omni's diffusers-style pipelines are not wired into (same "apply in place" shape as upstream's `sparse_nccl`); registered through the factory's runtime extension point, trainer side unchanged
- **Orchestrator** (`async_omni.py`): `AsyncOmni` exposes the four-phase API; lifecycle RPCs are routed only to stages with weight transfer enabled, and any per-stage error result raises (no silent partial success)
- **HTTP API** (`entrypoints/serve/weight_transfer_api.py`): four `POST` endpoints mirroring upstream's RLHF dev routes, mounted only when the feature is enabled (with a security warning in the server log)
- **Docs**: user guide (`docs/user_guide/weight_transfer.md`), config reference updates, and design RFC (`design/weight-transfer-engine-rfc.en.md`)

## Test Plan

**Unit tests** — `tests/test_weight_transfer_omni.py` (25 tests):

```bash
pytest tests/test_weight_transfer_omni.py -v
```

Covers config propagation (pipeline-wide field, deploy override, stage-scoped override), worker lifecycle (AR inheritance pinned, diffusion state machine, invalid-transition error handling, session reuse), and orchestrator semantics (error results raise, routing skips unconfigured stages, no-configured-stage is a hard error).

**E2E: generation verification (real payload path)** — `tests/e2e/features/rlhf_test/test_weight_transfer_changes_output.py`:

```bash
pytest tests/e2e/features/rlhf_test/test_weight_transfer_changes_output.py -v -s
```

In-process `AsyncOmni` engine; builds real CUDA IPC payloads exactly like `IPCTrainerWeightTransferEngine._send_unpacked` (upstream `reduce_tensor` + `IPCWeightTransferUpdateInfo`), ships them through the four-phase lifecycle, and asserts the generated image changes after a weight perturbation and returns after restore. Uses `tiny-random/Qwen-Image` on 1 GPU.

**E2E: HTTP API control plane** — `tests/e2e/features/rlhf_test/test_weight_transfer_http_api.py`:

```bash
pytest tests/e2e/features/rlhf_test/test_weight_transfer_http_api.py -v -s
```

Starts a server with `--weight-transfer-config`; verifies the trainer handshake payload (`{"packed": false}`) reaches the stage worker, and that invalid payloads / missing fields / protocol violations (update before start) are rejected instead of silently succeeding.

**vLLM Version:** vLLM 0.29

**vLLM-Omni Commit:** d4dba0ddb

## Test Result

Single-GPU remote machine (backend: `ipc`, model: `tiny-random/Qwen-Image`):

- `tests/test_weight_transfer_omni.py`: **25 passed**
- `tests/e2e/features/rlhf_test/test_weight_transfer_http_api.py`: **4 passed** — server started with `--weight-transfer-config`; the trainer handshake (`{"packed": false}`) reached the stage worker (config → engine creation → `init_weight_transfer_engine` all verified in the server log); invalid payloads, missing fields, and update-before-start were all rejected
- `tests/e2e/features/rlhf_test/test_weight_transfer_changes_output.py`: **passed** — real CUDA IPC payloads (helper-process handle export, mirroring the upstream trainer engine) shipped through the four-phase lifecycle: the live parameter matches the shipped tensor bit-exactly, a perturbed VAE-decoder weight changes the generated image (mean diff 0.028), and re-shipping the pristine weights restores the output bit-exactly (diff 0.0)

Notes:

- Unit tests run anywhere (no GPU required); both e2e tests require 1 CUDA GPU and download the tiny test model
- `nccl` / `sparse_nccl` backends are inherited from upstream vLLM unchanged; e2e covers `ipc` (single-GPU CI)
