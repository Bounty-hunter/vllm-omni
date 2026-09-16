# Weight Transfer Engine for vLLM-Omni

**Status**: Implemented  
**Authors**: vLLM-Omni Team  
**Created**: 2026-09  
**Updated**: 2026-09

## Summary

This RFC describes the implementation of weight transfer capability in vLLM-Omni, enabling dynamic model weight updates during inference for reinforcement learning (RL) training workflows. The design aligns with upstream vLLM's weight transfer architecture while adapting it for vLLM-Omni's multi-stage orchestration model.

## Motivation

Reinforcement Learning from Human Feedback (RLHF) and other RL-based training paradigms require frequent model weight updates during the inference phase. Traditional approaches restart the inference engine for each weight update, incurring significant overhead from model reloading and GPU initialization.

vLLM introduced a weight transfer mechanism in version 0.19+ to enable in-place weight updates without restarting the engine. This RFC brings equivalent functionality to vLLM-Omni, which orchestrates multi-stage pipelines (AR stage, diffusion stage) that each run independent vLLM/vLLM-Omni engine instances.

**Key requirements:**
- Support dynamic weight updates without engine restart
- Maintain alignment with upstream vLLM's four-phase protocol
- Enable per-stage weight transfer for multi-stage pipelines
- Provide HTTP API for integration with RL training frameworks

## Design

### Four-Phase Protocol

The weight transfer lifecycle follows upstream vLLM's proven four-phase protocol:

1. **Init** (`init_weight_transfer_engine`): Initialize the weight transfer backend and prepare communication channels
2. **Start** (`start_weight_update`): Begin a new weight update session and prepare workers to receive weight deltas
3. **Update** (`update_weights`): Stream weight tensors (names + data) to all workers
4. **Finish** (`finish_weight_update`): Finalize the update, swap in new weights, and close the session

This stateful protocol ensures consistency across distributed workers and enables efficient batching of weight updates.

### Multi-Stage Architecture

vLLM-Omni's architecture differs from upstream vLLM in its multi-stage orchestration:

```
┌─────────────────────────────────────────────────┐
│         AsyncOmni Orchestrator                  │
│  (entrypoints/async_omni.py)                   │
│                                                 │
│  ┌─────────────────────────────────────────┐  │
│  │ Weight Transfer Lifecycle Methods       │  │
│  │  - init_weight_transfer_engine()        │  │
│  │  - start_weight_update()                │  │
│  │  - update_weights()                     │  │
│  │  - finish_weight_update()               │  │
│  └──────────────┬──────────────────────────┘  │
│                 │ Forwards to all stages       │
│                 ▼                               │
│  ┌──────────────┴───────────────────────────┐ │
│  │ Stage Clients (AR, Diffusion)            │ │
│  │  - stage_client.weight_transfer_engine   │ │
│  │  - Wraps underlying EngineCore           │ │
│  └──────────────┬───────────────────────────┘ │
└─────────────────┼───────────────────────────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │  Worker Layer          │
      │  - ARWorker            │
      │  - DiffusionWorker     │
      │  - Exposes lifecycle   │
      └───────────────────────┘
```

**Key design decisions:**

1. **Orchestrator-level API**: The `AsyncOmni` class exposes the four-phase protocol at the top level
2. **Broadcast to stages**: Each lifecycle method is forwarded to all active stage clients
3. **Per-stage engines**: Each stage (AR, diffusion) maintains its own `WeightTransferEngine` instance
4. **Worker delegation**: Stage workers implement the lifecycle methods and delegate to their underlying vLLM engine

### Configuration Propagation

Weight transfer is enabled via `weight_transfer_config` in the engine configuration:

```python
engine = AsyncOmni(
    model="path/to/model",
    weight_transfer_config={
        "backend": "ipc",  # or "nccl", "sparse_nccl", "sharded_rdt"
    }
)
```

The configuration follows vLLM-Omni's config propagation model:

- **Pipeline-wide**: `weight_transfer_config` is a pipeline-wide field that applies to all stages by default
- **Stage override**: Can be overridden at stage level via `deploy_override` for stage-specific backends
- **Worker access**: Workers receive the config through their `EngineConfig` and initialize engines accordingly

### Supported Backends

The implementation inherits all backends registered in upstream vLLM (`vllm/distributed/weight_transfer/factory.py`), plus one omni-specific variant:

| Backend | Description | Use Case |
|---------|-------------|----------|
| `ipc` | CUDA IPC handles (zero-copy), layerwise-reload protocol | Single-node vLLM-native (AR) stages |
| `nccl` | NCCL broadcast from the trainer, layerwise-reload protocol | Multi-GPU trainers |
| `sparse_nccl` | Sparse tensor transfer over NCCL, applied in place | Large models with sparse updates |
| `omni_ipc` | IPC payloads, applied in place via pipeline `load_weights` | Diffusion stages (diffusers-style pipelines) |

`omni_ipc` is registered through the factory's runtime extension point (the same mechanism the factory docstring advertises for custom engines) and inherits the full IPC payload contract; only the lifecycle hooks differ — no layerwise meta-ization, mirroring upstream `sparse_nccl`'s "apply in place" design.

Backend selection is transparent to the orchestrator—each stage independently initializes its chosen backend.

### HTTP API

For integration with RL training frameworks (e.g., verl-omni), vLLM-Omni exposes HTTP endpoints mirroring the four-phase protocol. The routes are mounted **only when `weight_transfer_config` is enabled** (mirroring upstream's gated RLHF dev router):

```
POST /init_weight_transfer_engine
  Body: {"init_info": <backend handshake>}   # e.g. {"packed": false} for IPC

POST /start_weight_update
  (No body required)

POST /update_weights
  Body: {"update_info": <backend payload>}   # e.g. names/dtype_names/shapes/ipc_handles for IPC

POST /finish_weight_update
  (No body required)
```

The HTTP layer (`vllm_omni/entrypoints/serve/weight_transfer_api.py`) translates requests to `AsyncOmni` method calls. Payloads are backend-specific and are built by the trainer-side engines (`WeightTransferTrainerFactory.trainer_init` + `HTTPVLLMWeightSyncClient`); the HTTP layer treats them as opaque. Any stage-level failure raises on the HTTP call — a 200 means every enabled stage succeeded.

## Implementation Status

### Completed

- ✅ Config propagation: pipeline-wide deploy field + stage-level override, plumbed to the diffusion worker's `VllmConfig`
- ✅ AR stage worker lifecycle via upstream vLLM `gpu_worker` (inherited)
- ✅ Diffusion stage worker lifecycle with state machine; engine created post-model-load with the upstream factory signature
- ✅ Orchestrator-level API with selective routing (only configured stages) and strict error propagation
- ✅ HTTP API routes, gated on feature enablement, with error handling
- ✅ Unit tests covering config, workers, orchestrator
- ✅ End-to-end validation tests:
  - Generation output verification over the real IPC payload path (tiny-random/Qwen-Image)
  - HTTP API control-plane validation (handshake reaches the worker; invalid payloads rejected)

### Backend Support

| Backend | AR Stage | Diffusion Stage | Notes |
|---------|----------|-----------------|-------|
| `ipc` | ✅ | ✅ | Exercised by e2e tests |
| `nccl` | ✅ | ✅ | Inherited from vLLM |
| `sparse_nccl` | ✅ | ✅ | Inherited from vLLM |

## Testing Strategy

### Unit Tests (`tests/test_weight_transfer_omni.py`)

1. **Config propagation** (4 tests)
   - Verify `weight_transfer_config` is pipeline-wide
   - Test deploy override mechanism
   - Confirm stage-level config reaches workers

2. **Worker lifecycle** (15 tests)
   - AR worker exposes four methods
   - Diffusion worker state machine (init → start → update → finish)
   - Error handling (double start, update before start, etc.)
   - Session reusability after finish

3. **Orchestrator forwarding** (3 tests)
   - AsyncOmni exposes lifecycle methods
   - Methods correctly forward to all stage clients
   - Config initialization propagates to workers

### End-to-End Tests

1. **Generation verification** (`tests/e2e/features/rlhf_test/test_weight_transfer_changes_output.py`)
   - In-process `AsyncOmni` engine with `weight_transfer_config={"backend": "omni_ipc"}`
   - Builds real CUDA IPC payloads in a helper process (a process must not open its own exported handles), mirroring `IPCTrainerWeightTransferEngine._send_unpacked`
   - Ships the transformer backbone + VAE through the four-phase lifecycle; a perturbed VAE-decoder weight changes the generated image, and re-shipping the pristine weights restores the output bit-exactly (measured: live-param vs shipped diff 0.0, perturbed output diff 0.028, restored output diff 0.0)
   - Uses tiny-random/Qwen-Image for speed

2. **HTTP API validation** (`tests/e2e/features/rlhf_test/test_weight_transfer_http_api.py`)
   - Start vllm-omni server with `--weight-transfer-config` (inherited upstream CLI flag)
   - Verify the trainer handshake payload (`{"packed": false}`) reaches the stage worker via HTTP
   - Verify invalid payloads, missing fields, and protocol violations are rejected (no silent success)

## Alignment with Upstream vLLM

This implementation maintains strict alignment with current upstream vLLM main:

### Version Compatibility

- Weight transfer first appeared in vLLM 0.16; the backend ecosystem (nccl, sparse_nccl) grew through 0.19+
- **vLLM-Omni**: aligned with the vLLM 0.29 line that vLLM-Omni main tracks

### Protocol Fidelity

The four-phase protocol is preserved exactly:
1. Method names match upstream (`init_weight_transfer_engine`, `start_weight_update`, `update_weights`, `finish_weight_update`)
2. The worker engine is created with the upstream factory signature (`create_engine(config, vllm_config, device, model)`) after model load, mirroring upstream `gpu_worker`
3. Payload contracts are upstream's: init handshake carries backend wire params (e.g. `packed`), updates carry CUDA IPC handles / NCCL metadata — produced by the upstream trainer-side engines

### Worker Inheritance

- **AR stage**: served by upstream vLLM's `gpu_worker` — weight transfer support is inherited as-is
- **Diffusion stage**: implements the same lifecycle interface on top of the upstream engine classes

### Future-Proofing

By aligning with upstream vLLM's design:
- New backends added to vLLM automatically work in vLLM-Omni
- Upstream bug fixes and optimizations flow downstream
- Integration with vLLM-based tooling (verl, etc.) remains compatible

## Integration Example

### Trainer side (recommended)

```python
from vllm.distributed.weight_transfer.base import ModuleSource
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

client = HTTPVLLMWeightSyncClient(base_url="http://localhost:8000")
trainer_engine = WeightTransferTrainerFactory.trainer_init(
    IPCTrainerInitInfo(rank=0, packed=False),
    client=client,
    source=ModuleSource(training_model),
)

# Each round: generate rollouts, update the training model, ship weights.
for epoch in range(num_epochs):
    rollouts = collect_rollouts(client_base_url)
    weight_deltas = compute_policy_gradient(rollouts)
    apply_deltas(training_model, weight_deltas)
    trainer_engine.send_weights()
```

### Python API (orchestrator level, in-process)

```python
from vllm_omni.entrypoints.async_omni import AsyncOmni

engine = AsyncOmni(
    model="path/to/model",
    weight_transfer_config={"backend": "ipc"},
)

# IPC init handshake (the payload the trainer engine ships)
await engine.init_weight_transfer_engine({"packed": False})

# Each round mirrors the trainer's send_weights:
await engine.start_weight_update()
await engine.update_weights(update_info)  # backend payload (see trainer engines)
await engine.finish_weight_update()
```

### HTTP API (verl-omni integration)

The HTTP routes are driven by the upstream client, not hand-rolled requests:

```python
from vllm.distributed.weight_transfer.base import ModuleSource
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

client = HTTPVLLMWeightSyncClient(base_url="http://localhost:8000")
engine = WeightTransferTrainerFactory.trainer_init(
    IPCTrainerInitInfo(rank=0), client=client, source=ModuleSource(model)
)
engine.send_weights()  # init handshake + full update round trip over HTTP
```

## Alternatives Considered

### 1. Single-stage weight transfer (AR only)

**Rejected**: Diffusion models are increasingly used in RLHF (e.g., image generation with human preference feedback). Supporting only AR stage would limit applicability.

### 2. File-based weight transfer

**Rejected**: Writing weights to disk and reloading requires full model reloads per update; the four-phase protocol with IPC/NCCL backends updates weights in place.

### 3. Custom protocol (non-vLLM-compatible)

**Rejected**: Maintaining compatibility with upstream vLLM ensures:
- Access to upstream optimizations
- Compatibility with vLLM ecosystem (verl, etc.)
- Reduced maintenance burden

## Future Work

1. **Layerwise-reload adaptation for omni diffusion pipelines**: upstream's `initialize_layerwise_reload` moves every parameter to meta device at `start_weight_update` and restores it through per-parameter `weight_loader` wiring as chunks stream in; omni's diffusers-style pipelines are not wired into that protocol (`'ReplicatedLinear' object has no attribute 'weight'`). Short term this PR ships the `omni_ipc` backend, which applies received weights in place through the pipeline's `load_weights` (the same "no layerwise reload" shape as upstream's `sparse_nccl`). Wiring the pipelines into the reload protocol would additionally support quantized-weight updates on diffusion stages.
2. **Distributed weight transfer**: Exercise multi-node trainer scenarios with the `nccl` backends
3. **Sparse updates**: Optimize for LoRA/adapter-style updates that modify <1% of parameters (`sparse_nccl`)
4. **Async updates**: Allow weight updates to overlap with inference (double buffering)
5. **AR-stage config plumbing**: route `weight_transfer_config` through the structured AR stage config projection so LLM-only pipelines opt in without deploy-YAML-only configuration

## References

- Upstream vLLM `vllm/distributed/weight_transfer/` (engines, factory, clients)
- Upstream vLLM RLHF dev routes: `vllm/entrypoints/serve/dev/rlhf/api_router.py`
- [verl-omni](https://github.com/verl-project/verl-omni)
