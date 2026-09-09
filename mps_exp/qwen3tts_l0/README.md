# Qwen3-TTS L0: CUDA MPS off/on A/B (zero-copy)

Single-process experiment for `vllm_omni/deploy/qwen3_tts.yaml`. One Qwen3-TTS
serve owns two EngineCore subprocesses (stage0 talker, stage1 Code2Wav) on one
GPU. We toggle only the CUDA MPS daemon and sweep client concurrency, holding
every other knob identical, to measure how much cross-process kernel overlap
MPS recovers on this two-process-per-GPU workload.

## Why this shape

- vLLM-Omni runs **one EngineCore subprocess per stage**. A single-GPU
  multi-stage pipeline is therefore already several CUDA processes on one
  device, and without MPS their kernels mostly time-slice.
- This L0 isolates the cheapest possible win: **no second replica, no config
  change** — same `qwen3_tts.yaml`, MPS on vs off.
- If the single tuned replica leaves GPU idle (SM-active well under ~60%),
  MPS should raise both throughput and SM-active. If it is already compute
  bound, the numbers will not move and the same-GPU-DP direction is dead.

## Files

| file | role |
|---|---|
| `run_l0.sh` | driver; runs modes `off` then `on`; sweeps `CONC_LIST`; samples `nvidia-smi dmon`; prints a summary table |
| `bench_client.py` | seeded concurrent `/v1/audio/speech` load client (qps, latency p50/90/95/99, errors, audio bytes) |
| `mps_check.py` | verifies the running stage processes are really attached to the MPS pipe (a process that misses the pipe silently time-slices) |

## Run (on the remote omni box, inside the scheduler)

```bash
su - dyy -c 'source /data/dyy/env_setup.sh; dyy >/dev/null; \
  cd /data/dyy/code/vllm-omni && \
  gpu run --gpus 1 --timeout 40m --note "qwen3tts L0 mps A/B" -- \
    bash mps_exp/qwen3tts_l0/run_l0.sh'
```

Artifacts are written to `mps_exp/qwen3tts_l0/out/<timestamp>/`:

- `result_{off,on}_c{conc}.json` — one client result per mode/concurrency
- `dmon_{off,on}_c{conc}.csv` — GPU SM/mem samples during each run
- `server_{off,on}.log`, `bench_{off,on}.log`, `driver.log`
- the script's final summary table

## Env knobs

`MODEL` (default: first cached Qwen3-TTS CustomVoice, else the 1.7B id),
`PORT` (18091), `CONC_LIST="16 32 64"`, `N_MULT=6`, `WARMUP=6`,
`SRV_TIMEOUT=900`.

## Reading the result

Compare mode `off` vs `on` **at the same concurrency**:

- `qps` up and `sm%` up → MPS is recovering previously time-sliced kernel
  overlap. Direction worth pursuing (next: L1 same-GPU DP2).
- `qps` flat but `sm%` up → kernels overlap but the server is not the
  bottleneck (e.g. host dispatch); same-GPU DP still has headroom.
- `qps` and `sm%` both flat / `qps` down → single replica is compute bound or
  the stage pipeline leaves no idle to harvest; stop the MPS direction.

Also sanity-check the MPS-attach line in `driver.log` for the `on` mode: it
must report stage processes attached to the pipe, otherwise the `on` run was
really time-slicing and the comparison is void.
