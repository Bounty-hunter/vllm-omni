# Qwen-Image 2.1 native accuracy

This suite compares real-checkpoint native HTTP serving with an independent
Diffusers implementation. Each parametrized test owns its candidate, two fresh
reference generations, input copies, and a unique artifact directory. It never
uses another test's output as its reference.

## Coverage and reference

The representative `it2i` case uses the two existing
`tests/assets/qwen_image_edit/` images: a portrait cat costume template followed
by a square dog photograph. The prompt replaces the template's cat head with
the dog while retaining the costume. It exercises ordered multipart decoding,
aspect-preserving input resizing, Qwen3-VL vision/text encoding, condition-image
VAE encoding, positive and negative branches, prefix cache prefill/decode,
denoising, and RGBA PNG output. The additional `t2i` case covers the separate
no-image prompt template and latent path with CFG disabled. Extra seeds alone
would not exercise that branch.

The independent reference is the official
[Diffusers PR #14804](https://github.com/huggingface/diffusers/pull/14804) merge
commit `6256aa7666cedd47443adc8f82da9a10e110b09c` (version `0.41.0.dev0`). It is
installed in a separate virtual environment. Preflight checks its installed
VCS commit, the actual imported package and component paths, and Qwen3-VL
processor imports before loading either model. This pinned version implements
`use_kv_cache=True`.

The public checkpoint is `Qwen/Qwen-Image-2.1`, revision
`d26bb61231c349cf6b7896fa83353113880e1ba3`. Setup downloads that immutable snapshot
through the normal Hugging Face cache. `QWEN_IMAGE_21_MODEL` can instead point
to a complete frozen local snapshot; both runtimes receive the identical
absolute path. The artifact records whether it is a local override, config and
tokenizer hashes, and all selected weight shard sizes. Local overrides must
have separately retained checkpoint provenance; directory names and nonempty
shards do not prove weight integrity or identity with the public revision.

| Field | Effective reference and candidate contract |
| --- | --- |
| Inputs | Identical original RGB PNG bytes, pinned SHA-256, first template then dog |
| Input geometry | 1190x2032 -> 768x1344; 1024x1024 -> 1024x1024; area target 1024 squared, dimensions rounded to 32 |
| Vision / VAE preprocessing | Normal runtime interfaces; vision sees RGB on white, VAE sees opaque RGBA; no shared embeddings |
| Output | IT2I 768x1024; T2I 1024x1024; exactly one RGBA PNG, no output resizing or RGB conversion |
| Sampling | Seed 42, CUDA generator, BF16 noise; 40 FlowMatch Euler steps, checkpoint scheduler config and dynamic shift |
| RNG draw order | Condition VAE uses posterior mode (`argmax`), consuming no generator draws; then one noise draw of shape `(1,1,64,H/16,W/16)` on both sides |
| Guidance | IT2I `true_cfg_scale=4` with negative prompt; T2I `1`, no negative prompt; no `guidance_scale` |
| Precision | BF16 transformer, text encoder and VAE; single GPU, no quantization, tiling or CPU offload |
| DiT attention | Native `CUDNN_ATTN`; reference `QwenImage21AttnProcessor` with cuDNN SDPA for prefill and decode |
| Qwen3-VL encoder | HF SDPA with cuDNN enabled during the CUDA forward; vision/text RoPE constants initialized on CPU and transferred as FP32 |
| VAE attention | Ordinary runtime path, unchanged |
| Execution | Native `--enforce-eager`, request execution, no compilation or decode graph; prefix cache enabled on both sides |
| Deliberate differences | Native fused projections vs upstream separate projections; both use piecewise prefill with the same block-causal mask and cuDNN kernel family |

Native argument translation is in `entrypoints/openai/api_server.py`; the
generator is created by `diffusion/worker/diffusion_model_runner.py`. Model
preprocessing, VAE posterior mode, noise layout and scheduler construction are
in `diffusion/models/qwen_image_21/pipeline_qwen_image_21.py`. Reference effective
dtypes, scheduler, attention processor and timesteps are saved at runtime.
Native effective configuration is supported by these source traces and the
server console logs, not introspected from the running worker. Investigate any
unexpected loader/backend log before accepting calibration.
The reference wraps only the transformer's forward in
`sdpa_kernel(CUDNN_ATTENTION)`: its prefill explicitly passes `backend=None`,
so setting the processor's decode backend alone is insufficient. Unsupported
cuDNN paths fail rather than silently falling back to a different kernel.

Unvalidated branches include transparent input alpha, single-image edits,
three/four input images, automatic output dimensions, multiple outputs, chat
completions, step batching, graph/compile, quantization, distributed execution,
offload, VAE tiling and non-CUDA platforms. RGB fixtures cover RGBA output but
do not establish preservation of transparent input pixels.

## Environment and commands

Use Linux, one H100 80 GB (or a separately calibrated CUDA device with at least
80 GB), the repository's installed vLLM-Omni test environment, Git/network access
for reference provisioning, Hugging Face checkpoint access/cache, and enough
host RAM and disk for both packages and the complete checkpoint. No FFmpeg is
needed. `torchmetrics`, NumPy, Pillow, requests and pytest must be installed in
the serving interpreter. The reference inherits its exact torch/CUDA,
Transformers, Accelerate, Pillow and Hugging Face Hub versions; preflight requires
equality and records the actual imported Hub path.
Setup appends the serving interpreter's enabled `site-packages` and
`dist-packages` roots in their actual `sys.path` order using a `.pth` file.
This includes an enabled user-site ahead of system packages and works when
the serving interpreter itself belongs to a virtualenv. The reference's own
package directory stays first so its pinned Diffusers takes precedence.
Only Diffusers and the reference-only `torchao==0.16.0` are installed locally
using `--no-deps`; the latter supplies `FqnToConfig`, required by this Diffusers
revision even without quantization. Preflight verifies its pinned version.
The reference environment defaults to `tests/e2e/accuracy/qwen_image_21/.venv`,
covered by the repository's existing `.venv` ignore rule. Set
`QWEN_IMAGE_21_REFERENCE_ENV` to use a different reference environment directory.
The serving/shared environment is unchanged. Dependency versions and actual
import paths are part of the calibration scope; a serving image upgrade
requires revisiting the gate. No reference source is added to `PYTHONPATH`.

Run from the repository root:

```bash
source tests/e2e/accuracy/qwen_image_21/setup_reference.sh

# Collection with the exact CI selector; this does not run models or preflight.
pytest --collect-only -q tests/e2e/accuracy/qwen_image_21/test_qwen_image_21.py \
  -m "full_model and diffusion and H100 and cards_1" --run-level full_model

# First calibration: one representative case, reference repeated within the worker.
QWEN_IMAGE_21_CALIBRATE=1 pytest -sv \
  'tests/e2e/accuracy/qwen_image_21/test_qwen_image_21.py::test_qwen_image_21_matches_reference[it2i]' \
  -m "full_model and diffusion and H100 and cards_1" --run-level full_model

# Collect both distinct workloads on the intended CI host.
QWEN_IMAGE_21_CALIBRATE=1 pytest -sv \
  tests/e2e/accuracy/qwen_image_21/test_qwen_image_21.py \
  -m "full_model and diffusion and H100 and cards_1" --run-level full_model \
  --junitxml=tests/e2e/accuracy/artifacts/qwen_image_21/calibration.xml

# After calibration and user selection of thresholds.json: exact CI command.
pytest -sv tests/e2e/accuracy/qwen_image_21/test_qwen_image_21.py \
  -m "full_model and diffusion and H100 and cards_1" --run-level full_model \
  --junitxml=tests/e2e/accuracy/artifacts/qwen_image_21/pytest.xml
```

The intended node IDs end in `[it2i]` and `[t2i]`. Calibration mode computes
everything, then reports XFAIL with the evidence path; it is not an accuracy
pass. Missing assets, imports, malformed outputs and subprocess failures remain
failures. CI does not enable calibration mode. Until `thresholds.json` contains
selected bounds, its normal command fails explicitly before model loading.
The proposed Buildkite step has `skip: true` with the pending-calibration reason;
remove it only after measurements and user selection establish a runnable gate.

To rerun only a reference preflight for an already written request:

```bash
"$QWEN_IMAGE_21_REFERENCE_PYTHON" \
  tests/e2e/accuracy/qwen_image_21/run_reference.py \
  --request /absolute/path/to/request.json \
  --output-dir /absolute/path/to/new-preflight-directory --preflight-only
```

## Numerical alignment and gate status

The native pipeline initializes Qwen3-VL's non-checkpoint vision and text RoPE
buffers with the original Transformers constructors on CPU, then copies them
to their runtime devices. Initializing the same FP32 constants directly on
CUDA changes low bits that can amplify through the BF16 encoder. This small
initialization step applies after either encoder loading path and does not
replace checkpoint weights or reuse reference tensors.

vLLM's CUDA initialization disables cuDNN SDPA globally. The pipeline enables
it only during the Qwen3-VL CUDA forward and restores the previous flag in
`finally`, including on encoder failure. Non-CUDA execution leaves that flag
alone. This restores HF's encoder dispatch choices independently of the DiT
backend. It does not force cuDNN for every model or override the VAE.

The validation below uses the production loader and the normal pytest/HTTP
entrypoints, without temporary native diagnostic hooks. Regression tests cover
CPU rotary initialization under a foreign device context and backend/hook
cleanup on successful and failed forwards.

Recommend full-image **RGBA SSIM and PSNR**: all four channels converted to
float32 in `[0,1]`, without compositing or resizing. Both selected lower bounds
must pass. Alpha MAE is recorded separately to help diagnose transparency
errors. Reference-repeat metrics are also saved. Infinite PSNR is serialized
as the string `"inf"`, while assertions use the numeric value.

`thresholds.json` keeps separate IT2I and T2I tables with null bounds and status
`awaiting_calibration_and_user_selection`. Normal execution requires status
`selected` and finite numeric thresholds. Calibration mode saves the evidence
then reports XFAIL; failures before that point remain failures. The Buildkite
step stays inactive until the user selects a gate for the intended CI scope.

### Validation scope

On 2026-10-13, the integrated source passed 114 model/helper/pipeline-routing
regression tests in the remote serving environment. The exact CI selector
collected both accuracy cases. Their complete native/reference comparisons
finished in 308.85 seconds with exit 0 and **2 XFAIL** in calibration mode.
The XFAIL status records pending threshold selection; it is not a selected
numerical gate passing. The prior queue timeout is retained separately.

| Case | RGBA SSIM | PSNR (dB) | Alpha MAE | Decoded native/reference pixels |
| --- | --- | --- | --- | --- |
| Two-image IT2I, CFG 4 | 0.999989212 | infinity | 0 | Exactly equal |
| T2I, CFG 1 | 0.999996185 | infinity | 0 | Exactly equal |

Each case used one fresh native server and two reference calls within one
reference process. Each reference pair was pixel-identical too. PNG encodings
can have different hashes despite equal decoded RGBA arrays. Direct pixel
comparison found maximum error 0; the slight SSIM shortfall from 1 is metric
floating-point roundoff. All model outputs, metrics and subprocess statuses
were retained before the calibration XFAIL.

The measured environment was one NVIDIA L20X, Torch 2.13.0 / CUDA 13.0,
Transformers 5.14.1, vLLM 0.31.0, native Diffusers 0.40.0, the pinned reference
Diffusers 0.41.0.dev0, and torchmetrics 1.9.0. The frozen local checkpoint was
`790c92633540aa0cb11d9abf19eb46d861714758`; retained Hub metadata shows the same
runtime/weight blobs as the public revision, with only its README changed.
The executed source hashes matched the local pending changes. The source
snapshot, logs, JUnit, images and pixel verification are retained in workspace
`outputs/qwen21-integration-20261013/run2/`. Case artifact names are
`it2i-8q604f8m` and `t2i-m6fc8hvy`.

This establishes parity for the two fixed workloads on this L20X environment.
Target H100, other machines/dependency versions, and actual Buildkite execution
remain unvalidated. Earlier results with different dispatch/initialization
settings are not pooled into the calibration range.

### Threshold proposal awaiting user selection

For these fixed cases, propose **SSIM >= 0.99 and PSNR >= 40 dB** as provisional
engineering targets. Their SSIM gaps to the observed scores are 0.009989212
(IT2I) and 0.009996185 (T2I). With zero observed pixel error, there is no finite
measured PSNR margin: 40 dB instead expresses an allowed normalized RGBA RMSE
of 0.01, or 2.55 levels on the 8-bit scale. These targets allow small numerical
differences from the current exact match while retaining a high fidelity floor.
They are not bounds derived from observed cross-environment variability.

There is only one native observation per case in the final implementation.
Reference repeats within one process cannot establish cross-machine stability.
Validate on the target H100 image and sample supported machines/library
versions before claiming that this headroom covers normal CI variation. An
SSIM allowance of 0.05 is not justified by these measurements. Investigate a
changed numerical contract or unexplained mismatch before relaxing either
metric. The user selects the final values; `thresholds.json` remains unselected
and the CI step remains inactive. Keep a `default` row and use the shared
device resolver for evidenced SKU overrides; fallback does not establish
calibration on an untested device.

## CI and evidence

The proposed CUDA nightly X2I Accuracy step has an explicit file selector, H100
single-card preset, three-hour timeout, full-model runtime and artifact glob.
It is explicitly inactive while calibration and user threshold selection are
pending. Once its `skip: true` is removed, it runs when the containing group is
activated by `NIGHTLY=1` or `nightly-test`;
the scheduled bootstrap uses `main` plus `NIGHTLY=1`, and PR uploads apply the
source dependency filter. B200 mirroring is intentionally not advertised
without numerical evidence. Source dependencies include model/runtime code,
this entire suite, inputs, shared helpers and requirements. Shared-only helper
changes preserve broad fallback coverage; mixed diffs match explicit aliases.

Each invocation creates
`tests/e2e/accuracy/artifacts/qwen_image_21/<case>-<unique>/` with original input
copies, request, asset metadata, native launch/version data, HTTP response,
lossless candidate/reference PNG images, reference effective metadata and schedules,
reference/preflight logs and exit statuses, threshold policy and complete
metrics. Buildkite retains native server/pytest console logs and uploads the
artifact tree plus JUnit on failures too. Paths are unique; stale outputs cannot
satisfy a comparison. Preserve failed evidence before any controlled rerun.

```bash
python tools/pre_commit/check_test_marks.py \
  tests/e2e/accuracy/qwen_image_21/test_qwen_image_21.py
python .buildkite/common/scripts/upload_pipeline.py --all .buildkite/cuda/test-nightly.yml
python .buildkite/common/scripts/upload_pipeline.py .buildkite/cuda/test-nightly.yml
```

Rendering does not evaluate Buildkite activation or run GPU comparisons.
