# Head overlap and early output return on image and video requests

The experimental adapter compares three graphs on four L20X GPUs with Ulysses SP4, VAE4, FP8 DiT, BF16 FA3, `ulysses_a2a_permute=true` and `VLLM_OMNI_PACKED_QKV=1`. The native arm retains native grouped CE. The shared-whole arm uses one head group with the same output-slot ownership as the split. The automatic arm samples real layers, freezes the decision and replays a single graph per component signature.

Qwen Image 2.1 Turbo uses eight steps. Wan 2.2 A14B uses eight steps for T2V and four for I2V. Each timed call covers offline `Omni.generate` through decoded image or video return. Native HTTP CI, HSDP and HTTP video polling have a different execution path. These runs use one request at a time.

## Paired full requests

Positive percentages indicate lower average latency. R60 is commit `735e61e65`. The dense-grid runs add the video eligibility change in this branch. The independent Qwen repeat uses the original R60 checkout. Each six-round block rotates all three arms through forward and reversed orders. The intervals use paired resampling of whole blocks, with three blocks for 18 rounds, four for 24, six for 36 and twelve for 72. Paired requests use seed 42.

| Run | Workload | Rounds | Heads per rank | Native ms | Shared whole ms | Auto ms | Saved vs native [95%] | Saved vs shared whole [95%] |
|---|---|---:|---|---:|---:|---:|---:|---:|
| r60 | Qwen 1024², 2 refs | 24 | native whole | 1240.00 | 1227.65 | 1245.30 | -0.43% [-1.85, +0.53] | -1.44% [-3.12, -0.25] |
| r60 | Qwen 1536², no ref | 24 | native whole | 1101.22 | 1178.55 | 1126.23 | -2.27% [-15.62, +8.38] | +4.44% [-18.89, +22.52] |
| r60 | Qwen 1536², 2 refs | 24 | 1+7 | 1800.41 | 1781.87 | 1746.27 | +3.01% [+2.62, +3.39] | +2.00% [+1.51, +2.48] |
| r60 | Qwen 2048², 1 ref | 24 | 1+7 | 2290.11 | 2247.26 | 2202.13 | +3.84% [+3.63, +4.14] | +2.01% [+1.51, +2.37] |
| r60 | Wan T2V 832×480×33 | 18 | native whole | 2153.76 | 2176.68 | 2170.64 | -0.78% [-1.44, +0.23] | +0.28% [-0.51, +0.81] |
| r60 | Wan T2V 832×480×65 | 18 | native whole | 5053.02 | 5019.85 | 5061.68 | -0.17% [-2.10, +2.41] | -0.83% [-2.42, +0.60] |
| r60 | Wan I2V 832×480×81 | 18 | native whole | 7097.38 | 7210.81 | 7083.74 | +0.19% [-0.68, +0.72] | +1.76% [+0.83, +2.53] |
| dense-grid | Wan T2V 832×480×33 | 18 | 1+9 | 2177.70 | 2212.56 | 2106.94 | +3.25% [+2.29, +3.81] | +4.77% [+3.66, +6.89] |
| dense-grid | Wan T2V 832×480×65 | 18 | 5+5 | 4828.35 | 4829.57 | 4718.34 | +2.28% [+1.64, +3.48] | +2.30% [+1.70, +2.99] |
| dense-grid | Wan I2V 832×480×81 | 18 | 2+8 | 7159.59 | 7331.38 | 8067.80 | -12.69% [-40.29, +2.60] | -10.04% [-32.48, +3.47] |
| r60-repeat | Qwen 1536², no ref | 72 | native whole | 980.13 | 954.35 | 976.50 | +0.37% [+0.01, +0.76] | -2.32% [-2.58, -2.08] |
| dense-grid-repeat | Wan I2V 832×480×81 | 36 | 2+8 | 7129.98 | 7143.07 | 6963.68 | +2.33% [+1.82, +3.00] | +2.51% [+2.26, +2.80] |

The original R60 route admits no candidate on Wan because its metadata always includes a video layout and a grid hint. Dense FA3 uses neither for attention weighting. The extended gate accepts a single grid covering every distributed row, with no mask, padding, prefix or sparse semantics. The first two real denoising calls still decide whether a split is worthwhile. All four ranks agreed on every component. Every output at seeds 42, 43 and 44 matched the native arm byte for byte. The later audit observed zero Python core calls, with sampling events and control collectives frozen.

The first dense-grid I2V 81-frame attempt returned matching first-request outputs for all three arms. Its workers shut down during the native arm in the first paired round. The 81-frame measurement in the table uses a fresh, complete run with the same configuration and code. The failed attempt contributes no timing samples.

In the completed 18-round I2V comparison, the automatic arm won 13 paired requests. A 24.05 s request included a four-step denoising loop of about 21 s, while the other 17 auto requests spent under 5 s in denoising. The resulting arithmetic mean regressed by 12.69% against native. The first three steps elapsed in about 3.7 s. The progress clock reached 21 s at step four. Per-kernel attribution is unavailable. A fresh 36-round run independently selected 2+8 on both DiTs, won 34 of 36 pairs against native, and saved 166.30 ms (2.33%). Its longest measured automatic request was 7.25 s. Both runs are included in full.

## Communication budget

| Run and workload | Pure P2P ms | P2P / shared whole | End-to-end ms saved vs shared whole | Saved / P2P [95%] |
|---|---:|---:|---:|---:|
| r60, Qwen 1536², 2 refs | 59.16 | 3.32% | 35.60 | 60.18% [45.33, 75.03] |
| dense-grid, Wan T2V 832×480×33 | 136.86 | 6.19% | 105.62 | 77.17% [58.44, 113.02] |

The denominator is the union of outgoing CUPTI PtoP copies on each GPU in an independent, matching shared-whole request. Use the maximum rank time per request, then average two requests. The individual denominators were 59.19 and 59.12 ms for Qwen, 136.31 and 137.41 ms for Wan. This excludes barriers, local staging and gaps. The interval holds the historical denominator fixed. A net end-to-end saving divided by this duration is sensitive to variation elsewhere in the request. Other shapes lack a matching full-request denominator.

## First requests

Each case starts a fresh engine and runs auto first, followed by shared whole and native. These measurements include first-use initialization. The inclusive mean counts the first request and every paired request for that arm. Correctness-only calls at the end are stored separately.

| Run | Workload | First auto / shared / native (ms) | Inclusive auto / shared / native (ms) |
|---|---|---:|---:|
| r60 | Qwen 1024², 2 refs | 5342.30 / 1695.23 / 1403.02 | 1409.18 / 1246.36 / 1246.52 |
| r60 | Qwen 1536², no ref | 7920.47 / 2463.26 / 1127.03 | 1398.00 / 1229.94 / 1102.25 |
| r60 | Qwen 1536², 2 refs | 6102.27 / 2166.20 / 1881.58 | 1920.51 / 1797.25 / 1803.66 |
| r60 | Qwen 2048², 1 ref | 7860.82 / 3248.16 / 2470.73 | 2428.48 / 2287.29 / 2297.34 |
| r60 | Wan T2V 832×480×33 | 8214.84 / 2909.22 / 2593.79 | 2488.75 / 2215.24 / 2176.92 |
| r60 | Wan T2V 832×480×65 | 8627.63 / 8700.40 / 7793.65 | 5249.36 / 5213.56 / 5197.27 |
| r60 | Wan I2V 832×480×81 | 9814.96 / 7685.79 / 7555.93 | 7227.49 / 7235.81 / 7121.52 |
| dense-grid | Wan T2V 832×480×33 | 8788.79 / 2762.00 / 2429.16 | 2458.62 / 2241.47 / 2190.93 |
| dense-grid | Wan T2V 832×480×65 | 7848.25 / 5768.00 / 5838.37 | 4883.08 / 4878.96 / 4881.51 |
| dense-grid | Wan I2V 832×480×81 | 13932.46 / 7769.01 / 7683.45 | 8376.47 / 7354.42 / 7187.16 |
| r60-repeat | Qwen 1536², no ref | 6350.68 / 1297.23 / 1097.22 | 1050.12 / 959.05 / 981.74 |
| dense-grid-repeat | Wan I2V 832×480×81 | 14321.95 / 7861.57 / 7546.60 | 7162.55 / 7162.49 / 7141.24 |

## Reproduction

In an isolated checkout, apply the repository's three experiment adapters. Set `MODEL` to the corresponding local checkpoint and `OUT` to an empty output directory. The following reproduces the 33-frame video workload.

```bash
BENCH=$PWD/benchmarks/diffusion/head_overlap
for p in native-probe model-probe qwen21-block-graph; do git apply "$BENCH/$p.patch"; done
mkdir -p "$OUT"
export PYTHONPATH="$BENCH:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_OMNI_PACKED_QKV=1 HCMS_MATRIX_AUDIT="$OUT/audit"
export HCMS_COPY_SUBMITTER="$OUT/copy_submitter.so"
c++ -O2 -std=c++17 -shared -fPIC "$BENCH/copy_submitter.cpp" -o "$HCMS_COPY_SUBMITTER"
python "$BENCH/query_tail_case.py" "$BENCH/head3-wan-t2v-matrix.json" wan14b-480x832-f33-real-prefix "$OUT/case.json" --rounds 18
python "$BENCH/model_matrix.py" --model "$MODEL" --config "$OUT/case.json" --out "$OUT" --rounds 18 --warmups 1
```

The Qwen and Wan I2V inputs come from `head3-qwen-matrix.json` and `head3-wan-i2v-matrix.json`. Each case ID in the request CSV maps to its source template. Use 24 rounds for Qwen's first pass, 72 for the independent no-reference repeat, 18 for Wan and 36 for the independent 81-frame repeat. The runtime check `python "$BENCH/check_lazy_runtime.py"` covers eligibility and rejection of incompatible layouts, backend, and causal semantics.

[All timed requests, output hashes and seeds (gzip CSV)](evidence/query-tail-cross-shape-requests.csv.gz)
