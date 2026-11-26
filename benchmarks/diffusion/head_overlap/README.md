# Adaptive head overlap for dense diffusion attention

This offline experiment compares native Ulysses with an adaptive head split on a single NVLink node.
Both arms use grouped CE, packed QKV, the same graph boundary and the same inputs.
The optional shared-output single-group control isolates the effect of splitting from output-buffer ownership.

The first real DiT call discovers the QKV layouts. Across layers of the second call, the policy times whole
attention and one uneven split, including both exchanges. A query-tile wave estimate proposes the smallest
leading head group that fits the wave budget. When possible, the final group computes remote query rows first.
Its remote output returns while the local query tail attends to the full K/V. This requires dense, noncausal
BF16 FA3 on SM90. The Qwen decode path also accepts a replicated K/V prefix with zero joint query rows.
Wan supports a single fully occupied video grid.

At graph capture the workers discard CPU-starved samples, take the maximum rank time and compare paired cores.
Insufficient evidence keeps the native graph when every layout stays whole. A mixed graph uses a shared-output
whole plan for its unsplit layouts. Each real prefix call returns its output, and graph replay executes the frozen
schedule. The first request pays for candidate preparation, timing and capture.

## Run

Run from the repository root in the GPU environment with four peer-accessible GPUs and the chosen checkpoint.
The benchmark selects a worker extension for this engine. All timings cover offline decoded output from
`Omni.generate`. The first request for each arm, the interleaved comparisons and the extra correctness seeds
are labeled individually in `results.json`. Each worker writes its chosen partition and counts to `audit/`.

```bash
bench=benchmarks/diffusion/head_overlap
export VLLM_OMNI_PACKED_QKV=1
export HCMS_COPY_SUBMITTER="$PWD/copy_submitter.so"
c++ -O2 -std=c++17 -shared -fPIC "$bench/copy_submitter.cpp" -o "$HCMS_COPY_SUBMITTER"
python -m benchmarks.diffusion.head_overlap.benchmark --model "$MODEL" \
  --config "$bench/qwen_image_21.json" --case qwen21-turbo-1536x1536-steps8-ref2 \
  --rounds 24 --out "$OUT"
```

Use `wan_t2v.json` for 832×480 at 33 or 65 frames and `wan_i2v.json` for 81-frame image-to-video.
Pass `--shared-whole` to include the output-ownership control. Use a fresh engine and output directory per case.
The driver checks decoded output hashes, shapes and dtypes across all arms for seeds 42, 43 and 44.

```bash
python -m benchmarks.diffusion.head_overlap.check_policy
torchrun --standalone --nproc-per-node=2 --module benchmarks.diffusion.head_overlap.check_transport
```

## Historical measurements

The original four-L20X SP4/VAE4 experiment used FP8 DiTs and BF16 FA3. Relative to packed-QKV native CE,
the measured steady-state latency reductions were 3.01% for Qwen 1536² with two references, 3.84% for
2048² with one, and 3.25% / 2.28% for Wan T2V at 33 / 65 frames. Wan I2V at 81 frames regressed 12.69%
over 18 rounds including a 24 s outlier and improved 2.33% on an independent 36-round repeat.
Qwen 1024² with two references fell back to native and measured 0.43% slower over 24 rounds.
These results predate the current upstream transport and worker extension. The original report includes all
shapes, first-request timings, paired intervals and individual observations.
[Historical experiment and request data](https://github.com/Bounty-hunter/vllm-omni/blob/34e7f98434bf70d3f5b12d7ab4579d98fa798fab/benchmarks/diffusion/head_overlap/QUERY-TAIL-RESULTS.md)
