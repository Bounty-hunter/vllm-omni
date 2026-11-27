# Adaptive head overlap for dense diffusion attention

This offline experiment compares native Ulysses with an adaptive head split on a single NVLink node.
Both engines use `ulysses_a2a_backend="copy_engine"` and set `VLLM_OMNI_PACKED_QKV=1`. With this transport,
the native baseline uses the grouped Q/K/V CE exchange. A selected split sends the Q, K and V head
slices for each peer through the fixed-chunk CE runner. Both engines use the same graph boundary and inputs.

Enable the runtime policy with `--ulysses-a2a-backend copy_engine --ulysses-overlap-chunks auto`. The worker attaches
one policy to each declared DiT component. An eligible native forward discovers actual layouts. Later real
layers time native attention and one uneven split, including both exchanges and output ownership.
The ranks agree on a choice at a component boundary and keep native attention when evidence is weak.

This temporary experiment relearns the decision in every request; it retains only compiled kernels,
workspace and the agreed homogeneous layer structure. All small-first two-way partitions (up to 32
rank-local heads) are tried in three rotations. The query-wave estimate only orders those candidates;
it is not an asserted kernel grid. There is no query-tail split. Spare real layers remain native before
sampling so the host can build launch lead after preparation. Sampling still commits at the next
component boundary, which may wait for device events and one asynchronous rank-MAX reduction.
No probe forward is added, but first-use preparation and slow trial choices have real costs.

Producer-validated `[real, pad]` FlashAttention layouts retain their original cumulative lengths and
packed-varlen kernel when heads are partitioned. They require vLLM FA3 and `fa_deterministic=True`
in both arms, which pins split-KV to one partition. The default automatic split-KV heuristic can
change a small trial group's reduction and propagate numerical differences before selection.
Without this backend contract, packed layouts stay native. Masks, multi-document batches, sparse gates,
reference-cache hooks and quantized KV remain outside this experiment. Numerical coverage includes
`tests/diffusion/distributed/test_ulysses_packed_head_overlap.py` with bit-exact checks on two and four ranks. This is not a
claim that every packed model has an end-to-end speedup. New layouts during capture, compiled attention,
incompatible metadata, and unsupported GPUs use native attention.

## Run

From the repository root, use four NVLink GPUs, BF16 FA3 on SM90 and one of the supplied checkpoints.
The benchmark starts independent native and auto engines through the public configuration. It records the
first full request, repeated warm requests and decoded output hashes for seeds 42, 43 and 44.

```bash
bench=benchmarks/diffusion/head_overlap
export VLLM_OMNI_PACKED_QKV=1
python -m benchmarks.diffusion.head_overlap.benchmark --model "$MODEL" \\
  --config "$bench/qwen_image_21.json" --case qwen21-turbo-1536x1536-steps8-ref2 \\
  --rounds 24 --out "$OUT"
```

Use `wan_t2v.json` for 832×480 at 33 or 65 frames and `wan_i2v.json` for 81-frame image-to-video.
Run the selection and two-GPU transport tests with pytest.

```bash
pytest tests/diffusion/attention/test_ulysses_head_policy.py tests/diffusion/distributed/test_ulysses_ce_overlap.py
```

## Historical measurements

The original four-L20X SP4/VAE4 experiment used FP8 DiTs and BF16 FA3. Relative to packed-QKV native CE,
the measured steady-state latency reductions were 3.01% for Qwen 1536² with two references, 3.84% for
2048² with one, and 3.25% / 2.28% for Wan T2V at 33 / 65 frames. Wan I2V at 81 frames regressed 12.69%
over 18 rounds including a 24 s outlier and improved 2.33% on an independent 36-round repeat.
Qwen 1024² with two references fell back to native and measured 0.43% slower over 24 rounds.
These results predate the current upstream transport and native runtime integration. The original report includes all
shapes, first-request timings, paired intervals and individual observations.
[Historical experiment and request data](https://github.com/Bounty-hunter/vllm-omni/blob/34e7f98434bf70d3f5b12d7ab4579d98fa798fab/benchmarks/diffusion/head_overlap/QUERY-TAIL-RESULTS.md)
