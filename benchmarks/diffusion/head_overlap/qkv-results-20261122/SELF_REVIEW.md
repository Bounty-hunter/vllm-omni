# QKV experiment self-review

Snapshot: new work relative to `69ca5945a`; R16 measured source `ae925b009d6114241d64e499629eb754c09bda9c`.

Human intent is retained from the task: reduce diffusion request latency by
overlapping ready QKV communication with remaining producer work, preserve
precision, learn from real early layers, and keep future model adapters small.
The user requested a separate experimental branch and no PR.

| Check | Result |
| --- | --- |
| Scope | Opt-in benchmark prototype; production defaults and models unchanged |
| Source binding | Runtime AST matches R16/R15; R17 scheduling experiment reverted |
| Correctness | 1581 normal requests and 91 diagnostic requests match native Ulysses RGB hashes for seeds 42/43/44 |
| Device controls | 36 real-weight core controls, four ranks, exact QKV/core outputs |
| Ownership | 7 asymmetric producer/old-consumer stress jobs pass eager/graph replay |
| Policy | All-rank decisions agree; 24 actual core trials; no added probe forwards |
| Steady path | No Python core calls, sampling events, queries or control collectives |
| First shape | Preparation, exploration and capture costs reported; no zero-latency claim |
| Performance | 30-pair matrices; A/A fallbacks, smoke results and negative experiments retained |
| Quality | No new hot-path clone/deepcopy, broad exception swallow, manager or dependency |
| Examples | No new Python example paths; model glue is confined to the experiment |
| Contracts | No supported production API/default changes; fixed-case graph and backend limits documented |
| Static checks | Changed-scope pre-commit and diff whitespace checks; Windows suggestion hook skipped |
| Submission | No PR; AI assistance disclosed; human production review still outstanding |

The target of saving at least half the pure P2P time end to end is not met.
The new plan remains slower than the earlier head-split plan in the same R16
large-shape matrix. Physical causes of attention/other-component inflation
remain unproven. GQA, mask/varlen, TP, LoRA/reload, other quantizers and new
model families have not been qualified for this plan.
