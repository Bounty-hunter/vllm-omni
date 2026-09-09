#!/usr/bin/env python3
"""Concurrent load client for the OpenAI-compatible /v1/audio/speech endpoint.

Used by the Qwen3-TTS L0 MPS A/B experiment. Fires ``--num-prompts``
synthesis requests at a fixed concurrency against one vLLM-Omni server and
reports throughput (qps), latency percentiles, error count, and total audio
bytes. Deterministic across runs (seeded prompt list) so MPS on/off
comparisons are apples-to-apples.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import httpx

_EN_POOL = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "A journey of a thousand miles begins with a single step taken today.",
    "Please keep the meeting room quiet while the presenter shares the slides.",
    "The weather forecast predicts sunshine and gentle breezes for the afternoon.",
    "She carefully arranged the books on the shelf by author and title.",
    "Our team shipped the new feature ahead of schedule last Friday morning.",
    "The museum opens at nine and offers guided tours every hour on weekdays.",
    "A warm cup of tea and a good book make a perfect rainy afternoon.",
    "The train to the city departs from platform three every half hour.",
    "Learning a new language opens the door to a different way of thinking.",
    "The garden was full of colorful flowers blooming under the summer sun.",
    "He wrote a short note to remind himself of the important meeting tomorrow.",
    "The little cafe on the corner serves the best coffee in the neighborhood.",
    "Scientists announced a breakthrough in battery technology this morning.",
    "The children laughed as they chased each other around the playground.",
    "Please attach the signed document and reply to this email by noon.",
]

_ZH_POOL = [
    "今天天气很好，我们一起去公园散步吧。",
    "这个项目的进度比预期快了不少，值得庆祝。",
    "请把会议纪要发给所有参会人员，谢谢合作。",
    "科技进步正在改变我们的日常生活方式。",
    "他认真地检查了每一份报告，确保没有错误。",
]


def build_prompts(num: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    pool = _EN_POOL + _ZH_POOL
    out: list[str] = []
    for i in range(num):
        sent = pool[(i + seed) % len(pool)]
        # unique per-request marker keeps texts non-identical without
        # changing the audio-generation length profile much.
        out.append(f"Utterance {i}: {sent}")
    return out


async def _one(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    timeout_s: float,
) -> dict:
    t0 = time.perf_counter()
    ok, err, nbytes = False, "", 0
    try:
        async with client.stream(
            "POST", url, json=payload, timeout=httpx.Timeout(timeout_s, connect=30.0)
        ) as resp:
            if resp.status_code != 200:
                snippet = (await resp.aread())[:200].decode("utf-8", "replace")
                err = f"http_{resp.status_code}: {snippet}"
            else:
                nbytes = 0
                async for chunk in resp.aiter_bytes():
                    nbytes += len(chunk)
                ok = True
    except Exception as e:  # noqa: BLE001 - record and continue
        err = f"{type(e).__name__}: {e}"
    return {
        "ok": ok,
        "err": err,
        "latency_s": time.perf_counter() - t0,
        "bytes": nbytes,
    }


async def _run(
    base_url: str,
    model: str,
    voice: str,
    language: str | None,
    concurrency: int,
    num_prompts: int,
    timeout_s: float,
    seed: int,
) -> dict:
    url = f"{base_url}/v1/audio/speech"
    prompts = build_prompts(num_prompts, seed)
    payloads = [
        {"model": model, "input": text, "voice": voice, "response_format": "wav"}
        for text in prompts
    ]
    if language:
        for p in payloads:
            p["language"] = language

    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    async with httpx.AsyncClient(limits=limits) as client:
        async def _bound(p: dict) -> dict:
            async with sem:
                return await _one(client, url, p, timeout_s)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(_bound(p) for p in payloads))
        wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    lats = sorted(r["latency_s"] for r in ok)

    def pct(p: float) -> float:
        if not lats:
            return 0.0
        idx = min(len(lats) - 1, int(p * len(lats)))
        return lats[idx]

    return {
        "base_url": base_url,
        "model": model,
        "voice": voice,
        "language": language,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "seed": seed,
        "ok": len(ok),
        "errors": len(errs),
        "wall_s": round(wall, 3),
        "qps": round(len(ok) / wall, 3) if wall > 0 else 0.0,
        "total_audio_bytes": sum(r["bytes"] for r in ok),
        "latency_s": {
            "mean": round(statistics.fmean(lats), 3) if lats else 0.0,
            "p50": round(pct(0.50), 3),
            "p90": round(pct(0.90), 3),
            "p95": round(pct(0.95), 3),
            "p99": round(pct(0.99), 3),
            "max": round(lats[-1], 3) if lats else 0.0,
        },
        "first_error": errs[0]["err"][:300] if errs else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18091")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--voice", default="vivian")
    ap.add_argument("--language", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    res = asyncio.run(
        _run(
            args.base_url,
            args.model,
            args.voice,
            args.language,
            args.concurrency,
            args.num_prompts,
            args.timeout,
            args.seed,
        )
    )
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(
        f"conc={res['concurrency']} n={res['num_prompts']} "
        f"ok={res['ok']} err={res['errors']} wall={res['wall_s']}s "
        f"qps={res['qps']} p50={res['latency_s']['p50']}s "
        f"p95={res['latency_s']['p95']}s p99={res['latency_s']['p99']}s"
    )


if __name__ == "__main__":
    main()
