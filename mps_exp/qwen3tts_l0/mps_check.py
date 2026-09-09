#!/usr/bin/env python3
"""Check how many live processes are attached to a CUDA MPS pipe directory.

Usage: mps_check.py <pipe_dir> [cmdline-substring]

Scans /proc for processes whose environment sets CUDA_MPS_PIPE_DIRECTORY to
<pipe_dir> and (optionally) whose cmdline contains <cmdline-substring>
(e.g. the server port). Prints the matches and exits 0 if at least one match,
1 otherwise. A process that missed the pipe directory silently falls back to
time-slicing, so this is the real MPS-attach verification for an A/B run.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: mps_check.py <pipe_dir> [cmdline-substring]", file=sys.stderr)
        return 2
    pipe = sys.argv[1]
    needle = sys.argv[2] if len(sys.argv) > 2 else ""

    found: list[tuple[str, str]] = []
    for ent in os.listdir("/proc"):
        if not ent.isdigit():
            continue
        try:
            raw_env = open(f"/proc/{ent}/environ", "rb").read().split(b"\0")
            raw_cmd = open(f"/proc/{ent}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        env_val = ""
        for kv in raw_env:
            if not kv:
                continue
            k, _, v = kv.partition(b"=")
            if k == b"CUDA_MPS_PIPE_DIRECTORY":
                env_val = v.decode("utf-8", "replace")
                break
        if env_val != pipe:
            continue
        cmd = " ".join(c.decode("utf-8", "replace") for c in raw_cmd if c)
        if needle and needle not in cmd:
            continue
        found.append((ent, cmd[:160]))

    print(f"{len(found)} process(es) attached to MPS pipe {pipe}")
    for pid, cmd in found:
        print(f"  pid {pid}: {cmd}")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
