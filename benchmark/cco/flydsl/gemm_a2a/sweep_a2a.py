#!/usr/bin/env python3
"""Measure mori.ops.gemm_a2a's five modes on an idle box.

Three repeats per cell, median of the three medians reported. Each launch is a
fresh 8-rank process group, and back-to-back launches lose the SDMA queue
reclamation race often enough (anvil.cpp:237) that a bare loop cannot finish --
so every launch settles first and a lost race is retried rather than recorded
as a result.

Writes one JSON line per cell to the output file as it goes, so a run that is
interrupted still leaves everything it had measured.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.environ.get("MORI_REPO", "/workspace/mori")
BENCH = f"{REPO}/benchmark/cco/flydsl/gemm_a2a/bench_gemm_a2a.py"
PY = os.environ.get("MORI_PYTHON", sys.executable)
OVERLAY = os.environ.get("PYTHONPATH", "")

SETTLE = 25  # between launches, for SDMA queue teardown
RETRY_SETTLE = 90  # after a lost queue race
QUEUE_RACE = ("anvil.cpp", "Allgather operation failed", "ChildFailedError")


def launch(mode, m, n, k, extra, world=8, warmup=8, iters=21, timeout=2400):
    """One process group. Returns the RESULT_JSON dict, or None if it raced."""
    time.sleep(SETTLE)
    env = dict(os.environ)
    env.update(
        MORI_ENABLE_SDMA="1",
        MORI_SOCKET_IFNAME="lo",
        PYTHONPATH=OVERLAY,
    )
    cmd = [
        PY,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world}",
        BENCH,
        "--mode",
        mode,
        "-m",
        str(m),
        "-n",
        str(n),
        "-k",
        str(k),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        *extra,
    ]
    p = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout
    )
    out = p.stdout + p.stderr
    hit = [ln for ln in out.splitlines() if ln.startswith("RESULT_JSON ")]
    if hit:
        return json.loads(hit[-1].removeprefix("RESULT_JSON "))
    if any(t in out for t in QUEUE_RACE):
        return None
    # Something else went wrong; show enough to act on rather than swallowing it.
    tail = "\n".join(
        ln
        for ln in out.splitlines()
        if re.search(r"Error|error:|Traceback|assert|FAILED", ln)
    )[-1500:]
    raise RuntimeError(f"{mode} {extra} failed:\n{tail}")


def cell(mode, extra, args, repeats=3):
    """Repeat a configuration, returning every accepted sample."""
    got = []
    for i in range(repeats):
        for attempt in range(3):
            r = launch(
                mode,
                args.m,
                args.n,
                args.k,
                extra,
                warmup=args.warmup,
                iters=args.iters,
            )
            if r is not None:
                break
            print(f"    (queue race, retrying after {RETRY_SETTLE}s)", flush=True)
            time.sleep(RETRY_SETTLE)
        else:
            print("    (gave up on this repeat)", flush=True)
            continue
        if not r["validated"]:
            raise RuntimeError(f"{mode} {extra} did not validate: relL2={r['rel_l2']}")
        got.append(r["us"])
        print(f"    rep {i + 1}: {r['us']:.1f}us", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", type=int, default=2048)
    ap.add_argument("-n", type=int, default=18432)
    ap.add_argument("-k", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=21)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="/root/a2a_sweep.jsonl")
    args = ap.parse_args()

    plan = [
        ("gemm-only", []),
        ("split-lsa", []),
        ("split-sdma", []),
        ("fused-lsa", ["--rotated", "--n-stripe", "1"]),
        ("fused-lsa", ["--no-rotated"]),
        ("fused-lsa", ["--rotated", "--n-stripe", "3"]),
        ("fused-sdma", ["--chunks", "1"]),
        ("fused-sdma", ["--chunks", "2"]),
        ("fused-sdma", ["--chunks", "4"]),
        ("fused-sdma", ["--chunks", "8"]),
        ("fused-sdma", ["--chunks", "16"]),
    ]

    with open(args.out, "w") as fh:
        for mode, extra in plan:
            label = f"{mode} {' '.join(extra)}".strip()
            print(f"=== {label}", flush=True)
            try:
                samples = cell(mode, extra, args, args.repeats)
            except RuntimeError as e:
                print(f"    FAILED: {e}", flush=True)
                fh.write(json.dumps({"label": label, "error": str(e)[:400]}) + "\n")
                fh.flush()
                continue
            if not samples:
                fh.write(json.dumps({"label": label, "error": "no samples"}) + "\n")
                fh.flush()
                continue
            rec = {
                "label": label,
                "mode": mode,
                "extra": extra,
                "samples": samples,
                "median": statistics.median(samples),
                "spread_pct": (max(samples) - min(samples)) / min(samples) * 100,
            }
            print(
                f"    -> median {rec['median']:.1f}us, "
                f"spread {rec['spread_pct']:.1f}%",
                flush=True,
            )
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print("SWEEPDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
