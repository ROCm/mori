# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Sweep gemm+all_gather's knobs at one shape.

``sweep_models.py`` asks "which transport, across shapes"; this asks "which
setting of one transport, at one shape". The two questions this one exists for
are the ones all-gather raises and the other two collectives do not:

* **push or pull.** A broadcast can be driven from either end. The split pair
  differs in nothing else -- same bytes, same barrier, same GEMM -- so the gap
  between them is the direction and only the direction. ``gemm_ar``'s own
  gather leg found pull worth 5.6-10.5% over an SDMA push, and pull is also
  where a future low-precision wire would put its dequantize.
* **how much chunking costs.** Chunked pushing is what the fused path buys with
  its bookkeeping, so the ``--chunks`` ladder prices both halves of that trade
  at once.

One process per cell: a mori window and its SDMA queues are per-process, and a
stale counter set from one configuration electing on another's leftovers is a
failure that only shows up as a wrong answer.

    MORI_REPO=/workspace/reports/mori PYTHONPATH=$MORI_REPO/python \\
      python sweep_ag.py -m 2048 --out-dim 2048 -k 7168
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.environ.get("MORI_REPO", "/workspace/reports/mori")
BENCH = f"{REPO}/benchmark/cco/flydsl/gemm_ag/bench_gemm_ag.py"
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
        "--out-dim",
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
            # A timing whose answer is wrong is worth nothing, so this stops the
            # cell rather than recording it. `--no-put` reports validated=True
            # with relL2 nan by design -- it is a cost probe, not a transport.
            raise RuntimeError(f"{mode} {extra} did not validate: relL2={r['rel_l2']}")
        got.append(r["us"])
        print(f"    rep {i + 1}: {r['us']:.1f}us", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", type=int, default=2048)
    ap.add_argument("--out-dim", "-N", dest="n", type=int, default=2048)
    ap.add_argument("-k", type=int, default=7168)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=21)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="/root/ag_sweep.jsonl")
    args = ap.parse_args()

    plan = [
        # The ceiling, and the baseline the fusion is claimed against.
        ("gemm-only", []),
        ("gemm-to-window", []),
        ("split-rccl", []),
        # Direction, at the copy kernel's own defaults. One variable.
        ("split-lsa-push", []),
        ("split-lsa-pull", []),
        # Does the push loop's latency actually need hiding? unroll=1 is the
        # shape gemm_a2a's copy kernel has, where world independent loads
        # supplied the parallelism; a broadcast push has one load, so this
        # prices what replacing that costs.
        ("split-lsa-push", ["--push-unroll", "1"]),
        ("split-lsa-push", ["--push-unroll", "8"]),
        # sc0|sc1 versus letting the fabric-crossing access sit in L2.
        ("split-lsa-push", ["--no-lsa-uncached"]),
        ("split-lsa-pull", ["--no-lsa-uncached"]),
        # Grid for the copy kernel. LSA_BLOCK_CAP is inherited from an
        # all-reduce's access pattern and has never been measured on this one.
        ("split-lsa-pull", ["--copy-blocks", "8"]),
        ("split-lsa-pull", ["--copy-blocks", "48"]),
        # The fused LSA epilogue, and its two publication knobs.
        ("fused-lsa", []),
        ("fused-lsa", ["--peer-uncached"]),
        ("fused-lsa", ["--direct-fence", "all"]),
        # What the epilogue's bookkeeping costs with nothing transferred. The
        # answer is only interesting next to the line above it.
        ("fused-lsa", ["--no-put"]),
        # SDMA, split and fused, with the chunk ladder.
        ("split-sdma", []),
        ("split-sdma", ["--split-gemm", "epilogue"]),
        ("fused-sdma", ["--chunks", "1"]),
        ("fused-sdma", ["--chunks", "2"]),
        ("fused-sdma", ["--chunks", "4"]),
        ("fused-sdma", ["--chunks", "8"]),
        ("fused-sdma", ["--chunks", "4", "--sdma-queues", "4"]),
        ("fused-sdma", ["--chunks", "4", "--no-put"]),
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
