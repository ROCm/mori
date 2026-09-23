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
"""Sweep gemm+all_gather over real model shapes and sequence lengths.

One process per cell, because a mori window and its SDMA queues are per-process
and reusing them across shapes is how a stale counter set from one shape elects
on another's leftovers. The harness -- idle detection, retry on a lost queue
race, clock/temperature capture -- is ``sweep_models.py`` from ``gemm_a2a``
unchanged; a neighbour on the box is worth 9% on a kernel that does no
communication at all, so samples taken with company are discarded rather than
averaged, and the GPU clock and junction temperature are recorded next to
every sample for the same reason.

    MORI_REPO=/workspace/reports/mori PYTHONPATH=$MORI_REPO/python \\
      python sweep_models.py --rounds 3 --quant ptpc --out ag_models.jsonl
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

SETTLE = 25
RETRY_SETTLE = 90
# `ChildFailedError` on its own is too coarse to be a retry signal: torchrun
# reports *every* child failure that way, so an ImportError in the bench --
# which is what a missing PYTHONPATH overlay looks like -- was classified as a
# lost queue race and retried three times per cell, silently emptying a whole
# sweep. Only retry when a real SDMA-side marker is present.
QUEUE_RACE = ("anvil.cpp", "Allgather operation failed")

# M = S/P, so sweeping M is sweeping the sequence length: at P=8,
# M = 512..4096 is S = 4k/8k/16k/32k. That is the dimension this operator
# actually moves along in deployment.
#
# The first two shapes are gemm_a2a's, carried over unchanged so the two
# operators' tables can be read against each other -- same GEMM, same box, only
# the collective differs. The wkv_gate pair is the one this op exists for: the
# DeepSeek V4-Pro prefill CP chain, where each context-parallel rank scores its
# own tokens and then everyone needs every token's score. ratio-4 layers use
# N=2048 and ratio-128 layers N=1024.
#
# N=1024 at world_size=8 is a shape gemm_a2a *cannot run*: its layout requires
# N % (world*block_n) == 0, i.e. a multiple of 2048. All-gather shards nothing,
# so its only rule is N % block_n == 0. That is not a detail -- it is half the
# ratio-128 layers.
MODELS = {
    "70B": dict(n=10240, k=8192),
    "405B": dict(n=18432, k=16384),
    "wkv_gate-r4": dict(n=2048, k=7168),
    "wkv_gate-r128": dict(n=1024, k=7168),
}
MS = [512, 1024, 2048, 4096]
SHAPES = {f"{name}@M{m}": dict(m=m, **cfg) for m in MS for name, cfg in MODELS.items()}

# split-rccl is the baseline the fusion is claimed against -- the collective a
# model would actually call. gemm-only sizes the ceiling. The rest are the
# transports, each in its split and fused form, which is the comparison the
# whole operator exists to make.
CONFIGS = [
    ("gemm-only", []),
    ("split-rccl", []),
    ("split-lsa-push", []),
    ("split-lsa-pull", []),
    ("fused-lsa", []),
    ("split-sdma", []),
    ("fused-sdma", ["--chunks", "4"]),
]


def foreign_pids():
    """KFD processes that are not ours. Empty list means the box is ours.

    A neighbour costs more than it looks: a *pure GEMM* measured 179.8us with
    the box to itself and 196.3us with company, 9% on a kernel that does no
    communication at all. Samples taken with company are discarded rather than
    averaged -- averaging them is how a configuration came to measure 5153us
    four times running and 3508us four times running with a byte-identical
    kernel, which cost most of a day to not explain.
    """
    try:
        out = subprocess.run(
            ["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=60
        ).stdout
    except Exception:
        return ["?"]
    if "No KFD PIDs currently running" in out:
        return []
    return [
        ln.split()[0]
        for ln in out.splitlines()
        if re.match(r"^\d+\s", ln.strip()) and ln.strip().split()[0].isdigit()
    ]


def wait_for_idle(limit=40):
    """Block until nobody else is on the GPUs, or give up and say so."""
    for _ in range(limit):
        if not foreign_pids():
            return True
        time.sleep(30)
    return False


def gpu_state():
    """sclk and junction temperature of GPU 0, for the record."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showclocks", "--showtemp"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        sclk = re.search(r"GPU\[0\].*sclk clock level: \S+ \((\d+)Mhz\)", out)
        temp = re.search(r"GPU\[0\].*junction\) \(C\): ([\d.]+)", out)
        return (
            int(sclk.group(1)) if sclk else -1,
            float(temp.group(1)) if temp else -1.0,
        )
    except Exception:
        return (-1, -1.0)


def launch(mode, shape, extra, quant="ptpc", warmup=30, iters=21, timeout=3000):
    time.sleep(SETTLE)
    env = dict(os.environ)
    env.update(MORI_ENABLE_SDMA="1", MORI_SOCKET_IFNAME="lo", PYTHONPATH=OVERLAY)
    cmd = [
        PY,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        BENCH,
        "--mode",
        mode,
        "-m",
        str(shape["m"]),
        "--out-dim",
        str(shape["n"]),
        "-k",
        str(shape["k"]),
        "--quant",
        quant,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--phase-split",
        *extra,
    ]
    if not wait_for_idle():
        raise RuntimeError("gave up waiting for an idle box")
    pre = gpu_state()
    p = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout
    )
    out = p.stdout + p.stderr
    # Ours have exited by now, so anything left is a neighbour that was running
    # during the measurement.
    time.sleep(5)
    if foreign_pids():
        print("    (a neighbour appeared mid-run; discarding)", flush=True)
        return None
    hit = [ln for ln in out.splitlines() if ln.startswith("RESULT_JSON ")]
    if hit:
        r = json.loads(hit[-1].removeprefix("RESULT_JSON "))
        r["_clk_mhz"], r["_temp_c"] = pre
        return r
    if any(t in out for t in QUEUE_RACE):
        return None
    tail = "\n".join(
        ln for ln in out.splitlines() if re.search(r"Error|error:|Traceback|FAILED", ln)
    )[-1000:]
    raise RuntimeError(f"{mode} {extra} {shape} failed:\n{tail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3)
    # A full round is len(MODELS)*len(MS)*len(CONFIGS) processes, each paying
    # SETTLE plus a FlyDSL compile. Filtering is how a targeted question --
    # "does fusion pay at the long-sequence end?" -- gets answered in minutes
    # instead of a working day.
    ap.add_argument(
        "--models", default="", help="comma-separated subset of MODELS; all if empty"
    )
    ap.add_argument("--ms", default="", help="comma-separated subset of MS")
    ap.add_argument(
        "--modes", default="", help="comma-separated subset of CONFIGS' modes"
    )
    ap.add_argument("--quant", choices=("ptpc", "blockscale", "mxfp8"), default="ptpc")
    ap.add_argument("--out", default="ag_models.jsonl")
    args = ap.parse_args()

    models = args.models.split(",") if args.models else list(MODELS)
    ms = [int(x) for x in args.ms.split(",")] if args.ms else MS
    shapes = {
        f"{name}@M{m}": dict(m=m, **MODELS[name]) for m in ms for name in models
    }
    configs = (
        [c for c in CONFIGS if c[0] in args.modes.split(",")]
        if args.modes
        else CONFIGS
    )
    if not shapes or not configs:
        raise SystemExit("--models/--ms/--modes selected nothing")

    samples = {}
    with open(args.out, "w") as fh:
        for rnd in range(1, args.rounds + 1):
            print(f"######## round {rnd}/{args.rounds}", flush=True)
            for sname, shape in shapes.items():
                for mode, extra in configs:
                    key = f"{sname} {mode} {' '.join(extra)}".strip()
                    r = None
                    for _ in range(3):
                        try:
                            r = launch(mode, shape, extra, quant=args.quant)
                        except RuntimeError as e:
                            print(f"  {key}: FAILED {e}", flush=True)
                            r = "err"
                            break
                        if r is not None:
                            break
                        print(f"  {key}: queue race, retrying", flush=True)
                        time.sleep(RETRY_SETTLE)
                    if r in (None, "err"):
                        continue
                    if not r["validated"]:
                        print(f"  {key}: NOT VALIDATED {r['rel_l2']}", flush=True)
                        continue
                    rec = {
                        "round": rnd,
                        "key": key,
                        "shape": sname,
                        "mode": mode,
                        "extra": extra,
                        "us": r["us"],
                        "comm": r.get("phase_comm"),
                        "compute": r.get("phase_compute"),
                        "clk_mhz": r["_clk_mhz"],
                        "temp_c": r["_temp_c"],
                        **{k: r[k] for k in ("m", "n", "k", "rel_l2")},
                    }
                    samples.setdefault(key, []).append(r["us"])
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    c = f"  comm {r['phase_comm']:.1f}" if r.get("phase_comm") else ""
                    print(
                        f"  {key:<34}{r['us']:>9.1f}us{c}"
                        f"   [clk {r['_clk_mhz']}MHz {r['_temp_c']:.0f}C]",
                        flush=True,
                    )

    print("\n######## summary (median, spread across rounds)", flush=True)
    for key, xs in samples.items():
        sp = (max(xs) - min(xs)) / min(xs) * 100 if len(xs) > 1 else 0.0
        print(
            f"  {key:<34}{statistics.median(xs):>9.1f}us  spread {sp:>5.1f}%  n={len(xs)}",
            flush=True,
        )
    print("MODELSWEEPDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
