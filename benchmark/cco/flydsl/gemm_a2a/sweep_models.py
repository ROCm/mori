#!/usr/bin/env python3
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
"""Two real shapes: Llama-3.1-70B and Llama-3.1-405B QKV projection + Ulysses a2a.

    70B   M=2048  N=10240  K=8192    s=1280   slab 5.00 MiB
    405B  M=2048  N=18432  K=16384   s=2304   slab 9.00 MiB

N is the QKV width, K the hidden size, and s = N/8 the per-rank shard: for 70B
that is 8 q heads + 1 k + 1 v at 128 each, so a rank gets exactly one GQA group
and no padding is needed.

**Interleaved, not grouped.** Every configuration is measured once per round,
and the rounds are repeated. Grouping by configuration -- which is what the
earlier sweeps did -- puts any drift in the machine *between* the things being
compared; interleaving puts it inside each configuration's own samples, where
the spread makes it visible. This exists because one configuration measured
5153us four times running and 3508us four times running on a later day, with a
byte-identical kernel and an identical config, and nothing in between explained
it.

GPU clock and junction temperature are recorded next to every sample for the
same reason.
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

SETTLE = 25
RETRY_SETTLE = 90
# `ChildFailedError` on its own is too coarse to be a retry signal: torchrun
# reports *every* child failure that way, so an ImportError in the bench --
# which is what a missing PYTHONPATH overlay looks like -- was classified as a
# lost queue race and retried three times per cell, silently emptying a whole
# sweep. Only retry when a real SDMA-side marker is present.
QUEUE_RACE = ("anvil.cpp", "Allgather operation failed")

# M = S/P, so sweeping M is sweeping the sequence length: at P=8,
# M = 4096/8192/16384 is S = 32k/64k/128k. That is the dimension this operator
# actually moves along in deployment, and it is the one the earlier tables
# never varied -- they all sat at M=2048, which the M-curve says is the end
# where fusing pays least.
MODELS = {
    "70B": dict(n=10240, k=8192),
    "405B": dict(n=18432, k=16384),
}
MS = [2048, 4096, 8192, 16384]
SHAPES = {f"{name}@M{m}": dict(m=m, **cfg) for m in MS for name, cfg in MODELS.items()}

# gemm-only and split-sdma are carried over from the previous sweep as anchors:
# they let this run be placed against that table without assuming the two runs
# saw the same machine. split-rccl is the point of this one.
CONFIGS = [
    ("gemm-only", []),
    ("split-sdma", []),
    ("split-rccl", []),
    ("fused-sdma", ["--chunks", "4"]),
    ("fused-sdma", ["--chunks", "8"]),
]

#: ``--alt-tile``: the same GEMM-bearing configs at the tile ``default_block_m``
#: did *not* pick, so every table carries the evidence for its own tile choice.
ALT_CONFIGS = [
    ("gemm-only", []),
    ("fused-sdma", ["--chunks", "4"]),
    ("fused-sdma", ["--chunks", "8"]),
]


def alt_block_m(shape, quant):
    """The other legal ptpc tile for ``shape``, or None (mxfp8 has only one)."""
    if quant != "ptpc" or shape["m"] % 256:
        return None
    from mori.ops.gemm_a2a.layout import default_block_m

    return 128 if default_block_m(shape["m"], shape["n"], quant) == 256 else 256


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
        "-n",
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
    ap.add_argument("--quant", choices=("ptpc", "blockscale", "mxfp8"), default="ptpc")
    ap.add_argument("--out", default="a2a_models.jsonl")
    ap.add_argument(
        "--alt-tile",
        action="store_true",
        help="also run gemm-only and fused-sdma at the tile not picked by default",
    )
    args = ap.parse_args()

    samples = {}
    with open(args.out, "w") as fh:
        for rnd in range(1, args.rounds + 1):
            print(f"######## round {rnd}/{args.rounds}", flush=True)
            for sname, shape in SHAPES.items():
                alt = alt_block_m(shape, args.quant) if args.alt_tile else None
                configs = list(CONFIGS)
                if alt:
                    configs += [
                        (m, [*e, "--block-m", str(alt)]) for m, e in ALT_CONFIGS
                    ]
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
                        "block_m": r.get("block_m"),
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
                        f"  {key:<48}{r['us']:>9.1f}us{c}"
                        f"   [clk {r['_clk_mhz']}MHz {r['_temp_c']:.0f}C]",
                        flush=True,
                    )

    print("\n######## summary (median, spread across rounds)", flush=True)
    for key, xs in samples.items():
        sp = (max(xs) - min(xs)) / min(xs) * 100 if len(xs) > 1 else 0.0
        print(
            f"  {key:<48}{statistics.median(xs):>9.1f}us  spread {sp:>5.1f}%  n={len(xs)}",
            flush=True,
        )
    print("MODELSWEEPDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
