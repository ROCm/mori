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
"""fused-sdma under even and uneven chunk plans, interleaved by round.

Every arm of a cell runs once per round and the rounds repeat, so drift lands
inside each arm's samples rather than between arms. Plans are row tiles per
chunk (``--chunk-plan``); ``even:N`` means ``--chunks N``.

    python sweep_chunk_plans.py --rounds 3 --out plans.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).with_name("bench_gemm_a2a.py")

SHAPES = {"70B": (10240, 8192), "405B": (18432, 16384)}


def _rep(head, mid, tail):
    return ",".join(map(str, head + mid + tail))


#: (shape, m) -> arms. Row tiles are m/128; a wave is ~256/n_blocks rows
#: (6.4 for 70B, 3.6 for 405B), which is the granularity chunk completion
#: actually has, so the tapered tails stop at about one wave.
CELLS = {
    ("70B", 2048): [
        "even:4",
        "even:8",
        "6,6,4",
        "8,4,2,2",
        "4,4,4,2,2",
        "1,3,4,4,4",
        "1,3,8,3,1",
        "2,4,4,4,2",
    ],
    ("70B", 16384): [
        "even:8",
        "even:16",
        _rep([], [16] * 7, [8, 4, 4]),
        _rep([], [16] * 7, [8, 8]),
        _rep([4, 4, 8], [16] * 6, [8, 4, 4]),
        _rep([2, 2, 4, 8], [16] * 7, []),
    ],
    ("405B", 2048): [
        "even:8",
        "even:4",
        "4,4,4,2,2",
        "2,2,4,4,4",
        "2,2,4,4,2,2",
    ],
    ("405B", 16384): [
        "even:8",
        "even:16",
        _rep([], [16] * 7, [8, 4, 4]),
        _rep([], [16] * 7, [8, 4, 2, 2]),
        _rep([2, 2, 4, 8], [16] * 7, []),
        _rep([2, 2, 4, 8], [16] * 6, [8, 4, 2, 2]),
    ],
}


#: ``--suite tiles``: 128x256 against 256x256 under one quantisation, so the
#: tile and the quantisation stop being changed together. Arms are
#: ``mode:block_m:plan``; plans count row tiles of *that* block_m.
_T128 = {
    2048: ["even:4", "even:8", "6,6,4", "4,4,4,2,2"],
    8192: ["even:8", "even:16", _rep([], [8] * 7, [4, 2, 2])],
    16384: ["even:16", _rep([], [16] * 7, [8, 4, 4])],
}
_T256 = {
    2048: ["even:2", "even:4", "even:8", "4,2,1,1"],
    8192: ["even:4", "even:8", _rep([], [8] * 3, [4, 2, 2])],
    16384: ["even:8", "even:16", _rep([], [8] * 7, [4, 2, 2])],
}
TILE_CELLS = {
    (shape, m): ["gemm-only:128:", "gemm-only:256:"]
    + [f"fused-sdma:128:{p}" for p in _T128[m]]
    + [f"fused-sdma:256:{p}" for p in _T256[m]]
    for shape in SHAPES
    for m in (2048, 8192, 16384)
}


#: ``--suite quant``: ptpc against mxfp8 at the *same* 256x256 tile, so the
#: quantisation is the only thing that changes. Arms are
#: ``mode:block_m:plan:quant``.
_Q256 = {
    2048: ["even:4", "even:8"],
    8192: ["even:4", "even:8"],
    16384: ["even:8", "even:16"],
}
QUANT_CELLS = {
    (shape, m): [f"gemm-only:256::{q}" for q in ("ptpc", "mxfp8")]
    + [f"fused-sdma:256:{p}:{q}" for p in _Q256[m] for q in ("ptpc", "mxfp8")]
    for shape in SHAPES
    for m in (2048, 8192, 16384)
}


def wait_idle(poll_s=30):
    """Block until no KFD process is running; a neighbour changes the ranking."""
    waited = 0
    while True:
        out = subprocess.run(
            ["rocm-smi", "--showpids"], capture_output=True, text=True, check=False
        ).stdout
        if "No KFD PIDs currently running" in out:
            return waited
        time.sleep(poll_s)
        waited += poll_s


def run_arm(shape, m, plan, args, mode="fused-sdma", block_m=128, quant=None):
    n, k = SHAPES[shape]
    waited = wait_idle()
    if waited:
        print(f"  (waited {waited}s for an idle box)", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc}",
        str(BENCH),
        "--mode",
        mode,
        "--block-m",
        str(block_m),
        "-m",
        str(m),
        "-n",
        str(n),
        "-k",
        str(k),
        "--quant",
        quant or args.quant,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--phase-split",
    ]
    if plan.startswith("even:"):
        cmd += ["--chunks", plan.split(":", 1)[1]]
    elif plan:
        cmd += ["--chunk-plan", plan]
    env = dict(os.environ, MORI_ENABLE_SDMA="1", MORI_SOCKET_IFNAME="lo")
    for _ in range(3):
        # check=False: a failed launch is retried below, not raised.
        p = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        for line in p.stdout.splitlines():
            if line.startswith("RESULT_JSON "):
                return json.loads(line[len("RESULT_JSON ") :])
        # The SDMA queue reclamation race (see README) fails a launch now and
        # then; settle and retry rather than recording a hole.
        time.sleep(5)
    sys.stderr.write(p.stdout[-2000:] + p.stderr[-4000:])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cells", default="", help="e.g. 70B:2048,405B:16384")
    ap.add_argument("--quant", default="ptpc")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=21)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--suite", choices=("plans", "tiles", "quant"), default="plans")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    table = {
        "tiles": TILE_CELLS,
        "quant": QUANT_CELLS,
        "plans": {c: [f"fused-sdma:128:{p}" for p in v] for c, v in CELLS.items()},
    }[args.suite]
    cells = list(table)
    if args.cells:
        want = {(s, int(m)) for s, m in (c.split(":") for c in args.cells.split(","))}
        cells = [c for c in cells if c in want]
    with open(args.out, "a") as f:
        for r in range(args.rounds):
            for shape, m in cells:
                for arm in table[(shape, m)]:
                    mode, bm, rest = arm.split(":", 2)
                    # A trailing ":ptpc"/":mxfp8" names the quantisation; plans
                    # themselves may contain ":" ("even:4"), so split from the right.
                    quant = None
                    if rest.rsplit(":", 1)[-1] in ("ptpc", "mxfp8", "blockscale"):
                        rest, quant = rest.rsplit(":", 1)
                    plan = rest
                    res = run_arm(
                        shape, m, plan, args, mode=mode, block_m=int(bm), quant=quant
                    )
                    rec = {
                        "quant_arm": quant or args.quant,
                        "round": r,
                        "shape": shape,
                        "m": m,
                        "plan": plan,
                        "arm": arm,
                        "block_m": int(bm),
                    }
                    if res is None:
                        rec["error"] = True
                        print(f"r{r} {shape}@{m} {arm:44s} FAILED", flush=True)
                    else:
                        rec.update(res)
                        print(
                            f"r{r} {shape}@{m} {arm:44s} {res['us']:8.1f}us "
                            f"comm {res.get('phase_comm', float('nan')):6.1f} "
                            f"ok={res['validated']}",
                            flush=True,
                        )
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
    print("SWEEPDONE", flush=True)


if __name__ == "__main__":
    main()
