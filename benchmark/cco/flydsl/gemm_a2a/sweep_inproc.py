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
"""Every case of a sweep in one process, on one communicator.

``sweep_models.py`` launches torchrun once per case and settles 25 s between
launches, so a box spends most of a sweep with its GPUs idle -- which a cluster
that reclaims idle GPUs reads, correctly, as idle. Here one job builds every
case on the same communicator, window and dev-comm, interleaves the rounds
in-process, and times each case with ``bench_gemm_a2a``'s ``stream`` mode
(launches back to back, an event between each, for at least ``--min-seconds``),
so the GPUs stay busy from the first case to the last.

The window is sized for the largest case and zeroed before each one: the
election counters, call counter and credits are read before they are written.

Kernels are built once per process and reused by later rounds: FlyDSL traces a
kernel in Python on every build, cache hit or not, and that trace is most of a
case's wall time. Rows already in ``--out`` are skipped, so a stopped sweep
resumes where it was.

    torchrun --nproc_per_node=8 sweep_inproc.py --suite models --quant ptpc \\
        --alt-tile --rounds 3 --out models.jsonl
    torchrun --nproc_per_node=4 sweep_inproc.py --suite minimax --rounds 1 \\
        --round-offset 0 --out minimax.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_gemm_a2a as B  # noqa: E402

from mori.cco import (  # noqa: E402
    CCODevCommRequirements,
    Communicator,
    GDA_CONNECTION_NONE,
)
from mori.ops.gemm_a2a.layout import default_block_m  # noqa: E402

#: The two QKV shapes of the README's model tables, at P=8.
MODELS = {"70B": dict(n=10240, k=8192), "405B": dict(n=18432, k=16384)}
MS = (2048, 4096, 8192, 16384)
CONFIGS = [
    ("gemm-only", []),
    ("split-sdma", []),
    ("split-rccl", []),
    ("fused-sdma", ["--chunks", "4"]),
    ("fused-sdma", ["--chunks", "8"]),
]
ALT_CONFIGS = [
    ("gemm-only", []),
    ("fused-sdma", ["--chunks", "4"]),
    ("fused-sdma", ["--chunks", "8"]),
]

#: MiniMax-M3's QKV under Ulysses SP4 (aiter #6689's large path): dense is
#: Q|K|V per destination, 2304 columns; indexer adds IQ|IK, 2560, which mori can
#: only express with IK's 128 weight rows replicated per destination (+3.9%
#: compute against aiter's computed-once IK).
MINIMAX = {"dense": dict(n=9216, k=6144), "indexer": dict(n=10240, k=6144)}
MINIMAX_ARMS = {  # M -> (block_m, chunks); the first is GemmA2AOp's pick at P=4
    2048: [(128, 8), (128, 4), (256, 4)],
    4096: [(256, 8), (128, 8), (256, 4)],
    8192: [(256, 8), (128, 16), (256, 16)],
}


def _memo(fn):
    """``fn`` with its results kept per argument list (configs are frozen
    dataclasses, so their repr is their identity)."""
    built = {}

    def wrapped(*args, **kwargs):
        key = repr((args, sorted(kwargs.items())))
        if key not in built:
            built[key] = fn(*args, **kwargs)
        return built[key]

    return wrapped


for _name in (
    "compile_fused_gemm_a2a",
    "compile_gemm_local",
    "build_sdma_phases",
    "build_lsa_a2a",
    "build_lsa_barrier",
):
    setattr(B, _name, _memo(getattr(B, _name)))


def _done(path: str) -> set:
    """(round, key) of the rows ``path`` already holds."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(d["round"], d["key"]) for d in map(json.loads, f) if "us" in d}


def _cases(a, world: int, num_cus: int):
    out = []
    if a.suite == "models":
        for m in MS:
            for name, shape in MODELS.items():
                n, k = shape["n"], shape["k"]
                configs = [(mode, list(extra), 0) for mode, extra in CONFIGS]
                if a.alt_tile and a.quant == "ptpc" and m % 256 == 0:
                    pick = default_block_m(m, n, a.quant, num_cus=num_cus)
                    alt = 128 if pick == 256 else 256
                    configs += [(mode, list(extra), alt) for mode, extra in ALT_CONFIGS]
                for mode, extra, bm in configs:
                    key = f"{name}@M{m} {mode} {' '.join(extra)}".strip()
                    if bm:
                        key += f" --block-m {bm}"
                    argv = ["--mode", mode, "-m", str(m), "-n", str(n), "-k", str(k)]
                    argv += ["--quant", a.quant, "--block-m", str(bm), *extra]
                    if mode != "gemm-only":
                        argv += ["--phase-split"]
                    out.append((key, name, m, argv))
    else:
        for case, shape in MINIMAX.items():
            for m, arms in MINIMAX_ARMS.items():
                for i, (bm, c) in enumerate(arms):
                    key = f"{case}@M{m} fused-sdma bm{bm} c{c}" + (
                        " (op default)" if i == 0 else ""
                    )
                    argv = ["--mode", "fused-sdma", "-m", str(m), "-n", str(shape["n"])]
                    argv += ["-k", str(shape["k"]), "--block-m", str(bm)]
                    argv += ["--chunks", str(c)]
                    out.append((key, case, m, argv))
                if case == "dense":
                    for mode in ("gemm-only", "split-rccl"):
                        argv = ["--mode", mode, "-m", str(m), "-n", str(shape["n"])]
                        argv += ["-k", str(shape["k"]), "--block-m", "0"]
                        out.append((f"{case}@M{m} {mode}", case, m, argv))
    common = ["--warmup", str(a.warmup), "--iters", str(a.iters)]
    common += ["--timing", "stream", "--min-seconds", str(a.min_seconds)]
    return [
        (key, name, m, B.build_parser().parse_args(argv + common))
        for key, name, m, argv in out
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=("models", "minimax"), required=True)
    ap.add_argument("--quant", default="ptpc")
    ap.add_argument("--alt-tile", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--round-offset", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--min-seconds", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    local_rank, rank, world, uid = B._setup_distributed()
    B.TIMING, B.MIN_SECONDS = "stream", a.min_seconds
    num_cus = torch.cuda.get_device_properties(local_rank).multi_processor_count
    cases = _cases(a, world, num_cus)
    done = [_done(a.out) if rank == 0 else None]
    dist.broadcast_object_list(done, src=0)
    done = done[0]
    # Resolve each case's tile and size its window once, up front.
    sizes = []
    for _, _, _, args in cases:
        P = B._plan_case(args, world, rank, local_rank)
        sizes.append(P.alloc_bytes)
        del P
    torch.cuda.empty_cache()
    cap = max(sizes)
    with Communicator.init(
        world, rank, uid, per_rank_vmm=2 * cap + (256 << 20)
    ) as comm:
        mem = comm.alloc_mem(cap)
        win = comm.register_window(mem.ptr, mem.size)
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1
        dc = comm.create_dev_comm(reqs)
        out = open(a.out, "a") if rank == 0 else None
        for r in range(a.rounds):
            rnd = a.round_offset + r
            order = cases if rnd % 2 == 0 else cases[::-1]
            for key, name, m, args in order:
                if (rnd, key) in done:
                    continue
                P = B._plan_case(args, world, rank, local_rank)
                t0 = time.time()
                res = B._run_case(comm, mem, win, args, P, rank, world, dev_comm=dc)
                del P
                if rank == 0:
                    rec = {
                        "round": rnd,
                        "key": key,
                        "shape": name,
                        "suite": a.suite,
                        "wall_s": round(time.time() - t0, 1),
                        **res,
                    }
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(
                        f"r{rnd} {key:<48} {res['us']:>9.1f}us "
                        f"ok={res['validated']}",
                        flush=True,
                    )
        if out:
            out.close()
        win.close()
        mem.close()
    dist.barrier()
    if rank == 0:
        print("INPROCDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
