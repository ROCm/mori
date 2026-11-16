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
"""Time and check a fused GEMM + all-reduce that implements the gemm_ar_bf16 interface.

    torchrun --nproc_per_node=8 op_harness.py --impl <file.py:Class> --mode time -m 16384
    torchrun --nproc_per_node=8 op_harness.py --impl <file.py:Class> --mode check

It has the shape the FlyDSL benchmarks share (``build_parser()``, ``run(args)``, a
``RESULT_JSON`` line from rank 0), so ``run.py --arms`` interleaves it with any other
benchmark, and the existing operator and a candidate are timed and checked by the same
code. The interface is in ``interface.py``.

Operands depend only on (seed, rank, salt, M), so every rank rebuilds every rank's
operands and computes the fp32 reference locally: no collective is trusted to check a
collective.

``time``: one call per CUDA graph, replayed ``--iters`` times after ``--warmup``, median
per rank, maximum over ranks (the method of ``benchmark/cco/flydsl/gemm_ar``'s
benchmark), then one more call checked against the reference.

``check``: the target M plus ``--held-out`` values of M drawn from ``[1, m_max]`` with the
seed (one of them not a multiple of 256, and M=1). For each M: ``--repeat-calls`` eager
calls with fresh inputs, then a CUDA graph replayed ``--replays`` times with fresh inputs
copied in before each replay. A random per-rank delay of up to ``--skew-ms`` precedes
every call. With ``debug_preset_counters`` the same runs again with every counter preset
just below 2^32. Every call has a watchdog (``--call-timeout``); a call that does not
finish ends the process with status 3.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import interface  # noqa: E402

TOLERANCE = 6e-3
WRAP_PRESET = 2**32 - 3


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--impl", required=True, help="file.py:Class or package.module:Class"
    )
    p.add_argument("--mode", choices=("time", "check"), default="time")
    p.add_argument("-m", type=int, default=16384)
    p.add_argument("-n", type=int, default=7168)
    p.add_argument("-k", type=int, default=2048)
    p.add_argument("--m-max", type=int, default=0, help="0 = -m")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tolerance", type=float, default=TOLERANCE, help="largest rel_l2 vs fp32"
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=51)
    p.add_argument("--held-out", type=int, default=3)
    p.add_argument("--repeat-calls", type=int, default=3)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--skew-ms", type=float, default=5.0)
    p.add_argument("--call-timeout", type=float, default=120.0)
    p.add_argument("--json-out")
    return p


def load_impl(spec):
    """``path/to/file.py:Class`` (relative to the repository) or ``package.module:Class``."""
    target, _, attr = spec.rpartition(":")
    if not target or not attr:
        raise ValueError(f"--impl {spec!r}: expected <file.py or module>:<Class>")
    if target.endswith(".py"):
        path = Path(target)
        path = path if path.is_absolute() else REPO / path
        name = f"impl_{path.stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
        mspec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(mspec)
        sys.modules[name] = module
        mspec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    cls = getattr(module, attr)
    gaps = interface.missing(cls)
    if gaps:
        raise TypeError(f"{spec} lacks {', '.join(gaps)} (see interface.py)")
    return cls


def held_out_ms(m, m_max, count, seed):
    """M values the implementer did not choose: M=1, one off the 256 grid, the rest random."""
    rng = random.Random(seed * 7919 + 17)
    out = []
    if count >= 1:
        out.append(1)
    if count >= 2 and m_max > 257:
        x = rng.randrange(257, m_max)
        out.append(x if x % 256 else x - 1)
    while len(out) < count:
        out.append(rng.randrange(1, m_max + 1))
    return [x for x in dict.fromkeys(out) if x != m]


def _gen(seed, *parts):
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(
        int(hashlib.sha256(repr((seed, *parts)).encode()).hexdigest()[:15], 16)
    )
    return g


def activation(seed, rank, salt, m, k, device):
    import torch

    x = torch.randn(m, k, generator=_gen(seed, "a", rank, salt, m)) / 8
    return x.to(device=device, dtype=torch.bfloat16)


def weight(seed, rank, n, k, device):
    import torch

    x = torch.randn(n, k, generator=_gen(seed, "b", rank)) / 8
    return x.to(device=device, dtype=torch.bfloat16)


def reference(seed, world, salt, m, n, k, device):
    """The fp32 sum over ranks of a_r @ b_r.T, from every rank's operands rebuilt here."""
    import torch

    acc = torch.zeros(m, n, dtype=torch.float32, device=device)
    for r in range(world):
        acc += (
            activation(seed, r, salt, m, k, device).float()
            @ weight(seed, r, n, k, device).float().t()
        )
    return acc


def rel_l2(got, ref):
    import torch

    return (
        torch.linalg.vector_norm(got.float() - ref) / torch.linalg.vector_norm(ref)
    ).item()


def merge(per_rank):
    """Cases from every rank -> one list: the largest rel_l2 and pass only if every rank passed."""
    merged = []
    for cases in zip(*per_rank):
        first = dict(cases[0])
        first["rel_l2"] = max(c["rel_l2"] for c in cases)
        first["pass"] = all(c["pass"] for c in cases)
        first["fused_path_ran"] = (
            None
            if any(c["fused_path_ran"] is None for c in cases)
            else all(c["fused_path_ran"] for c in cases)
        )
        merged.append(first)
    return merged


class _Run:
    def __init__(self, args, rank, world):
        import torch

        self.args, self.rank, self.world = args, rank, world
        self.torch = torch
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.skew = random.Random(args.seed * 1_000_003 + rank)

    def finish(self, what):
        """Wait for the stream, with a deadline. Busy-polls: a sleep between polls lets the
        ranks start their next call up to a sleep apart, and a rank that starts early then
        waits for its peers inside the collective, which the timing would count."""
        event = self.torch.cuda.Event()
        event.record()
        deadline = time.time() + self.args.call_timeout
        polls = 0
        while not event.query():
            polls += 1
            if polls % 4096 == 0 and time.time() > deadline:
                print(
                    f"[rank {self.rank}] op_harness: {what} did not finish in {self.args.call_timeout}s",
                    flush=True,
                )
                if self.rank == 0:
                    record = {
                        "mode": self.args.mode,
                        "impl": self.args.impl,
                        "validated": False,
                        "hang": what,
                    }
                    print("RESULT_JSON " + json.dumps(record), flush=True)
                os._exit(3)

    def delay(self):
        if self.args.skew_ms > 0:
            time.sleep(self.skew.uniform(0, self.args.skew_ms) / 1000.0)

    def case(self, op, kind, m, salt, out):
        ref = reference(
            self.args.seed, self.world, salt, m, self.args.n, self.args.k, self.device
        )
        err = rel_l2(out, ref)
        fused = getattr(op, "fused_path_ran", None)
        ok = err <= self.args.tolerance and fused is not False
        return {
            "kind": kind,
            "m": m,
            "salt": salt,
            "rel_l2": err,
            "pass": ok,
            "fused_path_ran": fused,
        }

    def eager(self, op, kind, m, salts):
        cases = []
        for salt in salts:
            a = activation(self.args.seed, self.rank, salt, m, self.args.k, self.device)
            self.delay()
            out = op(a)
            self.finish(f"{kind} M={m} salt={salt}")
            cases.append(self.case(op, kind, m, salt, out.clone()))
        return cases

    def graph(self, op, kind, m, salts):
        torch = self.torch
        a_buf = activation(
            self.args.seed, self.rank, salts[0], m, self.args.k, self.device
        )
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(a_buf)
        torch.cuda.current_stream().wait_stream(side)
        self.finish(f"{kind} warm-up M={m}")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = op(a_buf)
        cases, check_at = [], {0, len(salts) // 2, len(salts) - 1}
        for i, salt in enumerate(salts):
            a_buf.copy_(
                activation(self.args.seed, self.rank, salt, m, self.args.k, self.device)
            )
            self.delay()
            g.replay()
            self.finish(f"{kind} replay {i} M={m}")
            if i in check_at:
                cases.append(self.case(op, kind, m, salt, out.clone()))
        del g
        return cases

    def check(self, op):
        a = self.args
        ms = [a.m] + held_out_ms(a.m, a.m_max, a.held_out, a.seed)
        cases = []
        for i, m in enumerate(ms):
            base = 1000 * (i + 1)
            cases += self.eager(op, "eager", m, range(base, base + a.repeat_calls))
            cases += self.graph(
                op, "graph", m, range(base + 100, base + 100 + a.replays)
            )
        wrap = "not covered"
        if hasattr(op, "debug_preset_counters"):
            self.torch.cuda.synchronize()
            op.debug_preset_counters(WRAP_PRESET)
            self.torch.cuda.synchronize()
            cases += self.eager(
                op, "wrap-eager", a.m, range(9000, 9000 + a.repeat_calls)
            )
            cases += self.graph(
                op, "wrap-graph", a.m, range(9100, 9100 + max(a.replays // 2, 3))
            )
            wrap = "covered"
        return ms, cases, wrap

    def time(self, op):
        torch, a = self.torch, self.args
        x = activation(a.seed, self.rank, 0, a.m, a.k, self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(a.warmup):
                op(x)
        torch.cuda.current_stream().wait_stream(side)
        self.finish("time warm-up")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            op(x)
        for _ in range(a.warmup):
            g.replay()
        self.finish("graph warm-up")
        start, end, samples = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
            [],
        )
        for _ in range(a.iters):
            start.record()
            g.replay()
            end.record()
            self.finish("timed replay")
            samples.append(start.elapsed_time(end) * 1000.0)
        del g
        out = op(x)
        self.finish("checked call")
        return statistics.median(samples), self.case(op, "time", a.m, 0, out.clone())


def _setup_distributed():
    import torch
    import torch.distributed as dist
    from mori.cco import Communicator, UniqueId

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return rank, world, UniqueId.from_bytes(payload[0])


def run(args):
    import torch.distributed as dist
    from mori.cco import Communicator

    args.m_max = args.m_max or args.m
    impl = load_impl(args.impl)
    rank, world, uid = _setup_distributed()
    vmm = 2 * impl.window_bytes_for(world, m_max=args.m_max, n=args.n, k=args.k) + (
        256 << 20
    )
    result = None
    with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
        op = impl(comm, n=args.n, k=args.k, m_max=args.m_max)
        try:
            op.set_weight(weight(args.seed, rank, args.n, args.k, "cuda"))
            if hasattr(op, "self_test"):
                op.self_test()
            r = _Run(args, rank, world)
            if args.mode == "time":
                us, case = r.time(op)
                per_rank = [None] * world
                dist.all_gather_object(per_rank, (us, case))
                times = [t for t, _ in per_rank]
                cases = merge([[c] for _, c in per_rank])
                critical = max(range(world), key=lambda i: times[i])
                record = {
                    "max_rank_time_us": times[critical],
                    "per_rank_time_us": times,
                    "critical_rank": critical,
                    "rel_l2": cases[0]["rel_l2"],
                    "fused_path_ran": cases[0]["fused_path_ran"],
                    "validated": cases[0]["pass"],
                    "timing": "graph",
                }
            else:
                ms, mine, wrap = r.check(op)
                per_rank = [None] * world
                dist.all_gather_object(per_rank, mine)
                cases = merge(per_rank)
                record = {
                    "ms": ms,
                    "cases": cases,
                    "failures": [c for c in cases if not c["pass"]],
                    "wrap": wrap,
                    "fused_path_ran": (
                        None
                        if any(c["fused_path_ran"] is None for c in cases)
                        else all(c["fused_path_ran"] for c in cases)
                    ),
                    "rel_l2_max": max(c["rel_l2"] for c in cases),
                    "validated": all(c["pass"] for c in cases),
                    "max_rank_time_us": None,
                }
            if rank == 0:
                result = {
                    "mode": args.mode,
                    "impl": args.impl,
                    "world_size": world,
                    "m": args.m,
                    "n": args.n,
                    "k": args.k,
                    "m_max": args.m_max,
                    "seed": args.seed,
                    "tolerance": args.tolerance,
                    **record,
                }
                print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
                if args.json_out:
                    Path(args.json_out).write_text(
                        json.dumps(result, indent=2, sort_keys=True) + "\n"
                    )
        finally:
            op.close()
    dist.barrier()
    return 0 if (result is None or result["validated"]) else 1


if __name__ == "__main__":
    try:
        sys.exit(run(build_parser().parse_args()))
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
