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
    torchrun --nproc_per_node=8 op_harness.py --impl existing --mode time -m 16384

``--impl existing`` (and ``existing-q8``, eight SDMA queues per peer) runs the existing
operator through the case's adapter, so a baseline run never names a hidden file.

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

``timeline``: for an implementation with ``timeline_read()`` (``interface.py``): warm-up and
graph capture as in ``time``, then ``--timeline-calls`` replays, reading the device stamps
after each. The events of every call on every rank go to ``--timeline-out``; the result
holds the phase summary of ``benchmark/cco/harness/timeline.py``. One more call is checked.

``trace``: the same replays under ``torch.profiler``, for any implementation, the existing
operator included: every kernel's name, start and duration, call by call on every rank,
written to ``--timeline-out`` in ``timeline.py``'s format (``<kernel>:start@<n>``,
``<kernel>:end@<n>`` for its n-th launch in the call). What runs on the copy engines does
not show; the kernels that post and wait for it do. In a graph whose kernels run on more
than one stream the profiler matches names to kernels by order and can mix them up (the
times stay right); ``--trace-eager`` traces eager calls instead, where every launch is
matched by its own correlation.

``--impl-kwargs '{"mode": "fused"}'`` passes keyword switches to the implementation's
constructor, and the ones it declares to ``window_bytes_for``, so a variant needs no class
of its own.
"""

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import tempfile
from collections import Counter
import os
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))

import interface  # noqa: E402


def _harness_module(name):
    """A module of benchmark/cco/harness, loaded under a name of its own."""
    path = HERE.parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cco_harness_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOLERANCE = 6e-3
WRAP_PRESET = 2**32 - 3
# The existing operator behind the interface. The adapter is hidden from the implementer
# (hidden_paths.txt), so baselines name it by alias.
ALIASES = {
    "existing": "existing.py:ExistingGemmAllReduce",
    "existing-q8": "existing.py:ExistingGemmAllReduceQ8",
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--impl",
        required=True,
        help="file.py:Class, package.module:Class, or an alias: " + ", ".join(ALIASES),
    )
    p.add_argument(
        "--mode", choices=("time", "check", "timeline", "trace"), default="time"
    )
    p.add_argument(
        "--impl-kwargs",
        default="{}",
        help="JSON object of keyword switches for the implementation",
    )
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
    p.add_argument("--timeline-calls", type=int, default=10)
    p.add_argument(
        "--trace-eager",
        action="store_true",
        help="--mode trace: trace eager calls, not graph replays",
    )
    p.add_argument(
        "--timeline-out", help="where --mode timeline or trace writes every event"
    )
    p.add_argument("--json-out")
    return p


def impl_kwargs(text):
    """``--impl-kwargs`` as a dict."""
    try:
        kwargs = json.loads(text or "{}")
    except ValueError as e:
        raise ValueError(f"--impl-kwargs: not JSON: {e}") from None
    if not isinstance(kwargs, dict):
        raise ValueError("--impl-kwargs: needs a JSON object")
    return kwargs


def accepted(fn, kwargs):
    """The part of ``kwargs`` that ``fn`` takes by keyword."""
    params = inspect.signature(fn).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _ts(e):
    return float(e.get("ts", 0.0))


def kernel_calls(trace, calls):
    """A chrome trace's kernels as ``calls`` lists of timeline events, one per replay.

    Kernels are grouped by the graph launch they were correlated with when the trace
    says so, and otherwise split at the ``calls - 1`` largest gaps between kernels.
    """
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    kernels = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X" and str(e.get("cat", "")).lower() == "kernel"
        ),
        key=_ts,
    )
    if not kernels:
        return []
    launches = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and "graphlaunch" in str(e.get("name", "")).lower()
            and "correlation" in (e.get("args") or {})
        ),
        key=_ts,
    )
    index = {e["args"]["correlation"]: i for i, e in enumerate(launches)}
    groups = {}
    if index and all(
        (k.get("args") or {}).get("correlation") in index for k in kernels
    ):
        for k in kernels:
            groups.setdefault(index[k["args"]["correlation"]], []).append(k)
        split = [groups[i] for i in sorted(groups)]
    else:
        gaps = sorted(
            range(1, len(kernels)),
            key=lambda i: _ts(kernels[i])
            - _ts(kernels[i - 1])
            - kernels[i - 1].get("dur", 0),
        )
        cuts = sorted(gaps[-(calls - 1) :]) if calls > 1 else []
        split, last = [], 0
        for c in cuts + [len(kernels)]:
            split.append(kernels[last:c])
            last = c
    out = []
    for group in split:
        t0 = min(_ts(k) for k in group)
        seen, rows = Counter(), []
        for k in group:
            name = str(k.get("name", "?")).replace("@", "_")
            i = seen[name]
            seen[name] += 1
            start = _ts(k) - t0
            rows.append([f"{name}:start@{i}", round(start, 3)])
            rows.append([f"{name}:end@{i}", round(start + float(k.get("dur", 0.0)), 3)])
        out.append(rows)
    return out


def normalize_events(events, ticks_per_us):
    """One call's stamps as ``[name, us]`` from the call's first stamp.

    ``events`` holds ``(name, ticks)`` pairs, or dicts with ``event`` and either ``us`` or
    ``ticks``; unset stamps (None, 0) are dropped.
    """
    rows = []
    for e in events:
        if isinstance(e, dict):
            name = e.get("event") or e.get("name")
            us = e.get("us")
            if us is None and e.get("ticks"):
                us = e["ticks"] / ticks_per_us
        else:
            name, ticks = e
            us = ticks / ticks_per_us if ticks else None
        if name and us is not None:
            rows.append([str(name), float(us)])
    t0 = min((us for _, us in rows), default=0.0)
    return [[name, round(us - t0, 3)] for name, us in rows]


def load_impl(spec):
    """``path/to/file.py:Class`` (relative to the repository), ``package.module:Class``,
    or an alias from ``ALIASES``."""
    if spec in ALIASES:
        spec = str(HERE / ALIASES[spec])
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

    def trace(self, op):
        torch, a = self.torch, self.args
        from torch.profiler import ProfilerActivity, profile

        x = activation(a.seed, self.rank, 0, a.m, a.k, self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(a.warmup):
                op(x)
        torch.cuda.current_stream().wait_stream(side)
        self.finish("trace warm-up")
        g = None
        if not a.trace_eager:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                op(x)
            for _ in range(a.warmup):
                g.replay()
            self.finish("graph warm-up")
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities) as prof:
            for _ in range(a.timeline_calls):
                if g is None:
                    op(x)
                else:
                    g.replay()
                self.finish("traced call")
        fd, path = tempfile.mkstemp(suffix=".json", prefix=f"op_harness_r{self.rank}_")
        os.close(fd)
        try:
            prof.export_chrome_trace(path)
            calls = kernel_calls(json.loads(Path(path).read_text()), a.timeline_calls)
        finally:
            os.unlink(path)
        del g
        out = op(x)
        self.finish("checked call")
        return calls, self.case(op, "trace", a.m, 0, out.clone())

    def timeline(self, op):
        torch, a = self.torch, self.args
        x = activation(a.seed, self.rank, 0, a.m, a.k, self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(a.warmup):
                op(x)
        torch.cuda.current_stream().wait_stream(side)
        self.finish("timeline warm-up")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            op(x)
        for _ in range(a.warmup):
            g.replay()
        self.finish("graph warm-up")
        ticks = float(getattr(op, "TIMELINE_TICKS_PER_US", 100.0))
        calls = []
        for _ in range(a.timeline_calls):
            g.replay()
            self.finish("timeline replay")
            calls.append(normalize_events(op.timeline_read(), ticks))
        del g
        out = op(x)
        self.finish("checked call")
        return calls, self.case(op, "timeline", a.m, 0, out.clone())


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
    kwargs = impl_kwargs(args.impl_kwargs)
    if args.mode == "timeline" and not hasattr(impl, "timeline_read"):
        raise SystemExit("op_harness: --mode timeline needs timeline_read()")
    if args.mode in ("timeline", "trace") and not args.timeline_out:
        raise SystemExit(f"op_harness: --mode {args.mode} needs --timeline-out")
    rank, world, uid = _setup_distributed()
    window = impl.window_bytes_for(
        world,
        m_max=args.m_max,
        n=args.n,
        k=args.k,
        **accepted(impl.window_bytes_for, kwargs),
    )
    vmm = 2 * window + (256 << 20)
    result = None
    with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
        op = impl(comm, n=args.n, k=args.k, m_max=args.m_max, **kwargs)
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
            elif args.mode in ("timeline", "trace"):
                calls, case = (r.timeline if args.mode == "timeline" else r.trace)(op)
                per_rank = [None] * world
                dist.all_gather_object(per_rank, (calls, case))
                cases = merge([[c] for _, c in per_rank])
                record = {
                    "timeline_calls": len(calls),
                    "timeline_out": args.timeline_out,
                    "rel_l2": cases[0]["rel_l2"],
                    "fused_path_ran": cases[0]["fused_path_ran"],
                    "validated": cases[0]["pass"],
                    "max_rank_time_us": None,
                }
                if rank == 0:
                    data = {
                        "source": (
                            "stamps"
                            if args.mode == "timeline"
                            else "kernel trace, "
                            + ("eager" if args.trace_eager else "graph")
                        ),
                        "impl": args.impl,
                        "impl_kwargs": kwargs,
                        "m": args.m,
                        "n": args.n,
                        "k": args.k,
                        "ranks": [c for c, _ in per_rank],
                    }
                    Path(args.timeline_out).write_text(json.dumps(data) + "\n")
                    summary = _harness_module("timeline").summarize(data)
                    record["call_us"] = summary["call_us"]
                    record["critical_rank"] = summary["critical_rank"]
                    record["phases"] = summary["phases"]
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
                    "impl_kwargs": kwargs,
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
