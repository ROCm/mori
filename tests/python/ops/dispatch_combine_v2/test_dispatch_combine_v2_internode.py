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
"""Internode correctness for the v2 EP op, over CCO/GDA.

Identity expert -- the received tokens are combined unchanged -- so
``combine[t] == U[t] * input[t]`` and ``out_weights[t] == U[t] * wts[t]``, where
U is the number of DISTINCT destination ranks token t routed to. The expectation
is analytic: a pass means "matches the intended semantics", not "matches some
other implementation", which is the property that has to survive the kernel being
rewritten underneath it.

TWO NODES ARE REQUIRED, and not merely for coverage -- the op refuses to build
otherwise. ``gpu_per_node < world_size`` is what selects the internode path, and
the op checks EP's node grouping against the communicator's LSA team; on one host
CCO reports the whole world as one team, so a config claiming two nodes is
rejected rather than silently running the intranode half. Lowering gpu_per_node
on a single host does not emulate this either: RAIL leaves same-host peers
without a QP.

    # rank 0 of 2, 8 GPUs each
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \\
        --master_addr=<ip> --master_port=<port> \\
        test_dispatch_combine_v2_internode.py --max-tokens 128

``tools/run_internode_test.sh`` drives both ranks; the CLI below is the subset of
the shmem harness's flags that means anything here.

COMPARING AGAINST THE v1 BENCH
------------------------------
The reference numbers come from ``run_bench_once`` in
``examples/ops/dispatch_combine/test_dispatch_combine_internode.py``, driven
through v1's op. The timed loop here is structured the same way and reports the
same three statistics, but the two harnesses do NOT build the same shape by
default. Every difference found by reading both, and what to pass to close it:

  what                     v1 bench          here (default)     to align
  ----------------------   ---------------   ----------------   ------------------
  combine weights          None (no fold)    None               (aligned)
  num_experts_per_rank     256 // world      --experts-per-rank --experts-per-rank 16
                                             = 32               at world 16
  scale_dim                32                --scale-dim = 0    --scale-dim 32
  scale_type_size          4                 4 when scale_dim   (aligned)
  warmup / rounds / drop   20 / 30 / 1       same, and CHECKED  (aligned)
  routing                  randperm[:topk]   same               (aligned)
  tokens per rank          max on every rank same               (aligned)
  statistic                avg over          same               (aligned)
                           rounds x ranks

Only ONE row of this table is enforced: `_report_loop_alignment` reads the three
loop constants out of that harness's source and warns, next to the numbers, when
they differ from ours. Every other row is a claim a reader has to re-check, and
one of them was wrong for as long as it was written -- `--rounds` sat at 3
against that harness's 30 while this row said "same", because a table asserting
three numbers had been checked for two. Prefer extending the check to adding a
row.

The two that move real work: the weight fold costs an extra peer read per
(token, destination) plus an accumulate in three kernels and a wider staging slot,
and v1's scale_dim=32 makes ITS dispatch carry 128 more bytes per token than a
--scale-dim 0 run here. They push in opposite directions, so a comparison that
leaves both unaligned is not bounded in either direction.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from mori.cco import Communicator
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8_e4m3_fnuz": torch.float8_e4m3fnuz,
    "fp8_e4m3": torch.float8_e4m3fn,
}
_FP8 = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)


class Dist:
    """torchrun/gloo bootstrap. gloo is only the courier for the cco unique id
    and the pass/fail counts; every byte of payload moves over cco."""

    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(self.local_rank)

    def bcast_uid(self, uid):
        objs = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

    def allreduce_sum(self, value):
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def allreduce_minmax(self, lo, hi):
        """Extremes across ranks, for the same best/worst the v1 harness prints."""
        t = torch.tensor([lo, -hi], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return int(t[0].item()), -int(t[1].item())

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()


def _parse_args(argv):
    p = argparse.ArgumentParser(description="v2 internode dispatch/combine test")
    p.add_argument("--cmd", default="test", choices=["test", "bench"])
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=7168)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--experts-per-rank", type=int, default=32)
    p.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    p.add_argument("--combine-dtype", default=None, choices=list(_DTYPES))
    p.add_argument("--quant-type", default="none", choices=["none", "fp8_direct_cast"])
    p.add_argument("--num-qp", type=int, default=2)
    # 30, matching _EP_ROUNDS in the examples harness -- and for the reason its
    # comment gives, which applies here with full force: the CCO/GDA path reaches
    # steady state slowly, so at 10 rounds its per-round jitter does not average
    # out and the mean swings ~20% run to run. This defaulted to 3 while claiming
    # alignment, which with --drop-rounds 1 keeps rounds 1 and 2 -- precisely the
    # two that harness documents as still elevated. A single spike then carries
    # half the average AND is the reported worst, so this side read a wide tail
    # against a 29-round mean. Not a small-sample caveat: a different estimator.
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--scale-dim", type=int, default=0)
    # The v1 bench calls combine with weights=None, so it does not pay for the
    # weight fold: an extra peer read per (token, destination) plus an accumulate
    # in three kernels, and a wider staging slot (combXferBytes = hidden + weights).
    # Off by default so a reading here is comparable to one from that harness;
    # --bench-weights measures the fold when that is what you want. The
    # correctness path always folds -- it is checking the weights.
    p.add_argument("--bench-weights", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--drop-rounds", type=int, default=1)
    # Diagnostic, matching _EP_PERROUND_SYNC in the examples harness. Re-aligns
    # the ranks every round. These kernels spin on their peers, so any host skew
    # shows up as KERNEL time; if this moves the numbers, the gap is skew rather
    # than kernel work. It folds the barrier wait into the next round's dispatch
    # window, so dispatch is not clean under it -- combine is.
    p.add_argument("--per-round-sync", action="store_true")
    # Both members of the LL / non-LL pair are compiled either way; this picks
    # which one runs. Default (None) leaves the backend's token-count rule alone,
    # which selects LL below 2048 tokens. Naming it explicitly is what makes a
    # benchmark number comparable to another harness's.
    p.add_argument("--kernel-type", default=None, choices=[None, "v1", "v1_ll"])
    return p.parse_args(argv)


def _gen_round(rng, cfg, ct, dev, dtype):
    """Seeded per-round input, routing and weights, so a failure is reproducible
    from the round number and the rank alone."""
    inp = torch.randn(ct, cfg.hidden_dim, generator=rng, device=dev).to(dtype)
    n_experts = cfg.world_size * cfg.num_experts_per_rank
    idx = torch.stack(
        [
            torch.randperm(n_experts, generator=rng, device=dev)[
                : cfg.num_experts_per_token
            ]
            for _ in range(ct)
        ]
    ).to(torch.int32)
    wts = torch.rand(ct, cfg.num_experts_per_token, generator=rng, device=dev)
    # Real scales when the transport is on. The dispatch send path copies
    # scale_dim * scale_type_size bytes per token whether or not a buffer was
    # handed in -- only the staging copy is guarded on the pointer -- so passing
    # None with scale_dim > 0 transports uninitialised staging. Same byte count,
    # but not something to measure against.
    scales = (
        torch.rand(ct, cfg.scale_dim, generator=rng, device=dev)
        if cfg.scale_dim
        else None
    )
    return inp, idx, wts, scales


def _verify_once(op, cfg, d, dev, a, comm, inp, idx, wts, sc):
    """One dispatch+combine against the analytic golden. True if this rank agrees.

    Same expectation as `--cmd test`: an identity expert makes combine[t] equal
    U[t] * input[t] over the DISTINCT destination ranks. Weights are always folded
    here even when the timed loop will not fold them -- an unchecked weight path
    is how this harness shipped a silent bug twice.
    """
    r = op.dispatch(inp, wts, sc, idx, return_routing=True)
    torch.cuda.synchronize()
    comm.barrier()
    x = r[0].to(cfg.combine_dtype) if cfg.is_asymmetric_dtype else r[0]
    out, out_w = op.combine(x, wts, routing=r[5])
    torch.cuda.synchronize()
    comm.barrier()

    ct = inp.shape[0]
    idx_c = idx.cpu()
    U = np.array(
        [
            len({int(idx_c[t, j]) // cfg.num_experts_per_rank for j in range(a.topk)})
            for t in range(ct)
        ]
    )
    Ut = torch.from_numpy(U).view(ct, 1).float()
    got = out.float().cpu()
    exp = Ut * inp.float().cpu()
    per_elem = inp.float().cpu().abs()
    eps = 3e-1 if (a.quant_type != "none" or cfg.dispatch_dtype in _FP8) else 8e-3
    ok = bool(((got - exp).abs() <= eps * Ut * per_elem.clamp(min=1.0)).all())
    ok_w = bool(((out_w.cpu() - Ut * wts.float().cpu()).abs() <= 2e-3 * Ut).all())
    if not (ok and ok_w) and d.rank == 0:
        print(f"#   pre-bench check: hidden_ok={ok} weights_ok={ok_w}", flush=True)
    return ok and ok_w


def _sclk_levels():
    """Active DPM shader-clock level of every GPU on this host, as a string.

    The amdgpu driver publishes the level table per card with a `*` on the
    active one; on this rig that is 88MHz idle / 500MHz / 1850MHz with
    power_dpm_force_performance_level=auto, so a run that never boosts is
    several times slower at identical code. Cards are not numbered contiguously
    (card0, card16, card24...) and the HIP device order does not have to match
    the DRM order, so this reports ALL of them rather than pretending to know
    which one is ours -- the question is whether the node boosted, not which
    card did. A plain sysfs read, microseconds, no subprocess.
    """
    import glob

    out = []
    for path in sorted(glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk")):
        try:
            for line in open(path):
                if "*" in line:
                    out.append(line.split(":")[1].split("*")[0].strip())
                    break
        except OSError:
            pass
    return " ".join(out) if out else "n/a"


def _v1_loop_defaults():
    """The examples harness's rounds/warmup/drop defaults, read from its source.

    The alignment table in this module's docstring used to ASSERT that these
    matched ours. It was written from intent, not from the file: --rounds sat at
    3 against that harness's 30 for as long as the row claimed "same", because
    nothing ever compared the two. A comment cannot notice when the other side
    moves, and neither can a reader who wrote the comment.

    Read, do not import. Importing that module pulls in the v1 op, and this test
    depends on v1 nowhere else -- buying a consistency check with a dependency on
    the thing we are trying to be independent of is a bad trade. Parsing three
    integer literals out of its AST costs nothing and keeps the coupling at zero.

    Returns ``{"rounds": int, "warmup": int, "drop_rounds": int}``, or ``{}`` if
    the file is missing or has been restructured -- a missing reference is a
    reason to skip the check, never to fail the run.
    """
    import ast

    path = os.path.join(
        _ROOT, "examples", "ops", "dispatch_combine", "test_dispatch_combine_internode.py"
    )
    names = {
        "_EP_ROUNDS": "rounds",
        "_EP_WARMUP": "warmup",
        "_EP_DROP_ROUNDS": "drop_rounds",
    }
    out = {}
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        key = names.get(getattr(tgt, "id", None))
        if key is None:
            continue
        # `int(os.environ.get("MORI_EP_ROUNDS") or "30")` -- the default is the
        # only digit-string literal in the expression.
        #
        # Accept both spellings of a string literal. Python >= 3.8 gives
        # ast.Constant; 3.6/3.7 give ast.Str, and the host python here is 3.6
        # while the container's is 3.12. Matching only Constant makes this
        # silently return {} on the older one -- which would disable the very
        # check whose absence caused the problem it exists to catch.
        for sub in ast.walk(node.value):
            lit = None
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                lit = sub.value
            elif sub.__class__.__name__ == "Str":
                lit = sub.s
            if lit is not None and lit.isdigit():
                out[key] = int(lit)
                break
    return out


def _report_loop_alignment(a, rank):
    """Say out loud when this harness's timed loop is shaped differently from the
    reference one. Printed with the numbers, not buried in a docstring, because
    the numbers are what gets quoted."""
    ref = _v1_loop_defaults()
    if not ref or rank != 0:
        return
    mine = {"rounds": a.rounds, "warmup": a.warmup, "drop_rounds": a.drop_rounds}
    off = {k: (mine[k], v) for k, v in ref.items() if mine.get(k) != v}
    if off:
        detail = " ".join(f"{k}={m}(ref {r})" for k, (m, r) in sorted(off.items()))
        print(
            f"# WARNING: timed loop differs from run_bench_once: {detail} -- "
            f"these numbers are not directly comparable to that harness's",
            flush=True,
        )


def _bench(op, cfg, d, dev, a, comm):
    """Per-phase latency, structured to match ``run_bench_once`` in
    ``examples/ops/dispatch_combine/test_dispatch_combine_internode.py``.

    Apple-to-apple means only the op call differs, so everything around it is
    copied from there rather than invented here:

    * ONE event before the loop, then three per round. Round i's dispatch window
      is [end of round i-1's combine, end of this dispatch]; its combine window
      is [end of the combine-input conversion, end of this combine]. The
      conversion sits between two events of its own and is charged to neither.
    * Nothing synchronises or barriers inside the timed loop. Events are stream
      markers, so a host that has to stop and build the next launch shows up as
      GPU idle inside the window; free-running lets the host stay ahead. (An
      earlier version here synced per round and read 460us at 4 tokens against a
      41us reference -- 11x of host overhead, none of it kernel.)
    * Warmup is untimed and ends on sync + barrier; the leading `drop_rounds`
      timed rounds are discarded because the first after a barrier is thundering
      herd, not kernel.
    * The reported number is the AVERAGE over rounds and ranks, as there.

    `wall` is reported alongside as an honesty check, not as part of the
    measurement: it is the whole loop's wall time per round. The host is running
    ahead only while wall stays at or below dispatch+combine. Above it, the host
    is the pacer and the phase numbers carry host stall.
    """
    _report_loop_alignment(a, d.rank)
    rng = torch.Generator(device=dev)
    rng.manual_seed(4242 + d.rank)
    ct = a.max_tokens
    inp, idx, wts, sc = _gen_round(rng, cfg, ct, dev, cfg.dispatch_dtype)
    if a.kernel_type is not None:
        op._internode_force_ll = a.kernel_type == "v1_ll"

    # Check BEFORE measuring, as bench_dispatch_combine does (it runs
    # run_test_once and asserts before run_bench_once). A configuration that is
    # silently wrong still produces timings, and this harness has shipped two of
    # those -- the weight fold reading the wrong buffer, and both legs compiled
    # with the dispatch dtype. One round, folding weights whatever --bench-weights
    # says, because the point is to check them.
    ok = _verify_once(op, cfg, d, dev, a, comm, inp, idx, wts, sc)
    bad = d.allreduce_sum(0 if ok else 1)
    if bad:
        if d.rank == 0:
            print(
                f"# BENCH ABORTED: {bad} of {d.world} ranks failed the "
                f"pre-bench check; the numbers below would be meaningless",
                flush=True,
            )
        return 1
    torch.cuda.synchronize()
    comm.barrier()

    # The examples harness's _convert_for_combine: the combine leg reads its
    # input as its own element type, so an asymmetric config has to cast first.
    def convert(x):
        return x.to(cfg.combine_dtype) if cfg.is_asymmetric_dtype else x

    cw = wts if a.bench_weights else None

    # Allocated before the warmup, where run_bench_once allocates them. (Priming
    # them with a record here was tried and did not move the tail, so it is not
    # done -- run_bench_once does not either.)
    n = a.rounds
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3 * n + 1)]

    for _ in range(a.warmup):
        r = op.dispatch(inp, wts, sc, idx, return_routing=True)
        op.combine(convert(r[0]), cw, routing=r[5])
    torch.cuda.synchronize()
    comm.barrier()

    # Causal test for "is the host pacing the GPU", opt-in. Busy-waits a known
    # number of microseconds on the HOST between the convert-end event and the
    # combine enqueue -- pure host time, no GPU work, no extra kernel. If the
    # host is running ahead of the GPU, the GPU still has queued work to chew on
    # and the measured combine barely moves. If the GPU is already waiting on the
    # host, the injected delay is added GPU idle inside the combine window and
    # the measured combine rises by about the injected amount. The slope of
    # measured-combine against injected microseconds is the answer, and it needs
    # no assumption about what any window "should" cost.
    inject = float(os.environ.get("MORI_EP_INJECT_HOST_US") or 0) / 1e6

    # Which round a garbage collection lands in, opt-in. The spikes are GLOBAL --
    # 12 to 16 of 16 ranks spike in the same round -- so the cause is something
    # every rank does on its own schedule, not a straggler one rank produces. A
    # generational GC is exactly that shape: it fires on allocation counts, and
    # every rank allocates the same objects per round, so they all reach the
    # threshold on the same round. This records the collections rather than
    # inferring them from an A/B. (gc.disable() was A/B'd before and read
    # negative, but that run had 2 kept rounds and a round-8 event was not in the
    # sample at all -- the test could not have detected what it was testing for.)
    gc_hits = []
    gc_cb = None
    _cur = [-1]
    if os.environ.get("MORI_EP_GC_TRACE"):
        import gc

        def gc_cb(phase, info):
            if phase == "stop":
                gc_hits.append((_cur[0], info.get("generation"), info.get("collected")))

        gc.callbacks.append(gc_cb)

    # Opt-in counterpart: freeze everything already alive and stop collecting for
    # the timed loop. gc.freeze() moves the existing objects to a permanent
    # generation so re-enabling later does not immediately pay for them.
    _no_gc = bool(os.environ.get("MORI_EP_NO_GC"))
    if _no_gc:
        import gc

        gc.collect()
        gc.freeze()
        gc.disable()

    _sclk_before = _sclk_levels() if os.environ.get("MORI_EP_SCLK") else None

    t0 = time.perf_counter()
    ev[0].record()

    # Host-side duration of each enqueue, opt-in via the same flag as the round
    # series. These calls are asynchronous: while the host runs ahead of the GPU
    # they return as soon as the work is queued, so this reads as pure Python
    # cost. Once the launch queue is full they BLOCK until the GPU retires
    # something, and the number jumps. That transition is the thing to look for
    # -- it says the loop stopped measuring the kernel and started measuring the
    # host, and it is invisible in the event timings, which report the same
    # wall-clock either way. perf_counter is ~50ns, far below what it resolves.
    hdisp = [0.0] * n
    hcomb = [0.0] * n

    for i in range(n):
        _cur[0] = i
        _h = time.perf_counter()
        r = op.dispatch(inp, wts, sc, idx, return_routing=True)
        hdisp[i] = (time.perf_counter() - _h) * 1e6
        ev[3 * i + 1].record()
        x = convert(r[0])
        ev[3 * i + 2].record()
        if inject:
            t = time.perf_counter()
            while time.perf_counter() - t < inject:
                pass
        _h = time.perf_counter()
        op.combine(x, cw, routing=r[5])
        hcomb[i] = (time.perf_counter() - _h) * 1e6
        ev[3 * i + 3].record()
        if a.per_round_sync:
            torch.cuda.synchronize()
            comm.barrier()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e6 / n
    if _sclk_before is not None and d.rank == 0:
        print(f"# SCLK before: {_sclk_before}", flush=True)
        print(f"# SCLK after:  {_sclk_levels()}", flush=True)
    if _no_gc:
        import gc

        gc.enable()
        gc.unfreeze()
    if gc_cb is not None:
        import gc

        gc.callbacks.remove(gc_cb)
        if gc_hits:
            print(
                "# GC r%d: " % d.rank
                + " ".join("round%d/gen%s/n%s" % h for h in gc_hits),
                flush=True,
            )

    # Host profile, opt-in. Whenever wall exceeds dispatch+combine the loop above
    # is host-paced, and then its phase numbers are wrapper cost rather than
    # kernel cost. This says which wrapper. It runs its OWN untimed loop so the
    # profiler's overhead can never land in a reported number.
    if os.environ.get("MORI_EP_HOST_PROFILE"):
        import cProfile
        import pstats

        # EVERY rank runs the loop; only rank 0 prints. dispatch and combine are
        # collectives -- a rank that enters them alone spins in the kernel's
        # wait-for-peers loop forever, which reads as one GPU pinned at 100% with
        # every other idle. Guarding the loop itself on rank 0 (rather than just
        # the reporting) is exactly that hang.
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(n):
            rp = op.dispatch(inp, wts, sc, idx, return_routing=True)
            op.combine(convert(rp[0]), cw, routing=rp[5])
        pr.disable()
        torch.cuda.synchronize()
        comm.barrier()
        if d.rank != 0:
            return 0
        # Rank 0 must leave through here too. The stats below are allreduces, and
        # every other rank has already returned -- rank 0 entering them alone is a
        # hang, then a nonzero exit with no BENCH line. Profiling is a diagnostic
        # mode, so ending the run after the profile is right; ending it on 15 of
        # 16 ranks is not.
        st = pstats.Stats(pr)
        rows = sorted(st.stats.items(), key=lambda kv: -kv[1][2])[:20]
        print(f"# HOST PROFILE tok={ct}  (us/round, sorted by self time)", flush=True)
        for (fn, ln, name), (_, nc, tt, _ct, _) in rows:
            print(
                f"#   self={tt / n * 1e6:8.1f}  cum={_ct / n * 1e6:8.1f}  "
                f"n={nc / n:5.1f}  {os.path.basename(fn)}:{ln}({name})",
                flush=True,
            )
        return 0

    keep = slice(a.drop_rounds, None)
    disp = [ev[3 * i].elapsed_time(ev[3 * i + 1]) * 1e3 for i in range(n)][keep]
    comb = [ev[3 * i + 2].elapsed_time(ev[3 * i + 3]) * 1e3 for i in range(n)][keep]
    # The third window, which neither harness reports and which is the honest
    # host-gap meter.
    #
    # The events TILE the timed region: ev[3i+3] ends round i's combine and IS
    # ev[3(i+1)], the start of round i+1's dispatch window. There is no gap
    # between windows for host time to hide in -- every microsecond between
    # ev[0] and ev[3n] is inside exactly one of the three. So if the host falls
    # behind, the GPU idles at the head of whichever segment it is waiting for
    # and that idle is charged to that window as if it were kernel time.
    #
    # This window contains ONE cast. Its kernel cost is a few microseconds at
    # these token counts and does not grow with anything we tune, so whatever it
    # reads above that is GPU idle waiting for the host -- which makes it a
    # measure of host lag that does not require trusting `wall`. (`wall` cannot
    # show this: wall - (dispatch + combine) is this window BY CONSTRUCTION, so
    # quoting that difference as evidence of host pacing assumes the conclusion.)
    conv = [ev[3 * i + 1].elapsed_time(ev[3 * i + 2]) * 1e3 for i in range(n)][keep]

    # Which round stalled, opt-in. EVERY rank prints its own series, because the
    # question the averages cannot answer is whether a spike lands on the same
    # round index across ranks or on one rank alone:
    #   same round, all ranks   -> a whole-round event; every rank waits for the
    #                              same thing, so the cause is host-side (a launch
    #                              that took longer to build, an allocation, GC)
    #                              or a fabric stall that stops everyone.
    #   one rank, one round     -> a straggler; the others are only showing the
    #                              spin-wait for it. Chasing the peak on the ranks
    #                              that merely waited leads nowhere.
    # These are collective kernels, so a single slow rank prints as a slow round
    # on all of them -- which is why the RANK-LOCAL series is what separates the
    # two, and rank 0's alone cannot.
    if os.environ.get("MORI_EP_ROUND_SERIES"):
        print(
            "# rounds r%d disp: " % d.rank + " ".join("%.0f" % x for x in disp),
            flush=True,
        )
        print(
            "# rounds r%d comb: " % d.rank + " ".join("%.0f" % x for x in comb),
            flush=True,
        )
        print(
            "# hostus r%d disp: " % d.rank
            + " ".join("%.0f" % x for x in hdisp[keep]),
            flush=True,
        )
        print(
            "# hostus r%d comb: " % d.rank
            + " ".join("%.0f" % x for x in hcomb[keep]),
            flush=True,
        )

    # AVERAGE over rounds x ranks, plus BEST and WORST over the same sample set --
    # the three numbers run_bench_once prints, so a reading here can be put beside
    # one from the v1 harness without converting estimators. The average is the
    # robust one; best/worst are extremes and noise-dominated, but they are what
    # makes a single stalled round visible. Reporting only a minimum hides exactly
    # that (an earlier version of this comparison did, and buried a 227us outlier).
    def _stats(v):
        m = d.allreduce_sum(int(sum(v) / len(v) * 1000)) / d.world / 1000
        lo, hi = d.allreduce_minmax(int(min(v) * 1000), int(max(v) * 1000))
        return m, lo / 1000, hi / 1000

    dm, dlo, dhi = _stats(disp)
    cm, clo, chi = _stats(comb)
    vm, _, _ = _stats(conv)
    if d.rank == 0:
        print(
            f"# BENCH tok={ct} dtype={a.dtype}->{a.combine_dtype or a.dtype} "
            f"hidden={cfg.hidden_dim} topk={cfg.num_experts_per_token} "
            f"kernel={a.kernel_type or 'auto'} "
            f"dispatch={dm:.1f}us [{dlo:.1f}/{dhi:.1f}] "
            f"combine={cm:.1f}us [{clo:.1f}/{chi:.1f}] "
            f"total={dm + cm:.1f}us [conv={vm:.1f}us wall={wall:.1f}us]",
            flush=True,
        )
    return 0


def main(argv):
    a = _parse_args(argv)
    d = Dist()
    rank, npes = d.rank, d.world
    dev = torch.device("cuda", d.local_rank)

    gpu_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if npes <= gpu_per_node:
        raise SystemExit(
            f"this test needs more than one node: world_size={npes} with "
            f"{gpu_per_node} GPUs per node is a single node"
        )

    dtype = _DTYPES[a.dtype]
    combine_dtype = _DTYPES[a.combine_dtype] if a.combine_dtype else None
    M = a.max_tokens

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = d.bcast_uid(uid)
    # internode_regions sizes the arena exactly; this is the VMM budget it is
    # carved out of, with room for the communicator's own resource window.
    win_bytes = npes * M * a.hidden_dim * 4 * 2 + (1 << 24)
    failures = 0
    with Communicator.init(
        npes, rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)
    ) as comm:
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=npes,
            hidden_dim=a.hidden_dim,
            max_num_inp_token_per_rank=M,
            num_experts_per_rank=a.experts_per_rank,
            num_experts_per_token=a.topk,
            data_type=dtype if combine_dtype is None else torch.bfloat16,
            dispatch_data_type=dtype if combine_dtype is not None else None,
            combine_data_type=combine_dtype,
            scale_dim=a.scale_dim,
            scale_type_size=4 if a.scale_dim else 0,
            quant_type=a.quant_type,
            gpu_per_node=gpu_per_node,
            num_qp_per_pe=a.num_qp,
            kernel_backend="hip",
        )
        op = EpDispatchCombineOp(cfg, comm)
        comm.barrier()

        if a.cmd == "bench":
            rc = _bench(op, cfg, d, dev, a, comm)
            op.close()
            d.shutdown()
            return rc

        rng = torch.Generator(device=dev)
        for r in range(a.rounds):
            rng.manual_seed(1234 + r * 977 + rank)
            ct = M
            inp, idx, wts, sc = _gen_round(rng, cfg, ct, dev, cfg.dispatch_dtype)

            recv_x, recv_w, recv_s, recv_i, total_recv, routing = op.dispatch(
                inp, wts, None, idx, return_routing=True
            )
            torch.cuda.synchronize()
            comm.barrier()

            # Identity expert: recv_x already holds the dispatched tokens.
            #
            # Converting it is the CALLER's job when the two legs have different
            # element types, exactly as v1's harness does it (_get_combine_input
            # -> _to_combine_dtype). The combine kernel's T is the combine dtype
            # and it reads inpTokenBuf as T*, so handing it the fp8 dispatch
            # output unconverted reinterprets fp8 bytes as bf16.
            combine_in = (
                recv_x.to(cfg.combine_dtype) if cfg.is_asymmetric_dtype else recv_x
            )
            out, out_w = op.combine(combine_in, wts, routing=routing)
            torch.cuda.synchronize()
            comm.barrier()

            idx_c = idx.cpu()
            U = np.array(
                [
                    len(
                        {
                            int(idx_c[t, j]) // cfg.num_experts_per_rank
                            for j in range(a.topk)
                        }
                    )
                    for t in range(ct)
                ]
            )
            Ut = torch.from_numpy(U).view(ct, 1).float()
            exp_w = Ut * wts.float().cpu()
            exp = Ut * inp.float().cpu()

            # The bound has to scale with U: combine sums U contributions in the
            # wire dtype, so the error grows with the number of terms and with the
            # magnitude being summed -- not with |expected| at that element, which
            # cancellation can make arbitrarily small. This is the same shape of
            # bound v1's harness uses (it scales its per-element bound by
            # unique_pes) rather than a flat allclose.
            got = out.float().cpu()
            per_elem = inp.float().cpu().abs()
            eps = 3e-1 if (a.quant_type != "none" or dtype in _FP8) else 8e-3
            bound = eps * Ut * per_elem.clamp(min=1.0)
            ok = bool(((got - exp).abs() <= bound).all())
            # Weights are transported as f32 and summed the same way, so their
            # bound is much tighter -- but still proportional to U.
            gw = out_w.cpu()
            ok_w = bool(((gw - exp_w).abs() <= 2e-3 * Ut).all())

            if not (ok and ok_w) and rank == 0:
                print(f"#   hidden_ok={ok} weights_ok={ok_w}", flush=True)
                ratio = (out_w.cpu() / wts.float().cpu().clamp(min=1e-6))[:4]
                print(
                    f"#   effective multiplier got_w/wts[0,:4]={ratio[0, :4].tolist()} "
                    f"(expected U[0]={int(U[0])})",
                    flush=True,
                )
                # The worst violator with everything needed to classify it:
                # a relative error near the wire dtype's half-ulp is rounding,
                # one far above it is not, and a `want` at the staging format's
                # saturation point says the partial sum clipped rather than
                # rounded.
                viol = (got - exp).abs() - bound
                if bool((viol > 0).any()):
                    fi = int(viol.argmax())
                    # NOT `t, d` -- `d` is the Dist handle in this scope.
                    vt, vd = fi // cfg.hidden_dim, fi % cfg.hidden_dim
                    g, e, pin = (
                        float(got[vt, vd]),
                        float(exp[vt, vd]),
                        float(per_elem[vt, vd]),
                    )
                    print(
                        f"#   worst: tok={vt} dim={vd} got={g:.6g} want={e:.6g} "
                        f"input={pin:.6g} U={int(Ut[vt])} "
                        f"|diff|={abs(g - e):.6g} bound={float(bound[vt, vd]):.6g} "
                        f"rel={abs(g - e) / max(abs(e), 1e-9):.4f} "
                        f"nviol={int((viol > 0).sum())}",
                        flush=True,
                    )
                want = exp
                print(
                    f"#   hidden: max|diff|={(got - want).abs().max():.4g} "
                    f"got[0,:4]={got[0, :4].tolist()} want[0,:4]={want[0, :4].tolist()} "
                    f"got_nonzero={int((got != 0).sum())}/{got.numel()}",
                    flush=True,
                )
                ww = exp_w
                # Distinguish "the kernel never wrote it" from "the base reads
                # the wrong place": read the region the kernel targets directly.
                from mori.tensor_utils import from_gpu_ptr

                for rn in (
                    "combine_out_weights",
                    "inp_weights",
                    "dispatch_out_weights",
                ):
                    v = from_gpu_ptr(
                        op.arena.local_ptr(rn), (min(8, a.topk * 2),), torch.float32
                    )
                    print(f"#   region {rn}[:8] = {v.cpu().tolist()}", flush=True)
                print(
                    f"#   weights: max|diff|={(gw - ww).abs().max():.4g} "
                    f"got[0,:4]={gw[0, :4].tolist()} want[0,:4]={ww[0, :4].tolist()} "
                    f"U[:4]={U[:4].tolist()} total_recv={int(total_recv[0])}",
                    flush=True,
                )
            errs = d.allreduce_sum(0 if (ok and ok_w) else 1)
            failures += errs
            if rank == 0:
                print(
                    f"# round {r} tokens={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                    f"({errs} of {npes} ranks disagree)",
                    flush=True,
                )
        op.close()

    d.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
