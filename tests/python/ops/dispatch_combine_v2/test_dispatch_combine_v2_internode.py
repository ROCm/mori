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

RoCE QoS: SET MORI_RDMA_TC AND MORI_RDMA_SL
-------------------------------------------
Unset, ``bnxt.cpp`` takes ``ReadRdmaServiceLevelEnv().value_or(1)`` and leaves
``grh.traffic_class`` alone, so every transfer goes out on SL 1 / DSCP 0 -- the
default lossy class -- no matter how the NIC and switch are programmed. The
values must match the fabric: on the 2-node bnxt rig here the NIC is programmed
``roce_dscp=0x28`` (40), so ``MORI_RDMA_TC=160`` (40 << 2) and
``MORI_RDMA_SL=5``. ``tools/env_setup.sh`` derives both from ROCE_DSCP/ROCE_PRIO
and exports them; its checked-in 26/3 are reference defaults its own comment says
to align with the switch. Confirm they arrived with ``MORI_APP_LOG_LEVEL=info``:
``bnxt attr.ah_attr.sl:5 attr.ah_attr.grh.traffic_class:160``.

Measured, it changes nothing HERE -- interleaved A/B over four pairs at 4 tokens
put 160/5 and 0/1 within noise of each other. That is expected rather than
contradictory: a lossless priority class buys nothing when the benchmark is the
only traffic on the fabric. It is worth setting because a number measured in the
wrong traffic class does not transfer to a shared one, not because it is a
speedup.

Why the small-token bench is bimodal per RUN, how to recognise it from the
per-rank ``hwal`` series (``MORI_EP_ROUND_SERIES=1``), and how to A/B on this
path without being fooled by it live in ``docs/EP_INTERNODE_V2_TAIL.md`` rather
than here, so this file stays close in shape to the examples harness it mirrors.

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
  num_experts_per_rank     256 // world      256 // world       (aligned)
  scale_dim                32                32                 (aligned)
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

Both of those defaults used to differ and had to be passed by hand, which is a
bad way to keep a comparison honest: scale_dim=32 makes dispatch carry 128 more
bytes per token, so forgetting it silently flattered this side. They now default
to v1's values. The remaining row that moves real work is the weight fold, which
is off in both by default -- it costs an extra peer read per (token, destination)
plus an accumulate in three kernels and a wider staging slot.
"""

import argparse
import ctypes
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

    def all_gather_rows(self, row):
        """(world, len(row)) float64 from each rank's `row`. gloo, so CPU."""
        t = torch.tensor(row, dtype=torch.float64)
        out = [torch.zeros_like(t) for _ in range(self.world)]
        dist.all_gather(out, t)
        return torch.stack(out)

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
    p.add_argument("--cmd", default="test", choices=["test", "bench", "tuning"])
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=7168)
    p.add_argument("--topk", type=int, default=8)
    # None -> 256 // world_size, which is what the v1 harness hardcodes
    # (examples/.../test_dispatch_combine_internode.py:576). world_size is not
    # known until the workers start, so the default has to be resolved there.
    p.add_argument("--experts-per-rank", type=int, default=None)
    p.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    p.add_argument("--combine-dtype", default=None, choices=list(_DTYPES))
    p.add_argument("--quant-type", default="none", choices=["none", "fp8_direct_cast"])
    p.add_argument("--num-qp", type=int, default=2)
    # 30, matching _EP_ROUNDS in the examples harness. Lower is not a
    # small-sample caveat but a different estimator: at --rounds 3 with
    # --drop-rounds 1 the two kept rounds are the ones that harness documents as
    # still in the CCO ramp, so one spike carries half the mean AND is the worst.
    p.add_argument("--rounds", type=int, default=30)
    # 32 to match v1, which hardcodes scale_dim=32 / scale_type_size=4. It is not
    # free -- it makes dispatch carry 128 more bytes per token -- which is exactly
    # why it should not be something a comparison has to remember to pass.
    p.add_argument("--scale-dim", type=int, default=32)
    p.add_argument("--tuning-scope", default="quick", choices=["quick", "full"])
    p.add_argument("--tuning-reps", type=int, default=3)
    # Smoke-test / bisect aid: stop after N candidates (0 = sweep all).
    p.add_argument("--tuning-limit", type=int, default=0)
    p.add_argument(
        "--tuning-phase", default="dispatch", choices=["dispatch", "combine"]
    )
    # Validation mode: sweep exactly one named candidate against the shipped
    # geometry. A sweep winner is chosen by a greedy chain of paired tests, each
    # with its own error; before it is written into the table it gets one long
    # head-to-head against what it would replace.
    p.add_argument("--tuning-candidate", default=None)
    # What a candidate is selected ON. "total" by default, and deliberately:
    # internode_tuning_configs.py records that the two phases are COUPLED -- a
    # dispatch with too few rdma blocks leaves the following combine ~18us slower
    # at 4/8 tokens -- so the shipped small-token dispatch is NOT the dispatch
    # argmin, it holds rdma high to keep the paired combine fast. Selecting a
    # dispatch geometry on dispatch time alone reproduces exactly the mistake
    # that comment warns about. "phase" is kept for looking at a phase in
    # isolation, which is a diagnostic, not a way to choose a table row.
    p.add_argument("--tuning-metric", default="total", choices=["total", "phase"])
    # Greedy chaining (v1's shape: a winner becomes the incumbent) is OFF by
    # default here. With a chain, one lucky early win moves the baseline and
    # every later candidate is judged against it, so the outcome depends on the
    # order noise arrived in: five repeats of the same 29-candidate sweep at 15
    # paired reps returned five different winners -- (80,53,4), (16,10,4) twice,
    # (16,8,4), (8,5,8) -- plus (32,21,8) on two earlier runs. Holding the
    # incumbent FIXED at the shipped geometry makes every candidate an
    # independent paired test against the thing it would replace, which is both
    # what we want to know and reproducible across repeats.
    p.add_argument("--tuning-greedy", action="store_true")
    # Workers to spawn per node. 8 by default, which means this harness expects
    # --nproc_per_node=1 and builds the rest of the ranks itself -- the same
    # process tree the examples harness uses, so the two are comparable without
    # remembering to pass a flag. --spawn 0 turns it off for the old shape (one
    # torchrun process per rank).
    p.add_argument("--spawn", type=int, default=8)
    # How much better a candidate must be, on the paired difference, to take
    # over. Whichever of the two is larger. Not 0: see the note in _tune.
    p.add_argument("--tuning-margin-us", type=float, default=1.5)
    p.add_argument("--tuning-margin-frac", type=float, default=0.02)
    # How much worst-of-reps regression a median win may carry, and how much
    # worst-of-reps improvement makes a median TIE interesting. Fraction of the
    # incumbent's median.
    p.add_argument("--tuning-tail-frac", type=float, default=0.10)
    # The v1 bench calls combine with weights=None, so it does not pay for the
    # weight fold: an extra peer read per (token, destination) plus an accumulate
    # in three kernels, and a wider staging slot (combXferBytes = hidden + weights).
    # Off by default so a reading here is comparable to one from that harness;
    # --bench-weights measures the fold when that is what you want. The
    # correctness path always folds -- it is checking the weights.
    p.add_argument("--bench-weights", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--drop-rounds", type=int, default=1)
    p.add_argument("--no-bench-tables", action="store_true")
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


def _verify_once(op, cfg, d, a, comm, inp, idx, wts, sc):
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
        _ROOT,
        "examples",
        "ops",
        "dispatch_combine",
        "test_dispatch_combine_internode.py",
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
        # The default is the only digit-string literal in the expression.
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                if sub.value.isdigit():
                    out[key] = int(sub.value)
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


def _geom_for_report(op, cfg, a):
    """The (block, rdma, warp) each phase actually launched with, for the table
    titles -- the point v1's `_launch_params_str` makes: a table that does not
    name its launch config cannot be matched back to the run that produced it.
    Read from the backend rather than from the config, because on the internode
    path cfg.dispatch_block_num is a dict key and not a grid."""
    try:
        # EpDispatchCombineOpHip IS the backend -- it subclasses the op.
        return (
            tuple(op._internode_geom_for("dispatch", a.max_tokens)),
            tuple(op._internode_geom_for("combine", a.max_tokens)),
        )
    except Exception:
        from mori.ops.dispatch_combine_v2.internode_tuning_configs import lookup

        t = lookup(
            cfg.world_size, cfg.hidden_dim, cfg.num_experts_per_token, a.max_tokens
        )
        if t:
            return tuple(t["dispatch"]), tuple(t["combine"])
        return (0, 0, 0), (0, 0, 0)


def _rdma_algo_token_count(idx, cfg, ll):
    """(token, destination-node) pairs this rank emits, DeepEP's definition and
    the numerator of the RDMA bandwidth column.

    Reimplemented rather than imported: this file keeps zero coupling to the v1
    harness, which is what lets the two be compared without one dragging the
    other's op in. The formula is v1's `compute_rdma_algo_token_count`.

    The LL kernel does not deduplicate across expert slots, so every token
    contributes one entry per node unconditionally.
    """
    nodes = cfg.world_size // cfg.gpu_per_node
    if ll:
        return idx.shape[0] * nodes
    per_node = cfg.num_experts_per_rank * cfg.gpu_per_node
    seen = torch.zeros(idx.shape[0], nodes, dtype=torch.bool, device=idx.device)
    seen.scatter_((idx // per_node).long().clamp_(0, nodes - 1), 1, True)
    return int(seen.sum().item())


def _phase_stats(col):
    """(worst, best, avg) over a (rounds, ranks) tensor, as v1's _compute_stats:
    worst/best are extremes over individual samples, avg is the grand mean."""
    return col.min().item(), col.max().item(), col.mean(dim=1).mean().item()


def _print_phase_table(title, rdma, xgmi, ll, lat):
    from prettytable import PrettyTable

    t = PrettyTable()
    t.title = title
    t.field_names = [
        "Metrics",
        "RDMA Bandwidth (GB/s)",
        "XGMI Bandwidth (GB/s)",
        "LL Bandwidth (GB/s)",
        "Latency (us)",
    ]
    r = lambda v: round(v, 2)
    # Bandwidth "Best" is the MAX and latency "Best" is the MIN, so the two
    # columns index the same tuple from opposite ends. v1 does this too; it is
    # the reason Best/Worst are not simply [1]/[0] throughout.
    t.add_rows(
        [
            ["Best", r(rdma[1]), r(xgmi[1]), r(ll[1]), r(lat[0])],
            ["Worst", r(rdma[0]), r(xgmi[0]), r(ll[0]), r(lat[1])],
            ["Average", r(rdma[2]), r(xgmi[2]), r(ll[2]), r(lat[2])],
        ]
    )
    print(t, flush=True)


def _report_tables(d, cfg, a, disp, comb, total_recv, idx, ll, geom):
    """v1's bench output: a per-round dump and the two performance tables.

    Every rank computes its OWN bandwidths from its own byte counts and the
    numbers are then gathered, which is what v1 does -- gathering durations and
    applying one rank's byte count to all of them would be wrong the moment the
    routing is not perfectly balanced.
    """
    ct = a.max_tokens
    d_elem = torch.tensor([], dtype=cfg.dispatch_dtype).element_size()
    c_elem = torch.tensor([], dtype=cfg.combine_dtype).element_size()
    d_bytes = total_recv * cfg.hidden_dim * d_elem
    c_bytes = total_recv * cfg.hidden_dim * c_elem
    rdma_tok = _rdma_algo_token_count(idx, cfg, ll)
    d_rdma_bytes = rdma_tok * cfg.hidden_dim * d_elem
    c_rdma_bytes = rdma_tok * cfg.hidden_dim * c_elem
    # LL packs a fixed slot per (token, expert) rather than only what routed, so
    # its wire bytes exceed the payload by this factor. v1 scales the XGMI
    # column by it to get the LL column.
    ll_scale = ct * cfg.num_experts_per_token / (total_recv + 1)

    # bw in GB/s from a duration in MICROseconds: bytes/1e9 / (us/1e6).
    bw = lambda b, us: b / (1000.0 * us) if us > 0 else 0.0
    row = []
    for dv, cv in zip(disp, comb):
        row += [
            bw(d_rdma_bytes, dv),
            bw(d_bytes, dv),
            dv,
            bw(c_rdma_bytes, cv),
            bw(c_bytes, cv),
            cv,
        ]
    g = d.all_gather_rows(row).reshape(d.world, len(disp), 6).permute(1, 0, 2)
    if d.rank != 0:
        return

    for i in range(g.shape[0]):
        rd = g[i]
        print(f"Round {i}", flush=True)
        for phase, cols in (
            (
                "dispatch",
                (
                    ("duration", 2, "us"),
                    ("rdma bandwidth", 0, "GB/s"),
                    ("bandwidth", 1, "GB/s"),
                ),
            ),
            (
                "combine",
                (
                    ("duration", 5, "us"),
                    ("rdma bandwidth", 3, "GB/s"),
                    ("bandwidth", 4, "GB/s"),
                ),
            ),
        ):
            for name, c, unit in cols:
                vals = [round(v, 2) for v in rd[:, c].tolist()]
                print(
                    f"  {phase} {name} {vals} avg {rd[:, c].mean():.2f} {unit}",
                    flush=True,
                )

    # Config header immediately above the tables. The `# BENCH` one-liner is
    # printed before the per-round dump, which at 30 rounds is 180 lines earlier
    # -- by the time the tables are on screen it has scrolled away, and a table
    # whose configuration you have to scroll to find is a table you will
    # eventually misattribute.
    nodes = cfg.world_size // cfg.gpu_per_node
    print(
        f"\n# CONFIG tok={ct} dtype={str(cfg.dispatch_dtype).split('.')[-1]}"
        f"->{str(cfg.combine_dtype).split('.')[-1]} hidden={cfg.hidden_dim} "
        f"topk={cfg.num_experts_per_token} kernel={'v1_ll' if ll else 'v1'} "
        f"world={cfg.world_size} nodes={nodes}x{cfg.gpu_per_node} "
        f"experts/rank={cfg.num_experts_per_rank} scale_dim={cfg.scale_dim} "
        f"qp={cfg.num_qp_per_pe}",
        flush=True,
    )
    print(
        f"# CONFIG dispatch block/rdma/warp={geom[0]}  combine={geom[1]}  "
        f"rounds={a.rounds} warmup={a.warmup}  "
        f"recv_tokens={total_recv} rdma_algo_tokens={rdma_tok}",
        flush=True,
    )

    for name, (cr, cx, cl), dt, gm, elem in (
        ("Dispatch", (0, 1, 2), cfg.dispatch_dtype, geom[0], d_elem),
        ("Combine", (3, 4, 5), cfg.combine_dtype, geom[1], c_elem),
    ):
        xg = _phase_stats(g[:, :, cx])
        _print_phase_table(
            f"{name} Performance ({str(dt).split('.')[-1]}) "
            f"block={gm[0]} warp={gm[2]} rdma={gm[1]} "
            f"~{ct * cfg.hidden_dim * elem / (1024 ** 2):.1f} MB/rank",
            _phase_stats(g[:, :, cr]),
            xg,
            tuple(v * ll_scale for v in xg),
            _phase_stats(g[:, :, cl]),
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

    # Check BEFORE measuring, as bench_dispatch_combine does: a silently wrong
    # configuration still produces timings. Always folds weights, whatever
    # --bench-weights says, because the point is to check them.
    ok = _verify_once(op, cfg, d, a, comm, inp, idx, wts, sc)
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

    total_recv = 0
    for i in range(a.warmup):
        r = op.dispatch(inp, wts, sc, idx, return_routing=True)
        if i == a.warmup - 1:
            # Read it here, not in the timed loop: .item() synchronises.
            torch.cuda.synchronize()
            total_recv = int(r[4][0].item())
        op.combine(convert(r[0]), cw, routing=r[5])
    torch.cuda.synchronize()

    # One pre-loop barrier, gloo, matching what the v1 harness does at this same
    # boundary. KEEP IT: replacing it with un-timed un-barriered rounds removed
    # the opening spike but put three of four runs in the slow regime.
    #
    # It is not a thundering herd, whatever the transient is: the measured exit
    # skew is 200-800us, i.e. several rounds, so the ranks are NOT released at
    # one instant. The first timed rounds run elevated regardless; drop them with
    # --drop-rounds rather than trying to warm them away, since warmup runs
    # BEFORE the barrier.
    dist.barrier()

    _series = bool(os.environ.get("MORI_EP_ROUND_SERIES"))

    # Which physical CPU this rank is actually ON, sampled around the loop.
    # sched_getaffinity only gives the ALLOWED set, and after the NUMA bind that
    # is 192 CPUs shared by four ranks -- so it cannot answer whether two ranks
    # landed on the two SMT siblings of one core, which is the mechanism that
    # produced the 2x host-loop time earlier. sched_getcpu can.
    def _cpu():
        try:
            return int(ctypes.CDLL("libc.so.6", use_errno=True).sched_getcpu())
        except Exception:
            return -1

    _cpu0 = _cpu()

    # Host time INSIDE the two calls, plus a host timestamp per round. The phase
    # events bracket the call, so whatever the wrapper does on the host before
    # the launch lands in the reported phase time and is indistinguishable from
    # kernel time there; this is what separates them. Off unless the series is
    # asked for -- perf_counter is only ~0.1us, but it would sit inside the
    # measured window and the default path should carry nothing it does not need.
    hd = [0.0] * n
    hc = [0.0] * n
    tr = [0.0] * (n + 1)

    _ep0 = time.time()
    t0 = time.perf_counter()
    ev[0].record()

    for i in range(n):
        if _series:
            tr[i] = time.perf_counter()
        r = op.dispatch(inp, wts, sc, idx, return_routing=True)
        if _series:
            hd[i] = (time.perf_counter() - tr[i]) * 1e6
        ev[3 * i + 1].record()
        x = convert(r[0])
        ev[3 * i + 2].record()
        if _series:
            _h = time.perf_counter()
        op.combine(x, cw, routing=r[5])
        if _series:
            hc[i] = (time.perf_counter() - _h) * 1e6
        ev[3 * i + 3].record()
    if _series:
        tr[n] = time.perf_counter()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e6 / n

    keep = slice(a.drop_rounds, None)
    disp = [ev[3 * i].elapsed_time(ev[3 * i + 1]) * 1e3 for i in range(n)][keep]
    comb = [ev[3 * i + 2].elapsed_time(ev[3 * i + 3]) * 1e3 for i in range(n)][keep]
    # The events TILE the timed region -- ev[3i+3] ends round i's combine and IS
    # ev[3(i+1)] -- so host time cannot hide between windows: if the host falls
    # behind, the GPU idles at the head of a segment and that idle is charged to
    # it as kernel time. This window holds one cast, whose cost is small and
    # fixed, so what it reads above that is host lag. (wall - (dispatch+combine)
    # IS this window by construction, so it cannot be used as evidence instead.)
    conv = [ev[3 * i + 1].elapsed_time(ev[3 * i + 2]) * 1e3 for i in range(n)][keep]

    # Which round stalled, opt-in. EVERY rank prints its own series: these are
    # spin-wait collectives, so one slow rank shows as a slow round on all of
    # them and only the rank-local series separates a straggler from a
    # whole-round event. See docs/EP_INTERNODE_V2_TAIL.md for how to read it.
    if _series:
        print(
            "# rounds r%d disp: " % d.rank + " ".join("%.0f" % x for x in disp),
            flush=True,
        )
        print(
            "# rounds r%d comb: " % d.rank + " ".join("%.0f" % x for x in comb),
            flush=True,
        )
        print(
            "# rounds r%d hdis: " % d.rank + " ".join("%.0f" % x for x in hd[keep]),
            flush=True,
        )
        print(
            "# rounds r%d hcom: " % d.rank + " ".join("%.0f" % x for x in hc[keep]),
            flush=True,
        )
        # Host wall per round and the convert window. With disp/comb/hdis/hcom
        # above, these close the accounting: a round whose hwal exceeds
        # disp+conv+comb has a hole somewhere the other series do not cover.
        print(
            "# rounds r%d conv: " % d.rank + " ".join("%.0f" % x for x in conv),
            flush=True,
        )
        # Epoch bounds of the timed loop, so an external sampler (clocks, NIC)
        # can be lined up with it. perf_counter has no epoch; time.time does.
        print(
            "# loop r%d t0=%.4f t1=%.4f" % (d.rank, _ep0, time.time()),
            flush=True,
        )
        print("# cpu  r%d: %d %d" % (d.rank, _cpu0, _cpu()), flush=True)
        hwal = [(tr[i + 1] - tr[i]) * 1e6 for i in range(n)][keep]
        print(
            "# rounds r%d hwal: " % d.rank + " ".join("%.0f" % x for x in hwal),
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

    # v1's bench output on top of ours: the per-round dump and the two
    # performance tables. Off with --no-bench-tables; it costs one all_gather
    # after the timed loop and nothing inside it.
    if not a.no_bench_tables:
        ll = bool(getattr(op, "_internode_force_ll", a.max_tokens <= 2048))
        geom = _geom_for_report(op, cfg, a)
        _report_tables(d, cfg, a, disp, comb, total_recv, idx, ll, geom)
    return 0


def _timed_pass(op, d, a, inp, idx, wts, sc, cw, convert, n, warm):
    """One warmup+timed block. Returns (dispatch_us, combine_us) as grand means over
    rounds x ranks, plus (worst_dispatch, worst_combine) as the worst single round."""
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3 * n + 1)]
    for _ in range(warm):
        r = op.dispatch(inp, wts, sc, idx, return_routing=True)
        op.combine(convert(r[0]), cw, routing=r[5])
    torch.cuda.synchronize()
    ev[0].record()
    for i in range(n):
        r = op.dispatch(inp, wts, sc, idx, return_routing=True)
        ev[3 * i + 1].record()
        x = convert(r[0])
        ev[3 * i + 2].record()
        op.combine(x, cw, routing=r[5])
        ev[3 * i + 3].record()
    torch.cuda.synchronize()
    keep = slice(a.drop_rounds, None)
    dv = [ev[3 * i].elapsed_time(ev[3 * i + 1]) * 1e3 for i in range(n)][keep]
    cv = [ev[3 * i + 2].elapsed_time(ev[3 * i + 3]) * 1e3 for i in range(n)][keep]
    gm = lambda v: d.allreduce_sum(int(sum(v) / len(v) * 1000)) / d.world / 1000
    # The worst ROUND, across ranks, alongside the grand means. Without it a
    # sweep cannot see the failure mode that matters here: a geometry whose
    # median round is identical but which spikes to 4-5x on one round in thirty.
    # A pass mean hides that (160us over 30 rounds moves the mean by 4us, inside
    # the noise) and a median over paired passes discards it entirely -- which is
    # exactly how (32,12,6) won the 8-token sweep and then lost the bench.
    _, wd = d.allreduce_minmax(0, int(max(dv) * 1000))
    _, wc = d.allreduce_minmax(0, int(max(cv) * 1000))
    return gm(dv), gm(cv), wd / 1000, wc / 1000


def _build_op(cfg, comm, dgeom, cgeom):
    """An op whose dispatch plans are compiled for `dgeom` and its combine plans
    for `cgeom`. Goes through the MORI_EP_*_GEOM hook the backend already exposes
    for sweeps: a geometry is a compile-time identity there, read once at build
    time, so it cannot be selected per launch the way v1's can.

    The two are SEPARATE because the shipped table gives them separate values
    (tokens 4: dispatch 64/32/8, combine 32/21/6). Driving both from one geometry
    means the sweep's incumbent is not the configuration actually shipped, so
    "beats the incumbent" would not mean "beats what we ship".
    """
    old = (os.environ.get("MORI_EP_DISP_GEOM"), os.environ.get("MORI_EP_COMB_GEOM"))
    os.environ["MORI_EP_DISP_GEOM"] = "%d,%d,%d" % dgeom
    os.environ["MORI_EP_COMB_GEOM"] = "%d,%d,%d" % cgeom
    try:
        return EpDispatchCombineOp(cfg, comm)
    finally:
        for k, v in zip(("MORI_EP_DISP_GEOM", "MORI_EP_COMB_GEOM"), old):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _med(xs):
    v = sorted(xs)
    return v[len(v) // 2]


def _tune(cfg, d, dev, a, comm):
    """Sweep launch geometries and report the winner for this token count.

    Structured after tuning_dispatch_combine in the examples harness -- same
    candidate construction (block doubling plus 8/16, warp list, rdma as a
    fraction of block), same selection metric (grand-mean latency) -- with ONE
    deliberate difference, which the numbers force.

    That harness measures each candidate once and compares the means. It can:
    its transport's max/mean is 1.19 and its run-to-run spread is a few percent.
    Ours is not: the same geometry measured back to back has produced 86us and
    137us total, and a straight sweep here already "found" 64,32,4 beating the
    shipped 64,32,8 by 5% on the mean and 30% on the worst -- which vanished
    entirely when the two were run alternately, four pairs. A one-shot sweep on
    this path selects noise and writes it into the table as if it were tuning.

    So the incumbent stays LIVE and every candidate is measured against it
    alternately, `reps` times each, comparing medians. Two ops hold symmetric
    windows at once; the loser is closed immediately. Drift, whatever its cause,
    then applies to both arms of every comparison instead of to whichever
    candidate happened to run during it.
    """
    sm = torch.cuda.get_device_properties(dev).multi_processor_count
    blocks = {b for b in (8, 16) if b < sm}
    p = 32
    while p <= sm:
        blocks.add(p)
        p <<= 1
    blocks.add(sm)
    warps = [4, 8, 16] if a.tuning_scope == "quick" else [4, 6, 8, 12, 16]

    def rdmas(bn):
        # rdma_block_num partitions the SAME grid between the blocks that talk to
        # the network and the rest, so the optimum is a fraction of block and can
        # sit anywhere in (0, 1). Three points was too coarse to say anything
        # about the shape -- and it could not even reach the shipped 32-token row,
        # whose rdma=48 against block=80 is 0.6 and is not 1/4, 1/2 or 2/3.
        # Eighths plus 2/3 covers it at a cost of ~70s a sweep against ~31s.
        frac = (
            (bn // 2, bn * 2 // 3)
            if a.tuning_scope == "quick"
            else (
                bn // 8,
                bn // 4,
                3 * bn // 8,
                bn // 2,
                5 * bn // 8,
                bn * 2 // 3,
                3 * bn // 4,
            )
        )
        return sorted({v for v in frac if 1 <= v < bn})

    cands = [(b, r, w) for b in sorted(blocks) for w in warps for r in rdmas(b)]
    if a.tuning_candidate:
        cands = [tuple(int(x) for x in a.tuning_candidate.split(","))]
    elif a.tuning_limit:
        cands = cands[: a.tuning_limit]

    from mori.ops.dispatch_combine_v2.internode_tuning_configs import lookup

    tbl = lookup(
        cfg.world_size, cfg.hidden_dim, cfg.num_experts_per_token, a.max_tokens
    )
    # The incumbent is the SHIPPED pair, so a win means "better than what we ship".
    inc_d = tuple(tbl["dispatch"]) if tbl else cands[0]
    inc_c = tuple(tbl["combine"]) if tbl else cands[0]
    phase = a.tuning_phase
    start = inc_d if phase == "dispatch" else inc_c
    if start in cands:
        cands.remove(start)

    if d.rank == 0:
        print(
            f"# TUNING tok={a.max_tokens} phase={phase} scope={a.tuning_scope} "
            f"reps={a.tuning_reps} sm={sm} candidates={len(cands)} "
            f"shipped dispatch={inc_d} combine={inc_c}",
            flush=True,
        )

    rng = torch.Generator(device=dev)
    rng.manual_seed(4242 + d.rank)
    inp, idx, wts, sc = _gen_round(rng, cfg, a.max_tokens, dev, cfg.dispatch_dtype)
    convert = (
        (lambda x: x.to(cfg.combine_dtype))
        if cfg.is_asymmetric_dtype
        else (lambda x: x)
    )
    cw = wts if a.bench_weights else None

    # Only the swept phase varies; the other stays at the shipped value, because
    # the two are coupled (a dispatch with too few rdma blocks leaves the combine
    # after it slower) and a per-phase argmin measured against a DIFFERENT other
    # phase does not carry over.
    geoms = lambda g: ((g, inc_c) if phase == "dispatch" else (inc_d, g))
    if a.tuning_metric == "total":
        pick = lambda dv, cv: dv + cv
    else:
        pick = (lambda dv, cv: dv) if phase == "dispatch" else (lambda dv, cv: cv)

    best_op = _build_op(cfg, comm, *geoms(start))
    if a.kernel_type is not None:
        best_op._internode_force_ll = a.kernel_type == "v1_ll"
    comm.barrier()
    best = start
    best_med = None
    fixed_wins = []  # non-greedy: every candidate that beat the fixed incumbent

    for k, cand in enumerate(cands):
        try:
            cand_op = _build_op(cfg, comm, *geoms(cand))
        except Exception as exc:  # a geometry the backend rejects is not a failure
            if d.rank == 0:
                print(f"#   [{k + 1}/{len(cands)}] {cand} rejected: {exc}", flush=True)
            continue
        if a.kernel_type is not None:
            cand_op._internode_force_ll = a.kernel_type == "v1_ll"
        comm.barrier()

        bt, ct_ = [], []
        bph, cph = [], []  # (dispatch, combine) per rep, to show the coupling
        bw, cw_ = [], []  # worst ROUND per pass, per arm
        for _ in range(a.tuning_reps):
            dv, cv, wd, wc = _timed_pass(
                best_op, d, a, inp, idx, wts, sc, cw, convert, a.rounds, a.warmup
            )
            bt.append(pick(dv, cv))
            bph.append((dv, cv))
            bw.append(pick(wd, wc))
            dv, cv, wd, wc = _timed_pass(
                cand_op, d, a, inp, idx, wts, sc, cw, convert, a.rounds, a.warmup
            )
            ct_.append(pick(dv, cv))
            cph.append((dv, cv))
            cw_.append(pick(wd, wc))
        bm, cm = sorted(bt)[len(bt) // 2], sorted(ct_)[len(ct_) // 2]
        # Worst of the paired reps, as a tail proxy. _timed_pass returns a grand
        # mean, so this is run-to-run spread rather than a worst ROUND -- which is
        # the right thing here anyway, since the risk being guarded against is a
        # geometry that lands in a bad regime more often.
        # MEDIAN of the per-pass worst ROUND, not the worst pass mean. The
        # median across passes keeps one unlucky pass from vetoing a candidate,
        # while the per-pass max is what makes a recurring single-round spike
        # visible at all.
        bmax = sorted(bw)[len(bw) // 2]
        cmax = sorted(cw_)[len(cw_) // 2]
        # PAIRED, not a difference of medians: the regime moves during a sweep
        # (the same incumbent geometry has read 84.9us on one candidate and
        # 126.0us on the next), and differencing within a rep cancels that. The
        # margin then floors the improvement; v1's equivalent defaults to 0,
        # which is safe at its 1.19x max/mean and not here.
        diffs = sorted(c - b_ for b_, c in zip(bt, ct_))
        lo, hi = diffs[0], diffs[-1]
        dmed = diffs[len(diffs) // 2]
        margin = max(a.tuning_margin_us, bm * a.tuning_margin_frac)
        tail_room = bm * a.tuning_tail_frac
        # A median win is not enough on its own. Selecting purely on the median
        # would accept a geometry that gains 2us at the median and gives back 30
        # at the worst, and this table is used for a latency-bound collective
        # where the worst round is what the caller waits for.
        win = dmed < -margin and (cmax - bmax) <= tail_room
        # And when the medians TIE, a clearly better worst is worth surfacing --
        # this is the "same average, better tail" rule, made explicit rather than
        # applied by hand after the fact.
        tie = abs(dmed) <= margin and (bmax - cmax) > tail_room
        if d.rank == 0:
            print(
                f"#   [{k + 1}/{len(cands)}] {cand} med={cm:6.1f}us vs "
                f"incumbent {best} med={bm:6.1f}us  paired={dmed:+6.1f}us "
                f"[{lo:+.1f},{hi:+.1f}] worst {cmax:.0f}/{bmax:.0f}  "
                f"{'WIN' if win else ('TAIL' if tie else '--')}"
                f"   d/c cand={_med(x for x, _ in cph):.1f}/{_med(y for _, y in cph):.1f}"
                f" inc={_med(x for x, _ in bph):.1f}/{_med(y for _, y in bph):.1f}",
                flush=True,
            )
        if win and a.tuning_greedy:
            best_op.close()
            best_op, best, best_med = cand_op, cand, cm
        else:
            cand_op.close()
            if not a.tuning_greedy and (win or tie):
                # Rank real median wins ahead of tail-only ties: the two keys
                # are different quantities and must not be sorted against each
                # other, or a -20us tail tie outranks a -3us median win.
                fixed_wins.append(
                    (
                        0 if win else 1,
                        dmed if win else (cmax - bmax),
                        cand,
                        cm,
                        bm,
                        win,
                        cmax,
                        bmax,
                    )
                )
            best_med = bm
        comm.barrier()

    best_op.close()
    if d.rank == 0 and not a.tuning_greedy:
        fixed_wins.sort()
        print(
            f"# TUNING tok={a.max_tokens} phase={phase}: {len(fixed_wins)} of "
            f"{len(cands)} candidates beat the fixed incumbent {start}",
            flush=True,
        )
        for _, dm, cd, cm2, bm2, w, cx, bx in fixed_wins[:5]:
            print(
                f"#   {'BEAT' if w else 'TAIL'} {cd} paired={dm:+.1f}us "
                f"(cand med={cm2:.1f} inc med={bm2:.1f} worst {cx:.0f}/{bx:.0f})",
                flush=True,
            )
        if fixed_wins:
            best = fixed_wins[0][2]
            best_med = fixed_wins[0][3]
    if d.rank == 0:
        d_out = best if phase == "dispatch" else inc_d
        c_out = best if phase == "combine" else inc_c
        print(
            f"# TUNING RESULT tok={a.max_tokens} phase={phase}: "
            f"block/rdma/warp={best} median {phase}={best_med:.1f}us "
            f"(shipped was {start})\n"
            f"#   table row: ({a.max_tokens}, {d_out[0]}, {d_out[1]}, {d_out[2]}, "
            f"{c_out[0]}, {c_out[1]}, {c_out[2]}),",
            flush=True,
        )
    return 0


def _spawn_entry(local_rank, argv, node_rank, nnodes, per_node):
    """One spawned worker. Rewrites the rank env, then re-enters main().

    torchrun gives ONE process per node here (RANK = node rank, WORLD_SIZE =
    node count), and the workers are children of it -- the shape the examples
    harness uses. Everything downstream reads its identity from the environment,
    so setting it here is the whole adaptation; no call site changes.

    LOCAL_WORLD_SIZE has to be rewritten too. torchrun sets it to 1 under
    --nproc_per_node=1, and main() derives gpu_per_node from it, which decides
    the node grouping the internode path is gated on.
    """
    os.environ["_MORI_EP_SPAWN_CHILD"] = "1"
    os.environ["RANK"] = str(node_rank * per_node + local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(nnodes * per_node)
    os.environ["LOCAL_WORLD_SIZE"] = str(per_node)
    rc = main(argv)
    if rc:
        raise SystemExit(rc)


def main(argv):
    a = _parse_args(argv)

    # --spawn N reproduces the examples harness's process topology: one torchrun
    # process per node that spawns N workers, instead of N torchrun processes.
    # Same kernels, same bench, different process tree -- which is worth being
    # able to switch because host time on this path converts to measured "kernel"
    # time about 1:1, so how the ranks are parented is not obviously neutral.
    if a.spawn and not os.environ.get("_MORI_EP_SPAWN_CHILD"):
        # Both topologies at once would be nprocs x spawn ranks per node, each
        # claiming a GPU index it does not own. Refuse rather than deadlock in
        # the rendezvous, and say which of the two to drop.
        lws = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if lws > 1:
            raise SystemExit(
                f"--spawn {a.spawn} with torchrun --nproc_per_node={lws}: that is "
                f"{lws * a.spawn} ranks per node. Use --nproc_per_node=1 (spawn "
                f"builds the ranks), or pass --spawn 0 to let torchrun do it."
            )
        node_rank = int(os.environ["RANK"])
        nnodes = int(os.environ["WORLD_SIZE"])
        torch.multiprocessing.spawn(
            _spawn_entry,
            args=(argv, node_rank, nnodes, a.spawn),
            nprocs=a.spawn,
            join=True,
        )
        return 0

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
            num_experts_per_rank=(
                a.experts_per_rank if a.experts_per_rank else 256 // npes
            ),
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

        if a.cmd == "tuning":
            # The sweep builds its own ops, one per candidate geometry; this one
            # only proved the config is constructible.
            op.close()
            rc = _tune(cfg, d, dev, a, comm)
            d.shutdown()
            return rc

        rng = torch.Generator(device=dev)
        for r in range(a.rounds):
            rng.manual_seed(1234 + r * 977 + rank)
            ct = M
            inp, idx, wts, sc = _gen_round(rng, cfg, ct, dev, cfg.dispatch_dtype)

            recv_x, recv_w, recv_s, recv_i, total_recv, routing = op.dispatch(
                inp, wts, sc, idx, return_routing=True
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
