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
``combine[t] == unique_pes[t] * input[t]`` and
``out_weights[t] == unique_pes[t] * wts[t]``, where ``unique_pes[t]`` is the
number of DISTINCT destination ranks token t routed to (the same name the v1
harness gives this quantity). The expectation
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
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \\
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

The small-token bench is bimodal per RUN, not per round: a run settles into a
fast or a slow regime and holds it, so a worst/mean ratio is a blend of two
levels rather than a tail. ``MORI_EP_ROUND_SERIES=1`` prints the per-rank
per-round host-wall series that shows which one this run is in (the print
carries a legend for its column tags). INTERLEAVE the arms of any
A/B on this path -- a block of runs can sit in one regime for reasons that have
nothing to do with the change under test.

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
        objects = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

    def allreduce_sum(self, value):
        tensor = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())

    def all_gather_rows(self, row):
        """(world, len(row)) float64 from each rank's `row`. gloo, so CPU."""
        tensor = torch.tensor(row, dtype=torch.float64)
        gathered = [torch.zeros_like(tensor) for _ in range(self.world)]
        dist.all_gather(gathered, tensor)
        return torch.stack(gathered)

    def allreduce_minmax(self, low, high):
        """Extremes across ranks, for the same best/worst the v1 harness prints."""
        tensor = torch.tensor([low, -high], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return int(tensor[0].item()), -int(tensor[1].item())

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="v2 internode dispatch/combine test")
    parser.add_argument(
        "--cmd", default="test", choices=["test", "bench", "tuning", "stress"]
    )
    # stress only: how many datasets to cycle, and how often to drain the queue.
    # v1 uses 128 and 128; matching them keeps the two soaks comparable.
    parser.add_argument("--stress-datasets", type=int, default=128)
    parser.add_argument("--stress-sync-interval", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    # None -> 256 // world_size, which is what the v1 harness hardcodes
    # (examples/.../test_dispatch_combine_internode.py:576). world_size is not
    # known until the workers start, so the default has to be resolved there.
    parser.add_argument("--experts-per-rank", type=int, default=None)
    # Where tokens go. "uniform" keeps every chunk full and so never exercises
    # the empty-chunk path; "skewed" and "local" do. See _generate_round.
    parser.add_argument(
        "--routing", default="uniform", choices=["uniform", "skewed", "local"]
    )
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    parser.add_argument("--combine-dtype", default=None, choices=list(_DTYPES))
    parser.add_argument(
        "--quant-type", default="none", choices=["none", "fp8_direct_cast"]
    )
    parser.add_argument("--num-qp", type=int, default=1)
    # 30, matching _EP_ROUNDS in the examples harness. Lower is not a
    # small-sample caveat but a different estimator: at --rounds 3 with
    # --drop-rounds 1 the two kept rounds are the ones that harness documents as
    # still in the CCO ramp, so one spike carries half the mean AND is the worst.
    parser.add_argument("--rounds", type=int, default=30)
    # 32 to match v1, which hardcodes scale_dim=32 / scale_type_size=4. It is not
    # free -- it makes dispatch carry 128 more bytes per token -- which is exactly
    # why it should not be something a comparison has to remember to pass.
    parser.add_argument("--scale-dim", type=int, default=32)
    parser.add_argument("--tuning-scope", default="quick", choices=["quick", "full"])
    parser.add_argument("--tuning-reps", type=int, default=3)
    # Smoke-test / bisect aid: stop after N candidates (0 = sweep all).
    parser.add_argument("--tuning-limit", type=int, default=0)
    parser.add_argument(
        "--tuning-phase", default="dispatch", choices=["dispatch", "combine"]
    )
    # Validation mode: sweep exactly one named candidate against the shipped
    # geometry. A sweep winner is chosen by a greedy chain of paired tests, each
    # with its own error; before it is written into the table it gets one long
    # head-to-head against what it would replace.
    parser.add_argument("--tuning-candidate", default=None)
    # What a candidate is selected ON. "total" by default, and deliberately:
    # internode_tuning_configs.py records that the two phases are COUPLED -- a
    # dispatch with too few rdma blocks leaves the following combine ~18us slower
    # at 4/8 tokens -- so the shipped small-token dispatch is NOT the dispatch
    # argmin, it holds rdma high to keep the paired combine fast. Selecting a
    # dispatch geometry on dispatch time alone reproduces exactly the mistake
    # that comment warns about. "phase" is kept for looking at a phase in
    # isolation, which is a diagnostic, not a way to choose a table row.
    parser.add_argument("--tuning-metric", default="total", choices=["total", "phase"])
    # Greedy chaining (v1's shape: a winner becomes the incumbent) is OFF by
    # default here. With a chain, one lucky early win moves the baseline and
    # every later candidate is judged against it, so the outcome depends on the
    # order noise arrived in: five repeats of the same 29-candidate sweep at 15
    # paired reps returned five different winners -- (80,53,4), (16,10,4) twice,
    # (16,8,4), (8,5,8) -- plus (32,21,8) on two earlier runs. Holding the
    # incumbent FIXED at the shipped geometry makes every candidate an
    # independent paired test against the thing it would replace, which is both
    # what we want to know and reproducible across repeats.
    parser.add_argument("--tuning-greedy", action="store_true")
    # Workers to spawn per node. 8 by default, which means this harness expects
    # --nproc_per_node=1 and builds the rest of the ranks itself -- the same
    # process tree the examples harness uses, so the two are comparable without
    # remembering to pass a flag. --spawn 0 turns it off for the old shape (one
    # torchrun process per rank).
    parser.add_argument("--spawn", type=int, default=8)
    # How much better a candidate must be, on the paired difference, to take
    # over. Whichever of the two is larger. Not 0: see the note in _tune.
    parser.add_argument("--tuning-margin-us", type=float, default=1.5)
    parser.add_argument("--tuning-margin-frac", type=float, default=0.02)
    # How much worst-of-reps regression a median win may carry, and how much
    # worst-of-reps improvement makes a median TIE interesting. Fraction of the
    # incumbent's median.
    parser.add_argument("--tuning-tail-frac", type=float, default=0.10)
    # The v1 bench calls combine with weights=None, so it does not pay for the
    # weight fold: an extra peer read per (token, destination) plus an accumulate
    # in three kernels, and a wider staging slot (combXferBytes = hidden + weights).
    # Off by default so a reading here is comparable to one from that harness;
    # --bench-weights measures the fold when that is what you want. The
    # correctness path always folds -- it is checking the weights.
    parser.add_argument("--bench-weights", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--drop-rounds", type=int, default=1)
    parser.add_argument("--no-bench-tables", action="store_true")
    # v2 and v2_ll are separate kernels. Naming one compiles only that one and
    # runs it at every token count, which is what makes a benchmark number
    # comparable to another harness's. "auto" compiles both and picks per launch
    # at --auto-ll-max-tokens.
    parser.add_argument(
        "--kernel-type", default="auto", choices=["auto", "v2", "v2_ll"]
    )
    parser.add_argument("--auto-ll-max-tokens", type=int, default=512)
    return parser.parse_args(argv)


def _generate_round(rng, cfg, num_tokens, device, dtype, routing="uniform"):
    """Seeded per-round input, routing and weights, so a failure is reproducible
    from the round number and the rank alone.

    ``routing`` shapes WHERE the tokens go, which is what decides whether the
    cross-node chunk protocol is exercised at all:

    ``uniform``  every token draws topk experts from the whole world. With topk
                 well above the node count, essentially every token reaches every
                 node, so every chunk a peer polls for does arrive. This is the
                 easy case and it hides anything to do with an EMPTY chunk.
    ``skewed``   one token in eight draws its experts remotely, the rest locally.
                 The remote node then receives far fewer tokens than the sender's
                 chunk space, so most chunks are empty and the receiver must rely
                 on the per-node token count to retire them.
    ``local``    every expert is on the sender's own node, so a peer receives
                 NOTHING and every one of its chunks is empty. The extreme.
    """
    inp = torch.randn(num_tokens, cfg.hidden_dim, generator=rng, device=device).to(
        dtype
    )
    num_experts = cfg.world_size * cfg.num_experts_per_rank
    if routing == "uniform":
        idx = torch.stack(
            [
                torch.randperm(num_experts, generator=rng, device=device)[
                    : cfg.num_experts_per_token
                ]
                for _ in range(num_tokens)
            ]
        ).to(torch.int32)
    else:
        experts_per_node = cfg.gpu_per_node * cfg.num_experts_per_rank
        my_node = cfg.rank // cfg.gpu_per_node
        num_nodes = cfg.world_size // cfg.gpu_per_node
        rows = []
        for _ in range(num_tokens):
            remote = routing == "skewed" and (
                int(torch.randint(0, 8, (1,), generator=rng, device=device).item()) == 0
            )
            if remote and num_nodes > 1:
                node_offset = int(
                    torch.randint(
                        1, num_nodes, (1,), generator=rng, device=device
                    ).item()
                )
                node = (my_node + node_offset) % num_nodes
            else:
                node = my_node
            pool = torch.randperm(experts_per_node, generator=rng, device=device)
            rows.append(pool[: cfg.num_experts_per_token] + node * experts_per_node)
        idx = torch.stack(rows).to(torch.int32)
    wts = torch.rand(
        num_tokens, cfg.num_experts_per_token, generator=rng, device=device
    )
    # Real scales when the transport is on. The dispatch send path copies
    # scale_dim * scale_type_size bytes per token whether or not a buffer was
    # handed in -- only the staging copy is guarded on the pointer -- so passing
    # None with scale_dim > 0 transports uninitialised staging. Same byte count,
    # but not something to measure against.
    scales = (
        torch.rand(num_tokens, cfg.scale_dim, generator=rng, device=device)
        if cfg.scale_dim
        else None
    )
    return inp, idx, wts, scales


def _verify_once(op, cfg, dist_handle, args, comm, inp, idx, wts, sc):
    """One dispatch+combine against the analytic golden. True if this rank agrees.

    Same expectation as `--cmd test`: an identity expert makes combine[t] equal
    unique_pes[t] * input[t] over the DISTINCT destination ranks. Weights are
    always folded here even when the timed loop will not fold them -- an
    unchecked weight path is how this harness shipped a silent bug twice.
    """
    dispatch_out = op.dispatch(inp, wts, sc, idx, return_routing=True)
    torch.cuda.synchronize()
    comm.barrier()
    combine_input = (
        dispatch_out[0].to(cfg.combine_dtype)
        if cfg.is_asymmetric_dtype
        else dispatch_out[0]
    )
    out, out_w = op.combine(combine_input, wts, routing=dispatch_out[5])
    torch.cuda.synchronize()
    comm.barrier()

    num_tokens = inp.shape[0]
    idx_cpu = idx.cpu()
    unique_pes = np.array(
        [
            len(
                {
                    int(idx_cpu[t, j]) // cfg.num_experts_per_rank
                    for j in range(args.topk)
                }
            )
            for t in range(num_tokens)
        ]
    )
    unique_pes_column = torch.from_numpy(unique_pes).view(num_tokens, 1).float()
    got = out.float().cpu()
    expected = unique_pes_column * inp.float().cpu()
    input_magnitude = inp.float().cpu().abs()
    eps = 3e-1 if (args.quant_type != "none" or cfg.dispatch_dtype in _FP8) else 8e-3
    hidden_ok = bool(
        (
            (got - expected).abs()
            <= eps * unique_pes_column * input_magnitude.clamp(min=1.0)
        ).all()
    )
    weights_ok = bool(
        (
            (out_w.cpu() - unique_pes_column * wts.float().cpu()).abs()
            <= 2e-3 * unique_pes_column
        ).all()
    )
    if not (hidden_ok and weights_ok) and dist_handle.rank == 0:
        print(
            f"#   pre-bench check: hidden_ok={hidden_ok} weights_ok={weights_ok}",
            flush=True,
        )
    return hidden_ok and weights_ok


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
    defaults = {}
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        key = names.get(getattr(target, "id", None))
        if key is None:
            continue
        # The default is the only digit-string literal in the expression.
        for sub_node in ast.walk(node.value):
            if isinstance(sub_node, ast.Constant) and isinstance(sub_node.value, str):
                if sub_node.value.isdigit():
                    defaults[key] = int(sub_node.value)
                    break
    return defaults


def _report_loop_alignment(args, rank):
    """Say out loud when this harness's timed loop is shaped differently from the
    reference one. Printed with the numbers, not buried in a docstring, because
    the numbers are what gets quoted."""
    reference = _v1_loop_defaults()
    if not reference or rank != 0:
        return
    mine = {
        "rounds": args.rounds,
        "warmup": args.warmup,
        "drop_rounds": args.drop_rounds,
    }
    differences = {
        name: (mine[name], value)
        for name, value in reference.items()
        if mine.get(name) != value
    }
    if differences:
        detail = " ".join(
            f"{name}={mine_value}(ref {reference_value})"
            for name, (mine_value, reference_value) in sorted(differences.items())
        )
        print(
            f"# WARNING: timed loop differs from run_bench_once: {detail} -- "
            f"these numbers are not directly comparable to that harness's",
            flush=True,
        )


def _geometry_for_report(op, cfg, args):
    """The (block, rdma, warp) each phase actually launched with, for the table
    titles -- the point v1's `_launch_params_str` makes: a table that does not
    name its launch config cannot be matched back to the run that produced it.
    Read from the backend rather than from the config, because on the internode
    path cfg.dispatch_block_num is a dict key and not a grid."""
    try:
        # EpDispatchCombineOpHip IS the backend -- it subclasses the op.
        return (
            tuple(op._internode_geom_for("dispatch", args.max_tokens)),
            tuple(op._internode_geom_for("combine", args.max_tokens)),
        )
    except Exception:
        from mori.ops.dispatch_combine_v2.internode_tuning_configs import lookup

        table_row = lookup(
            cfg.world_size, cfg.hidden_dim, cfg.num_experts_per_token, args.max_tokens
        )
        if table_row:
            return tuple(table_row["dispatch"]), tuple(table_row["combine"])
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
    experts_per_node = cfg.num_experts_per_rank * cfg.gpu_per_node
    seen = torch.zeros(idx.shape[0], nodes, dtype=torch.bool, device=idx.device)
    seen.scatter_(1, (idx // experts_per_node).long(), True)
    return int(seen.sum().item())


def _phase_stats(column):
    """(worst, best, avg) over a (rounds, ranks) tensor, as v1's _compute_stats:
    worst/best are extremes over individual samples, avg is the grand mean."""
    return column.min().item(), column.max().item(), column.mean(dim=1).mean().item()


def _print_phase_table(title, rdma, xgmi, ll, latency):
    from prettytable import PrettyTable

    table = PrettyTable()
    table.title = title
    table.field_names = [
        "Metrics",
        "RDMA Bandwidth (GB/s)",
        "XGMI Bandwidth (GB/s)",
        "LL Bandwidth (GB/s)",
        "Latency (us)",
    ]

    def rounded(value):
        return round(value, 2)

    # Bandwidth "Best" is the MAX and latency "Best" is the MIN, so the two
    # columns index the same tuple from opposite ends. v1 does this too; it is
    # the reason Best/Worst are not simply [1]/[0] throughout.
    table.add_rows(
        [
            [
                "Best",
                rounded(rdma[1]),
                rounded(xgmi[1]),
                rounded(ll[1]),
                rounded(latency[0]),
            ],
            [
                "Worst",
                rounded(rdma[0]),
                rounded(xgmi[0]),
                rounded(ll[0]),
                rounded(latency[1]),
            ],
            [
                "Average",
                rounded(rdma[2]),
                rounded(xgmi[2]),
                rounded(ll[2]),
                rounded(latency[2]),
            ],
        ]
    )
    print(table, flush=True)


def _report_tables(
    dist_handle, cfg, args, dispatch_us, combine_us, total_recv, idx, ll, geometry
):
    """v1's bench output: a per-round dump and the two performance tables.

    Every rank computes its OWN bandwidths from its own byte counts and the
    numbers are then gathered, which is what v1 does -- gathering durations and
    applying one rank's byte count to all of them would be wrong the moment the
    routing is not perfectly balanced.
    """
    num_tokens = args.max_tokens
    dispatch_elem_size = torch.tensor([], dtype=cfg.dispatch_dtype).element_size()
    combine_elem_size = torch.tensor([], dtype=cfg.combine_dtype).element_size()
    dispatch_bytes = total_recv * cfg.hidden_dim * dispatch_elem_size
    combine_bytes = total_recv * cfg.hidden_dim * combine_elem_size
    rdma_tokens = _rdma_algo_token_count(idx, cfg, ll)
    dispatch_rdma_bytes = rdma_tokens * cfg.hidden_dim * dispatch_elem_size
    combine_rdma_bytes = rdma_tokens * cfg.hidden_dim * combine_elem_size
    # LL packs a fixed slot per (token, expert) rather than only what routed, so
    # its wire bytes exceed the payload by this factor. v1 scales the XGMI
    # column by it to get the LL column.
    ll_scale = num_tokens * cfg.num_experts_per_token / (total_recv + 1)

    # Bandwidth in GB/s from a duration in MICROseconds: bytes/1e9 / (us/1e6).
    def bandwidth(num_bytes, microseconds):
        return num_bytes / (1000.0 * microseconds) if microseconds > 0 else 0.0

    row = []
    for dispatch_time, combine_time in zip(dispatch_us, combine_us):
        row += [
            bandwidth(dispatch_rdma_bytes, dispatch_time),
            bandwidth(dispatch_bytes, dispatch_time),
            dispatch_time,
            bandwidth(combine_rdma_bytes, combine_time),
            bandwidth(combine_bytes, combine_time),
            combine_time,
        ]
    gathered = (
        dist_handle.all_gather_rows(row)
        .reshape(dist_handle.world, len(dispatch_us), 6)
        .permute(1, 0, 2)
    )
    if dist_handle.rank != 0:
        return

    for i in range(gathered.shape[0]):
        round_data = gathered[i]
        print(f"Round {i}", flush=True)
        for phase, columns in (
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
            for name, column, unit in columns:
                values = [round(v, 2) for v in round_data[:, column].tolist()]
                print(
                    f"  {phase} {name} {values} avg "
                    f"{round_data[:, column].mean():.2f} {unit}",
                    flush=True,
                )

    # Config header immediately above the tables. The `# BENCH` one-liner is
    # printed before the per-round dump, which at 30 rounds is 180 lines earlier
    # -- by the time the tables are on screen it has scrolled away, and a table
    # whose configuration you have to scroll to find is a table you will
    # eventually misattribute.
    nodes = cfg.world_size // cfg.gpu_per_node
    print(
        f"\n# CONFIG tok={num_tokens} dtype={str(cfg.dispatch_dtype).split('.')[-1]}"
        f"->{str(cfg.combine_dtype).split('.')[-1]} hidden={cfg.hidden_dim} "
        f"topk={cfg.num_experts_per_token} kernel={'v2_ll' if ll else 'v2'} "
        f"world={cfg.world_size} nodes={nodes}x{cfg.gpu_per_node} "
        f"experts/rank={cfg.num_experts_per_rank} scale_dim={cfg.scale_dim} "
        f"qp={cfg.num_qp_per_pe}",
        flush=True,
    )
    print(
        f"# CONFIG dispatch block/rdma/warp={geometry[0]}  combine={geometry[1]}  "
        f"rounds={args.rounds} warmup={args.warmup}  "
        f"recv_tokens={total_recv} rdma_algo_tokens={rdma_tokens}",
        flush=True,
    )

    for (
        name,
        (rdma_column, xgmi_column, latency_column),
        phase_dtype,
        phase_geometry,
        elem_size,
    ) in (
        ("Dispatch", (0, 1, 2), cfg.dispatch_dtype, geometry[0], dispatch_elem_size),
        ("Combine", (3, 4, 5), cfg.combine_dtype, geometry[1], combine_elem_size),
    ):
        xgmi_stats = _phase_stats(gathered[:, :, xgmi_column])
        _print_phase_table(
            f"{name} Performance ({str(phase_dtype).split('.')[-1]}) "
            f"block={phase_geometry[0]} warp={phase_geometry[2]} "
            f"rdma={phase_geometry[1]} "
            f"~{num_tokens * cfg.hidden_dim * elem_size / (1024 ** 2):.1f} MB/rank",
            _phase_stats(gathered[:, :, rdma_column]),
            xgmi_stats,
            tuple(stat * ll_scale for stat in xgmi_stats),
            _phase_stats(gathered[:, :, latency_column]),
        )


def _bench(op, cfg, dist_handle, device, args, comm):
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
    _report_loop_alignment(args, dist_handle.rank)
    rng = torch.Generator(device=device)
    rng.manual_seed(4242 + dist_handle.rank)
    num_tokens = args.max_tokens
    inp, idx, wts, sc = _generate_round(
        rng, cfg, num_tokens, device, cfg.dispatch_dtype, args.routing
    )

    # Check BEFORE measuring, as bench_dispatch_combine does: a silently wrong
    # configuration still produces timings. Always folds weights, whatever
    # --bench-weights says, because the point is to check them.
    verified = _verify_once(op, cfg, dist_handle, args, comm, inp, idx, wts, sc)
    failed_ranks = dist_handle.allreduce_sum(0 if verified else 1)
    if failed_ranks:
        if dist_handle.rank == 0:
            print(
                f"# BENCH ABORTED: {failed_ranks} of {dist_handle.world} ranks "
                f"failed the pre-bench check; the numbers below would be meaningless",
                flush=True,
            )
        return 1
    torch.cuda.synchronize()
    comm.barrier()

    # The examples harness's _convert_for_combine: the combine leg reads its
    # input as its own element type, so an asymmetric config has to cast first.
    def convert(tensor):
        return tensor.to(cfg.combine_dtype) if cfg.is_asymmetric_dtype else tensor

    combine_weights = wts if args.bench_weights else None

    # Allocated before the warmup, where run_bench_once allocates them. (Priming
    # them with a record here was tried and did not move the tail, so it is not
    # done -- run_bench_once does not either.)
    num_rounds = args.rounds
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3 * num_rounds + 1)]

    total_recv = 0
    for i in range(args.warmup):
        dispatch_out = op.dispatch(inp, wts, sc, idx, return_routing=True)
        if i == args.warmup - 1:
            # Read it here, not in the timed loop: .item() synchronises.
            torch.cuda.synchronize()
            total_recv = int(dispatch_out[4][0].item())
        op.combine(convert(dispatch_out[0]), combine_weights, routing=dispatch_out[5])
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

    print_series = bool(os.environ.get("MORI_EP_ROUND_SERIES"))

    # Which physical CPU this rank is actually ON, sampled around the loop.
    # sched_getaffinity only gives the ALLOWED set, and after the NUMA bind that
    # is 192 CPUs shared by four ranks -- so it cannot answer whether two ranks
    # landed on the two SMT siblings of one core, which is the mechanism that
    # produced the 2x host-loop time earlier. sched_getcpu can.
    def current_cpu():
        try:
            return int(ctypes.CDLL("libc.so.6", use_errno=True).sched_getcpu())
        except Exception:
            return -1

    cpu_at_start = current_cpu()

    # Host time INSIDE the two calls, plus a host timestamp per round. The phase
    # events bracket the call, so whatever the wrapper does on the host before
    # the launch lands in the reported phase time and is indistinguishable from
    # kernel time there; this is what separates them. Off unless the series is
    # asked for -- perf_counter is only ~0.1us, but it would sit inside the
    # measured window and the default path should carry nothing it does not need.
    host_dispatch_us = [0.0] * num_rounds
    host_combine_us = [0.0] * num_rounds
    round_start = [0.0] * (num_rounds + 1)

    epoch_at_start = time.time()
    loop_start = time.perf_counter()
    events[0].record()

    for i in range(num_rounds):
        if print_series:
            round_start[i] = time.perf_counter()
        dispatch_out = op.dispatch(inp, wts, sc, idx, return_routing=True)
        if print_series:
            host_dispatch_us[i] = (time.perf_counter() - round_start[i]) * 1e6
        events[3 * i + 1].record()
        combine_input = convert(dispatch_out[0])
        events[3 * i + 2].record()
        if print_series:
            combine_call_start = time.perf_counter()
        op.combine(combine_input, combine_weights, routing=dispatch_out[5])
        if print_series:
            host_combine_us[i] = (time.perf_counter() - combine_call_start) * 1e6
        events[3 * i + 3].record()
    if print_series:
        round_start[num_rounds] = time.perf_counter()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - loop_start) * 1e6 / num_rounds

    keep = slice(args.drop_rounds, None)
    dispatch_us = [
        events[3 * i].elapsed_time(events[3 * i + 1]) * 1e3 for i in range(num_rounds)
    ][keep]
    combine_us = [
        events[3 * i + 2].elapsed_time(events[3 * i + 3]) * 1e3
        for i in range(num_rounds)
    ][keep]
    # The events TILE the timed region -- events[3i+3] ends round i's combine and
    # IS events[3(i+1)] -- so host time cannot hide between windows: if the host
    # falls behind, the GPU idles at the head of a segment and that idle is charged
    # to it as kernel time. This window holds one cast, whose cost is small and
    # fixed, so what it reads above that is host lag. (wall - (dispatch+combine)
    # IS this window by construction, so it cannot be used as evidence instead.)
    convert_us = [
        events[3 * i + 1].elapsed_time(events[3 * i + 2]) * 1e3
        for i in range(num_rounds)
    ][keep]

    # Which round stalled, opt-in. EVERY rank prints its own series: these are
    # spin-wait collectives, so one slow rank shows as a slow round on all of
    # them and only the rank-local series separates a straggler from a
    # whole-round event. A rank whose host wall runs at about twice its peers'
    # is the signature of two ranks sharing one physical core.
    if print_series:
        # The series carry short tags so the columns line up across sixteen
        # ranks. Print what they mean rather than making the reader come here.
        if dist_handle.rank == 0:
            print(
                "# series legend, one value per round, microseconds:\n"
                "#   disp  dispatch kernel time (GPU events)\n"
                "#   comb  combine kernel time (GPU events)\n"
                "#   conv  the dtype cast between them, when the legs differ\n"
                "#   hdis  host time inside the op.dispatch() call\n"
                "#   hcom  host time inside the op.combine() call\n"
                "#   hwal  host wall clock for the whole round\n"
                "# A round whose hwal exceeds disp+conv+comb has a gap the other\n"
                "# series do not account for. A rank whose hwal is ~2x its peers'\n"
                "# is two ranks on the two SMT siblings of one physical core.",
                flush=True,
            )
        print(
            "# rounds r%d disp: " % dist_handle.rank
            + " ".join("%.0f" % value for value in dispatch_us),
            flush=True,
        )
        print(
            "# rounds r%d comb: " % dist_handle.rank
            + " ".join("%.0f" % value for value in combine_us),
            flush=True,
        )
        print(
            "# rounds r%d hdis: " % dist_handle.rank
            + " ".join("%.0f" % value for value in host_dispatch_us[keep]),
            flush=True,
        )
        print(
            "# rounds r%d hcom: " % dist_handle.rank
            + " ".join("%.0f" % value for value in host_combine_us[keep]),
            flush=True,
        )
        # Host wall per round and the convert window; with the four above they
        # close the accounting. See the legend.
        print(
            "# rounds r%d conv: " % dist_handle.rank
            + " ".join("%.0f" % value for value in convert_us),
            flush=True,
        )
        # Epoch bounds of the timed loop, so an external sampler (clocks, NIC)
        # can be lined up with it. perf_counter has no epoch; time.time does.
        print(
            "# loop r%d t0=%.4f t1=%.4f"
            % (dist_handle.rank, epoch_at_start, time.time()),
            flush=True,
        )
        print(
            "# cpu  r%d: %d %d" % (dist_handle.rank, cpu_at_start, current_cpu()),
            flush=True,
        )
        host_wall_us = [
            (round_start[i + 1] - round_start[i]) * 1e6 for i in range(num_rounds)
        ][keep]
        print(
            "# rounds r%d hwal: " % dist_handle.rank
            + " ".join("%.0f" % value for value in host_wall_us),
            flush=True,
        )

    # AVERAGE over rounds x ranks, plus BEST and WORST over the same sample set --
    # the three numbers run_bench_once prints, so a reading here can be put beside
    # one from the v1 harness without converting estimators. The average is the
    # robust one; best/worst are extremes and noise-dominated, but they are what
    # makes a single stalled round visible. Reporting only a minimum hides exactly
    # that (an earlier version of this comparison did, and buried a 227us outlier).
    def _stats(values):
        mean = (
            dist_handle.allreduce_sum(int(sum(values) / len(values) * 1000))
            / dist_handle.world
            / 1000
        )
        low, high = dist_handle.allreduce_minmax(
            int(min(values) * 1000), int(max(values) * 1000)
        )
        return mean, low / 1000, high / 1000

    dispatch_mean, dispatch_low, dispatch_high = _stats(dispatch_us)
    combine_mean, combine_low, combine_high = _stats(combine_us)
    convert_mean, _, _ = _stats(convert_us)
    # Report the family that RAN, not the request: under "auto" the request does
    # not name one, and a number is only comparable to another harness's if the
    # kernel behind it is named.
    kernel_ran = "v2_ll" if op._internode_use_ll(num_tokens) else "v2"
    if dist_handle.rank == 0:
        print(
            f"# BENCH tok={num_tokens} "
            f"dtype={args.dtype}->{args.combine_dtype or args.dtype} "
            f"hidden={cfg.hidden_dim} topk={cfg.num_experts_per_token} "
            f"kernel={kernel_ran}{'(auto)' if args.kernel_type == 'auto' else ''} "
            f"dispatch={dispatch_mean:.1f}us [{dispatch_low:.1f}/{dispatch_high:.1f}] "
            f"combine={combine_mean:.1f}us [{combine_low:.1f}/{combine_high:.1f}] "
            f"total={dispatch_mean + combine_mean:.1f}us "
            f"[conv={convert_mean:.1f}us wall={wall:.1f}us]",
            flush=True,
        )

    # v1's bench output on top of ours: the per-round dump and the two
    # performance tables. Off with --no-bench-tables; it costs one all_gather
    # after the timed loop and nothing inside it.
    if not args.no_bench_tables:
        ll = op._internode_use_ll(args.max_tokens)
        geometry = _geometry_for_report(op, cfg, args)
        _report_tables(
            dist_handle,
            cfg,
            args,
            dispatch_us,
            combine_us,
            total_recv,
            idx,
            ll,
            geometry,
        )
    return 0


def _timed_pass(
    op,
    dist_handle,
    args,
    inp,
    idx,
    wts,
    sc,
    combine_weights,
    convert,
    num_rounds,
    num_warmup,
):
    """One warmup+timed block. Returns (dispatch_us, combine_us) as grand means over
    rounds x ranks, plus (worst_dispatch, worst_combine) as the worst single round."""
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3 * num_rounds + 1)]
    for _ in range(num_warmup):
        dispatch_out = op.dispatch(inp, wts, sc, idx, return_routing=True)
        op.combine(convert(dispatch_out[0]), combine_weights, routing=dispatch_out[5])
    torch.cuda.synchronize()
    events[0].record()
    for i in range(num_rounds):
        dispatch_out = op.dispatch(inp, wts, sc, idx, return_routing=True)
        events[3 * i + 1].record()
        combine_input = convert(dispatch_out[0])
        events[3 * i + 2].record()
        op.combine(combine_input, combine_weights, routing=dispatch_out[5])
        events[3 * i + 3].record()
    torch.cuda.synchronize()
    keep = slice(args.drop_rounds, None)
    dispatch_us = [
        events[3 * i].elapsed_time(events[3 * i + 1]) * 1e3 for i in range(num_rounds)
    ][keep]
    combine_us = [
        events[3 * i + 2].elapsed_time(events[3 * i + 3]) * 1e3
        for i in range(num_rounds)
    ][keep]

    def grand_mean(values):
        return (
            dist_handle.allreduce_sum(int(sum(values) / len(values) * 1000))
            / dist_handle.world
            / 1000
        )

    # The worst ROUND, across ranks, alongside the grand means. Without it a
    # sweep cannot see the failure mode that matters here: a geometry whose
    # median round is identical but which spikes to 4-5x on one round in thirty.
    # A pass mean hides that (160us over 30 rounds moves the mean by 4us, inside
    # the noise) and a median over paired passes discards it entirely -- which is
    # exactly how (32,12,6) won the 8-token sweep and then lost the bench.
    _, worst_dispatch = dist_handle.allreduce_minmax(0, int(max(dispatch_us) * 1000))
    _, worst_combine = dist_handle.allreduce_minmax(0, int(max(combine_us) * 1000))
    return (
        grand_mean(dispatch_us),
        grand_mean(combine_us),
        worst_dispatch / 1000,
        worst_combine / 1000,
    )


def _build_op(cfg, comm, dispatch_geometry, combine_geometry):
    """An op whose dispatch plans are compiled for `dispatch_geometry` and its
    combine plans for `combine_geometry`. Goes through the MORI_EP_*_GEOM hook
    the backend already exposes for sweeps: a geometry is a compile-time identity
    there, read once at build time, so it cannot be selected per launch the way
    v1's can.

    The two are SEPARATE because the shipped table gives them separate values
    (tokens 4: dispatch 64/32/8, combine 32/21/6). Driving both from one geometry
    means the sweep's incumbent is not the configuration actually shipped, so
    "beats the incumbent" would not mean "beats what we ship".
    """
    previous = (
        os.environ.get("MORI_EP_DISP_GEOM"),
        os.environ.get("MORI_EP_COMB_GEOM"),
    )
    os.environ["MORI_EP_DISP_GEOM"] = "%d,%d,%d" % dispatch_geometry
    os.environ["MORI_EP_COMB_GEOM"] = "%d,%d,%d" % combine_geometry
    try:
        return EpDispatchCombineOp(cfg, comm)
    finally:
        for name, value in zip(("MORI_EP_DISP_GEOM", "MORI_EP_COMB_GEOM"), previous):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _stress(op, cfg, dist_handle, device, args, comm):
    """v1's stress case: cycle pre-generated rounds with VARYING token counts.

    Deliberately different from --cmd bench, which sends max_tokens every round
    from one buffer. Here each rank draws a fresh count in [1, max_tokens] per
    dataset, so the soak covers what a fixed-size bench cannot: ragged and
    unequal loads, the chunk protocol's empty-chunk path, and -- under
    internode_kernel="auto" -- the per-call switch between the v2 and v2_ll
    families, since the counts fall on both sides of the crossover.

    No per-round verification: this looks for hangs, faults and protocol drift
    over many rounds, and checking every round would hide them behind the
    check's own synchronisation. The queue is drained every
    --stress-sync-interval rounds rather than every round for the same reason.

    v1 follows its soak with a CUDA-graph phase. Not reproduced here: graph
    capture over the internode plan sequence is untested, and adding it to a
    soak would confuse a capture bug with a protocol one.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(20260910 + dist_handle.rank)
    max_tokens = args.max_tokens

    counts = torch.randint(
        1, max_tokens + 1, [args.stress_datasets], generator=rng, device=device
    ).tolist()
    datasets = [
        _generate_round(rng, cfg, int(n), device, cfg.dispatch_dtype, args.routing)
        for n in counts
    ]
    used_ll = {n: op._internode_use_ll(n) for n in set(counts)}
    if dist_handle.rank == 0:
        families = sorted({("v2_ll" if v else "v2") for v in used_ll.values()})
        print(
            f"# STRESS rounds={args.rounds} datasets={args.stress_datasets} "
            f"tokens=[{min(counts)},{max(counts)}] of {max_tokens} "
            f"routing={args.routing} kernel={args.kernel_type} "
            f"families exercised: {'+'.join(families)}",
            flush=True,
        )

    dist.barrier()
    started = time.time()
    for i in range(args.rounds):
        inp, idx, wts, sc = datasets[i % len(datasets)]
        dispatch_out = op.dispatch(inp, wts, sc, idx, return_routing=True)
        combine_input = (
            dispatch_out[0].to(cfg.combine_dtype)
            if cfg.is_asymmetric_dtype
            else dispatch_out[0]
        )
        # None, not wts: v1's soak combines without weights
        # (run_combine(op, combine_input, None, indices)), and since want_weights
        # is now honoured, passing them would make this a different kernel path
        # from the case it is meant to mirror.
        op.combine(combine_input, None, routing=dispatch_out[5])
        if i % args.stress_sync_interval == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    comm.barrier()

    if dist_handle.rank == 0:
        print(
            f"# STRESS OK: {args.rounds} rounds in {time.time() - started:.1f}s",
            flush=True,
        )
    return 0


def _tune(cfg, dist_handle, device, args, comm):
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
    num_cus = torch.cuda.get_device_properties(device).multi_processor_count
    blocks = {b for b in (8, 16) if b < num_cus}
    power_of_two = 32
    while power_of_two <= num_cus:
        blocks.add(power_of_two)
        power_of_two <<= 1
    blocks.add(num_cus)
    warps = [4, 8, 16] if args.tuning_scope == "quick" else [4, 6, 8, 12, 16]

    def rdma_block_counts(block_count):
        # rdma_block_num partitions the SAME grid between the blocks that talk to
        # the network and the rest, so the optimum is a fraction of block and can
        # sit anywhere in (0, 1). Three points was too coarse to say anything
        # about the shape -- and it could not even reach the shipped 32-token row,
        # whose rdma=48 against block=80 is 0.6 and is not 1/4, 1/2 or 2/3.
        # Eighths plus 2/3 covers it at a cost of ~70s a sweep against ~31s.
        fractions = (
            (block_count // 2, block_count * 2 // 3)
            if args.tuning_scope == "quick"
            else (
                block_count // 8,
                block_count // 4,
                3 * block_count // 8,
                block_count // 2,
                5 * block_count // 8,
                block_count * 2 // 3,
                3 * block_count // 4,
            )
        )
        return sorted({count for count in fractions if 1 <= count < block_count})

    candidates = [
        (block, rdma, warp)
        for block in sorted(blocks)
        for warp in warps
        for rdma in rdma_block_counts(block)
    ]
    if args.tuning_candidate:
        candidates = [tuple(int(x) for x in args.tuning_candidate.split(","))]
    elif args.tuning_limit:
        candidates = candidates[: args.tuning_limit]

    from mori.ops.dispatch_combine_v2.internode_tuning_configs import lookup

    table_row = lookup(
        cfg.world_size, cfg.hidden_dim, cfg.num_experts_per_token, args.max_tokens
    )
    # The incumbent is the SHIPPED pair, so a win means "better than what we ship".
    incumbent_dispatch = tuple(table_row["dispatch"]) if table_row else candidates[0]
    incumbent_combine = tuple(table_row["combine"]) if table_row else candidates[0]
    phase = args.tuning_phase
    shipped = incumbent_dispatch if phase == "dispatch" else incumbent_combine
    if shipped in candidates:
        candidates.remove(shipped)

    if dist_handle.rank == 0:
        print(
            f"# TUNING tok={args.max_tokens} phase={phase} "
            f"scope={args.tuning_scope} reps={args.tuning_reps} cus={num_cus} "
            f"candidates={len(candidates)} shipped dispatch={incumbent_dispatch} "
            f"combine={incumbent_combine}",
            flush=True,
        )

    rng = torch.Generator(device=device)
    rng.manual_seed(4242 + dist_handle.rank)
    inp, idx, wts, sc = _generate_round(
        rng, cfg, args.max_tokens, device, cfg.dispatch_dtype
    )
    convert = (
        (lambda tensor: tensor.to(cfg.combine_dtype))
        if cfg.is_asymmetric_dtype
        else (lambda tensor: tensor)
    )
    combine_weights = wts if args.bench_weights else None

    # Only the swept phase varies; the other stays at the shipped value, because
    # the two are coupled (a dispatch with too few rdma blocks leaves the combine
    # after it slower) and a per-phase argmin measured against a DIFFERENT other
    # phase does not carry over.
    def geometries(geometry):
        return (
            (geometry, incumbent_combine)
            if phase == "dispatch"
            else (incumbent_dispatch, geometry)
        )

    if args.tuning_metric == "total":

        def metric(dispatch_value, combine_value):
            return dispatch_value + combine_value

    elif phase == "dispatch":

        def metric(dispatch_value, combine_value):
            return dispatch_value

    else:

        def metric(dispatch_value, combine_value):
            return combine_value

    best_op = _build_op(cfg, comm, *geometries(shipped))
    comm.barrier()
    best = shipped
    best_median = None
    fixed_wins = []  # non-greedy: every candidate that beat the fixed incumbent

    for index, candidate in enumerate(candidates):
        try:
            candidate_op = _build_op(cfg, comm, *geometries(candidate))
        except Exception as exc:  # a geometry the backend rejects is not a failure
            if dist_handle.rank == 0:
                print(
                    f"#   [{index + 1}/{len(candidates)}] {candidate} "
                    f"rejected: {exc}",
                    flush=True,
                )
            continue
        comm.barrier()

        incumbent_metrics, candidate_metrics = [], []
        # (dispatch, combine) per rep, to show the coupling
        incumbent_phases, candidate_phases = [], []
        incumbent_worsts, candidate_worsts = [], []  # worst ROUND per pass, per arm
        for _ in range(args.tuning_reps):
            mean_dispatch, mean_combine, worst_dispatch, worst_combine = _timed_pass(
                best_op,
                dist_handle,
                args,
                inp,
                idx,
                wts,
                sc,
                combine_weights,
                convert,
                args.rounds,
                args.warmup,
            )
            incumbent_metrics.append(metric(mean_dispatch, mean_combine))
            incumbent_phases.append((mean_dispatch, mean_combine))
            incumbent_worsts.append(metric(worst_dispatch, worst_combine))
            mean_dispatch, mean_combine, worst_dispatch, worst_combine = _timed_pass(
                candidate_op,
                dist_handle,
                args,
                inp,
                idx,
                wts,
                sc,
                combine_weights,
                convert,
                args.rounds,
                args.warmup,
            )
            candidate_metrics.append(metric(mean_dispatch, mean_combine))
            candidate_phases.append((mean_dispatch, mean_combine))
            candidate_worsts.append(metric(worst_dispatch, worst_combine))
        incumbent_median = sorted(incumbent_metrics)[len(incumbent_metrics) // 2]
        candidate_median = sorted(candidate_metrics)[len(candidate_metrics) // 2]
        # Worst of the paired reps, as a tail proxy. _timed_pass returns a grand
        # mean, so this is run-to-run spread rather than a worst ROUND -- which is
        # the right thing here anyway, since the risk being guarded against is a
        # geometry that lands in a bad regime more often.
        # MEDIAN of the per-pass worst ROUND, not the worst pass mean. The
        # median across passes keeps one unlucky pass from vetoing a candidate,
        # while the per-pass max is what makes a recurring single-round spike
        # visible at all.
        incumbent_worst = sorted(incumbent_worsts)[len(incumbent_worsts) // 2]
        candidate_worst = sorted(candidate_worsts)[len(candidate_worsts) // 2]
        # PAIRED, not a difference of medians: the regime moves during a sweep
        # (the same incumbent geometry has read 84.9us on one candidate and
        # 126.0us on the next), and differencing within a rep cancels that. The
        # margin then floors the improvement; v1's equivalent defaults to 0,
        # which is safe at its 1.19x max/mean and not here.
        diffs = sorted(
            candidate_value - incumbent_value
            for incumbent_value, candidate_value in zip(
                incumbent_metrics, candidate_metrics
            )
        )
        low, high = diffs[0], diffs[-1]
        median_diff = diffs[len(diffs) // 2]
        margin = max(args.tuning_margin_us, incumbent_median * args.tuning_margin_frac)
        tail_room = incumbent_median * args.tuning_tail_frac
        # A median win is not enough on its own. Selecting purely on the median
        # would accept a geometry that gains 2us at the median and gives back 30
        # at the worst, and this table is used for a latency-bound collective
        # where the worst round is what the caller waits for.
        win = median_diff < -margin and (candidate_worst - incumbent_worst) <= tail_room
        # And when the medians TIE, a clearly better worst is worth surfacing --
        # this is the "same average, better tail" rule, made explicit rather than
        # applied by hand after the fact.
        tie = (
            abs(median_diff) <= margin
            and (incumbent_worst - candidate_worst) > tail_room
        )
        if dist_handle.rank == 0:
            candidate_dispatch_median = _median(
                dispatch_time for dispatch_time, _ in candidate_phases
            )
            candidate_combine_median = _median(
                combine_time for _, combine_time in candidate_phases
            )
            incumbent_dispatch_median = _median(
                dispatch_time for dispatch_time, _ in incumbent_phases
            )
            incumbent_combine_median = _median(
                combine_time for _, combine_time in incumbent_phases
            )
            print(
                f"#   [{index + 1}/{len(candidates)}] {candidate} "
                f"med={candidate_median:6.1f}us vs "
                f"incumbent {best} med={incumbent_median:6.1f}us  "
                f"paired={median_diff:+6.1f}us "
                f"[{low:+.1f},{high:+.1f}] "
                f"worst {candidate_worst:.0f}/{incumbent_worst:.0f}  "
                f"{'WIN' if win else ('TAIL' if tie else '--')}"
                f"   d/c cand={candidate_dispatch_median:.1f}"
                f"/{candidate_combine_median:.1f}"
                f" inc={incumbent_dispatch_median:.1f}"
                f"/{incumbent_combine_median:.1f}",
                flush=True,
            )
        if win and args.tuning_greedy:
            best_op.close()
            best_op, best, best_median = candidate_op, candidate, candidate_median
        else:
            candidate_op.close()
            if not args.tuning_greedy and (win or tie):
                # Rank real median wins ahead of tail-only ties: the two keys
                # are different quantities and must not be sorted against each
                # other, or a -20us tail tie outranks a -3us median win.
                fixed_wins.append(
                    (
                        0 if win else 1,
                        median_diff if win else (candidate_worst - incumbent_worst),
                        candidate,
                        candidate_median,
                        incumbent_median,
                        win,
                        candidate_worst,
                        incumbent_worst,
                    )
                )
            best_median = incumbent_median
        comm.barrier()

    best_op.close()
    if dist_handle.rank == 0 and not args.tuning_greedy:
        fixed_wins.sort()
        print(
            f"# TUNING tok={args.max_tokens} phase={phase}: {len(fixed_wins)} of "
            f"{len(candidates)} candidates beat the fixed incumbent {shipped}",
            flush=True,
        )
        for (
            _,
            paired_diff,
            candidate,
            candidate_median,
            incumbent_median,
            is_win,
            candidate_worst,
            incumbent_worst,
        ) in fixed_wins[:5]:
            print(
                f"#   {'BEAT' if is_win else 'TAIL'} {candidate} "
                f"paired={paired_diff:+.1f}us "
                f"(cand med={candidate_median:.1f} inc med={incumbent_median:.1f} "
                f"worst {candidate_worst:.0f}/{incumbent_worst:.0f})",
                flush=True,
            )
        if fixed_wins:
            best = fixed_wins[0][2]
            best_median = fixed_wins[0][3]
    if dist_handle.rank == 0:
        dispatch_row = best if phase == "dispatch" else incumbent_dispatch
        combine_row = best if phase == "combine" else incumbent_combine
        print(
            f"# TUNING RESULT tok={args.max_tokens} phase={phase}: "
            f"block/rdma/warp={best} median {phase}={best_median:.1f}us "
            f"(shipped was {shipped})\n"
            f"#   table row: ({args.max_tokens}, {dispatch_row[0]}, "
            f"{dispatch_row[1]}, {dispatch_row[2]}, "
            f"{combine_row[0]}, {combine_row[1]}, {combine_row[2]}),",
            flush=True,
        )
    return 0


def _spawn_entry(local_rank, argv, node_rank, num_nodes, ranks_per_node):
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
    os.environ["RANK"] = str(node_rank * ranks_per_node + local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(num_nodes * ranks_per_node)
    os.environ["LOCAL_WORLD_SIZE"] = str(ranks_per_node)
    return_code = main(argv)
    if return_code:
        raise SystemExit(return_code)


def main(argv):
    args = _parse_args(argv)

    # --spawn N reproduces the examples harness's process topology: one torchrun
    # process per node that spawns N workers, instead of N torchrun processes.
    # Same kernels, same bench, different process tree -- which is worth being
    # able to switch because host time on this path converts to measured "kernel"
    # time about 1:1, so how the ranks are parented is not obviously neutral.
    if args.spawn and not os.environ.get("_MORI_EP_SPAWN_CHILD"):
        # Both topologies at once would be nprocs x spawn ranks per node, each
        # claiming a GPU index it does not own. Refuse rather than deadlock in
        # the rendezvous, and say which of the two to drop.
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if local_world_size > 1:
            raise SystemExit(
                f"--spawn {args.spawn} with torchrun "
                f"--nproc_per_node={local_world_size}: that is "
                f"{local_world_size * args.spawn} ranks per node. Use "
                f"--nproc_per_node=1 (spawn builds the ranks), or pass "
                f"--spawn 0 to let torchrun do it."
            )
        node_rank = int(os.environ["RANK"])
        num_nodes = int(os.environ["WORLD_SIZE"])
        torch.multiprocessing.spawn(
            _spawn_entry,
            args=(argv, node_rank, num_nodes, args.spawn),
            nprocs=args.spawn,
            join=True,
        )
        return 0

    dist_handle = Dist()
    rank, npes = dist_handle.rank, dist_handle.world
    device = torch.device("cuda", dist_handle.local_rank)

    gpu_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if npes <= gpu_per_node:
        raise SystemExit(
            f"this test needs more than one node: world_size={npes} with "
            f"{gpu_per_node} GPUs per node is a single node"
        )

    dtype = _DTYPES[args.dtype]
    combine_dtype = _DTYPES[args.combine_dtype] if args.combine_dtype else None
    max_tokens = args.max_tokens

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = dist_handle.bcast_uid(uid)
    # internode_regions sizes the arena exactly; this is the VMM budget it is
    # carved out of, with room for the communicator's own resource window.
    win_bytes = npes * max_tokens * args.hidden_dim * 4 * 2 + (1 << 24)
    failures = 0
    with Communicator.init(
        npes, rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)
    ) as comm:
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=npes,
            hidden_dim=args.hidden_dim,
            max_num_inp_token_per_rank=max_tokens,
            num_experts_per_rank=(
                args.experts_per_rank if args.experts_per_rank else 256 // npes
            ),
            num_experts_per_token=args.topk,
            data_type=dtype if combine_dtype is None else torch.bfloat16,
            dispatch_data_type=dtype if combine_dtype is not None else None,
            combine_data_type=combine_dtype,
            scale_dim=args.scale_dim,
            scale_type_size=4 if args.scale_dim else 0,
            quant_type=args.quant_type,
            gpu_per_node=gpu_per_node,
            num_qp_per_pe=args.num_qp,
            internode_kernel=args.kernel_type,
            internode_auto_ll_max_tokens=args.auto_ll_max_tokens,
            kernel_backend="hip",
        )
        op = EpDispatchCombineOp(cfg, comm)
        comm.barrier()

        if args.cmd == "bench":
            return_code = _bench(op, cfg, dist_handle, device, args, comm)
            op.close()
            dist_handle.shutdown()
            return return_code

        if args.cmd == "stress":
            return_code = _stress(op, cfg, dist_handle, device, args, comm)
            op.close()
            dist_handle.shutdown()
            return return_code

        if args.cmd == "tuning":
            # The sweep builds its own ops, one per candidate geometry; this one
            # only proved the config is constructible.
            op.close()
            return_code = _tune(cfg, dist_handle, device, args, comm)
            dist_handle.shutdown()
            return return_code

        rng = torch.Generator(device=device)
        for round_index in range(args.rounds):
            rng.manual_seed(1234 + round_index * 977 + rank)
            num_tokens = max_tokens
            inp, idx, wts, sc = _generate_round(
                rng, cfg, num_tokens, device, cfg.dispatch_dtype, args.routing
            )

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
            combine_input = (
                recv_x.to(cfg.combine_dtype) if cfg.is_asymmetric_dtype else recv_x
            )
            out, out_w = op.combine(combine_input, wts, routing=routing)
            torch.cuda.synchronize()
            comm.barrier()

            idx_cpu = idx.cpu()
            unique_pes = np.array(
                [
                    len(
                        {
                            int(idx_cpu[t, j]) // cfg.num_experts_per_rank
                            for j in range(args.topk)
                        }
                    )
                    for t in range(num_tokens)
                ]
            )
            unique_pes_column = torch.from_numpy(unique_pes).view(num_tokens, 1).float()
            expected_weights = unique_pes_column * wts.float().cpu()
            expected = unique_pes_column * inp.float().cpu()

            # The bound has to scale with unique_pes: combine sums that many
            # contributions in the wire dtype, so the error grows with the number
            # of terms and with the magnitude being summed -- not with |expected|
            # at that element, which cancellation can make arbitrarily small. This
            # is the same shape of bound v1's harness uses (it scales its
            # per-element bound by unique_pes) rather than a flat allclose.
            got = out.float().cpu()
            input_magnitude = inp.float().cpu().abs()
            eps = 3e-1 if (args.quant_type != "none" or dtype in _FP8) else 8e-3
            bound = eps * unique_pes_column * input_magnitude.clamp(min=1.0)
            hidden_ok = bool(((got - expected).abs() <= bound).all())
            # Weights are transported as f32 and summed the same way, so their
            # bound is much tighter -- but still proportional to unique_pes.
            got_weights = out_w.cpu()
            weights_ok = bool(
                (
                    (got_weights - expected_weights).abs() <= 2e-3 * unique_pes_column
                ).all()
            )

            if not (hidden_ok and weights_ok) and rank == 0:
                print(f"#   hidden_ok={hidden_ok} weights_ok={weights_ok}", flush=True)
                ratio = (out_w.cpu() / wts.float().cpu().clamp(min=1e-6))[:4]
                print(
                    f"#   effective multiplier got_w/wts[0,:4]={ratio[0, :4].tolist()} "
                    f"(expected unique_pes[0]={int(unique_pes[0])})",
                    flush=True,
                )
                # The worst violator with everything needed to classify it:
                # a relative error near the wire dtype's half-ulp is rounding,
                # one far above it is not, and a `want` at the staging format's
                # saturation point says the partial sum clipped rather than
                # rounded.
                violation = (got - expected).abs() - bound
                if bool((violation > 0).any()):
                    flat_index = int(violation.argmax())
                    worst_token = flat_index // cfg.hidden_dim
                    worst_dim = flat_index % cfg.hidden_dim
                    got_value, want_value, input_value = (
                        float(got[worst_token, worst_dim]),
                        float(expected[worst_token, worst_dim]),
                        float(input_magnitude[worst_token, worst_dim]),
                    )
                    print(
                        f"#   worst: tok={worst_token} dim={worst_dim} "
                        f"got={got_value:.6g} want={want_value:.6g} "
                        f"input={input_value:.6g} "
                        f"unique_pes={int(unique_pes_column[worst_token])} "
                        f"|diff|={abs(got_value - want_value):.6g} "
                        f"bound={float(bound[worst_token, worst_dim]):.6g} "
                        f"rel={abs(got_value - want_value) / max(abs(want_value), 1e-9):.4f} "
                        f"num_violations={int((violation > 0).sum())}",
                        flush=True,
                    )
                want = expected
                print(
                    f"#   hidden: max|diff|={(got - want).abs().max():.4g} "
                    f"got[0,:4]={got[0, :4].tolist()} want[0,:4]={want[0, :4].tolist()} "
                    f"got_nonzero={int((got != 0).sum())}/{got.numel()}",
                    flush=True,
                )
                want_weights = expected_weights
                # Distinguish "the kernel never wrote it" from "the base reads
                # the wrong place": read the region the kernel targets directly.
                from mori.tensor_utils import from_gpu_ptr

                for region_name in (
                    "combine_out_weights",
                    "inp_weights",
                    "dispatch_out_weights",
                ):
                    region_values = from_gpu_ptr(
                        op.arena.local_ptr(region_name),
                        (min(8, args.topk * 2),),
                        torch.float32,
                    )
                    print(
                        f"#   region {region_name}[:8] = "
                        f"{region_values.cpu().tolist()}",
                        flush=True,
                    )
                print(
                    f"#   weights: "
                    f"max|diff|={(got_weights - want_weights).abs().max():.4g} "
                    f"got[0,:4]={got_weights[0, :4].tolist()} "
                    f"want[0,:4]={want_weights[0, :4].tolist()} "
                    f"unique_pes[:4]={unique_pes[:4].tolist()} "
                    f"total_recv={int(total_recv[0])}",
                    flush=True,
                )
            failed_ranks = dist_handle.allreduce_sum(
                0 if (hidden_ok and weights_ok) else 1
            )
            failures += failed_ranks
            if rank == 0:
                print(
                    f"# round {round_index} tokens={num_tokens}: "
                    f"{'PASS' if failed_ranks == 0 else 'FAIL'} "
                    f"({failed_ranks} of {npes} ranks disagree)",
                    flush=True,
                )
        op.close()

    dist_handle.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
