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

THE INTERNODE PATH NEEDS TWO NODES. ``gpu_per_node < world_size`` is what selects
it, and the op checks EP's node grouping against the communicator's LSA team; on
one host CCO reports the whole world as one team, so a config claiming two nodes
is rejected rather than silently running the intranode half. Lowering
gpu_per_node on a single host does not emulate this either: RAIL leaves
same-host peers without a QP. On one node (world_size == gpu_per_node) this
harness builds the intranode op instead; only ``--cmd test`` and
``--cmd tuning`` accept that, and bench and stress exit.

    # rank 0 of 2, 8 GPUs each
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \\
        --master_addr=<ip> --master_port=<port> \\
        test_dispatch_combine_v2_internode.py --max-tokens 128

``tools/run_internode_test.sh`` with ``--entry`` set to this file runs one node's
side (``--rank 0`` or ``1``; call it once per node). The CLI below keeps the
shmem harness's flag names where the two mean the same thing, and adds its own,
among them routing, process topology, active QPs, tuning, bench and stress.

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
import dataclasses
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.distributed as dist
from mori.cco import Communicator
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, _HERE)
from _data import decode_fp4, encode_fp4  # noqa: E402

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8_e4m3_fnuz": torch.float8_e4m3fnuz,
    "fp8_e4m3": torch.float8_e4m3fn,
}
_FP8 = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
# Dispatch only: the op reduces combine in bf16 or fp32. Absent from older torch.
_FP4 = getattr(torch, "float4_e2m1fn_x2", None)
if _FP4 is not None:
    _DTYPES["fp4"] = _FP4
_DTYPE_NAMES = {dtype: name for name, dtype in _DTYPES.items()}


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
    # Node index (-1 = none) whose ranks all send max_tokens in every dataset: a
    # full-capacity sender, the case where the LL receive can finish on chunk
    # flags alone.
    parser.add_argument("--stress-full-node", type=int, default=-1)
    # Check every Nth round's combine output in place (0 = never). No sync is
    # added between that round's dispatch and combine.
    parser.add_argument("--stress-check-interval", type=int, default=0)
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
        "--routing",
        default="uniform",
        choices=["uniform", "skewed", "local", "replica"],
    )
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    # Unset: the dispatch dtype, or bf16 under an fp4 dispatch.
    parser.add_argument(
        "--combine-dtype",
        default=None,
        choices=[name for name in _DTYPES if name != "fp4"],
    )
    parser.add_argument(
        "--quant-type", default="none", choices=["none", "fp8_direct_cast"]
    )
    # QPs ALLOCATED per peer (num_qp_per_pe). How many a kernel sends on is a
    # separate choice: --active-qps, else the tuning table, else min(2, this).
    parser.add_argument("--num-qp", type=int, default=1)
    # Pin the active QP count for every bucket and both phases (cfg.active_qps).
    parser.add_argument("--active-qps", type=int, default=None)
    # Extra counts compiled for op.set_active_qps(), comma-separated. Under
    # --cmd test the rounds cycle through them, then through the table's own
    # choice, so a switch between rounds is checked against the golden.
    parser.add_argument("--active-qp-counts", default=None)
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
        "--tuning-phase", default="dispatch", choices=["dispatch", "combine", "pair"]
    )
    # Validation mode: sweep exactly one named candidate against the shipped
    # geometry. A sweep picks its winner from many short paired tests, each with
    # its own error; before it is written into the table it gets one long
    # head-to-head against what it would replace. block,rdma,warp[,active_qps];
    # under --tuning-phase pair, dispatch,combine QP counts per pair, separated by
    # '/' (or ';', which a remote shell would split on).
    parser.add_argument("--tuning-candidate", default=None)
    # Active QP counts to cross with every geometry, e.g. 1,2,4,8. Each must be
    # within --num-qp, the allocation. Unset keeps the shipped count.
    parser.add_argument("--tuning-qps", default=None)
    # "shipped" keeps each phase's table geometry and varies only the active QP
    # count (--tuning-qps). --tuning-phase pair always does: it crosses the
    # dispatch and combine counts, because the op drains the larger of the two,
    # so neither phase's best count is independent of the other's.
    parser.add_argument(
        "--tuning-geometry", default="sweep", choices=["sweep", "shipped"]
    )
    # Dispatch geometry can affect the following combine, so select on the
    # pair total by default. "phase" is available for individual-op diagnostics.
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
    # Saving is opt-in. Confirm sweep winners in independent paired runs before
    # adopting them; the sweep alone does not establish a stable improvement.
    parser.add_argument("--tuning-save", action="store_true")
    # Where --tuning-save writes. Default: this repo's copy of the table, not the
    # installed package's -- a row is only useful once it is in a git diff
    # somebody can read (same convention as the v1 bench's `auto`).
    parser.add_argument("--tuning-config-dir", default=None)
    # The v1 bench calls combine with weights=None, so it does not pay for the
    # weight fold: an extra peer read per (token, destination) plus an accumulate
    # in three kernels, and a wider staging slot (combXferBytes = hidden + weights).
    # Off by default so a reading here is comparable to one from that harness;
    # --bench-weights measures the fold when that is what you want. The
    # correctness path always folds -- it is checking the weights.
    parser.add_argument("--bench-weights", action="store_true")
    # Hand combine the op's combine_in_view(), filled with the expert output (in
    # bench, inside the untimed conversion window), as a zero-copy caller does.
    parser.add_argument("--combine-zero-copy", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--drop-rounds", type=int, default=1)
    parser.add_argument("--no-bench-tables", action="store_true")
    # v2 and v2_ll are separate kernels. Naming one compiles only that one and
    # runs it at every token count, which is what makes a benchmark number
    # comparable to another harness's. "auto" compiles both and picks per launch
    # at --auto-ll-max-tokens.
    # "asyncll" is a different family rather than a third internode kernel: it
    # sets async_ll on the config instead of internode_kernel, and it addresses
    # peers individually where v2/v2_ll aggregate through a same-rail proxy. That
    # makes it the one choice here that needs a cross-rail-capable fabric.
    parser.add_argument(
        "--kernel-type", default="auto", choices=["auto", "v2", "v2_ll", "asyncll"]
    )
    parser.add_argument("--auto-ll-max-tokens", type=int, default=512)
    # Dispatch also builds aiter's moe_sorting tables with this block_size_M; --cmd
    # test checks them against a torch reference and checks the expert-output zeroing.
    parser.add_argument("--moe-sort-block-m", type=int, default=0)
    # Rank 0 saves round 0's received rows and sort tables here, to check against aiter.
    parser.add_argument("--moe-sort-dump", default=None)
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
    ``replica``  one shared expert fused in as a routed one, the way ATOM lays it
                 out (SharedExpertMode.LOCAL_REPLICA): every rank's last expert
                 slot is a replica of the shared expert and each token's last
                 column is its own rank's replica; the other topk - 1 columns are
                 distinct routed experts drawn uniformly over every rank's other
                 slots. DeepSeek-V3/R1 at EP16 is 17 experts a rank, topk 9.
    """
    inp = torch.randn(num_tokens, cfg.hidden_dim, generator=rng, device=device)
    # fp4 takes the bf16 draw rounded to e2m1, so its routing is the bf16 run's.
    inp = encode_fp4(inp.to(torch.bfloat16)) if dtype == _FP4 else inp.to(dtype)
    num_experts = cfg.world_size * cfg.num_experts_per_rank
    if routing == "replica":
        routed_per_rank = cfg.num_experts_per_rank - 1
        routed = torch.stack(
            [
                torch.randperm(
                    cfg.world_size * routed_per_rank, generator=rng, device=device
                )[: cfg.num_experts_per_token - 1]
                for _ in range(num_tokens)
            ]
        )
        # Routed expert p of rank p // routed_per_rank sits in the slot after the
        # replicas of the ranks before it.
        routed = routed + routed // routed_per_rank
        shared = torch.full(
            (num_tokens, 1),
            cfg.rank * cfg.num_experts_per_rank + routed_per_rank,
            dtype=routed.dtype,
            device=device,
        )
        idx = torch.cat((routed, shared), dim=1).to(torch.int32)
    elif routing == "uniform":
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


def _payload_f32(cfg, tensor):
    """A dispatch payload's values as f32: fp4 decoded, every other dtype cast."""
    return decode_fp4(tensor, torch.float32) if cfg.is_fp4 else tensor.float()


def _combine_input(cfg, dispatched):
    """What the identity expert hands combine: the dispatched tokens in the combine
    dtype. The combine kernel reads its input as that type, so converting is the
    caller's job -- an fp8 dispatch is cast, an fp4 one decoded."""
    if cfg.is_fp4:
        return decode_fp4(dispatched, cfg.combine_dtype)
    return dispatched.to(cfg.combine_dtype) if cfg.is_asymmetric_dtype else dispatched


class _Golden(NamedTuple):
    unique_pes: torch.Tensor  # (tokens, 1): distinct destination ranks per token
    hidden: torch.Tensor
    hidden_bound: torch.Tensor
    weights: torch.Tensor | None
    weights_bound: torch.Tensor | None


def _golden(cfg, args, inp, idx, wts=None):
    """What an identity expert's combine must return, with its error bounds.

    combine[t] sums one copy of input[t] per DISTINCT destination rank, so the
    golden is unique_pes[t] * input[t]. The bound has to scale with unique_pes:
    the sum has that many terms in the wire dtype, so its error grows with their
    number and magnitude -- not with |golden| at that element, which cancellation
    can make arbitrarily small. v1's harness bounds the same way. Weights are
    summed in f32, so theirs is much tighter; both are None without ``wts``.
    """
    pes = idx.long() // cfg.num_experts_per_rank
    unique = (
        torch.stack(
            [(pes[:, j : j + 1] != pes[:, :j]).all(dim=1) for j in range(pes.shape[1])],
            dim=1,
        )
        .sum(dim=1, keepdim=True)
        .float()
    )
    eps = 3e-1 if (args.quant_type != "none" or cfg.dispatch_dtype in _FP8) else 8e-3
    source = _payload_f32(cfg, inp)
    return _Golden(
        unique,
        unique * source,
        eps * unique * source.abs().clamp(min=1.0),
        None if wts is None else unique * wts.float(),
        None if wts is None else 2e-3 * unique,
    )


def _check_moe_sort(op, cfg, recv_i, recv_w, total_recv):
    """op.moe_sort() against aiter's moe_sorting contract, built here in torch from
    the rows dispatch delivered. Order within an expert is not part of it (the
    rows are placed by atomics), so each expert's entries compare as a set."""
    bm, epr, topk = (
        cfg.moe_sort_block_m,
        cfg.num_experts_per_rank,
        cfg.num_experts_per_token,
    )
    ids, weights, expert_ids, num_valid, _ = op.moe_sort()
    recv = int(total_recv.item())
    idx = recv_i[:recv].long()
    mine = (idx >= 0) & (idx // epr == cfg.rank)
    rows, slots = mine.nonzero(as_tuple=True)
    local = idx[rows, slots] % epr
    want_ids = rows | (slots << 24)
    want_w = recv_w[:recv][rows, slots].float()
    counts = torch.bincount(local, minlength=epr)
    padded = (counts + bm - 1) // bm * bm
    off = torch.cumsum(padded, 0) - padded
    total = int(padded.sum())
    problems = []
    if num_valid.tolist() != [total, recv]:
        problems.append(f"num_valid_ids={num_valid.tolist()} want {[total, recv]}")
    else:
        want_eids = torch.repeat_interleave(
            torch.arange(epr, device=idx.device), padded // bm
        )
        if not torch.equal(expert_ids[: total // bm].long(), want_eids):
            problems.append("sorted_expert_ids differ")
        # Per expert: its count of real rows (any order), then the padding.
        pos = torch.arange(total, device=idx.device)
        expert_of = torch.repeat_interleave(
            torch.arange(epr, device=idx.device), padded
        )
        real = (pos - off[expert_of]) < counts[expert_of]
        got_ids, got_w = ids[:total].long(), weights[:total].float()
        pad_id = recv | (topk << 24)
        if not (
            bool((got_ids[~real] == pad_id).all()) and bool((got_w[~real] == 0).all())
        ):
            problems.append("padding rows differ")
        key_got = expert_of[real] * (1 << 40) + got_ids[real]
        key_want = local * (1 << 40) + want_ids
        order_got, order_want = key_got.argsort(), key_want.argsort()
        if not torch.equal(key_got[order_got], key_want[order_want]):
            problems.append(
                f"sorted ids differ ({int(real.sum())} vs {key_want.numel()} rows)"
            )
        elif not torch.equal(got_w[real][order_got], want_w[order_want]):
            problems.append("sorted weights differ")
    if problems:
        print(f"#   rank {cfg.rank} moe_sort: {'; '.join(problems)}", flush=True)
    return not problems


def _dispatch(op, inp, wts, sc, idx, peak=None):
    return op.dispatch(
        inp,
        wts,
        sc,
        idx,
        return_routing=True,
        max_num_inp_token_across_ranks=inp.shape[0] if peak is None else peak,
    )


def _verify_once(op, cfg, dist_handle, args, comm, inp, idx, wts, sc):
    """One dispatch+combine against the analytic golden. True if this rank agrees.

    Same expectation as `--cmd test`: an identity expert makes combine[t] equal
    unique_pes[t] * input[t] over the DISTINCT destination ranks. Weights are
    always folded here even when the timed loop will not fold them -- an
    unchecked weight path is how this harness shipped a silent bug twice.
    """
    dispatch_out = _dispatch(op, inp, wts, sc, idx)
    torch.cuda.synchronize()
    comm.barrier()
    combine_input = _combine_input(cfg, dispatch_out[0])
    # AsyncLL is the exception to "always fold here": its combine payload is
    # hidden-only, so the request raises rather than returning a tensor. The
    # hidden check below still runs, which is the half that can be wrong.
    folds_weights = not getattr(cfg, "async_ll", False)
    out, out_w = op.combine(
        combine_input, wts if folds_weights else None, routing=dispatch_out[5]
    )
    torch.cuda.synchronize()
    comm.barrier()

    golden = _golden(cfg, args, inp, idx, wts)
    hidden_ok = bool(((out.float() - golden.hidden).abs() <= golden.hidden_bound).all())
    weights_ok = out_w is None or bool(
        ((out_w.float() - golden.weights).abs() <= golden.weights_bound).all()
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


def _geometry_for_report(op, num_tokens):
    """Report the same HIP configuration used to select the launch callable:
    (block, rdma, warp, active_qps) per phase, after any set_active_qps()."""
    selected = op.kernel_config(num_tokens, num_tokens)
    return tuple(
        (spec.block_num, spec.rdma_block_num, spec.warp_per_block, spec.active_qps)
        for spec in (selected.dispatch, selected.combine)
    )


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
    # Transport widths: packed FP4 moves hidden_dim / 2 bytes per token.
    dispatch_token_bytes = cfg.token_nbytes
    combine_token_bytes = cfg.combine_token_nbytes
    dispatch_bytes = total_recv * dispatch_token_bytes
    combine_bytes = total_recv * combine_token_bytes
    rdma_tokens = _rdma_algo_token_count(idx, cfg, ll)
    dispatch_rdma_bytes = rdma_tokens * dispatch_token_bytes
    combine_rdma_bytes = rdma_tokens * combine_token_bytes
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
        f"qp_allocated={cfg.num_qp_per_pe}",
        flush=True,
    )
    print(
        f"# CONFIG dispatch block/rdma/warp/active_qps={geometry[0]}  "
        f"combine={geometry[1]}  "
        f"rounds={args.rounds} warmup={args.warmup}  "
        f"recv_tokens={total_recv} rdma_algo_tokens={rdma_tokens}",
        flush=True,
    )

    for (
        name,
        (rdma_column, xgmi_column, latency_column),
        phase_dtype,
        phase_geometry,
        token_bytes,
    ) in (
        ("Dispatch", (0, 1, 2), cfg.dispatch_dtype, geometry[0], dispatch_token_bytes),
        ("Combine", (3, 4, 5), cfg.combine_dtype, geometry[1], combine_token_bytes),
    ):
        xgmi_stats = _phase_stats(gathered[:, :, xgmi_column])
        _print_phase_table(
            f"{name} Performance ({str(phase_dtype).split('.')[-1]}) "
            f"block={phase_geometry[0]} warp={phase_geometry[2]} "
            f"rdma={phase_geometry[1]} qps={phase_geometry[3]} "
            f"~{num_tokens * token_bytes / (1024 ** 2):.1f} MB/rank",
            _phase_stats(gathered[:, :, rdma_column]),
            xgmi_stats,
            tuple(stat * ll_scale for stat in xgmi_stats),
            _phase_stats(gathered[:, :, latency_column]),
        )


def _family_that_ran(op, num_tokens, max_tokens):
    """Which kernel family served this call, as kernel_config() resolves it.

    AsyncLL is reported with its transport: sdma and lsa are different kernels
    behind the one schedule entry.
    """
    if getattr(op.cfg, "async_ll", False):
        return f"asyncll({op.cfg.async_ll_transport})"
    return op.kernel_config(num_tokens, max_tokens).dispatch.family


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

    def convert(tensor):
        if args.combine_zero_copy:
            zero_copy = op.combine_in_view()
            zero_copy[: tensor.shape[0]].copy_(_combine_input(cfg, tensor))
            return zero_copy
        return _combine_input(cfg, tensor)

    combine_weights = wts if args.bench_weights else None

    # Allocated before the warmup, where run_bench_once allocates them. (Priming
    # them with a record here was tried and did not move the tail, so it is not
    # done -- run_bench_once does not either.)
    num_rounds = args.rounds
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3 * num_rounds + 1)]

    total_recv = 0
    # Warm with the SAME tensor lifetime the timed loop uses. Passing the
    # converted tensor inline makes it a temporary that dies at the end of the
    # statement, so warmup only ever keeps one alive, while the timed loop keeps
    # two: it binds the result to a name, and Python evaluates the right-hand
    # side before rebinding, so the previous round's tensor is still referenced.
    # Warming one block deep leaves the caching allocator holding one where two
    # are needed, and the second timed round pays a fresh hipMalloc -- ~100us of
    # HOST time. The timing events tile the timed region, so that host stall is
    # charged to the combine window as GPU idle and reads as a first round twice
    # as slow as every round after it.
    warmup_combine_input = None
    for i in range(args.warmup):
        dispatch_out = _dispatch(op, inp, wts, sc, idx)
        if i == args.warmup - 1:
            # Read it here, not in the timed loop: .item() synchronises.
            torch.cuda.synchronize()
            total_recv = int(dispatch_out[4][0].item())
        warmup_combine_input = convert(dispatch_out[0])
        op.combine(warmup_combine_input, combine_weights, routing=dispatch_out[5])
    # Release before the timed loop: left bound, it keeps a THIRD tensor alive
    # and just moves the allocation into the measured region.
    del warmup_combine_input
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
        dispatch_out = _dispatch(op, inp, wts, sc, idx)
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
    kernel_ran = _family_that_ran(op, num_tokens, num_tokens)
    if dist_handle.rank == 0:
        print(
            f"# BENCH tok={num_tokens} "
            f"dtype={_DTYPE_NAMES.get(cfg.dispatch_dtype)}->"
            f"{_DTYPE_NAMES.get(cfg.combine_dtype)} "
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
        ll = (
            op.kernel_config(args.max_tokens, args.max_tokens).dispatch.family
            == "v2_ll"
        )
        geometry = _geometry_for_report(op, args.max_tokens)
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
    rounds = _timed_rounds(
        op, inp, idx, wts, sc, combine_weights, convert, num_rounds, num_warmup
    )[args.drop_rounds :]
    dispatch_us = [dispatch for dispatch, _ in rounds]
    combine_us = [combine for _, combine in rounds]

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
    """An op whose dispatch plans run `dispatch_geometry` and its combine plans
    `combine_geometry`. Goes through the MORI_EP_*_GEOM hook the backend already
    exposes for sweeps: it is read once, when the op resolves its schedule, so
    each candidate needs an op of its own and cannot be selected per launch the
    way v1's can.

    The two are SEPARATE because the shipped tables give them separate values
    (MI308X InterNodeV2, 256/8 hidden 7168 bf16 at 4 tokens: dispatch 64/42/4,
    combine 16/10/4). Driving both from one geometry means the sweep's incumbent
    is not the configuration actually shipped, so "beats the incumbent" would not
    mean "beats what we ship".

    The intranode backend has no such hook; there the geometry is pinned through
    the config fields, which also drops the tuned schedule.
    """
    # The backend logs every op's full schedule at INFO; a sweep builds one op
    # per candidate, and its own report already names each geometry.
    backend_log = logging.getLogger("mori.ops.dispatch_combine_v2.hip_backend")
    backend_level = backend_log.level
    backend_log.setLevel(logging.WARNING)
    try:
        if not cfg.is_internode:
            pinned = dataclasses.replace(
                cfg,
                schedule=None,
                dispatch_block_num=dispatch_geometry[0],
                warp_num_per_block=dispatch_geometry[2],
                combine_block_num=combine_geometry[0],
                combine_warp_num_per_block=combine_geometry[2],
            )
            return EpDispatchCombineOp(pinned, comm)
        previous = (
            os.environ.get("MORI_EP_DISP_GEOM"),
            os.environ.get("MORI_EP_COMB_GEOM"),
        )
        os.environ["MORI_EP_DISP_GEOM"] = ",".join(map(str, dispatch_geometry))
        os.environ["MORI_EP_COMB_GEOM"] = ",".join(map(str, combine_geometry))
        try:
            return EpDispatchCombineOp(cfg, comm)
        finally:
            for name, value in zip(
                ("MORI_EP_DISP_GEOM", "MORI_EP_COMB_GEOM"), previous
            ):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    finally:
        backend_log.setLevel(backend_level)


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _timed_rounds(
    op,
    inp,
    idx,
    wts,
    sc,
    combine_weights,
    convert,
    num_rounds,
    num_warmup,
    *,
    barrier=False,
):
    """Warmup, then num_rounds timed dispatch+combine pairs on this rank:
    (dispatch_us, combine_us) per round. ``barrier`` lines the ranks up before,
    not inside, the timed loop, as v1 does for the metrics it reports."""
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3 * num_rounds + 1)]
    # Same tensor lifetime as the timed loop; see the warmup in _bench for why
    # the inline temporary costs the second timed round a hipMalloc. It matters
    # more for the tuner, which breaks ties on the worst single round against a
    # default --tuning-margin-us of 1.5.
    warmup_combine_input = None
    for _ in range(num_warmup):
        dispatch_out = _dispatch(op, inp, wts, sc, idx)
        warmup_combine_input = convert(dispatch_out[0])
        op.combine(warmup_combine_input, combine_weights, routing=dispatch_out[5])
    del warmup_combine_input
    torch.cuda.synchronize()
    if barrier:
        dist.barrier()
    events[0].record()
    for i in range(num_rounds):
        dispatch_out = _dispatch(op, inp, wts, sc, idx)
        events[3 * i + 1].record()
        combine_input = convert(dispatch_out[0])
        events[3 * i + 2].record()
        op.combine(combine_input, combine_weights, routing=dispatch_out[5])
        events[3 * i + 3].record()
    torch.cuda.synchronize()
    return [
        (
            events[3 * i].elapsed_time(events[3 * i + 1]) * 1e3,
            events[3 * i + 2].elapsed_time(events[3 * i + 3]) * 1e3,
        )
        for i in range(num_rounds)
    ]


def _summarize_metric_samples(
    gathered, *, num_passes, num_rounds, drop_rounds, max_tokens, topk, ll
):
    """Rounded tuning metrics from the CPU all-gather rows.

    Each rank contributes four metadata values followed by interleaved
    dispatch/combine microseconds in pass-major, round-major order. Metadata is
    received tokens, algorithm RDMA tokens, dispatch bytes/token and combine
    bytes/token. Bandwidth is averaged after dividing each rank's own bytes by
    each retained latency. The LL scale is rank 0's, applied after that average.
    """
    rows = [list(row) for row in gathered]
    sample_count = len(rows) * num_passes * (num_rounds - drop_rounds)
    ll_scale_rank0 = max_tokens * topk / (rows[0][0] + 1)
    metrics = {}
    for phase_index, phase in enumerate(("dispatch", "combine")):
        latencies, rdma_bandwidths, xgmi_bandwidths = [], [], []
        for row in rows:
            recv_tokens, rdma_tokens = row[:2]
            width = row[2 + phase_index]
            for pass_index in range(num_passes):
                for round_index in range(drop_rounds, num_rounds):
                    offset = 4 + 2 * (pass_index * num_rounds + round_index)
                    latency = row[offset + phase_index]
                    latencies.append(latency)
                    rdma_bandwidths.append(rdma_tokens * width / (1000 * latency))
                    xgmi_bandwidths.append(recv_tokens * width / (1000 * latency))
        latency = math.fsum(latencies) / sample_count
        rdma = math.fsum(rdma_bandwidths) / sample_count
        xgmi = math.fsum(xgmi_bandwidths) / sample_count
        ll_bandwidth = xgmi * ll_scale_rank0
        values = {
            "bandwidth_gbps": ll_bandwidth if ll else rdma,
            "avg_rdma_bandwidth_gbps": rdma,
            "avg_xgmi_bandwidth_gbps": xgmi,
            "avg_ll_bandwidth_gbps": ll_bandwidth,
            "avg_latency_us": latency,
        }
        metrics[phase] = {
            **{name: round(value, 2) for name, value in values.items()},
            "bandwidth_metric": "grand_mean",
        }
    return metrics


def _row_metrics(op, cfg, args, dist_handle, rng_data, convert, ll):
    """Final metrics with v1's grand mean, separate from search estimators.

    Each rank computes bandwidth with its own payload and each retained round's
    latency, then all samples across passes, ranks, and rounds are averaged.
    The headline is RDMA for v2 and LL for v2_ll. LL uses the global XGMI mean
    times rank 0's scale, exactly as v1's saved Average row does.

    One CPU all-gather follows all synchronized timing passes.
    """
    if (
        args.tuning_reps < 1
        or args.rounds < 1
        or not 0 <= args.drop_rounds < args.rounds
    ):
        raise ValueError("Metric sampling requires passes and retained rounds")
    if args.warmup < 0:
        raise ValueError("Metric warmup must be nonnegative")
    inp, idx, wts, sc, combine_weights = rng_data
    dispatch_out = _dispatch(op, inp, wts, sc, idx)
    torch.cuda.synchronize()
    total_recv = int(dispatch_out[4].item())
    combine_input = convert(dispatch_out[0])
    # Actual storage widths handle packed FP4 dispatch and unpacked BF16 combine.
    dispatch_bytes = inp.shape[-1] * inp.element_size()
    combine_bytes = combine_input.shape[-1] * combine_input.element_size()
    op.combine(combine_input, combine_weights, routing=dispatch_out[5])
    torch.cuda.synchronize()
    del combine_input, dispatch_out
    rdma_tokens = _rdma_algo_token_count(idx, cfg, ll)
    row = [total_recv, rdma_tokens, dispatch_bytes, combine_bytes]
    for _ in range(args.tuning_reps):
        samples = _timed_rounds(
            op,
            inp,
            idx,
            wts,
            sc,
            combine_weights,
            convert,
            args.rounds,
            args.warmup,
            barrier=True,
        )
        row.extend(value for sample in samples for value in sample)
    gathered = dist_handle.all_gather_rows(row).tolist()
    return _summarize_metric_samples(
        gathered,
        num_passes=args.tuning_reps,
        num_rounds=args.rounds,
        drop_rounds=args.drop_rounds,
        max_tokens=args.max_tokens,
        topk=cfg.num_experts_per_token,
        ll=ll,
    )


def _active_qp_schedule(cfg):
    """The counts --active-qp-counts switches among, then None (back to the
    table), or None when the flag is not given."""
    return list(cfg.active_qp_counts) + [None] if cfg.active_qp_counts else None


def _active_qp_for(schedule, round_index, cfg, phase_offset):
    """The count this rank selects for one phase of one round.

    Deliberately NOT in step. The combine barrier only carries markers between a
    rank and its same-local-rank peer on each other node, so the offset is the
    NODE index: with two nodes, those peers take adjacent schedule entries, and
    phase_offset 1 puts combine one entry ahead of its own dispatch. Those are
    the two cases the combine barrier has to tolerate. They actually differ only
    where adjacent entries resolve to different counts: the trailing None means
    each bucket's own count, which can equal a neighbour (e.g. 2,4 with a
    default of 2). 1,2,4,8 at the default of 2 differs at every step.
    """
    node = cfg.rank // cfg.gpu_per_node
    return schedule[(round_index + node + phase_offset) % len(schedule)]


def _stress(op, cfg, dist_handle, device, args, comm):
    """v1's stress case: cycle pre-generated rounds with VARYING token counts.

    Deliberately different from --cmd bench, which sends max_tokens every round
    from one buffer. Here each rank draws a fresh count in [1, max_tokens] per
    dataset, so the soak covers what a fixed-size bench cannot: ragged and
    unequal loads, the chunk protocol's empty-chunk path, and -- under
    internode_kernel="auto" -- the per-call switch between the v2 and v2_ll
    families, since every eighth dataset keeps all ranks at or below the crossover.

    By default no per-round verification: this looks for hangs, faults and
    protocol drift over many rounds, and checking every round would hide them
    behind the check's own synchronisation. The queue is drained every
    --stress-sync-interval rounds rather than every round for the same reason.
    --stress-check-interval N compares every Nth round's combine output with the
    identity-expert expectation after the round, never between its dispatch and
    combine, so a combine that starts before dispatch state has landed is caught
    rather than masked. --stress-full-node makes one node a full-capacity sender.

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
    ll = min(cfg.internode_auto_ll_max_tokens, max_tokens)
    if cfg.internode_kernel == "auto" and not cfg.async_ll and ll > 0:
        counts = [(n - 1) % ll + 1 if j % 8 == 0 else n for j, n in enumerate(counts)]
    if dist_handle.rank // cfg.gpu_per_node == args.stress_full_node:
        counts = [max_tokens] * len(counts)
    peaks = dist_handle.all_gather_rows(counts).max(dim=0).values.long().tolist()
    datasets = [
        _generate_round(rng, cfg, int(n), device, cfg.dispatch_dtype, args.routing)
        for n in counts
    ]
    expected = []
    if args.stress_check_interval:
        # combine(weights=None), so only the hidden half is checked.
        for inp, idx, _, _ in datasets:
            golden = _golden(cfg, args, inp, idx)
            expected.append((golden.hidden, golden.hidden_bound))
    failed_checks = checks = 0
    if dist_handle.rank == 0:
        families = sorted({_family_that_ran(op, n, p) for n, p in zip(counts, peaks)})
        print(
            f"# STRESS rounds={args.rounds} datasets={args.stress_datasets} "
            f"tokens=[{min(counts)},{max(counts)}] of {max_tokens} "
            f"routing={args.routing} kernel={args.kernel_type} "
            f"families exercised: {'+'.join(families)}",
            flush=True,
        )

    # Active QP switching as in --cmd test, on top of the ragged per-rank token
    # counts; see _active_qp_for.
    qp_schedule = _active_qp_schedule(cfg)
    dist.barrier()
    started = time.time()
    for i in range(args.rounds):
        inp, idx, wts, sc = datasets[i % len(datasets)]
        if qp_schedule:
            op.set_active_qps(_active_qp_for(qp_schedule, i, cfg, 0))
        dispatch_out = _dispatch(op, inp, wts, sc, idx, peaks[i % len(datasets)])
        combine_input = _combine_input(cfg, dispatch_out[0])
        if qp_schedule:
            op.set_active_qps(_active_qp_for(qp_schedule, i, cfg, 1))
        # None, not wts: v1's soak combines without weights
        # (run_combine(op, combine_input, None, indices)), and since want_weights
        # is now honoured, passing them would make this a different kernel path
        # from the case it is meant to mirror.
        out, _ = op.combine(combine_input, None, routing=dispatch_out[5])
        if args.stress_check_interval and i % args.stress_check_interval == 0:
            golden, tolerance = expected[i % len(datasets)]
            checks += 1
            failed_checks += int(
                not bool(((out.float() - golden).abs() <= tolerance).all())
            )
        if i % args.stress_sync_interval == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    comm.barrier()

    failed_ranks = dist_handle.allreduce_sum(int(failed_checks > 0))
    if dist_handle.rank == 0:
        verdict = "OK" if failed_ranks == 0 else "FAILED"
        print(
            f"# STRESS {verdict}: {args.rounds} rounds in {time.time() - started:.1f}s"
            + (
                f"; {checks} checked rounds per rank, {failed_ranks} rank(s) with "
                "mismatches"
                if args.stress_check_interval
                else ""
            ),
            flush=True,
        )
    return 0 if failed_ranks == 0 else 1


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

    if cfg.is_internode:
        geometry_candidates = [
            (block, rdma, warp)
            for block in sorted(blocks)
            for warp in warps
            for rdma in rdma_block_counts(block)
        ]
    else:
        # Intranode has no RDMA split; the middle field stays 0.
        geometry_candidates = [
            (block, 0, warp) for block in sorted(blocks) for warp in warps
        ]

    from mori.ops.dispatch_combine_v2.hip_tuning_configs import (
        internode_kernel_family,
        lookup_internode,
        pick_bucket,
        resolve_active_qps,
    )
    from mori.ops.tuning_config import DTYPE_TO_CONFIG_STR

    # Each phase's rules are keyed by ITS OWN leg's dtype, as v1 keys its two
    # files: an fp8-dispatch + bf16-combine op writes fp8_e4m3 dispatch rules and
    # bf16 combine rules. So the two strings can differ, and the saved row has to
    # use the one for the phase being swept.
    dtype_keys = {
        "dispatch": DTYPE_TO_CONFIG_STR.get(cfg.dispatch_dtype),
        "combine": DTYPE_TO_CONFIG_STR.get(cfg.combine_dtype),
    }
    kernel_family = (
        internode_kernel_family(cfg, max_tokens=args.max_tokens)
        if cfg.is_internode
        else "intra"
    )
    if args.tuning_save or args.tuning_config_dir:
        # Read the incumbent from the SAME table the winner will be written to.
        # The kernels run from site-packages while --tuning-save writes the
        # repo's copy, so without this a sweep would be judged against a row it
        # is not about to replace -- and the second phase of a shape would not
        # see the first phase's result at all.
        os.environ["MORI_EP_V2_TUNING_DIR"] = str(_tuning_table_dir(args))
    if cfg.is_internode:
        table_row = lookup_internode(
            cfg.world_size,
            cfg.hidden_dim,
            cfg.num_experts_per_token,
            args.max_tokens,
            cfg.dispatch_dtype,
            cfg.combine_dtype,
            kernel_family=kernel_family,
            experts_per_rank=cfg.num_experts_per_rank,
            cu_count=num_cus,
        )
    else:
        # Re-resolve the schedule: the caller's config was built before
        # MORI_EP_V2_TUNING_DIR pointed at the table this run writes.
        current = dataclasses.replace(
            cfg,
            schedule=None,
            dispatch_block_num=None,
            warp_num_per_block=None,
            combine_block_num=None,
            combine_warp_num_per_block=None,
        )
        row = (
            pick_bucket(current.schedule, args.max_tokens)
            if current.schedule
            else (
                None,
                current.dispatch_block_num,
                current.warp_num_per_block,
                current.combine_block_num,
                current.combine_warp_num_per_block,
            )
        )
        table_row = {
            "dispatch": (row[1], 0, row[2], None),
            "combine": (row[3], 0, row[4], None),
        }

    def incumbent(geometry):
        # The count the op itself would run for this geometry, by the backend's
        # own rules: a configured --active-qps wins, an open count takes the
        # default, and a table count is clamped to the allocation. Named
        # explicitly, so the incumbent and the candidates compare like with like.
        # Intranode sends on no QP, so its count stays None.
        block, rdma, warp = geometry[:3]
        if not cfg.is_internode:
            return (block, rdma, warp, None)
        active_qps = geometry[3] if len(geometry) > 3 else None
        return (block, rdma, warp, resolve_active_qps(cfg, active_qps))

    def checked(active_qps):
        # A count the sweep was ASKED for is rejected, not clamped: a clamped
        # candidate would report a winner it never ran. Same rule as the env pin.
        if not cfg.is_internode or not 1 <= active_qps <= cfg.num_qp_per_pe:
            raise SystemExit(
                f"active QP count {active_qps} is outside the allocation "
                f"[1, --num-qp={cfg.num_qp_per_pe}] or not internode"
            )
        return active_qps

    # The incumbent is the shipped row where there is one (clamped to this GPU's
    # CU count, as the op clamps it), and otherwise the config's fallback
    # geometry as given. That fallback is not clamped here, so on a part with
    # fewer CUs than its block count (MI308X: 80 against 96) the incumbent runs a
    # wider grid than the op a user builds would. Not candidates[0] -- that is
    # just the smallest grid in the sweep, so on a shape with no row yet the
    # whole sweep would be scored against a geometry nobody runs, and "beats the
    # incumbent" would not mean "beats what we ship".
    #
    # Per phase, because a phase may have no rule: the tuner sweeps one phase
    # against the other's current value, so a bucket grows a phase at a time.
    def _incumbent(phase_name, config_geometry):
        tuned = table_row.get(phase_name) if table_row else None
        return incumbent(tuple(tuned) if tuned else config_geometry)

    incumbent_dispatch = _incumbent(
        "dispatch",
        (cfg.dispatch_block_num, cfg.dispatch_rdma_block_num, cfg.warp_num_per_block),
    )
    incumbent_combine = _incumbent(
        "combine",
        (
            cfg.combine_block_num,
            cfg.combine_rdma_block_num,
            cfg.combine_warp_num_per_block,
        ),
    )
    phase = args.tuning_phase
    pair = phase == "pair"
    if pair:
        # A candidate is a (dispatch, combine) geometry pair.
        shipped = (incumbent_dispatch, incumbent_combine)
    else:
        shipped = incumbent_dispatch if phase == "dispatch" else incumbent_combine
    if pair and (
        not cfg.is_internode or not (args.tuning_qps or args.tuning_candidate)
    ):
        raise SystemExit(
            "--tuning-phase pair is internode only and needs --tuning-qps or "
            "--tuning-candidate"
        )
    if args.tuning_geometry == "shipped" and not pair:
        geometry_candidates = [shipped[:3]]

    # --tuning-qps crosses every geometry with every active count. Without it the
    # sweep keeps the shipped count and varies the geometry alone, as before.
    active_qp_candidates = (
        [checked(int(count)) for count in args.tuning_qps.split(",")]
        if args.tuning_qps
        else ([] if pair else [shipped[3]])
    )
    if pair:
        # Table geometry for both phases; every (dispatch, combine) count pair,
        # or just the pairs --tuning-candidate names ("1,4/1,2": dispatch,combine
        # counts per pair) to confirm a sweep's winners with more --tuning-reps.
        count_pairs = (
            [
                tuple(checked(int(x)) for x in item.split(","))
                for item in re.split(r"[;/]", args.tuning_candidate)
            ]
            if args.tuning_candidate
            else [
                (dq, cq) for dq in active_qp_candidates for cq in active_qp_candidates
            ]
        )
        candidates = [
            (incumbent_dispatch[:3] + (dq,), incumbent_combine[:3] + (cq,))
            for dq, cq in count_pairs
        ]
    else:
        candidates = [
            geometry + (active_qps,)
            for geometry in geometry_candidates
            for active_qps in active_qp_candidates
        ]
    candidate_fields = ()
    if args.tuning_candidate and not pair:
        candidate_fields = tuple(int(x) for x in args.tuning_candidate.split(","))
        # Three fields keep the shipped count, so a validation run changes the
        # geometry alone; a fourth names the count.
        candidates = [
            candidate_fields[:3]
            + (
                (
                    checked(candidate_fields[3])
                    if len(candidate_fields) > 3
                    else shipped[3]
                ),
            )
        ]
    elif args.tuning_limit:
        candidates = candidates[: args.tuning_limit]
    if shipped in candidates:
        candidates.remove(shipped)
    # With --tuning-save an empty list is a refresh run: measure and save the
    # shipped geometry's row again.
    if not candidates and not args.tuning_save:
        if dist_handle.rank == 0:
            print(
                f"# TUNING tok={args.max_tokens}: nothing to compare -- every "
                f"candidate is the shipped {shipped}",
                flush=True,
            )
        return 0
    # Whether the reported and saved rules name an active QP count: only when
    # this sweep measured the count, or the shipped rule already names one.
    # Otherwise the count is this run's resolved default, and recording it would
    # pin production to it. A pair sweep always measures the counts: it requires
    # --tuning-qps or --tuning-candidate, and either names them.
    # A bucket can hold one phase's rule before the other is tuned.
    table_names_count = bool(
        table_row
        and any(
            table_row[name] is not None and table_row[name][3] is not None
            for name in ("dispatch", "combine")
        )
    )
    report_active_qps = (
        pair or bool(args.tuning_qps) or len(candidate_fields) > 3 or table_names_count
    )

    if dist_handle.rank == 0:
        print(
            f"# TUNING tok={args.max_tokens} phase={phase} "
            f"scope={args.tuning_scope} reps={args.tuning_reps} cus={num_cus} "
            f"candidates={len(candidates)} qp_allocated={cfg.num_qp_per_pe} "
            f"shipped block/rdma/warp/active_qps dispatch={incumbent_dispatch} "
            f"combine={incumbent_combine}",
            flush=True,
        )

    rng = torch.Generator(device=device)
    rng.manual_seed(4242 + dist_handle.rank)
    inp, idx, wts, sc = _generate_round(
        rng, cfg, args.max_tokens, device, cfg.dispatch_dtype, args.routing
    )

    def convert(tensor):
        return _combine_input(cfg, tensor)

    combine_weights = wts if args.bench_weights else None

    # Only the swept phase varies; the other stays at the shipped value, because
    # the two are coupled (a dispatch with too few rdma blocks leaves the combine
    # after it slower) and a per-phase argmin measured against a DIFFERENT other
    # phase does not carry over. Under --tuning-phase pair both phases keep their
    # shipped geometry and only the active QP counts vary, as (dispatch, combine)
    # pairs.
    def geometries(geometry):
        if pair:
            return geometry
        return (
            (geometry, incumbent_combine)
            if phase == "dispatch"
            else (incumbent_dispatch, geometry)
        )

    if args.tuning_metric == "total" or pair:

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
    if not args.tuning_greedy and fixed_wins:
        # Sorted on every rank, not only rank 0: the winner has to be agreed by
        # EVERY rank before anyone builds an op for it, since building one is
        # collective. fixed_wins is derived from allreduced timings, so the
        # ordering is identical on every rank and no communication is needed.
        fixed_wins.sort()
    if dist_handle.rank == 0 and not args.tuning_greedy:
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
    if not args.tuning_greedy and fixed_wins:
        best = fixed_wins[0][2]
        best_median = fixed_wins[0][3]

    if pair:
        dispatch_row, combine_row = best
    else:
        dispatch_row = best if phase == "dispatch" else incumbent_dispatch
        combine_row = best if phase == "combine" else incumbent_combine

    # Measure the pair that will actually be written, on every rank: the row's
    # bandwidth and latency fields are v1's, and v1's mean them for the
    # configuration it shipped. Reusing the sweep's own numbers would mean the
    # winner's stats came from a comparison against a live second op holding a
    # second symmetric window, which is not the state a user runs in.
    metrics = None
    if args.tuning_save:
        final_op = _build_op(cfg, comm, dispatch_row, combine_row)
        comm.barrier()
        metrics = _row_metrics(
            final_op,
            cfg,
            args,
            dist_handle,
            (inp, idx, wts, sc, combine_weights),
            convert,
            ll=kernel_family == "v2_ll",
        )
        final_op.close()
        comm.barrier()

    if dist_handle.rank == 0:
        # best_median is None when nothing was measured against the incumbent:
        # either a refresh run (no candidates left) or every candidate was
        # rejected by the backend. Formatting it unguarded is how that path
        # used to die.
        if best_median is not None:
            # pair always selects on the total.
            metric_name = "total" if pair else args.tuning_metric
            selection = f"median {metric_name}={best_median:.1f}us"
        elif not candidates:
            selection = "no candidate measured (refresh of the shipped geometry)"
        else:
            selection = (
                f"all {len(candidates)} candidates rejected, shipped geometry kept"
            )
        # The rules as --tuning-save writes them: the swept phase's, or both
        # under pair; the active count only where the rule records it.
        rule_fields = "block/rdma/warp" + ("/active_qps" if report_active_qps else "")
        rule_width = 4 if report_active_qps else 3
        rules = "; ".join(
            f"{name} {rule_fields}={row[:rule_width]}"
            for name, row in (
                (("dispatch", dispatch_row), ("combine", combine_row))
                if pair
                else ((phase, best),)
            )
        )
        print(
            f"# TUNING RESULT tok={args.max_tokens} phase={phase}: "
            # Named for the metric, not the phase: --tuning-metric defaults to
            # the dispatch+combine total, so "median dispatch=725us" read as a
            # dispatch time is off by roughly a factor of two.
            f"block/rdma/warp/active_qps={best} {selection} (shipped was {shipped}, "
            f"qp_allocated={cfg.num_qp_per_pe})\n"
            f"#   {kernel_family} rules num_tokens={args.max_tokens}: {rules}",
            flush=True,
        )
        if args.tuning_save:
            # pair writes both phases' rules; each names its measured count.
            for saved_phase, row in (
                (("dispatch", dispatch_row), ("combine", combine_row))
                if pair
                else ((phase, best),)
            ):
                _save_tuning_row(
                    cfg,
                    args,
                    phase=saved_phase,
                    dtype_key=dtype_keys[saved_phase],
                    kernel_family=kernel_family,
                    geometry=row,
                    metrics=metrics[saved_phase],
                    record_active_qps=report_active_qps and cfg.is_internode,
                )
    return 0


def _tuning_table_dir(args):
    """Where --tuning-save writes, as a Path.

    Defaults to THIS REPO's table rather than the installed package's: the
    kernels run from site-packages, but a tuning row is only useful once it is a
    diff somebody can read, and writing into site-packages leaves it nowhere. Same
    reasoning as the v1 bench's `config_path auto`. parents[4] is the repo root
    from tests/python/ops/dispatch_combine_v2/.
    """
    if args.tuning_config_dir:
        return Path(args.tuning_config_dir)
    return (
        Path(__file__).resolve().parents[4]
        / "python"
        / "mori"
        / "ops"
        / "dispatch_combine_v2"
        / "tuning_configs"
    )


def _save_tuning_row(
    cfg,
    args,
    *,
    phase,
    dtype_key,
    kernel_family,
    geometry,
    metrics,
    record_active_qps=False,
):
    """Merge one swept phase's rule into the model's table for this kernel
    family and EP size (hip_tuning_configs.v2_table_filename), which holds
    every expert count, top-k and phase.

    The rule names its active QP count only when the sweep measured it (or the
    shipped rule already named one); otherwise the count is this run's resolved
    default, and writing it would pin production to it.

    Only the phase that was swept is written, as that phase's rule, so the
    other phase's rule is untouched by construction -- writing the incumbent
    back as if it were a result would claim a measurement that was never made.
    """
    from mori.ops.dispatch_combine_v2.hip_tuning_configs import save_internode_result

    block_num, rdma_block_num, warp_per_block = geometry[:3]
    if kernel_family == "intra":
        # One node has no RDMA leg and no LL scale; xGMI is the transport.
        metrics = {
            "bandwidth_gbps": metrics["avg_xgmi_bandwidth_gbps"],
            "avg_xgmi_bandwidth_gbps": metrics["avg_xgmi_bandwidth_gbps"],
            "avg_latency_us": metrics["avg_latency_us"],
            "bandwidth_metric": metrics["bandwidth_metric"],
        }
    entry = {
        "dtype": dtype_key,
        "num_tokens": args.max_tokens,
        # An fp4 dispatch rule is keyed by the packed width, as lookup reads it.
        "hidden_dim": (
            cfg.hidden_dim // 2
            if phase == "dispatch" and cfg.is_fp4
            else cfg.hidden_dim
        ),
        "topk": cfg.num_experts_per_token,
        # world_size x this is the model's expert count, which the rule stores
        # as num_experts, so two expert counts at the same (dtype, hidden,
        # topk, num_tokens) are two rules rather than one overwriting the other.
        "experts_per_rank": cfg.num_experts_per_rank,
        "block_num": block_num,
        "rdma_block_num": rdma_block_num,
        "warp_per_block": warp_per_block,
        **({"active_qps": geometry[3]} if record_active_qps else {}),
        # v1's metric field set, same names and same definitions (_row_metrics).
        **metrics,
    }
    save_internode_result(
        cfg.world_size,
        phase,
        entry,
        kernel_family=kernel_family,
        directory=_tuning_table_dir(args),
    )


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
    # One node builds the intranode op; only test and tuning support that.
    if npes <= gpu_per_node and args.cmd not in ("test", "tuning"):
        raise SystemExit(
            f"--cmd {args.cmd} needs more than one node: world_size={npes} with "
            f"{gpu_per_node} GPUs per node is a single node"
        )

    dtype = _DTYPES[args.dtype]
    combine_dtype = _DTYPES[args.combine_dtype] if args.combine_dtype else None
    if dtype == _FP4 and combine_dtype is None:
        combine_dtype = torch.bfloat16
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
            active_qps=args.active_qps,
            active_qp_counts=(
                tuple(int(count) for count in args.active_qp_counts.split(","))
                if args.active_qp_counts
                else None
            ),
            # AsyncLL is selected by its own flag; internode_kernel keeps its
            # default so the config stays valid (it is simply not read).
            async_ll=args.kernel_type == "asyncll",
            internode_kernel=(
                "auto" if args.kernel_type == "asyncll" else args.kernel_type
            ),
            internode_auto_ll_max_tokens=args.auto_ll_max_tokens,
            kernel_backend="hip",
            moe_sort_block_m=args.moe_sort_block_m,
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

        qp_schedule = _active_qp_schedule(cfg)
        rng = torch.Generator(device=device)
        prev_sort_rows = 0
        for round_index in range(args.rounds):
            rng.manual_seed(1234 + round_index * 977 + rank)
            num_tokens = max_tokens
            inp, idx, wts, sc = _generate_round(
                rng, cfg, num_tokens, device, cfg.dispatch_dtype, args.routing
            )

            if qp_schedule:
                op.set_active_qps(_active_qp_for(qp_schedule, round_index, cfg, 0))
            if args.moe_sort_block_m:
                # Stand in for the last FC2: dirty every row it could have written,
                # the received rows plus the one its padding rows land on.
                moe_out = op.moe_sort()[4]
                moe_out[:prev_sort_rows] = 1
            recv_x, recv_w, recv_s, recv_i, total_recv, routing = _dispatch(
                op, inp, wts, sc, idx
            )
            torch.cuda.synchronize()
            comm.barrier()
            sort_ok = True
            if args.moe_sort_block_m:
                sort_ok = _check_moe_sort(op, cfg, recv_i, recv_w, total_recv)
                prev_sort_rows = int(total_recv.item()) + 1
                if args.moe_sort_dump and rank == 0 and round_index == 0:
                    tables = op.moe_sort()
                    torch.save(
                        dict(
                            recv_i=recv_i.cpu(),
                            recv_w=recv_w.cpu(),
                            total_recv=int(total_recv.item()),
                            block_m=args.moe_sort_block_m,
                            epr=cfg.num_experts_per_rank,
                            rank=rank,
                            tables=[t.cpu() for t in tables[:4]],
                        ),
                        args.moe_sort_dump,
                    )
                if bool(moe_out.any()):
                    print(
                        f"#   rank {rank} moe_sort: expert output not zeroed",
                        flush=True,
                    )
                    sort_ok = False

            # Identity expert: recv_x already holds the dispatched tokens.
            #
            # Converting it is the CALLER's job when the two legs have different
            # element types, exactly as v1's harness does it (_get_combine_input
            # -> _to_combine_dtype). The combine kernel's T is the combine dtype
            # and it reads inpTokenBuf as T*, so handing it the fp8 dispatch
            # output unconverted reinterprets fp8 bytes as bf16.
            combine_input = _combine_input(cfg, recv_x)
            if args.combine_zero_copy:
                # The expert wrote its received rows into combine_in_view(); combine
                # has nothing to copy.
                zero_copy = op.combine_in_view()
                rows = int(total_recv.item())
                zero_copy[:rows].copy_(combine_input[:rows])
                combine_input = zero_copy
            if qp_schedule:
                op.set_active_qps(_active_qp_for(qp_schedule, round_index, cfg, 1))
            # `wts` is a REQUEST for the weight fold, and AsyncLL cannot serve
            # one: its combine payload is hidden-only, so there is no per-token
            # metadata to accumulate (v1's AsyncLL is the same). Asking anyway
            # raises; the weight half of the check is skipped for that family
            # rather than compared against a tensor it never writes.
            # Zero-copy rounds alternate with no weight fold, the combine that
            # skips CombineSync, so both launch paths are checked.
            wants_weights = not getattr(cfg, "async_ll", False) and not (
                args.combine_zero_copy and round_index % 2 == 1
            )
            out, out_w = op.combine(
                combine_input, wts if wants_weights else None, routing=routing
            )
            torch.cuda.synchronize()
            comm.barrier()

            golden = _golden(cfg, args, inp, idx, wts)
            unique_pes_column = golden.unique_pes.cpu()
            unique_pes = unique_pes_column.view(-1)
            expected = golden.hidden.cpu()
            expected_weights = golden.weights.cpu()
            bound = golden.hidden_bound.cpu()
            got = out.float().cpu()
            input_magnitude = _payload_f32(cfg, inp).cpu().abs()
            hidden_ok = bool(((got - expected).abs() <= bound).all())
            got_weights = out_w.cpu() if out_w is not None else None
            weights_ok = got_weights is None or bool(
                (
                    (got_weights - expected_weights).abs() <= golden.weights_bound.cpu()
                ).all()
            )

            if not (hidden_ok and weights_ok) and rank == 0:
                print(f"#   hidden_ok={hidden_ok} weights_ok={weights_ok}", flush=True)
                ratio = (
                    (got_weights / wts.float().cpu().clamp(min=1e-6))[:4]
                    if got_weights is not None
                    else None
                )
                print(
                    f"#   effective multiplier got_w/wts[0,:4]="
                    f"{ratio[0, :4].tolist() if ratio is not None else 'n/a'} "
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
                if got_weights is None:
                    # Nothing to diff: this family was asked for hidden only, so
                    # a weight failure here would be the hidden check's.
                    print(
                        "#   weights: not requested (AsyncLL folds none)",
                        flush=True,
                    )
                else:
                    want_weights = expected_weights
                    # Distinguish "the kernel never wrote it" from "the base
                    # reads the wrong place": read the region it targets.
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
                0 if (hidden_ok and weights_ok and sort_ok) else 1
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
