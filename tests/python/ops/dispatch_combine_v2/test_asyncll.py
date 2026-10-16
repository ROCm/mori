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
"""AsyncLL (hip backend, ccoSdma transport) correctness on one node.

A torchrun script, not a pytest file -- it needs eight ranks and a cco
communicator, like its neighbours here.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 test_asyncll.py

Env: HIDDEN, TOPK, EPR, M, LOOP (at least 3), SHAPES to override the swept list,
TRANSPORT=sdma|lsa, and GRAPH_REPLAYS (default 32; 0 skips the graph regression).

What is checked, and why these and not more:

  * **arrival count** -- every source token routed to any expert of mine arrives
    exactly once. Top-k agnostic, so it is the check that survives a top-k the
    positional comparison cannot handle, and an orphan or duplicated slot breaks
    it. This is the invariant v1's lane-tiling regression was found with.
  * **origin decode** -- the reverse map must decode to a live (rank, token).
    It catches a slot that was allocated but never written, which the count alone
    does not: an orphan leaves the recorded origin at whatever was there before.
  * **round trip** -- an identity expert must return each token scaled by its
    number of DISTINCT destination ranks. That is the end-to-end statement, and
    it is what fails if dedup, the slot maps or the reduction disagree.
  * **received fields** -- hidden, indices and optional top-k weights match the
    source row exactly, including a zero-input rank receiving its peers' weights.

Both split and ordinary APIs run weighted -> unweighted -> weighted calls on one
op. A small graph additionally replays fresh payload/weights with a zero-input
receiver and with all ranks empty. Graph replay is checked after every call.

The swept shapes are chosen for the two things that have actually broken:
top-k that does not divide the wavefront (64 % 6 == 4, DeepSeek-V4's shape), and
ranks that are handed no tokens at all.
"""

import os
import sys

import torch
import torch.distributed as dist
from mori.cco.communicator import Communicator
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

# (hidden, topk, experts_per_rank, max_tokens)
_DEFAULT_SHAPES = [
    (2048, 8, 8, 32),  # small and square
    (7168, 8, 32, 128),  # the shape the bench and the tuning tables use
    (4096, 6, 48, 128),  # 64 % 6 == 4: trailing lanes alias the next warp's token
    (4096, 9, 32, 96),  # 64 % 9 == 1
    (2048, 12, 16, 64),  # 64 % 12 == 4
]


def _shapes():
    raw = os.environ.get("SHAPES")
    if not raw:
        if "HIDDEN" in os.environ or "TOPK" in os.environ:
            return [
                (
                    int(os.environ.get("HIDDEN", "7168")),
                    int(os.environ.get("TOPK", "8")),
                    int(os.environ.get("EPR", "32")),
                    int(os.environ.get("M", "128")),
                )
            ]
        return _DEFAULT_SHAPES
    return [tuple(int(x) for x in row.split(",")) for row in raw.split(";")]


class Dist:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def bcast_uid(self, uid):
        objs = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

    def allreduce_sum(self, value):
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t)
        return int(t.item())


def _routing(rank, world, m, topk, epr, dev, gen, num_tokens):
    """Random top-k over every expert, plus the per-rank token count."""
    total_experts = epr * world
    idx = torch.stack(
        [
            torch.randperm(total_experts, generator=gen, device=dev)[:topk]
            for _ in range(m)
        ]
    ).to(torch.int32)
    return idx[:num_tokens]


def _gather_inputs(x, indices, weights, world):
    """Gather actual source rows; GPU and CPU RNGs need not generate equal data."""
    local = (
        x.shape[0],
        indices.cpu(),
        x.cpu(),
        None if weights is None else weights.cpu(),
    )
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    return gathered


def _check_dispatch(
    d, out, out_indices, out_weights, total, reverse, gathered, m, epr, tag
):
    """Count, origin, and exact source-row comparisons after dispatch completes."""
    recv = int(total[0].item())
    invalid = torch.tensor([int(not 0 <= recv <= out.shape[0])], dtype=torch.int64)
    dist.all_reduce(invalid)
    if invalid.item():
        raise RuntimeError(f"{tag}: receive count outside the output capacity")

    failures = 0
    expected = sum(
        d.rank in {int(e) // epr for e in row.tolist()}
        for _, source_indices, _, _ in gathered
        for row in source_indices
    )
    if recv != expected:
        print(f"[rank {d.rank}] {tag}: recv={recv}, expected={expected}", flush=True)
        failures += 1

    origins = reverse[:recv].cpu().to(torch.int64)
    source_ranks, source_tokens = origins // m, origins % m
    bad_origin = sum(
        not (0 <= pe < d.world and 0 <= token < gathered[pe][0])
        for pe, token in zip(source_ranks.tolist(), source_tokens.tolist())
    )
    if bad_origin:
        print(f"[rank {d.rank}] {tag}: {bad_origin} invalid source origins", flush=True)
        failures += 1

    # None means the caller did not request weight transport; its output weights
    # are not defined. The next weighted call still has to replace every row.
    has_weights = gathered[d.rank][3] is not None
    actual = {"hidden": out[:recv].cpu(), "indices": out_indices[:recv].cpu()}
    if has_weights:
        actual["weights"] = out_weights[:recv].cpu()
    for pe, (source_n, source_idx, source_x, source_w) in enumerate(gathered):
        selected = (
            (source_ranks == pe) & (source_tokens >= 0) & (source_tokens < source_n)
        )
        if not selected.any().item():
            continue
        token_ids = source_tokens[selected]
        expected_fields = {
            "hidden": source_x[token_ids],
            "indices": source_idx[token_ids],
        }
        if has_weights:
            expected_fields["weights"] = source_w[token_ids]
        for field, want in expected_fields.items():
            got = actual[field][selected]
            if not torch.equal(got, want):
                bad_rows = int((got != want).any(dim=1).sum().item())
                print(
                    f"[rank {d.rank}] {tag}: recv {field} mismatch in {bad_rows} "
                    f"rows from rank {pe}",
                    flush=True,
                )
                failures += 1
    return recv, failures


def run_shape(d, comm, hidden, topk, epr, m, loops):
    rank, world = d.rank, d.world
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=hidden,
        max_num_inp_token_per_rank=m,
        num_experts_per_rank=epr,
        num_experts_per_token=topk,
        data_type=torch.bfloat16,
        kernel_backend="hip",
        async_ll=True,
        async_ll_transport=os.environ.get("TRANSPORT", "sdma"),
    )
    failures = 0
    with EpDispatchCombineOp(cfg, comm) as op:
        for it in range(loops):
            # A different token count per rank per iteration, including zero:
            # a rank with no tokens still has to publish its count or every peer
            # waits on it forever, and the per-token divisions must not divide by
            # that zero either.
            counts = [((r + it) * 7) % (m + 1) for r in range(world)]
            n = counts[rank]
            gen = torch.Generator(device=dev).manual_seed(1234 + rank * 17 + it)
            x = torch.randn(m, hidden, generator=gen, device=dev, dtype=torch.bfloat16)
            w = torch.rand(m, topk, generator=gen, device=dev, dtype=torch.float32)
            idx_full = _routing(rank, world, m, topk, epr, dev, gen, m)
            idx, xin, win = idx_full[:n], x[:n], w[:n]

            weights = None if it % 3 == 1 else win
            gathered = _gather_inputs(xin, idx, weights, world)
            idx_cpu = gathered[rank][1]
            for split in (True, False):
                api = "split" if split else "ordinary"
                tag = f"h={hidden} k={topk} it={it} api={api} weights={weights is not None}"
                dispatch = op.dispatch_send if split else op.dispatch
                out, out_weights, _, out_indices, total, routing = dispatch(
                    xin, weights, None, idx, return_routing=True
                )
                if split:
                    op.dispatch_recv()
                torch.cuda.synchronize()
                dist.barrier()
                recv, bad = _check_dispatch(
                    d,
                    out,
                    out_indices,
                    out_weights,
                    total,
                    routing.disp_tok_id_to_src_tok_id_local,
                    gathered,
                    m,
                    epr,
                    tag,
                )
                failures += bad

                combine = op.combine_send if split else op.combine
                cout, _ = combine(out[:recv].clone(), None, idx, routing=routing)
                if split:
                    op.combine_recv()
                torch.cuda.synchronize()
                dist.barrier()

                if n:
                    uniq = torch.tensor(
                        [len({int(e) // epr for e in row.tolist()}) for row in idx_cpu],
                        dtype=torch.float32,
                    )
                    want = (gathered[rank][2].float() * uniq[:, None]).to(
                        torch.bfloat16
                    )
                    got = cout[:n].cpu()
                    if not torch.allclose(
                        got.float(), want.float(), atol=1e-2, rtol=1e-2
                    ):
                        diff = (got.float() - want.float()).abs()
                        print(
                            f"[rank {rank}] {tag}: combine max |diff| {diff.max().item():.4f}",
                            flush=True,
                        )
                        failures += 1

            # Asking for the weight fold has to FAIL, not return zeros: on this
            # API a non-None `weights` is a REQUEST, and AsyncLL's combine
            # payload carries no per-token metadata to fold. Every call above
            # passes None, which is how a wrong weight branch sat here unnoticed
            # until the cross-node harness -- which requests weights
            # unconditionally -- ran into it.
            #
            # Last in the iteration, and only on the first one: the raise comes
            # from inside _combine_impl, AFTER it has staged `input` into the
            # out_tok region, so running it before the round trip above would
            # have it write through a buffer the measured path reads.
            if it == 0:
                try:
                    op.combine_send(out[:recv].clone(), win, idx, routing=routing)
                except NotImplementedError:
                    pass
                else:
                    print(
                        f"[rank {rank}] h={hidden} k={topk}: combine accepted a "
                        "weight-fold request it cannot serve",
                        flush=True,
                    )
                    failures += 1
    return failures


def run_graph(d, comm, empty_all, replays):
    """Small exact-data regression; all allocations precede graph replay."""
    hidden, topk, epr, m = 2048, 6, 8, 16
    n = 0 if empty_all or d.rank == 0 else m
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", d.rank)))
    cfg = EpDispatchCombineConfig(
        rank=d.rank,
        world_size=d.world,
        hidden_dim=hidden,
        max_num_inp_token_per_rank=m,
        num_experts_per_rank=epr,
        num_experts_per_token=topk,
        data_type=torch.bfloat16,
        kernel_backend="hip",
        async_ll=True,
        async_ll_transport=os.environ.get("TRANSPORT", "sdma"),
    )
    # With at least two ranks, peers deterministically route live tokens to rank
    # zero, whose empty weights tensor has a null data_ptr on HIP.
    idx_cpu = torch.arange(n * topk, dtype=torch.int32).view(n, topk) % (d.world * epr)
    indices = idx_cpu.to(dev)
    base_x = (
        (torch.arange(n).view(n, 1) % 2)
        + (torch.arange(hidden).view(1, hidden) % 2)
        + 2 * (d.rank % 4)
        + 1
    )
    base_w = (
        torch.arange(n * topk, dtype=torch.float32).view(n, topk) + d.rank * 100000 + 1
    )
    x = torch.empty((n, hidden), dtype=torch.bfloat16, device=dev)
    weights = torch.empty((n, topk), dtype=torch.float32, device=dev)
    uniq = torch.tensor(
        [len(set((row // epr).tolist())) for row in idx_cpu],
        dtype=torch.float32,
    ).view(n, 1)
    failures = 0
    scenario = "all_zero" if empty_all else "zero_source_receives"
    with EpDispatchCombineOp(cfg, comm) as op:
        stage = op.combine_in_view()
        last = {}

        def update(generation):
            # Hidden <= 40 and at most six contributions: the identity sum is
            # <= 240 and exact in bf16. Float32 weight watermarks change on every
            # replay, even when the hidden watermark wraps after 32 generations.
            x.copy_((base_x + generation % 32).to(torch.bfloat16))
            weights.copy_(base_w + generation * 1024)

        def pair():
            out, out_weights, _, out_indices, total, routing = op.dispatch_send(
                x, weights, None, indices, return_routing=True
            )
            op.dispatch_recv()
            # Capture the lazy reverse-map clone after recv. Its GPU copy then
            # executes on EACH replay; reading a cached host routing snapshot
            # only after the first replay would retain the previous slot order.
            reverse = routing.disp_tok_id_to_src_tok_id_local
            stage.copy_(out)
            cout, _ = op.combine_send(stage, routing=routing)
            op.combine_recv()
            last.update(
                out=out,
                weights=out_weights,
                indices=out_indices,
                total=total,
                reverse=reverse,
                combined=cout,
            )

        def check(generation):
            torch.cuda.synchronize()
            dist.barrier()
            gathered = _gather_inputs(x, indices, weights, d.world)
            tag = f"graph={scenario} generation={generation}"
            recv, bad = _check_dispatch(
                d,
                last["out"],
                last["indices"],
                last["weights"],
                last["total"],
                last["reverse"],
                gathered,
                m,
                epr,
                tag,
            )
            if not empty_all and d.world > 1 and d.rank == 0 and recv == 0:
                print(
                    f"[rank 0] {tag}: zero-input rank received no peer tokens",
                    flush=True,
                )
                bad += 1
            want = (gathered[d.rank][2].float() * uniq).to(torch.bfloat16)
            if not torch.equal(last["combined"].cpu(), want):
                print(
                    f"[rank {d.rank}] {tag}: exact identity round trip failed",
                    flush=True,
                )
                bad += 1
            return bad

        update(0)
        pair()
        failures += check("prime")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pair()
        torch.cuda.synchronize()
        dist.barrier()
        for generation in range(1, replays + 1):
            update(generation)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            failures += check(generation)
        del graph
        torch.cuda.synchronize()
        dist.barrier()
    return failures


def main():
    d = Dist()
    shapes = _shapes()
    loops = max(3, int(os.environ.get("LOOP", "3")))
    graph_replays = int(os.environ.get("GRAPH_REPLAYS", "32"))
    if graph_replays < 0:
        raise ValueError("GRAPH_REPLAYS must be nonnegative")

    failures = 0
    # One communicator PER SHAPE, each with its own unique id, rather than one for
    # the whole run. Each shape's arena is a differently-sized window, and on
    # ROCm 7.0-7.2.x mapping several of those into one reservation trips a clr
    # bug: hipMemSetAccess validates coverage by walking the sub-buffers from
    # index 0 regardless of the pointer it was given, so every window after the
    # first fails with "invalid argument" (known-issues skill, Issue 2). Nothing
    # to do with these kernels -- the same failure reproduces on an untouched
    # tree, in test_ep_backend_parity, which likewise builds two ops on one
    # communicator -- but the shapes are the point of this test, so it is
    # structured to stay out of the bug's way.
    for hidden, topk, epr, m in shapes:
        uid = Communicator.get_unique_id() if d.rank == 0 else None
        uid = d.bcast_uid(uid)
        win_bytes = d.world * hidden * m * 2 * 4 + (1 << 24)
        with Communicator.init(
            d.world, d.rank, uid, per_rank_vmm=6 * win_bytes + (1 << 28)
        ) as comm:
            failures += run_shape(d, comm, hidden, topk, epr, m, loops)
        if d.rank == 0:
            print(
                f"# asyncll hidden={hidden} topk={topk} epr={epr} m={m}: done",
                flush=True,
            )

    if graph_replays:
        for empty_all in (False, True):
            uid = Communicator.get_unique_id() if d.rank == 0 else None
            uid = d.bcast_uid(uid)
            win_bytes = d.world * 2048 * 16 * 2 * 4 + (1 << 24)
            with Communicator.init(
                d.world, d.rank, uid, per_rank_vmm=6 * win_bytes + (1 << 28)
            ) as comm:
                failures += run_graph(d, comm, empty_all, graph_replays)
            if d.rank == 0:
                print(
                    f"# asyncll graph empty_all={empty_all}: {graph_replays} replays done",
                    flush=True,
                )

    total = d.allreduce_sum(failures)
    if d.rank == 0:
        print(
            f"# {'PASS' if total == 0 else 'FAIL'}: {total} failed checks across "
            f"{d.world} ranks, {len(shapes)} shapes x {loops} iterations x 2 APIs, "
            f"2 graph scenarios x {graph_replays} replays",
            flush=True,
        )
    dist.destroy_process_group()
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
