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
"""Simulated EP dispatch/combine (MORI_EP_SIM_ENABLE=1): runs without a GPU."""
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mori.ops.dispatch_combine import (
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
    EpDispatchCombineOp,
)

HIDDEN, TOPK, EXPERTS_PER_RANK, MAX_TOKENS = 256, 4, 8, 128
DISPATCH_US, COMBINE_US = 3000, 5000
ENV = {
    "MORI_EP_SIM_ENABLE": "1",
    "MORI_EP_SIM_DISPATCH_US": str(DISPATCH_US),
    "MORI_EP_SIM_COMBINE_US": str(COMBINE_US),
}


@pytest.fixture(autouse=True)
def sim_env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _op(rank=0, world=1, kernel=EpDispatchCombineKernelType.IntraNode):
    return EpDispatchCombineOp(
        EpDispatchCombineConfig(
            data_type=torch.bfloat16,
            rank=rank,
            world_size=world,
            hidden_dim=HIDDEN,
            scale_dim=0,
            scale_type_size=0,
            max_token_type_size=2,
            max_num_inp_token_per_rank=MAX_TOKENS,
            num_experts_per_rank=EXPERTS_PER_RANK,
            num_experts_per_token=TOPK,
            kernel_type=kernel,
            gpu_per_node=8,
        )
    )


def _routing(rank, world):
    """Rank `rank`'s tokens, skewed: rank 0's experts are 4x as likely as the others'."""
    g = torch.Generator().manual_seed(rank)
    n = 16 * (rank + 1)
    p = torch.ones(world * EXPERTS_PER_RANK)
    p[:EXPERTS_PER_RANK] = 4
    ids = torch.stack([torch.multinomial(p, TOPK, generator=g) for _ in range(n)])
    ids[::5, 1:] = -1  # some tokens use fewer experts
    return ids.to(torch.int32)


def _expected(rank, world):
    lo, hi = rank * EXPERTS_PER_RANK, (rank + 1) * EXPERTS_PER_RANK
    rows = [
        r
        for s in range(world)
        for r in _routing(s, world)
        if ((r >= lo) & (r < hi)).any()
    ]
    ids = torch.stack(rows) if rows else torch.empty(0, TOPK, dtype=torch.int32)
    hits = ids[(ids >= lo) & (ids < hi)] - lo
    return ids, torch.bincount(hits, minlength=EXPERTS_PER_RANK).to(torch.int32)


def _inputs(n):
    return torch.zeros(n, HIDDEN, dtype=torch.bfloat16), torch.ones(n, TOPK)


def _check_dispatch(op, outputs, rank, world):
    out, out_w, out_s, out_i, total = outputs
    want_ids, want_count = _expected(rank, world)
    m = want_ids.size(0)
    assert int(total.item()) == m
    assert out.shape == (m, HIDDEN) and out_w.shape == (m, TOPK) and out_s is None
    assert torch.equal(out_i, want_ids)
    assert torch.equal(op.local_expert_count, want_count)


def test_single_rank_latency_and_shapes():
    op = _op()
    ids = _routing(0, 1)
    x, w = _inputs(ids.size(0))
    t0 = time.monotonic()
    outputs = op.dispatch(x, w, None, ids)
    assert time.monotonic() - t0 >= DISPATCH_US / 1e6
    _check_dispatch(op, outputs, 0, 1)

    t0 = time.monotonic()
    out, out_w = op.combine(outputs[0], outputs[1], ids)
    assert time.monotonic() - t0 >= COMBINE_US / 1e6
    assert out.shape == (ids.size(0), HIDDEN) and out_w.shape == (ids.size(0), TOPK)


def test_asyncll_recv_waits_out_the_rest():
    op = _op(kernel=EpDispatchCombineKernelType.AsyncLL)
    ids = _routing(0, 1)
    x, w = _inputs(ids.size(0))
    t0 = time.monotonic()
    op.dispatch(x, w, None, ids)
    assert time.monotonic() - t0 < DISPATCH_US / 2e6
    op.dispatch_recv()
    assert time.monotonic() - t0 >= DISPATCH_US / 1e6

    op.combine(x, w, ids)
    time.sleep(2 * COMBINE_US / 1e6)  # work that hides the transfer
    t0 = time.monotonic()
    op.combine_recv()
    assert time.monotonic() - t0 < 1e-3


def test_latency_must_be_set(monkeypatch):
    monkeypatch.delenv("MORI_EP_SIM_COMBINE_US")
    with pytest.raises(RuntimeError, match="MORI_EP_SIM_COMBINE_US"):
        _op()


LATE_RANK, LATE_S = 3, 0.02


def _rank_main(rank, world, port):
    import os

    os.environ.update(ENV)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    op = _op(rank, world)
    ids = _routing(rank, world)
    x, w = _inputs(ids.size(0))
    for late in (False, True):
        dist.barrier()
        if late and rank == LATE_RANK:
            time.sleep(LATE_S)
        t0 = time.monotonic()
        outputs = op.dispatch(x, w, None, ids)
        t1 = time.monotonic()
        _check_dispatch(op, outputs, rank, world)
        ends = [None] * world
        dist.all_gather_object(ends, t1)
        assert max(ends) - min(ends) < 3e-3, ends
        if late and rank != LATE_RANK:
            # less the spread of barrier exits across ranks
            assert t1 - t0 >= LATE_S + DISPATCH_US / 1e6 - 1e-3
        out, _ = op.combine(outputs[0], outputs[1], ids)
        assert out.shape == (ids.size(0), HIDDEN)
    dist.destroy_process_group()


def test_ranks_see_their_own_imbalance():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_rank_main, args=(4, port), nprocs=4)
