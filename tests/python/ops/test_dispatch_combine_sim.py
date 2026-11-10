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
import os
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

HIDDEN, TOPK, EXPERTS_PER_RANK, MAX_TOKENS = 256, 4, 8, 256
DISPATCH_US, COMBINE_US = 3000, 5000
ENV = {
    "MORI_EP_SIM_ENABLE": "1",
    "MORI_EP_SIM_DISPATCH_US": str(DISPATCH_US),
    "MORI_EP_SIM_COMBINE_US": str(COMBINE_US),
}
# Layer 0 uniform, layer 1 skewed with expert 0 (on rank 0) the hottest.
ROUTING = "- alpha: 0\n- alpha: 2\n"


@pytest.fixture(autouse=True)
def sim_env(monkeypatch):
    monkeypatch.delenv("MORI_EP_SIM_ROUTING", raising=False)
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


def _inputs(n):
    # The simulator draws its own routing: these indices are ignored.
    return (
        torch.zeros(n, HIDDEN, dtype=torch.bfloat16),
        torch.ones(n, TOPK),
        torch.full((n, TOPK), -1, dtype=torch.int32),
    )


def _check_dispatch(op, outputs, rank, world):
    out, out_w, out_s, out_i, total = outputs
    m = int(total.item())
    assert out.shape == (m, HIDDEN) and out_w.shape == (m, TOPK) and out_s is None
    assert out_i.shape == (m, TOPK)
    lo, hi = rank * EXPERTS_PER_RANK, (rank + 1) * EXPERTS_PER_RANK
    assert ((out_i >= 0) & (out_i < world * EXPERTS_PER_RANK)).all()
    assert all(len(set(row.tolist())) == TOPK for row in out_i)
    local = (out_i >= lo) & (out_i < hi)
    assert local.any(dim=1).all()
    want = torch.bincount(out_i[local] - lo, minlength=EXPERTS_PER_RANK)
    assert torch.equal(op.local_expert_count, want.to(torch.int32))
    return m


def test_single_rank_latency_and_shapes():
    op = _op()
    x, w, idx = _inputs(16)
    t0 = time.monotonic()
    outputs = op.dispatch(x, w, None, idx)
    assert time.monotonic() - t0 >= DISPATCH_US / 1e6
    assert _check_dispatch(op, outputs, 0, 1) == 16  # one rank receives every token

    t0 = time.monotonic()
    out, out_w = op.combine(outputs[0], outputs[1], idx)
    assert time.monotonic() - t0 >= COMBINE_US / 1e6
    assert out.shape == (16, HIDDEN) and out_w.shape == (16, TOPK)


def test_asyncll_recv_waits_out_the_rest():
    op = _op(kernel=EpDispatchCombineKernelType.AsyncLL)
    x, w, idx = _inputs(16)
    t0 = time.monotonic()
    op.dispatch(x, w, None, idx)
    assert time.monotonic() - t0 < DISPATCH_US / 2e6
    op.dispatch_recv()
    assert time.monotonic() - t0 >= DISPATCH_US / 1e6

    op.combine(x, w, idx)
    time.sleep(2 * COMBINE_US / 1e6)  # work that hides the transfer
    t0 = time.monotonic()
    op.combine_recv()
    assert time.monotonic() - t0 < 1e-3


def test_config_errors(monkeypatch, tmp_path):
    monkeypatch.delenv("MORI_EP_SIM_COMBINE_US")
    with pytest.raises(RuntimeError, match="MORI_EP_SIM_COMBINE_US"):
        _op()
    monkeypatch.setenv("MORI_EP_SIM_COMBINE_US", "1")
    bad = tmp_path / "routing.yaml"
    bad.write_text("- alpah: 1\n")
    monkeypatch.setenv("MORI_EP_SIM_ROUTING", str(bad))
    with pytest.raises(RuntimeError, match="unknown field 'alpah'"):
        _op()


LATE_RANK, LATE_S = 3, 0.02


def _rank_main(rank, world, port, routing):
    os.environ.update(ENV, MORI_EP_SIM_ROUTING=routing)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    op = _op(rank, world)
    n = 64 * (rank + 1)
    x, w, idx = _inputs(n)
    sent = [None] * world
    dist.all_gather_object(sent, n)
    for step in range(4):
        late = step == 3
        dist.barrier()
        if late and rank == LATE_RANK:
            time.sleep(LATE_S)
        t0 = time.monotonic()
        outputs = op.dispatch(x, w, None, idx)
        t1 = time.monotonic()
        got = [None] * world
        dist.all_gather_object(
            got,
            (
                _check_dispatch(op, outputs, rank, world),
                t1,
                int(op.local_expert_count.sum()),
            ),
        )
        recv, ends, hits = zip(*got)
        assert sum(hits) == sum(sent) * TOPK  # every (token, expert) pair lands once
        if step % 2 == 0:  # uniform layer
            assert max(recv) < 1.3 * min(recv), recv
        else:  # expert 0's rank is hot
            assert recv[0] > 1.5 * max(recv[1:]), recv
        assert max(ends) - min(ends) < 3e-3, ends
        if late and rank != LATE_RANK:
            # less the spread of barrier exits across ranks
            assert t1 - t0 >= LATE_S + DISPATCH_US / 1e6 - 1e-3
        out, _ = op.combine(outputs[0], outputs[1], idx)
        assert out.shape == (n, HIDDEN)
    dist.destroy_process_group()


def test_routing_layers_and_imbalance(tmp_path):
    routing = tmp_path / "routing.yaml"
    routing.write_text(ROUTING)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_rank_main, args=(4, port, str(routing)), nprocs=4)
