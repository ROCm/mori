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
"""Simulated EP dispatch/combine (MORI_EP_SIM_ENABLE=1): no GPU is needed.

No token moves; the checks are on what the simulator does compute -- received-token counts,
local expert counts and timing -- against the routing the ranks actually used.
"""
import json
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mori import cpp as mori_cpp
from mori.ops.dispatch_combine import (
    _EP_SIM_CATALOG,
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
    EpDispatchCombineOp,
    _make_cpp_config,
)
from mori.tensor_utils import dtype_to_int

HIDDEN = 7168
TOPK = 8
NUM_EXPERTS = 256
BF16 = dtype_to_int(torch.bfloat16)


def _config(
    rank, world, kernel=EpDispatchCombineKernelType.IntraNode, tokens=512, topk=TOPK
):
    return EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world,
        hidden_dim=HIDDEN,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=NUM_EXPERTS // world,
        num_experts_per_token=topk,
        kernel_type=kernel,
        gpu_per_node=8,
    )


def _rank_hit(num_experts, per_rank, topk):
    miss = 1.0
    for i in range(topk):
        miss *= (num_experts - per_rank - i) / (num_experts - i)
    return 1.0 - miss


def _routing(rank, n, seed=1234):
    g = torch.Generator().manual_seed(seed + rank)
    return torch.stack(
        [torch.randperm(NUM_EXPERTS, generator=g)[:TOPK] for _ in range(n)]
    ).to(torch.int32)


def _catalog_curve(**match):
    yaml = pytest.importorskip("yaml")
    with open(_EP_SIM_CATALOG) as f:
        for e in yaml.safe_load(f):
            if all(e.get(k) == v for k, v in match.items()):
                toks = [int(t) for t in str(e["tokens"]).split(",")]
                lats = [float(v) for v in str(e["latency_us"]).split(",")]
                return toks, lats
    raise AssertionError(f"no catalog curve {match}")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _sim_env(monkeypatch):
    monkeypatch.setenv("MORI_EP_SIM_ENABLE", "1")
    monkeypatch.setenv("MORI_EP_SIM_GPU", "mi355x")
    for var in (
        "MORI_EP_SIM_ROUTING",
        "MORI_EP_SIM_GROUP",
        "MORI_EP_SIM_SYNC",
        "MORI_EP_SIM_CATALOG",
        "MORI_EP_SIM_TIME_SCALE",
        "MORI_EP_SIM_TRACE",
    ):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------------------------
# Timing model
# --------------------------------------------------------------------------------------------
def test_estimate_returns_catalog_points():
    toks, lats = _catalog_curve(
        gpu="mi355x",
        kernel="IntraNode",
        ep_size=8,
        phase="dispatch",
        dtype="bf16",
        topk=6,
    )
    cfg = _make_cpp_config(_config(0, 8, topk=6))
    for t, lat in zip(toks, lats):
        est = mori_cpp.ep_sim_estimate(
            cfg, "dispatch", BF16, HIDDEN, t, False, _EP_SIM_CATALOG
        )
        assert est["latency_us"] == pytest.approx(lat, rel=1e-9)
        assert est["tokens"] == pytest.approx(t)
        assert est["source"].startswith(_EP_SIM_CATALOG)
    mid = (toks[2] + toks[3]) // 2
    est = mori_cpp.ep_sim_estimate(
        cfg, "dispatch", BF16, HIDDEN, mid, False, _EP_SIM_CATALOG
    )
    assert lats[2] < est["latency_us"] < lats[3]


def test_estimate_scales_other_shapes_by_bytes():
    cfg = _make_cpp_config(_config(0, 8, topk=6))
    full = mori_cpp.ep_sim_estimate(
        cfg, "dispatch", BF16, HIDDEN, 1024, False, _EP_SIM_CATALOG
    )
    half = mori_cpp.ep_sim_estimate(
        cfg, "dispatch", BF16, 4096, 1024, False, _EP_SIM_CATALOG
    )
    ratio = (4096 * 2 + 6 * 8 + 4) / (HIDDEN * 2 + 6 * 8 + 4)
    assert "scaled" in half["source"]
    assert half["tokens"] == pytest.approx(1024 * ratio)
    assert half["latency_us"] < full["latency_us"]


def test_estimate_without_gpu_uses_analytic_model(monkeypatch):
    monkeypatch.delenv("MORI_EP_SIM_GPU")
    cfg = _make_cpp_config(_config(0, 8))
    est = mori_cpp.ep_sim_estimate(
        cfg, "dispatch", BF16, HIDDEN, 1024, False, _EP_SIM_CATALOG
    )
    assert est["source"].startswith("analytic")
    pairs = 1024 * 8 * _rank_hit(NUM_EXPERTS, 32, TOPK)
    expect = 25 + pairs * (HIDDEN * 2 + TOPK * 8 + 4) / 350 / 1000
    assert est["latency_us"] == pytest.approx(expect, rel=1e-6)


def test_unknown_gpu_is_rejected(monkeypatch):
    monkeypatch.setenv("MORI_EP_SIM_GPU", "mi999x")
    cfg = _make_cpp_config(_config(0, 8))
    with pytest.raises(RuntimeError, match="mi355x"):
        mori_cpp.ep_sim_estimate(
            cfg, "dispatch", BF16, HIDDEN, 64, False, _EP_SIM_CATALOG
        )


# --------------------------------------------------------------------------------------------
# One process, no rendezvous (MORI_EP_SIM_SYNC=0)
# --------------------------------------------------------------------------------------------
def test_uniform_routing_ignores_indices(monkeypatch):
    monkeypatch.setenv("MORI_EP_SIM_SYNC", "0")
    monkeypatch.setenv("MORI_EP_SIM_ROUTING", "uniform")
    op = EpDispatchCombineOp(_config(0, 8))
    n = 512
    x = torch.zeros(n, HIDDEN, dtype=torch.bfloat16)
    garbage = torch.full((n, TOPK), 9999, dtype=torch.int32)
    out, w, s, i, total = op.dispatch(x, torch.rand(n, TOPK), None, garbage)
    assert out.shape == (op.max_num_tokens_to_recv(), HIDDEN) and s is None
    assert int(total.item()) == round(n * 8 * _rank_hit(NUM_EXPERTS, 32, TOPK))
    comb, _ = op.combine(out, None, garbage)
    assert comb.shape == (512, HIDDEN)


def test_out_of_range_expert_names_the_uniform_switch(monkeypatch):
    monkeypatch.setenv("MORI_EP_SIM_SYNC", "0")
    op = EpDispatchCombineOp(_config(0, 8))
    bad = torch.full((4, TOPK), NUM_EXPERTS, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="MORI_EP_SIM_ROUTING=uniform"):
        op.dispatch(torch.zeros(4, HIDDEN, dtype=torch.bfloat16), None, None, bad)


def test_time_scale_shortens_the_sleep(monkeypatch):
    monkeypatch.setenv("MORI_EP_SIM_SYNC", "0")
    monkeypatch.setenv("MORI_EP_SIM_TIME_SCALE", "10")
    op = EpDispatchCombineOp(_config(0, 8, tokens=4096))
    n = 4096
    x = torch.zeros(n, HIDDEN, dtype=torch.bfloat16)
    idx = _routing(0, n)
    op.dispatch(x, None, None, idx)  # first call allocates the outputs
    t0 = time.monotonic()
    for _ in range(5):
        op.dispatch(x, None, None, idx)
    wall_us = (time.monotonic() - t0) * 1e6 / 5
    st = op.sim_last_stats()
    assert st["model_us"] > 500
    assert wall_us < st["model_us"] / 2
    assert st["elapsed_us"] >= st["model_us"] * 0.99


def test_unsupported_features_raise(monkeypatch):
    monkeypatch.setenv("MORI_EP_SIM_SYNC", "0")
    op = EpDispatchCombineOp(_config(0, 8))
    x = torch.zeros(4, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        op.dispatch(x, None, None, _routing(0, 4), return_routing=True)
    with pytest.raises(NotImplementedError):
        op.get_dispatch_src_token_pos()
    with pytest.raises(ValueError, match="AsyncLL"):
        op.dispatch_recv()


# --------------------------------------------------------------------------------------------
# Several processes: real routing, shared-memory rendezvous
# --------------------------------------------------------------------------------------------
def _expected(all_idx, rank, world):
    per_rank = NUM_EXPERTS // world
    recv = sum(int(((i // per_rank) == rank).any(dim=1).sum()) for i in all_idx)
    lec = torch.zeros(per_rank, dtype=torch.int64)
    for i in all_idx:
        local = i[(i // per_rank) == rank].long() - rank * per_rank
        lec += torch.bincount(local, minlength=per_rank)
    return recv, lec.tolist()


def _intranode_worker(rank, world, port, out_dir):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    op = EpDispatchCombineOp(_config(rank, world))
    n = 300 if rank == 5 else 512
    x = torch.zeros(n, HIDDEN, dtype=torch.bfloat16)
    idx = _routing(rank, n)
    all_idx = [None] * world
    dist.all_gather_object(all_idx, idx)
    exp_recv, exp_lec = _expected(all_idx, rank, world)
    res = {"rank": rank, "exp_recv": exp_recv, "exp_lec": exp_lec}

    dist.barrier()
    out, _, _, _, total = op.dispatch(
        x, torch.rand(n, TOPK), None, idx, call_local_expert_count=True
    )
    res["done"] = time.monotonic()
    res["recv"] = int(total.item())
    res["lec"] = op.local_expert_count.tolist()
    res["dispatch"] = op.sim_last_stats()
    op.combine(out, None, idx)
    res["combine"] = op.sim_last_stats()

    # Rank 3 arrives 20 ms late: everyone else waits for it, as on the real barrier.
    dist.barrier()
    if rank == 3:
        time.sleep(0.02)
    op.dispatch(x, None, None, idx)
    res["late"] = op.sim_last_stats()
    op.combine(out, None, idx)
    with open(f"{out_dir}/{rank}.json", "w") as f:
        json.dump(res, f)
    dist.barrier()
    dist.destroy_process_group()


def _run(worker, world, tmp_path):
    mp.spawn(worker, args=(world, _free_port(), str(tmp_path)), nprocs=world, join=True)
    return [json.load(open(tmp_path / f"{r}.json")) for r in range(world)]


def test_intranode_counts_and_barrier(tmp_path):
    world = 8
    res = _run(_intranode_worker, world, tmp_path)
    for r in res:
        assert r["recv"] == r["exp_recv"], r["rank"]
        assert r["lec"] == r["exp_lec"], r["rank"]
        d = r["dispatch"]
        assert d["source"].startswith(_EP_SIM_CATALOG), d["source"]
        assert d["elapsed_us"] >= d["model_us"] * 0.99
        assert r["combine"]["phase"] == "combine"
        assert r["combine"]["recv_tokens"] == (300 if r["rank"] == 5 else 512)
    # Every rank wakes at the same deadline.
    done = [r["done"] for r in res]
    assert (max(done) - min(done)) * 1e6 < 2000
    # The late rank waits only for its own transfer; the others also wait for it.
    late = res[3]["late"]
    assert late["elapsed_us"] < late["model_us"] + 3000
    for r in res:
        if r["rank"] != 3:
            assert r["late"]["elapsed_us"] > 20000 + late["model_us"] * 0.5, r["rank"]


def _asyncll_worker(rank, world, port, out_dir):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    op = EpDispatchCombineOp(
        _config(rank, world, kernel=EpDispatchCombineKernelType.AsyncLL)
    )
    n = 512
    x = torch.zeros(n, HIDDEN, dtype=torch.bfloat16)
    idx = _routing(rank, n, seed=99)
    all_idx = [None] * world
    dist.all_gather_object(all_idx, idx)
    exp_recv, _ = _expected(all_idx, rank, world)
    res = {"rank": rank, "exp_recv": exp_recv}

    dist.barrier()
    t0 = time.monotonic()
    *_, total = op.dispatch(x, None, None, idx)
    res["send_us"] = (time.monotonic() - t0) * 1e6
    try:
        op.dispatch(x, None, None, idx)
        res["double_post"] = "accepted"
    except RuntimeError as e:
        res["double_post"] = str(e)
    op.dispatch_recv()
    res["recv"] = int(total.item())
    res["idle"] = op.sim_last_stats()
    op.combine(x, None, idx)
    op.combine_recv()

    # Work between send and recv hides the transfer.
    dist.barrier()
    op.dispatch(x, None, None, idx)
    time.sleep(max(3 * res["idle"]["model_us"] / 1e6, 0.02))
    t0 = time.monotonic()
    op.dispatch_recv()
    res["hidden_recv_us"] = (time.monotonic() - t0) * 1e6
    op.combine(x, None, idx)
    op.combine_recv()
    with open(f"{out_dir}/{rank}.json", "w") as f:
        json.dump(res, f)
    dist.barrier()
    dist.destroy_process_group()


def test_asyncll_send_recv_overlap(tmp_path):
    world = 16
    res = _run(_asyncll_worker, world, tmp_path)
    for r in res:
        idle = r["idle"]
        assert idle["source"].startswith(_EP_SIM_CATALOG), idle["source"]
        assert r["recv"] == r["exp_recv"], r["rank"]
        assert r["send_us"] < idle["model_us"] / 2
        assert "not received" in r["double_post"]
        assert idle["elapsed_us"] >= idle["model_us"] * 0.99
        assert r["hidden_recv_us"] < 1000


def test_rendezvous_needs_a_name(monkeypatch):
    with pytest.raises(RuntimeError, match="MORI_EP_SIM_GROUP"):
        EpDispatchCombineOp(_config(0, 8))
