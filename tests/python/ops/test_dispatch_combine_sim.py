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
"""Simulated EP dispatch/combine (MORI_EP_SIM_ENABLE=1).

Standalone mode runs without a GPU. Runtime mode needs a simulated HIP runtime, e.g. Sundial's
virtual GPU (GPU-free): MORI_EP_SIM_RUNTIME_LIB=<libvgpu.so> pytest -k test_runtime_mode, or
LD_PRELOAD=<libvgpu.so> VGPU_NUM_DEVICES=4 python3 test_dispatch_combine_sim.py --runtime [--graph] [--v2]
(--v2, dispatch_combine_v2: also VGPU_VMM=1)
"""
import os
import socket
import subprocess
import sys
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
    "MORI_EP_SIM_MODE": "standalone",
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


def test_runtime_mode_needs_a_simulated_runtime(monkeypatch):
    from mori import cpp
    from mori.ops.dispatch_combine import ep_sim_mode

    if cpp.ep_sim_runtime_available():
        pytest.skip("a simulated runtime is loaded")
    monkeypatch.delenv("MORI_EP_SIM_MODE")
    assert ep_sim_mode() == "standalone"
    monkeypatch.setenv("MORI_EP_SIM_MODE", "runtime")
    with pytest.raises(RuntimeError, match="simulated HIP runtime"):
        ep_sim_mode()


def test_runtime_mode():
    lib = os.environ.get("MORI_EP_SIM_RUNTIME_LIB")
    if not lib:
        pytest.skip("MORI_EP_SIM_RUNTIME_LIB not set")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MORI_EP_SIM")}
    for extra in ([], ["--graph"], ["--v2"], ["--v2", "--graph"]):
        r = subprocess.run(
            [sys.executable, __file__, "--runtime"] + extra,
            # VGPU_VMM: libvgpu's virtual memory API, which mori.cco (v2) needs
            env=dict(env, LD_PRELOAD=lib, VGPU_VMM="1" if "--v2" in extra else "0"),
            capture_output=True,
            text=True,
            timeout=1200,
        )
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


RT_WORLD, RT_EPR, RT_MAX_TOK, RT_HIDDEN = 4, 8, 64, 512


def _rt_op(rank, version):
    if version == 1:
        import mori

        mori.shmem.shmem_torch_process_group_init("default")
        return EpDispatchCombineOp(
            EpDispatchCombineConfig(
                data_type=torch.bfloat16,
                rank=rank,
                world_size=RT_WORLD,
                hidden_dim=RT_HIDDEN,
                scale_dim=0,
                scale_type_size=4,
                max_token_type_size=2,
                max_num_inp_token_per_rank=RT_MAX_TOK,
                num_experts_per_rank=RT_EPR,
                num_experts_per_token=TOPK,
                gpu_per_node=RT_WORLD,
            )
        )
    from mori.cco import Communicator
    from mori.ops import dispatch_combine_v2 as v2

    uid = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = Communicator.init(RT_WORLD, rank, uid[0], per_rank_vmm=1 << 30)
    op = v2.EpDispatchCombineOp(
        v2.EpDispatchCombineConfig(
            rank=rank,
            world_size=RT_WORLD,
            hidden_dim=RT_HIDDEN,
            max_num_inp_token_per_rank=RT_MAX_TOK,
            num_experts_per_rank=RT_EPR,
            num_experts_per_token=TOPK,
            data_type=torch.bfloat16,
            combine_mode="gather",
        ),
        comm,
    )
    comm.barrier()
    return op


def _rt_rank(rank, port, graph, version):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        MORI_EP_SIM_ENABLE="1",
        MORI_EP_SIM_MODE="runtime",
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=RT_WORLD)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    import mori
    from mori.tensor_utils import from_gpu_ptr

    op = _rt_op(rank, version)
    dev = torch.device("cuda", rank)
    gen = torch.Generator().manual_seed(rank)
    fails = []

    def expect(name, ok):
        if not ok:
            fails.append(name)

    def check(tag, out, idx, eager):
        """Every output against the routing all ranks drew (into their top-k buffers)."""
        mine = idx.cpu().tolist()
        expect(
            f"{tag} drawn routing",
            all(
                len(set(r)) == TOPK and 0 <= min(r) <= max(r) < RT_WORLD * RT_EPR
                for r in mine
            ),
        )
        routing = [None] * RT_WORLD
        dist.all_gather_object(routing, mine)
        rows = [
            r for ids in routing for r in ids if any(e // RT_EPR == rank for e in r)
        ]
        src = [
            (j, t)
            for j, ids in enumerate(routing)
            for t, r in enumerate(ids)
            if any(e // RT_EPR == rank for e in r)
        ]
        expect(f"{tag} out_indices", out[3][: len(rows)].cpu().tolist() == rows)
        if version == 1:
            lec = [sum(r.count(rank * RT_EPR + i) for r in rows) for i in range(RT_EPR)]
            expect(f"{tag} lec", op.local_expert_count.cpu().tolist() == lec)
        if eager:  # (a v1 graph's combine has reset total_recv)
            expect(f"{tag} total_recv", int(out[4].item()) == len(rows))
            if version == 1:
                ptr, n = mori.cpp.get_dispatch_src_token_pos(op._handle)
                stride = op.max_num_tokens_to_send()
            else:
                ptr, n = op.arena.local_ptr(op._region("recv_to_src_token")), len(rows)
                stride = RT_MAX_TOK
            flat = from_gpu_ptr(ptr, (max(n, 1),), torch.int32)[:n].cpu().tolist()
            expect(f"{tag} src map", [(f // stride, f % stride) for f in flat] == src)
        return mine

    def dispatch(x, w, idx):
        if version == 1:
            return op.dispatch(x, w, None, idx, call_local_expert_count=True)
        return op.dispatch(x, w, None, idx, return_routing=True)

    def combine(out, idx):
        if version == 1:
            return op.combine(out[0], None, idx)
        x = op.combine_in_view()[: out[0].shape[0]]
        return op.combine(x, None, out[3], routing=out[5])

    def inputs(n):
        return (
            torch.empty(n, RT_HIDDEN, dtype=torch.bfloat16, device=dev),
            torch.ones(n, TOPK, dtype=torch.float32, device=dev),
            torch.zeros(n, TOPK, dtype=torch.int32, device=dev),
        )

    for it in range(3):
        n = (
            0
            if (rank == 1 and it == 1)
            else int(torch.randint(1, RT_MAX_TOK + 1, (1,), generator=gen))
        )
        x, w, idx = inputs(n)
        if rank == 0 and it == 2:
            time.sleep(1.0)
        t0 = time.perf_counter()
        out = dispatch(x, w, idx)
        if rank != 0 and it == 2:  # rank 0 is late: launches do not wait for it
            expect("no wait at launch", time.perf_counter() - t0 < 0.2)
        check(f"eager{it}", out, idx, eager=True)
        combine(out, idx)
        torch.cuda.synchronize()
        if version == 1:
            expect(f"eager{it} combine resets total_recv", int(out[4].item()) == 0)

    if graph:
        x, w, idx = inputs(16)
        g, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                out = dispatch(x, w, idx)
                combine(out, idx)
        drawn = []
        for r in range(3):
            g.replay()
            torch.cuda.synchronize()
            drawn.append(check(f"replay{r}", out, idx, eager=False))
        expect("each replay draws new routing", drawn[0] != drawn[1] != drawn[2])
    dist.barrier()
    dist.destroy_process_group()
    assert not fails, fails


if __name__ == "__main__" and "--runtime" in sys.argv:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = (port, "--graph" in sys.argv, 2 if "--v2" in sys.argv else 1)
    mp.spawn(_rt_rank, args=args, nprocs=RT_WORLD)
    print("runtime mode OK")
