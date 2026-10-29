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
"""LL128 calls next to the schedule's on one op (hip backend, gfx1250 intranode).

An op whose capacity is past the LL128 table's max_tok (hip_backend.ll128_layout) runs its
calls of at most max_tok tokens a rank on LL128 plans with regions of their own, and the
rest on the schedule's. One sequence of call sizes crosses max_tok both ways, and runs

- synced, a barrier after every dispatch and combine: each call takes the plans its size
  says, delivers every source row, id, weight and scale row to the slot its reverse map
  names (src_rank * M + src_tok on both plans), and a bf16 combine returns U * x;
- back to back, eager and then replaying one graph a call size;
- as snapshot handles, two dispatches before both combines, for each pairing of the sides,
  a barrier after every call as test_routing_snapshot.py has: a peer's next schedule
  dispatch writes into this rank's receive rows, and its next schedule combine restages
  the rows this rank reads. Gather ops only: every push-quant op scatters, and dispatch
  rejects snapshot=True there.

Every combine of the last two equals the synced one bit for bit. The expert is the identity
on the received rows (fp8 / fp4 decoded to bf16), so a combine row is U copies of one row,
U the token's distinct destination ranks.

    torchrun --standalone --nproc_per_node=4 test_ll128_split.py
    QUANTS=fp4_blockwise M=1024 torchrun --standalone --nproc_per_node=8 test_ll128_split.py
"""
import os
import sys

import torch
import torch.distributed as dist

import mori.cco as cco
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp
from mori.ops.dispatch_combine_v2.hip_backend import ll128_layout

HIDDEN = 7168
TOPK = 6
EPR = int(os.environ.get("EPR", 8))
M = int(os.environ.get("M", 512))
_CASES = {
    "none": ("none", torch.bfloat16, 0, 1),
    "none_fp8": ("none", torch.float8_e4m3fn, HIDDEN // 128, 4),
    "fp8_blockwise": ("fp8_blockwise", torch.float8_e4m3fn, HIDDEN // 32, 1),
    "fp4_blockwise": ("fp4_blockwise", torch.float4_e2m1fn_x2, HIDDEN // 32, 1),
}
QUANTS = os.environ.get("QUANTS", ",".join(_CASES)).split(",")
ROUNDS = int(os.environ.get("ROUNDS", 2))
_SCALE_DTYPE = {1: torch.uint8, 4: torch.float32}
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def make_cfg(case, rank, world):
    q, dt, scale_dim, scale_size = _CASES[case]
    bf16 = dt is torch.bfloat16
    return EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=HIDDEN,
        max_num_inp_token_per_rank=M,
        num_experts_per_rank=EPR,
        num_experts_per_token=TOPK,
        data_type=torch.bfloat16,
        dispatch_data_type=None if bf16 else dt,
        combine_data_type=None if bf16 else torch.bfloat16,
        scale_dim=scale_dim,
        scale_type_size=scale_size,
        kernel_backend="hip",
        quant_type=q,
    )


def call_sizes(lim):
    once = [1, M, lim, lim + 1, 2, M, lim, 1, lim + 1, (lim + 1 + M) // 2, lim]
    return once * ROUNDS


def make_inputs(cfg, i, src, n):
    """Call i's tokens, ids, weights and scale rows on rank src, the same wherever made."""
    g = torch.Generator().manual_seed(7919 * i + src)
    dt = cfg.dispatch_dtype
    if dt is torch.float4_e2m1fn_x2:
        x = torch.randint(0, 256, (n, HIDDEN // 2), generator=g, dtype=torch.uint8)
        x = x.view(dt)
    else:
        x = torch.randn(n, HIDDEN, generator=g).to(dt)
    idx = torch.stack(
        [torch.randperm(cfg.world_size * EPR, generator=g)[:TOPK] for _ in range(n)]
    ).to(torch.int32)
    w = torch.rand(n, TOPK, generator=g) + 0.5
    s = None
    if cfg.scale_dim:
        width = cfg.scale_dim * cfg.scale_type_size
        s = torch.randint(0, 256, (n, width), generator=g, dtype=torch.uint8)
        s = s.view(_SCALE_DTYPE[cfg.scale_type_size])
    return x, idx, w, s


def copy_of(t):
    return None if t is None else t.view(torch.uint8).clone().view(t.dtype)


def expert(recv, lut):
    """The identity on the received rows, as the bf16 rows combine takes."""
    if recv.dtype is torch.bfloat16:
        return recv
    if recv.dtype is torch.float4_e2m1fn_x2:
        b = recv.view(torch.uint8).long()
        return torch.stack((lut[b & 15], lut[b >> 4]), -1).view(recv.shape[0], HIDDEN)
    return recv.to(torch.bfloat16)


def run(case, cfg, comm, dev, lut, lim, sizes, args):
    rank, world = cfg.rank, cfg.world_size
    push = cfg.quant_type != "none"
    op = EpDispatchCombineOp(cfg, comm)
    checks = {}

    def check(name, ok):
        checks[name] = checks.get(name, True) and bool(ok)

    def sync():
        torch.cuda.synchronize()
        comm.barrier()

    def rows(n):
        return world * (lim if n <= lim else M)

    def sources(i, n):
        return [make_inputs(cfg, i, src, n) for src in range(world)]

    def keys(srcs):
        return sorted(
            src * M + t
            for src, (_, idx, _, _) in enumerate(srcs)
            for t in ((idx.long() // EPR) == rank).any(1).nonzero().flatten().tolist()
        )

    def delivered(srcs, rev, out, oidx, ow, osc):
        sel = [(int(v) // M, int(v) % M) for v in rev]
        if not sel:
            return True
        got = [out, oidx, ow] + ([osc] if cfg.scale_dim else [])
        for k, t in enumerate(got):
            want = torch.stack([srcs[a][k][b].view(torch.uint8) for a, b in sel])
            t = t[: len(sel)].view(torch.uint8).cpu()
            if not torch.equal(t[:, : want.shape[1]], want):
                return False
        return True

    refs = []
    for i, n in enumerate(sizes):
        x, idx, w, s = args[i]
        out, ow, osc, oidx, total, h = op.dispatch(x, w, s, idx, return_routing=True)
        sync()
        tot = int(total.item())
        rev = h.disp_tok_id_to_src_tok_id_local[:tot].cpu()
        srcs = sources(i, n)
        check("synced_path_by_size", out.shape[0] == rows(n))
        check("synced_reverse_map", sorted(rev.tolist()) == keys(srcs))
        check("synced_delivery", delivered(srcs, rev, out, oidx, ow, osc))
        o, _ = op.combine(expert(out, lut), routing=h)
        sync()
        refs.append(o.clone())
        if not push:
            mine = srcs[rank]
            u = torch.zeros(n, world).scatter_(1, mine[1].long() // EPR, 1.0)
            want = u.sum(1, keepdim=True) * mine[0].float()
            check(
                "synced_bf16_sum",
                torch.allclose(o.float().cpu(), want, atol=2e-2, rtol=2e-2),
            )

    def pair(i):
        x, idx, w, s = args[i]
        out, _, _, _, _, h = op.dispatch(x, w, s, idx, return_routing=True)
        o, _ = op.combine(expert(out, lut), routing=h)
        return o.clone()

    got = [pair(i) for i in range(len(sizes))]
    sync()
    check("eager_back_to_back", all(map(torch.equal, got, refs)))

    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for i, n in enumerate(sizes):
        if n in graphs:
            continue
        x, idx, w, s = st = tuple(copy_of(t) for t in args[i])
        so = torch.empty(n, HIDDEN, dtype=torch.bfloat16, device=dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            out, _, _, _, _, h = op.dispatch(x, w, s, idx, return_routing=True)
            o, _ = op.combine(expert(out, lut), routing=h)
            so.copy_(o)
        graphs[n] = (g, st, so)
    sync()
    got = []
    for i, n in enumerate(sizes):
        g, st, so = graphs[n]
        for d, t in zip(st, args[i]):
            if d is not None:
                d.view(torch.uint8).copy_(t.view(torch.uint8))
        g.replay()
        got.append(so.clone())
    sync()
    check("graph_back_to_back", all(map(torch.equal, got, refs)))

    small = [i for i, n in enumerate(sizes) if n <= lim]
    large = [i for i, n in enumerate(sizes) if n > lim]
    pairs = (small[:2], (small[0], large[0]), (large[0], small[0]), large[:2])
    for a, b in () if push else pairs:
        side = "_".join("ll128" if sizes[i] <= lim else "sched" for i in (a, b))
        ins, handles = [], []
        for i in (a, b):
            x, idx, w, s = args[i]
            out, _, _, _, _, h = op.dispatch(
                x, w, s, idx, return_routing=True, snapshot=True
            )
            check(f"snapshot_path_{side}", out.shape[0] == rows(sizes[i]))
            ins.append(expert(out, lut).clone())
            handles.append(h)
            sync()
        for i, y, h in zip((a, b), ins, handles):
            o = op.combine(y, routing=h)[0].clone()
            sync()
            check(f"snapshot_combine_{side}", torch.equal(o, refs[i]))
        ha = handles[0]
        if ha.recv_maps is not None:
            tot = int(ha.total_recv_token_num.item())
            rev = ha.disp_tok_id_to_src_tok_id_local[:tot].cpu()
            want = keys(sources(a, sizes[a]))
            check(f"snapshot_own_reverse_map_{side}", sorted(rev.tolist()) == want)

    op.close()
    failures = 0
    for name, ok in checks.items():
        bad = torch.tensor([0 if ok else 1], dtype=torch.int32)
        dist.all_reduce(bad)
        failures += int(bad.item() != 0)
        if rank == 0:
            verdict = "PASS" if bad.item() == 0 else "FAIL"
            print(f"# LL128SPLIT[{case}] {name}: {verdict}", flush=True)
    return failures


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    lut = torch.tensor(_E2M1 + [-v for v in _E2M1], dtype=torch.bfloat16, device=dev)

    # Inputs before the communicator: comm_create leaves a HIP error latched that
    # the next torch GPU call would report (see test_ep_backend_parity.py).
    plans = []
    for q in QUANTS:
        cfg = make_cfg(q, rank, world)
        lay = ll128_layout(cfg)
        if lay is None or lay["max_tok"] >= M:
            if rank == 0:
                print(f"# LL128SPLIT[{q}]: SKIP (no LL128 calls below M={M})")
            continue
        lim = lay["max_tok"]
        sizes = call_sizes(lim)
        args = [
            tuple(
                None if t is None else t.to(dev) for t in make_inputs(cfg, i, rank, n)
            )
            for i, n in enumerate(sizes)
        ]
        plans.append((q, cfg, lim, sizes, args))
        if rank == 0:
            print(f"# LL128SPLIT[{q}] EP{world} M={M} max_tok={lim} sizes={sizes}")
    torch.cuda.synchronize()

    obj = [cco.Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    vmm = 2 * (world * M * HIDDEN * 2 * 2 + (16 << 20)) + (1 << 30)
    comm = cco.Communicator.init(world, rank, obj[0], vmm)

    failures = 0
    for q, cfg, lim, sizes, args in plans:
        failures += run(q, cfg, comm, dev, lut, lim, sizes, args)

    if rank == 0:
        print(f"# total ll128 split failures: {failures}", flush=True)
    comm.destroy()
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
