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
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--scale-dim", type=int, default=0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--drop-rounds", type=int, default=1)
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
    return inp, idx, wts


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
    rng = torch.Generator(device=dev)
    rng.manual_seed(4242 + d.rank)
    ct = a.max_tokens
    inp, idx, wts = _gen_round(rng, cfg, ct, dev, cfg.dispatch_dtype)
    if a.kernel_type is not None:
        op._internode_force_ll = a.kernel_type == "v1_ll"

    # The examples harness's _convert_for_combine: the combine leg reads its
    # input as its own element type, so an asymmetric config has to cast first.
    def convert(x):
        return x.to(cfg.combine_dtype) if cfg.is_asymmetric_dtype else x

    for _ in range(a.warmup):
        r = op.dispatch(inp, wts, None, idx, return_routing=True)
        op.combine(convert(r[0]), wts, routing=r[5])
    torch.cuda.synchronize()
    comm.barrier()

    n = a.rounds
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3 * n + 1)]
    t0 = time.perf_counter()
    ev[0].record()
    for i in range(n):
        r = op.dispatch(inp, wts, None, idx, return_routing=True)
        ev[3 * i + 1].record()
        x = convert(r[0])
        ev[3 * i + 2].record()
        op.combine(x, wts, routing=r[5])
        ev[3 * i + 3].record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e6 / n

    keep = slice(a.drop_rounds, None)
    disp = [ev[3 * i].elapsed_time(ev[3 * i + 1]) * 1e3 for i in range(n)][keep]
    comb = [ev[3 * i + 2].elapsed_time(ev[3 * i + 3]) * 1e3 for i in range(n)][keep]
    dm = sum(disp) / len(disp)
    cm = sum(comb) / len(comb)
    dm = d.allreduce_sum(int(dm * 1000)) / d.world / 1000
    cm = d.allreduce_sum(int(cm * 1000)) / d.world / 1000
    if d.rank == 0:
        print(
            f"# BENCH tok={ct} dtype={a.dtype}->{a.combine_dtype or a.dtype} "
            f"hidden={cfg.hidden_dim} topk={cfg.num_experts_per_token} "
            f"kernel={a.kernel_type or 'auto'} "
            f"dispatch={dm:.1f}us combine={cm:.1f}us total={dm + cm:.1f}us "
            f"[wall={wall:.1f}us]",
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
            scale_type_size=1 if a.scale_dim else 0,
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
            inp, idx, wts = _gen_round(rng, cfg, ct, dev, cfg.dispatch_dtype)

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
