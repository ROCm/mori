"""aiter MegaMoEV2 stage1 (and the whole layer) in the mode this process got: fixed-slot
(no compact plan) or compact. Which one is set by ``AITER_MEGA_FIXED_SLOT_MAX_MTPR`` at import
(fixed-slot needs 48 experts per rank and mtpr <= that value), so run it once per mode:

    AITER_MEGA_FIXED_SLOT_MAX_MTPR=2047 torchrun --nproc_per_node=8 aiter_mode_bench.py --bs 512
    torchrun --nproc_per_node=8 aiter_mode_bench.py --bs 512           # default 255: compact

stage1 = quant + dispatch + GEMM1, the part test_recv_layout.py compares:
  fixed-slot: quantize() then _run_fused_stage1()
  compact:    _run_compact_prepare(quant_input=x) then _run_fused_stage1(prepared=True)
Each timed per call (sync + barrier between calls), eager and in a CUDA graph. ``--save``
writes the layer's output per rank, ``--compare`` diffs this run's against a saved one.
"""

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "11_flydsl_mega_moe_sdma")
)

import main as ex11  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=ex11.NETWORKS, default="flash")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--save", default="")
    ap.add_argument("--compare", default="")
    args = ap.parse_args()

    rank, world, device = ex11._setup()
    net, moe = ex11._build_moe(args, args.bs, rank, world, device)
    fixed = bool(moe._s1_fixed_slot)
    x, wts, ids = ex11._make_inputs(
        args.bs, net["model_dim"], net["experts"], net["topk"], rank, args.seed, device
    )
    out = moe(x, wts, ids)[: args.bs].clone()
    moe(x, wts, ids)
    torch.cuda.synchronize()
    config = moe._active_config

    def stage1():
        fs = fx.Stream(torch.cuda.current_stream().cuda_stream)
        if fixed:
            xq, sc = moe.quantize(x)
            moe._run_fused_stage1(xq, wts, sc, ids, stream=fs, config=config.stage1)
        else:
            moe._run_compact_prepare(ids, args.bs, config.stage1, fs, quant_input=x)
            moe._run_fused_stage1(
                moe._s1_quant_x[: args.bs], wts, moe._s1_quant_scale[: args.bs], ids,
                stream=fs, config=config.stage1, prepared=True,
            )

    def layer():
        moe(x, wts, ids)

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def timed(fn):
        t = []
        for it in range(args.iters + 3):
            torch.cuda.synchronize()
            dist.barrier()
            ev[0].record()
            fn()
            ev[1].record()
            torch.cuda.synchronize()
            if it >= 3:
                t.append(ev[0].elapsed_time(ev[1]))
        return torch.tensor(t).median().item()

    def graphed(fn):
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        return g

    r = [timed(stage1), timed(layer)]
    g1, g2 = graphed(stage1), graphed(layer)
    r += [timed(g1.replay), timed(g2.replay)]
    del g1, g2

    if args.save:
        torch.save(out.cpu(), f"{args.save}_r{rank}.pt")
    diff = torch.zeros(2, dtype=torch.float64)
    if args.compare:
        other = torch.load(f"{args.compare}_r{rank}.pt").to(device)
        d = (out.float() - other.float()).abs()
        diff[0] = float(torch.equal(out, other))
        diff[1] = float(d.max())
    med = torch.tensor(r, dtype=torch.float64)
    dist.all_reduce(med)
    med /= world
    eq, mx = diff[0:1].clone(), diff[1:2].clone()
    dist.all_reduce(eq)
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    diff = torch.cat([eq, mx])
    if rank == 0:
        s_e, l_e, s_g, l_g = med.tolist()
        mode = "fixed-slot" if fixed else "compact"
        line = (
            f"AITER-MODE {args.network} bs={args.bs} mtpr={moe.mtpr} {mode} "
            f"(sbm={config.stage1.sort_block_m}): stage1 eager {s_e:.3f} graph {s_g:.3f} ms | "
            f"layer eager {l_e:.3f} graph {l_g:.3f} ms"
        )
        if args.compare:
            line += f" | output == {args.compare}: {int(diff[0])}/{world} ranks, max |diff| {diff[1]:.3g}"
        print(line, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
