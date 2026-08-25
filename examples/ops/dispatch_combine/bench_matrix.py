# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Repeated-trial benchmark harness for EP dispatch/combine.

Purpose: the stock ``--cmd bench`` reports a single 10-round grand mean, which
is not enough to separate a real effect from run-to-run noise at small token
counts.  This harness runs T independent trials per config, reports the median
and IQR across trials, and lets dispatch and combine use *different* launch
geometries so their coupling can be measured directly.

Reuses EpDispatchCombineTestCase from the stock example (imported as a module,
so argv is staged before import).

Usage (per node, launched by torchrun with --nproc_per_node=1):
    RANK=<node_rank> python bench_matrix.py --matrix <json> --trials 5 ...
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_own = argparse.ArgumentParser(add_help=False)
_own.add_argument("--matrix", type=str, required=True)
_own.add_argument("--trials", type=int, default=5)
_own.add_argument("--rounds", type=int, default=10)
_own.add_argument("--barrier-per-round", action="store_true")
_own.add_argument("--max-tokens", type=int, default=4)
_own.add_argument("--hidden-dim", type=int, default=6144)
_own.add_argument("--kernel-type", type=str, default="v1_ll")
_own.add_argument("--num-qp", type=int, default=2)
_own.add_argument("--dtype", type=str, default="fp8_e4m3_fnuz")
_own.add_argument("--combine-dtype", type=str, default="bf16")
_own.add_argument("--quant-type", type=str, default="none")
_own.add_argument("--out", type=str, default=None)
_own.add_argument("--label", type=str, default="")
_own.add_argument(
    "--cpu-profile",
    action="store_true",
    help="cProfile the host-side enqueue loop.",
)
_own.add_argument(
    "--no-batch-launch",
    action="store_true",
    help="Expand _launch_multi into individual _launch calls (no events). The "
    "delta vs normal measures what kernel-launch batching is worth, i.e. the "
    "headroom available from merging kernels.",
)
_own.add_argument(
    "--profile-kernels",
    action="store_true",
    help="Time each individual kernel launch (dispatch and combine are several "
    "kernels each; the phase totals alone cannot say which one dominates).",
)
OWN, _rest = _own.parse_known_args()

# Stage argv for the stock example's module-level parse_args().  Restored right
# after the import: torch's spawn start method snapshots sys.argv and replays it
# in the children, which re-run this file and need the real flags back.
_REAL_ARGV = list(sys.argv)
sys.argv = [
    "test_dispatch_combine_internode.py",
    "--cmd", "bench",
    "--kernel-type", OWN.kernel_type,
    "--num-qp", str(OWN.num_qp),
    "--max-tokens", str(OWN.max_tokens),
    "--hidden-dim", str(OWN.hidden_dim),
    "--dtype", OWN.dtype,
    "--combine-dtype", OWN.combine_dtype,
    "--quant-type", OWN.quant_type,
    "--skip-verify",
]

sys.path.insert(0, str(_HERE))
import test_dispatch_combine_internode as ex  # noqa: E402

sys.argv = _REAL_ARGV

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def _timed_trial(case, op, test_data, rounds, disp_cfg, comb_cfg, barrier_per_round):
    """One trial: `rounds` back-to-back dispatch+combine, returns (disp_us, comb_us) lists.

    Events are recorded per phase so a dispatch measurement never spans the
    previous combine (unlike the stock harness, where events[3i] is the prior
    combine's end event).
    """
    all_rank_num_token, all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales = test_data
    r = case.rank

    ev_d0 = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    ev_d1 = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    ev_c0 = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    ev_c1 = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]

    torch.cuda.synchronize()
    dist.barrier()

    _t0 = time.perf_counter()
    for i in range(rounds):
        if barrier_per_round:
            torch.cuda.synchronize()
            dist.barrier()
        ev_d0[i].record()
        disp = case.run_dispatch(
            op, all_rank_input[r], all_rank_weights[r], all_rank_scales[r],
            all_rank_indices[r],
            block_num=disp_cfg[0], warp_per_block=disp_cfg[1], rdma_block_num=disp_cfg[2],
        )
        ev_d1[i].record()
        combine_input = case._convert_for_combine(disp[0])
        ev_c0[i].record()
        case.run_combine(
            op, combine_input, None, all_rank_indices[r],
            block_num=comb_cfg[0], warp_per_block=comb_cfg[1], rdma_block_num=comb_cfg[2],
        )
        ev_c1[i].record()
    enqueue_wall_us = (time.perf_counter() - _t0) * 1e6
    torch.cuda.synchronize()
    total_wall_us = (time.perf_counter() - _t0) * 1e6

    disp_us = [ev_d0[i].elapsed_time(ev_d1[i]) * 1000.0 for i in range(rounds)]
    comb_us = [ev_c0[i].elapsed_time(ev_c1[i]) * 1000.0 for i in range(rounds)]
    _timed_trial.last_enqueue_us = enqueue_wall_us / rounds
    _timed_trial.last_wall_us = total_wall_us / rounds
    return disp_us, comb_us


def _gather_mean(case, value):
    """All-gather a scalar and return the mean across ranks (slowest-rank view uses max)."""
    local = torch.tensor([value], dtype=torch.float64)
    out = [torch.zeros(1, dtype=torch.float64) for _ in range(case.world_size)]
    dist.all_gather(out, local)
    vals = [o.item() for o in out]
    return sum(vals) / len(vals), max(vals)


def _profile_kernels(case, op, test_data, matrix, global_rank):
    """Per-kernel timing: wrap op._launch so every kernel gets its own event pair.

    dispatch and combine are each several kernels on one stream, so a phase
    total cannot distinguish "the compute kernel is slow" from "we are sitting
    in a barrier kernel waiting for the slowest peer".
    """
    all_rank_num_token, all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales = test_data
    r = case.rank
    trace = []
    orig_launch = op._launch
    orig_launch_multi = op._launch_multi

    def _one(name, grid, block, shm, stream, args_ptr):
        # _launch_multi carries grid/block as bare ints; _launch wants tuples.
        g = (grid,) if isinstance(grid, int) else grid
        b = (block,) if isinstance(block, int) else block
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        orig_launch(name, g, b, shm, stream, args_ptr)
        e1.record()
        trace.append((name, g, e0, e1))

    def traced(name, grid, block, shm, stream, args_ptr):
        _one(name, grid, block, shm, stream, args_ptr)

    # v1_ll issues its kernels through _launch_multi (one batched call), so the
    # group must be expanded into individual launches to attribute time.
    def traced_multi(names, grids, blocks, shms, stream, args_ptr):
        for n, g, b, s in zip(names, grids, blocks, shms):
            _one(n, g, b, s, stream, args_ptr)

    for entry in matrix:
        disp_cfg = tuple(entry["disp"])
        comb_cfg = tuple(entry["comb"])

        # warm up this geometry with the real launcher
        _timed_trial(case, op, test_data, 3, disp_cfg, comb_cfg, False)

        op._launch = traced
        op._launch_multi = traced_multi
        trace.clear()
        torch.cuda.synchronize()
        dist.barrier()
        rounds = OWN.rounds
        for _ in range(rounds):
            if OWN.barrier_per_round:
                torch.cuda.synchronize()
                dist.barrier()
            d = case.run_dispatch(
                op, all_rank_input[r], all_rank_weights[r], all_rank_scales[r],
                all_rank_indices[r],
                block_num=disp_cfg[0], warp_per_block=disp_cfg[1], rdma_block_num=disp_cfg[2],
            )
            case.run_combine(
                op, case._convert_for_combine(d[0]), None, all_rank_indices[r],
                block_num=comb_cfg[0], warp_per_block=comb_cfg[1], rdma_block_num=comb_cfg[2],
            )
        torch.cuda.synchronize()
        op._launch = orig_launch
        op._launch_multi = orig_launch_multi

        per_kernel = {}
        for name, grid, e0, e1 in trace:
            per_kernel.setdefault((name, grid), []).append(e0.elapsed_time(e1) * 1000.0)

        # slowest-rank view: max across ranks of this rank's median
        rows = []
        for (name, grid), xs in per_kernel.items():
            med = statistics.median(xs)
            local = torch.tensor([med], dtype=torch.float64)
            out = [torch.zeros(1, dtype=torch.float64) for _ in range(case.world_size)]
            dist.all_gather(out, local)
            vals = [o.item() for o in out]
            rows.append((name, grid, max(vals), sum(vals) / len(vals), min(vals)))

        if global_rank == 0:
            print(f"\n=== per-kernel [{entry['name']}] disp={disp_cfg} comb={comb_cfg} "
                  f"tokens={OWN.max_tokens} ===", flush=True)
            print(f"{'kernel':<52} {'grid':>7} {'slowest':>9} {'mean':>9} {'fastest':>9}", flush=True)
            total = 0.0
            for name, grid, mx, mean, mn in rows:
                total += mx
                print(f"{name:<52} {str(grid[0]):>7} {mx:9.2f} {mean:9.2f} {mn:9.2f}", flush=True)
            print(f"{'TOTAL (sum of slowest)':<52} {'':>7} {total:9.2f}", flush=True)


def main(local_rank, num_node, gpu_per_node):
    node_rank = int(os.environ["RANK"])
    global_rank = node_rank * gpu_per_node + local_rank
    world_size = num_node * gpu_per_node

    disp_dtype = ex._DATA_TYPE_MAP[OWN.dtype]
    comb_dtype = ex._DATA_TYPE_MAP[OWN.combine_dtype]

    case = ex.EpDispatchCombineTestCase(
        global_rank, gpu_per_node, world_size, OWN.max_tokens,
        OWN.kernel_type, OWN.num_qp, OWN.quant_type, disp_dtype,
        hidden_dim=OWN.hidden_dim, combine_dtype=comb_dtype,
    )
    case.setup()

    matrix = json.loads(OWN.matrix)
    op = ex.mori.ops.EpDispatchCombineOp(case.config)
    test_data = case.gen_test_data(
        max_num_token=OWN.max_tokens, use_max_token_num=True, only_my_rank=True
    )

    if OWN.cpu_profile:
        import cProfile, pstats, io as _io
        entry = matrix[0]
        dc, cc = tuple(entry["disp"]), tuple(entry["comb"])
        _timed_trial(case, op, test_data, 5, dc, cc, False)
        r = case.rank
        (_, ari, ari_in, arw, ars) = test_data

        def _loop():
            for _ in range(50):
                d = case.run_dispatch(op, ari_in[r], arw[r], ars[r], ari[r],
                                      block_num=dc[0], warp_per_block=dc[1], rdma_block_num=dc[2])
                ci = case._convert_for_combine(d[0])
                case.run_combine(op, ci, None, ari[r],
                                 block_num=cc[0], warp_per_block=cc[1], rdma_block_num=cc[2])

        pr = cProfile.Profile()
        torch.cuda.synchronize(); dist.barrier()
        pr.enable(); _loop(); pr.disable()
        torch.cuda.synchronize()
        if global_rank == 0:
            buf = _io.StringIO()
            pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(22)
            print("\n=== host-side cost per 50 dispatch+combine rounds ===")
            print(buf.getvalue(), flush=True)
        case.cleanup()
        return

    if OWN.profile_kernels:
        _profile_kernels(case, op, test_data, matrix, global_rank)
        case.cleanup()
        return

    if OWN.no_batch_launch:
        _orig_lm = op._launch_multi
        _orig_l = op._launch

        def _unbatched(names, grids, blocks, shms, stream, args_ptr):
            for n, g, b, sm in zip(names, grids, blocks, shms):
                _orig_l(n, (g,) if isinstance(g, int) else g,
                        (b,) if isinstance(b, int) else b, sm, stream, args_ptr)

        op._launch_multi = _unbatched

    results = []
    for entry in matrix:
        name = entry["name"]
        disp_cfg = tuple(entry["disp"])   # (block_num, warp_per_block, rdma_block_num)
        comb_cfg = tuple(entry["comb"])

        # warmup for this geometry
        _timed_trial(case, op, test_data, 3, disp_cfg, comb_cfg, False)

        trial_disp, trial_comb, trial_enq, trial_wall = [], [], [], []
        for _ in range(OWN.trials):
            d_us, c_us = _timed_trial(
                case, op, test_data, OWN.rounds, disp_cfg, comb_cfg, OWN.barrier_per_round
            )
            # per-trial: mean over rounds on this rank, then max across ranks
            _, d_slow = _gather_mean(case, statistics.mean(d_us))
            _, c_slow = _gather_mean(case, statistics.mean(c_us))
            trial_disp.append(d_slow)
            trial_comb.append(c_slow)
            trial_enq.append(_timed_trial.last_enqueue_us)
            trial_wall.append(_timed_trial.last_wall_us)

        if global_rank == 0:
            def _stat(xs):
                xs = sorted(xs)
                return {
                    "median": round(statistics.median(xs), 2),
                    "min": round(xs[0], 2),
                    "max": round(xs[-1], 2),
                    "spread_pct": round((xs[-1] - xs[0]) / statistics.median(xs) * 100, 1),
                    "all": [round(x, 2) for x in xs],
                }
            rec = {
                "name": name, "disp": list(disp_cfg), "comb": list(comb_cfg),
                "dispatch_us": _stat(trial_disp), "combine_us": _stat(trial_comb),
            }
            results.append(rec)
            print(
                f"[{name}] disp={disp_cfg} comb={comb_cfg}\n"
                f"    dispatch  median={rec['dispatch_us']['median']:8.2f} us  "
                f"spread={rec['dispatch_us']['spread_pct']:5.1f}%  {rec['dispatch_us']['all']}\n"
                f"    cpu-enqueue/round={statistics.median(trial_enq):7.2f} us   "
                f"wall/round={statistics.median(trial_wall):7.2f} us\n"
                f"    combine   median={rec['combine_us']['median']:8.2f} us  "
                f"spread={rec['combine_us']['spread_pct']:5.1f}%  {rec['combine_us']['all']}",
                flush=True,
            )

    if global_rank == 0 and OWN.out:
        with open(OWN.out, "w") as f:
            json.dump({"label": OWN.label, "max_tokens": OWN.max_tokens,
                       "hidden_dim": OWN.hidden_dim, "results": results}, f, indent=2)
        print(f"\nwrote {OWN.out}", flush=True)

    case.cleanup()


if __name__ == "__main__":
    gpu_per_node = int(os.environ.get("GPU_PER_NODE", 8))
    num_node = int(os.environ["WORLD_SIZE"])
    torch.multiprocessing.spawn(
        main, args=(num_node, gpu_per_node), nprocs=gpu_per_node, join=True
    )
