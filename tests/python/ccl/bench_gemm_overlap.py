#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# the benchmark (THE point of SDMA: compute/comm overlap).
#
# Runs a GEMM loop on a dedicated COMPUTE stream concurrently with an AllGather
# and measures the OVERLAPPED TOTAL time (gemm + AG), for:
#   (a) RCCL  all_gather_into_tensor  — steals CUs from the GEMM
#   (b) mori hier SDMA AllGather      — copy engines, no CU contention
# Because mori moves intra-node bytes on the SDMA copy engines (XGMI) instead of
# CUs, the GEMM keeps its CUs and the overlapped total time must be STRICTLY
# LOWER than RCCL's at the sizes where the AG would otherwise contend.
#
# Per size the GEMM iteration count is auto-tuned so the GEMM solo time ~ the AG
# solo time, i.e. the two genuinely overlap (so contention, if any, is visible).
# Bit-exact vs torch is asserted per size (zero tolerance) to keep correctness
# green inside the perf harness. rank 0 emits logs/sweep_gemm_overlap.csv.
#
# Launch:  bash scripts/build_and_test.sh C xnode tests/python/ccl/bench_gemm_overlap.py
import argparse
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import mori.shmem as shmem  # noqa: E402
from mori.ccl import HierAllGather  # noqa: E402

_OUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
_LOGS_DIR = os.path.join(_OUT_ROOT, "logs")
_DEFAULT_SIZES_MB = [4, 8, 16, 32, 64, 128, 256, 512]


def _dtype_of(name):
    return {"fp32": torch.float32, "bf16": torch.bfloat16,
            "fp16": torch.float16}[name]


def _make_input(dtype, numel, rank, device):
    base = torch.arange(numel, device=device, dtype=torch.float32)
    return (base + rank * 131.0).to(dtype)


def _host_time(fn, reps, warmup):
    """Wall time of `fn` bracketed by full device sync — captures work enqueued
    on ANY stream (the compute stream + the AG stream both finish before the
    second sync), so it is the true OVERLAPPED total."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts), sum(ts) / len(ts)


def _bench_size(handle, dtype, numel, rank, world_size, device, reps, warmup,
                gemm_n, compute_stream, gemm_dtype, gemm_iters_fixed):
    inp = _make_input(dtype, numel, rank, device)
    out_mori = torch.empty(numel * world_size, dtype=dtype, device=device)
    out_ref = torch.empty(numel * world_size, dtype=dtype, device=device)
    main_stream = torch.cuda.current_stream()

    # GEMM operands. Default bf16 so the matmul maps to the MFMA matrix cores:
    # that path is COMPUTE(CU)-bound with low HBM traffic (arithmetic intensity
    # ~gemm_n/3 flop/byte), which is the textbook compute/comm-overlap scenario. It isolates
    # the SDMA advantage cleanly: RCCL's AG steals CUs from the GEMM, while
    # mori's SDMA moves intra-node bytes on the copy engines and leaves the CUs
    # (and thus the GEMM) untouched. An fp32 GEMM instead contends on HBM
    # bandwidth (it does not use the fast matrix cores), which masks the
    # CU-freeing advantage — that was why T46 saw 32/64MB regress.
    a = torch.randn(gemm_n, gemm_n, device=device, dtype=gemm_dtype)
    b = torch.randn(gemm_n, gemm_n, device=device, dtype=gemm_dtype)

    # ---- correctness gate (zero tolerance) ----
    dist.all_gather_into_tensor(out_ref, inp)
    assert handle(inp, out_mori, numel, main_stream), "HierAllGather failed"
    main_stream.synchronize()
    torch.cuda.synchronize()
    bx = bool(torch.equal(out_mori, out_ref))
    if not bx:
        raise AssertionError(f"bit-exact MISMATCH dtype={dtype} numel={numel}")

    def rccl_ag():
        dist.all_gather_into_tensor(out_ref, inp)

    def mori_ag():
        assert handle(inp, out_mori, numel, main_stream)

    # ---- solo AG times (no overlap) to size the GEMM loop ----
    def solo(ag):
        ag(); main_stream.synchronize()
    r_solo, _ = _host_time(lambda: solo(rccl_ag), reps, warmup)
    m_solo, _ = _host_time(lambda: solo(mori_ag), reps, warmup)

    # one-GEMM time to pick iters so gemm_solo ~ max(AG solo).
    def one_gemm():
        torch.matmul(a, b)
    g1_min, _ = _host_time(one_gemm, max(reps, 3), warmup)
    if gemm_iters_fixed > 0:
        # Fixed-compute mode: model a real layer whose GEMM work does NOT scale
        # with the AG size. At small AG sizes the auto-tuned loop collapses to
        # iters~1 (the AG solo time is tiny), so there is no compute on the CUs
        # for RCCL's AG to steal from and the total just measures AG launch
        # latency (where mori's fixed overhead loses). A fixed, meaningful GEMM
        # is the textbook compute/comm-overlap case: the same CU load is present at EVERY AG
        # size, so RCCL's CU-stealing AG contends while mori's copy-engine SDMA
        # does not — making the no-CU-contention advantage visible at 4/8MB too.
        iters = gemm_iters_fixed
    else:
        target_ms = max(r_solo, m_solo)
        iters = max(1, int(round(target_ms / max(g1_min, 1e-3))))

    def gemm_loop():
        with torch.cuda.stream(compute_stream):
            for _ in range(iters):
                torch.matmul(a, b)

    # ---- overlapped totals: GEMM loop (compute stream) + AG (main stream) ----
    def overlap(ag):
        compute_stream.wait_stream(main_stream)
        gemm_loop()
        ag()
        main_stream.wait_stream(compute_stream)

    rccl_tot, _ = _host_time(lambda: overlap(rccl_ag), reps, warmup)
    mori_tot, _ = _host_time(lambda: overlap(mori_ag), reps, warmup)
    return rccl_tot, mori_tot, r_solo, m_solo, iters, bx


def _worker(rank, world_size, ranks_per_node, device, sizes_mb, dtypes, reps,
            warmup, gemm_n, gemm_dtype, gemm_iters_fixed):
    max_bytes = max(sizes_mb) * 1024 * 1024
    per_rank_bytes = max_bytes + 4096
    need = per_rank_bytes * world_size * 3 + per_rank_bytes + (1 << 28)
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", str(need))
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank
    # VALIDATED-NEGATIVE -- host-blocking barriers do NOT enable
    # GEMM overlap. Hypothesis was: the default ON-DEVICE spin barriers
    # (stream_ring/stream_intra) busy-wait on CUs during cross-node arrivals,
    # contending with the GEMM, so switching to HOST-BLOCKING barriers
    # (MORI_HIER_STREAM_RING=0 / STREAM_INTRA=0) would free the CUs. Measured A/B
    # (true xnode, 16/64/256MB) REFUTED it: host-sync mori_total got WORSE
    # (64MB 9.89ms vs default 7.25ms) AND still no overlap (total ~= gemm+solo).
    # Root cause is deeper: the inter-node phase is CU-DRIVEN RDMA (GPU threads
    # post/poll WQEs) and dominates at >=16MB, so the AG occupies CUs regardless
    # of barrier mechanism; a full-occupancy GEMM then serializes against it.
    # SDMA's copy-engine (no-CU) advantage only covers the INTRA phase, a minority
    # of cross-node cost at large sizes. So the default device-barrier path is the
    # honest overlap config. Set MORI_HIER_OVERLAP_HOSTSYNC=1 to re-run the A/B.
    if os.environ.get("MORI_HIER_OVERLAP_HOSTSYNC", "0") not in ("0", "false", "False"):
        os.environ["MORI_HIER_STREAM_RING"] = "0"
        os.environ["MORI_HIER_STREAM_INTRA"] = "0"
    handle = HierAllGather(
        my_pe=rank, npes=world_size, ranks_per_node=ranks_per_node,
        input_buffer_size=per_rank_bytes,
        output_buffer_size=per_rank_bytes * world_size,
        copy_output_to_user=True,
    )
    compute_stream = torch.cuda.Stream()
    if rank == 0:
        print(f"[gemm-ovlp] world={world_size} rpn={ranks_per_node} "
              f"num_nodes={handle.num_nodes} sizes_mb={sizes_mb} "
              f"gemm_n={gemm_n} gemm_dtype={gemm_dtype} reps={reps}")
    rows = []
    try:
        for dname in dtypes:
            dtype = _dtype_of(dname)
            itemsize = torch.tensor([], dtype=dtype).element_size()
            for mb in sizes_mb:
                numel = (mb * 1024 * 1024) // itemsize
                r_tot, m_tot, r_solo, m_solo, iters, bx = _bench_size(
                    handle, dtype, numel, rank, world_size, device, reps,
                    warmup, gemm_n, compute_stream, gemm_dtype,
                    gemm_iters_fixed)
                if rank == 0:
                    win = "mori" if m_tot < r_tot else "RCCL"
                    print(f"[gemm-ovlp] {dname} {mb}MB iters={iters} | "
                          f"rccl_total={r_tot:.3f}ms mori_total={m_tot:.3f}ms | "
                          f"solo rccl={r_solo:.3f} mori={m_solo:.3f} | "
                          f"win={win} | bitexact={bx}")
                    rows.append((mb, dname, r_tot, m_tot, r_solo, m_solo,
                                 iters, int(bx)))
                dist.barrier()
        if rank == 0:
            os.makedirs(_LOGS_DIR, exist_ok=True)
            csv = os.path.join(_LOGS_DIR, "sweep_gemm_overlap.csv")
            with open(csv, "w") as f:
                f.write("size_mb,dtype,gemm_rccl_total_ms,gemm_sdma_total_ms,"
                        "rccl_solo_ms,mori_solo_ms,gemm_iters,bitexact\n")
                for r in rows:
                    f.write("%d,%s,%.4f,%.4f,%.4f,%.4f,%d,%d\n" % r)
            print(f"[gemm-ovlp] wrote {csv}")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        del handle
        dist.barrier()
        shmem.shmem_finalize()


def main():
    p = argparse.ArgumentParser(description="GEMM-overlap AllGather sweep")
    p.add_argument("--sizes-mb", type=int, nargs="+", default=_DEFAULT_SIZES_MB)
    p.add_argument("--dtypes", type=str, nargs="+", default=["fp32"])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--gemm-n", type=int, default=4096,
                   help="square GEMM dimension run on the compute stream")
    p.add_argument("--gemm-dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"],
                   help="GEMM operand dtype; bf16/fp16 use MFMA matrix cores "
                        "(compute/CU-bound, the clean compute/comm-overlap contention case)")
    p.add_argument("--gemm-iters", type=int, default=0,
                   help="fixed GEMM loop count for ALL sizes (0=auto-tune to "
                        "match AG solo time). A fixed value models a real "
                        "model layer's compute that does not scale with the AG, "
                        "exposing the SDMA no-CU-contention win at small sizes.")
    args = p.parse_args()

    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    os.environ.setdefault("MORI_SDMA_NUM_CHANNELS", "1")

    assert "RANK" in os.environ, "launch under torchrun (build_and_test.sh xnode)"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank,
                            world_size=world_size, device_id=device)
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    try:
        _worker(rank, world_size, ranks_per_node, device, args.sizes_mb,
                args.dtypes, args.reps, args.warmup, args.gemm_n,
                _dtype_of(args.gemm_dtype), args.gemm_iters)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
