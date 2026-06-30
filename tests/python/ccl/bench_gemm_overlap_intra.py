#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# the benchmark ISOLATION experiment (reviewer item #1, 2026-06-29).
#
# The cross-node GEMM-overlap bench (bench_gemm_overlap.py) shows mori LOSES to
# RCCL at every size: the inter-node phase is CU-driven RDMA (GPU threads
# post/poll WQEs), so the AllGather occupies CUs regardless of barrier mechanism
# and serializes against a concurrent GEMM. The reviewer asked: does the
# "SDMA frees CUs for the GEMM" thesis hold AT ALL?  The only place it CAN hold
# is the PURE INTRA-NODE path (XGMI SDMA copy engines, NO RDMA, NO CUs for the
# data move). This bench isolates exactly that:
#
#   single node, world=4 ranks, AllgatherSdma (pure SDMA),
#   under a concurrent compute-stream GEMM, vs RCCL all_gather_into_tensor.
#
# If mori SDMA total < RCCL total here, the copy-engine/CU-contention thesis is
# REAL and the cross-node loss is purely the RDMA inter-node phase (a scoped,
# honest result). If mori still loses single-node, the thesis is dead. Either
# way this is the decisive measurement.
#
# Self-spawning single-node (mp.spawn, like test_allgather.py) so it runs under
# the harness `intra` launch model (one docker exec, no torchrun/2-node setup).
# Bit-exact vs torch asserted per size (zero tolerance). rank 0 writes
# logs/sweep_gemm_overlap_intra.csv.
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import mori.shmem as shmem  # noqa: E402
from mori.ccl import AllgatherSdma  # noqa: E402
from tests.python.utils import TorchDistContext, get_free_port  # noqa: E402

_OUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
_LOGS_DIR = os.path.join(_OUT_ROOT, "logs")
_DEFAULT_SIZES_MB = [4, 8, 16, 32, 64, 128, 256, 512]


def _make_input(numel, rank, device):
    base = torch.arange(numel, device=device, dtype=torch.float32)
    return base + rank * 131.0


def _host_time(fn, reps, warmup):
    """Wall time of fn bracketed by full device sync — captures work enqueued on
    ANY stream (compute + AG both finish before the 2nd sync), so it is the true
    OVERLAPPED total."""
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


def _bench_size(handle, numel, rank, world_size, device, reps, warmup, gemm_n,
                compute_stream, gemm_dtype):
    inp = _make_input(numel, rank, device)
    out_mori = torch.empty(numel * world_size, dtype=torch.float32, device=device)
    out_ref = torch.empty(numel * world_size, dtype=torch.float32, device=device)
    main_stream = torch.cuda.current_stream()

    # The copy-engine-overlap thesis can ONLY win when the concurrent compute is
    # FLOP-bound (HBM-light), leaving HBM bandwidth free for the SDMA copy. fp32
    # square matmul is HBM-heavy ( -> serialized). bf16/fp16 MFMA on gfx950
    # runs ~10x the FLOPS/byte, so for the same wall-time it moves far fewer HBM
    # bytes -> the decisive steelman. Preallocate the output to kill per-iter alloc
    # HBM traffic. AG stays fp32 (bit-exact gate unchanged).
    a = torch.randn(gemm_n, gemm_n, device=device, dtype=gemm_dtype)
    b = torch.randn(gemm_n, gemm_n, device=device, dtype=gemm_dtype)
    gemm_out = torch.empty(gemm_n, gemm_n, device=device, dtype=gemm_dtype)

    # ---- correctness gate (zero tolerance) ----
    dist.all_gather_into_tensor(out_ref, inp)
    assert handle(inp, out_mori, numel, main_stream), "AllgatherSdma failed"
    main_stream.synchronize()
    torch.cuda.synchronize()
    bx = bool(torch.equal(out_mori, out_ref))
    if not bx:
        raise AssertionError(f"bit-exact MISMATCH numel={numel}")

    def rccl_ag():
        dist.all_gather_into_tensor(out_ref, inp)

    def mori_ag():
        assert handle(inp, out_mori, numel, main_stream)

    # ---- solo AG times to size the GEMM loop ----
    def solo(ag):
        ag(); main_stream.synchronize()
    r_solo, _ = _host_time(lambda: solo(rccl_ag), reps, warmup)
    m_solo, _ = _host_time(lambda: solo(mori_ag), reps, warmup)

    def one_gemm():
        torch.matmul(a, b, out=gemm_out)
    g1_min, _ = _host_time(one_gemm, max(reps, 3), warmup)
    target_ms = max(r_solo, m_solo)
    iters = max(1, int(round(target_ms / max(g1_min, 1e-3))))

    def gemm_loop():
        with torch.cuda.stream(compute_stream):
            for _ in range(iters):
                torch.matmul(a, b, out=gemm_out)

    # ---- overlapped totals: GEMM loop (compute stream) + AG (main stream) ----
    # MORI_OVLP_AG_FIRST=1 issues the AG BEFORE the GEMM so the SDMA copy engines
    # get a head start (the steelman of the copy-engine thesis: otherwise the GEMM
    # grabs all CUs first and even the tiny SDMA kicker kernel starves behind it).
    ag_first = os.environ.get("MORI_OVLP_AG_FIRST", "0") not in ("0", "", "false")

    def overlap(ag):
        compute_stream.wait_stream(main_stream)
        if ag_first:
            ag()
            gemm_loop()
        else:
            gemm_loop()
            ag()
        main_stream.wait_stream(compute_stream)

    rccl_tot, _ = _host_time(lambda: overlap(rccl_ag), reps, warmup)
    mori_tot, _ = _host_time(lambda: overlap(mori_ag), reps, warmup)
    return rccl_tot, mori_tot, r_solo, m_solo, iters, bx


_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _worker(rank, world_size, port, sizes_mb, reps, warmup, gemm_n, gemm_dtype_s):
    gemm_dtype = _DTYPES[gemm_dtype_s]
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    os.environ.setdefault("MORI_SDMA_NUM_CHANNELS", "1")
    device = torch.device("cuda", rank)
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port,
                          device_id=rank):
        max_bytes = max(sizes_mb) * 1024 * 1024
        per_rank_bytes = max_bytes + 4096
        need = per_rank_bytes * world_size * 3 + (1 << 28)
        os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", str(need))
        shmem.shmem_torch_process_group_init("default")
        assert shmem.shmem_mype() == rank

        handle = AllgatherSdma(
            my_pe=rank, npes=world_size,
            input_buffer_size=per_rank_bytes,
            output_buffer_size=per_rank_bytes * world_size,
            copy_output_to_user=True,
        )
        compute_stream = torch.cuda.Stream()
        if rank == 0:
            print(f"[gemm-ovlp-intra] world={world_size} (single node, pure SDMA) "
                  f"sizes_mb={sizes_mb} gemm_n={gemm_n} gemm_dtype={gemm_dtype_s} "
                  f"reps={reps}")
        rows = []
        try:
            for mb in sizes_mb:
                numel = (mb * 1024 * 1024) // 4
                r_tot, m_tot, r_solo, m_solo, iters, bx = _bench_size(
                    handle, numel, rank, world_size, device, reps, warmup,
                    gemm_n, compute_stream, gemm_dtype)
                if rank == 0:
                    win = "mori" if m_tot < r_tot else "RCCL"
                    print(f"[gemm-ovlp-intra] {mb}MB iters={iters} | "
                          f"rccl_total={r_tot:.3f}ms mori_total={m_tot:.3f}ms | "
                          f"solo rccl={r_solo:.3f} mori={m_solo:.3f} | "
                          f"win={win} | bitexact={bx}")
                    rows.append((mb, r_tot, m_tot, r_solo, m_solo, iters, int(bx)))
                dist.barrier()
            if rank == 0:
                os.makedirs(_LOGS_DIR, exist_ok=True)
                suffix = "" if gemm_dtype_s == "fp32" else f"_{gemm_dtype_s}"
                csv = os.path.join(
                    _LOGS_DIR, f"sweep_gemm_overlap_intra{suffix}.csv")
                with open(csv, "w") as f:
                    f.write("size_mb,gemm_rccl_total_ms,gemm_sdma_total_ms,"
                            "rccl_solo_ms,mori_solo_ms,gemm_iters,bitexact\n")
                    for r in rows:
                        f.write("%d,%.4f,%.4f,%.4f,%.4f,%d,%d\n" % r)
                print(f"[gemm-ovlp-intra] wrote {csv}")
        finally:
            torch.cuda.synchronize()
            dist.barrier()
            del handle
            dist.barrier()
            shmem.shmem_finalize()


def main():
    p = argparse.ArgumentParser(description="Intra-node GEMM-overlap AG sweep")
    p.add_argument("--sizes-mb", type=int, nargs="+", default=_DEFAULT_SIZES_MB)
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--gemm-n", type=int, default=4096)
    p.add_argument("--gemm-dtype", choices=list(_DTYPES.keys()), default="fp32",
                   help="bf16/fp16 = FLOP-bound MFMA GEMM (HBM-light, the "
                        "copy-engine-overlap steelman); fp32 = HBM-heavy")
    args = p.parse_args()
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    port = get_free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(args.world_size, port, args.sizes_mb, args.reps, args.warmup,
              args.gemm_n, args.gemm_dtype),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
