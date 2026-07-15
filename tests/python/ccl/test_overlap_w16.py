#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# CROSS-NODE overlap UT for the hp_sdma path (this PR is cross-node, world>=16).
#
# THESIS. hp_sdma = host-proxy AllGather: cross-node leg CPU-posted (CU-free),
# intra-node leg on the XGMI SDMA copy engine (CU-free). RCCL all_gather runs a
# CU-resident ncclDevKernel that stays on the GPU for the whole (slow) cross-node
# round-trip. So when MANY AllGathers overlap a compute stream, RCCL's resident
# kernels squeeze the concurrent GEMMs while hp_sdma leaves the GPU to compute.
#
# METRIC. We report the GEMMs' OWN completion time (compute-stream CUDA events)
# while N AllGathers run concurrently, for RCCL vs hp_sdma. LOWER = the collective
# steals less GPU from compute. We deliberately time ONLY the GEMMs (not total
# wall) so the AG's own host/latency cost is excluded -- the question is purely
# "how much does the concurrent collective disturb the compute". A single
# cross-node AG barely disturbs (RCCL cross-node AG is network-bound, GPU-light);
# the effect emerges with MANY concurrent AGs (the E2E regime), which is why N
# defaults to 50. HARD bit-exact gate (torch.equal vs RCCL) before timing.
# Set MORI_OVERLAP_ASSERT=1 to assert hp_sdma GEMM time < RCCL GEMM time.
import argparse, os, sys, time, traceback
import torch
import torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import mori.shmem as shmem  # noqa: E402


def _env_on(n, d="0"): return os.environ.get(n, d) not in ("0", "", "false", "False")


def _build_handle(rank, ws, rpn, per_rank_bytes, device):
    from mori.ccl.host_proxy_ag import HostProxyHierAllGather
    cap = max(per_rank_bytes, int(os.environ.get("MORI_FSDP_HOSTPROXY_CAP_MB", "512")) * (1 << 20))
    return HostProxyHierAllGather(my_pe=rank, npes=ws, ranks_per_node=rpn,
                                  output_buffer_size=cap * ws, device=device)


def _wall(fn, reps, warmup):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


def _worker(rank, ws, rpn, device, size_mb, nops, gemm_n, reps, warmup):
    per = size_mb * 1024 * 1024 + 4096
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", str(per * ws * 3 + (1 << 28)))
    shmem.shmem_torch_process_group_init("default")
    async_hp = _env_on("MORI_HOSTPROXY_ASYNC", "0")
    if async_hp: os.environ.setdefault("MORI_HOSTPROXY_ASYNC_RING", "2")
    h = _build_handle(rank, ws, rpn, per, device)
    numel = (size_mb * 1024 * 1024) // 4
    R = 4
    inp = [(torch.arange(numel, device=device, dtype=torch.float32) + rank * 131.0 + i) for i in range(R)]
    outm = [torch.empty(numel * ws, dtype=torch.float32, device=device) for _ in range(R)]
    outr = [torch.empty(numel * ws, dtype=torch.float32, device=device) for _ in range(R)]
    a = torch.randn(gemm_n, gemm_n, device=device, dtype=torch.bfloat16)
    b = torch.randn(gemm_n, gemm_n, device=device, dtype=torch.bfloat16)
    g = torch.empty(gemm_n, gemm_n, device=device, dtype=torch.bfloat16)
    main = torch.cuda.current_stream(); comm = torch.cuda.Stream(); comp = torch.cuda.Stream()
    # bit-exact gate
    dist.all_gather_into_tensor(outr[0], inp[0]); assert h(inp[0], outm[0], numel, main)
    main.synchronize(); torch.cuda.synchronize()
    bx = bool(torch.equal(outm[0], outr[0]))
    assert bx, "bitexact MISMATCH"

    gi = int(os.environ.get("MORI_MANYOP_GEMM_ITERS", "1"))
    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)

    # Measure the GEMMs' OWN completion time (compute-stream events) while the AGs
    # run concurrently. ag_pre: launch AGs on the comm stream BEFORE timing (RCCL,
    # GPU kernels). ag_mid: run the host-proxy AG loop AFTER the GEMMs are queued
    # (mori, CPU-driven, overlaps the GPU GEMMs). disturbance = under/solo - 1.
    def timed_gemm(ag_pre=None, ag_mid=None):
        def once():
            comm.wait_stream(main); comp.wait_stream(main)
            if ag_pre is not None: ag_pre()
            ev0.record(comp)
            with torch.cuda.stream(comp):
                for _ in range(nops):
                    for _ in range(gi): torch.matmul(a, b, out=g)
            ev1.record(comp)
            if ag_mid is not None: ag_mid()
            main.wait_stream(comm); main.wait_stream(comp)
            torch.cuda.synchronize(); dist.barrier()
            return ev0.elapsed_time(ev1)
        for _ in range(warmup): once()
        return min(once() for _ in range(reps))

    def rccl_pre():
        with torch.cuda.stream(comm):
            for i in range(nops):
                dist.all_gather_into_tensor(outr[i % R], inp[i % R])

    def mori_mid():
        if async_hp:
            for i in range(nops):
                hh = h.call_async(inp[i % R], outm[i % R], numel, main); h._complete(hh)
        else:
            for i in range(nops):
                assert h(inp[i % R], outm[i % R], numel, main)

    g_rccl = timed_gemm(ag_pre=rccl_pre)
    g_mori = timed_gemm(ag_mid=mori_mid)
    if rank == 0:
        speedup = g_rccl / g_mori if g_mori > 0 else 0.0
        win = "hp_sdma" if g_mori < g_rccl else "RCCL"
        print(f"[overlap-w16] nops={nops} shard={size_mb}MB gemm_n={gemm_n} | "
              f"GEMM time under concurrent AllGather: rccl={g_rccl:.2f}ms "
              f"hp_sdma={g_mori:.2f}ms | hp_sdma {speedup:.2f}x faster | win={win} | bitexact={bx}")
        if os.environ.get("MORI_OVERLAP_ASSERT", "0") not in ("0", "", "false"):
            assert g_mori < g_rccl, (
                f"overlap thesis FAILED: hp_sdma GEMM {g_mori:.2f}ms >= RCCL {g_rccl:.2f}ms")
            print("test_overlap_w16: PASSED")
    torch.cuda.synchronize(); dist.barrier(); del h; dist.barrier(); shmem.shmem_finalize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size-mb", type=int, default=8)
    p.add_argument("--nops", type=int, default=50)
    p.add_argument("--gemm-n", type=int, default=2048)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    rank = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"])
    lr = int(os.environ.get("LOCAL_RANK", rank)); rpn = int(os.environ.get("LOCAL_WORLD_SIZE", ws))
    torch.cuda.set_device(lr); device = torch.device(f"cuda:{lr}")
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=ws, device_id=device)
    torch._C._distributed_c10d._register_process_group("default", torch.distributed.group.WORLD)
    try:
        _worker(rank, ws, rpn, device, a.size_mb, a.nops, a.gemm_n, a.reps, a.warmup)
    finally:
        if dist.is_initialized(): dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc(); raise SystemExit(1)
