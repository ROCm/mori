# MI350X IntraNode Profiler Analysis

Date: 2026-07-10
Hardware: 8x MI350X (gfx950, 256 CUs, max 32 waves/CU = 8192 wave slots)

Note: Runtime trace profiling (`--cmd profile`) requires `ENABLE_PROFILER=ON` build, which is not available in this environment. This analysis is based on static occupancy calculations from the tuned configs.

## 1. Dispatch Occupancy

Theoretical occupancy = (block_num × warp_per_block) / 8192 wave slots.

### EP8

| Tokens | Dtype | bn/wpb | Total warps | Warps/CU | Occupancy | BW (GB/s) |
|--------|-------|--------|-------------|----------|-----------|-----------|
| 64 | fp8_e4m3 | 128/4 | 512 | 2.0 | 6% | 76.0 |
| 128 | fp8_e4m3 | 208/5 | 1040 | 4.1 | 13% | 121.5 |
| 256 | fp8_e4m3 | 256/4 | 1024 | 4.0 | 12% | 166.6 |
| 512 | fp8_e4m3 | 228/6 | 1368 | 5.3 | 17% | 221.3 |
| 1024 | fp8_e4m3 | 256/8 | 2048 | 8.0 | 25% | 272.0 |
| 2048 | fp8_e4m3 | 208/8 | 1664 | 6.5 | 20% | 314.4 |
| 4096 | fp8_e4m3 | 256/8 | 2048 | 8.0 | 25% | 342.4 |
| 64 | fp4 | 128/4 | 512 | 2.0 | 6% | 35.7 |
| 512 | fp4 | 216/10 | 2160 | 8.4 | 26% | 128.8 |
| 4096 | fp4 | 184/15 | 2760 | 10.8 | 34% | 237.4 |

**Observation:** Dispatch occupancy is deliberately low (6-34%). This is XGMI bandwidth-bound, not compute-bound — more warps don't help because the bottleneck is cross-GPU data transfer, not ALU throughput. The sweep correctly converges on configs that have just enough warps to keep XGMI links busy without wasting register/LDS resources.

### EP2 — Higher occupancy needed

| Tokens | Dtype | bn/wpb | Warps/CU | Occupancy | BW (GB/s) |
|--------|-------|--------|----------|-----------|-----------|
| 2048 | fp4 | 256/16 | 16.0 | 50% | 81.1 |
| 4096 | fp4 | 255/16 | 15.9 | 50% | 87.9 |
| 2048 | fp8_e4m3 | 128/16 | 8.0 | 25% | 99.1 |
| 4096 | fp8_e4m3 | 128/16 | 8.0 | 25% | 103.2 |

EP2 dispatch prefers higher occupancy (25-50%) than EP8 (6-25%). With only 1 peer, each GPU must process all its own data — more compute parallelism is needed.

## 2. Combine Shared Memory Pressure

Estimated shared memory per block ≈ worldSize × warp_per_block × 64 × sizeof(element).

### EP8 Combine

| Tokens | Quant | Path | bn/wpb | Est shmem/block | BW (GB/s) |
|--------|-------|------|--------|-----------------|-----------|
| 64 | fp8_direct_cast | non-P2P | 112/8 | 16 KB | 149.4 |
| 512 | fp8_direct_cast | non-P2P | 224/16 | 32 KB | 525.9 |
| 4096 | fp8_direct_cast | non-P2P | 256/16 | 32 KB | 642.2 |
| 64 | none | P2P | 64/8 | 16 KB | 170.2 |
| 4096 | none | P2P | 56/15 | 30 KB | 435.5 |

**Key pattern:** Non-P2P combine at large token counts uses 32 KB shared memory (wpb=16). MI350X has 64 KB LDS per CU, so 32 KB allows 2 blocks per CU — sufficient for the bn=224-256 configs. No shared memory pressure issues observed.

P2P combine uses smaller bn (56-88) with 16-30 KB shmem — well within limits.

### Shared memory scales with EP size

| EP | Max shmem/block | Observation |
|----|-----------------|-------------|
| EP8 | 32 KB | 2 blocks/CU possible |
| EP4 | 16 KB | 4 blocks/CU possible |
| EP2 | 8 KB | 8 blocks/CU possible |

Lower EP sizes use less shared memory per block, allowing more concurrent blocks per CU. This is consistent with the tuned configs: EP2 combine can use more blocks without hitting LDS limits.

## 3. Config Pattern Summary

| Pattern | Explanation |
|---------|-------------|
| Dispatch occupancy 6-34% | XGMI bandwidth-bound — more warps don't help |
| EP2 dispatch occupancy higher (25-50%) | Single peer requires more compute parallelism |
| Combine non-P2P prefers bn=224 | Slightly below CU count (256) for cleaner scheduling |
| Combine P2P prefers bn=56-88 | ~1 block per GPU peer, minimizes contention |
| wpb=8-16 for combine | Balances LDS usage vs warp-level parallelism |
| wpb=4-10 for dispatch | Lower warp count sufficient; XGMI-limited |

## 4. Recommendations

1. **No LDS pressure issues** — all configs fit comfortably within MI350X's 64 KB/CU LDS
2. **Barrier stalls** cannot be analyzed without trace profiling (`ENABLE_PROFILER=ON` build). If investigating latency outliers, rebuild with profiling enabled and use `analyze_ep_kernel_trace.py`
3. The `bn=256` performance cliff on MI350X (seen in EP2 fp4/256 tokens and EP4 fp8dc/nP2P/128 tokens) warrants investigation — may be related to CU scheduling when block count exactly matches CU count
