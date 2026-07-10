# EP8 IntraNodeLL Tuning Results — MI350X (gfx950, 256 CUs)

Date: 2026-07-10
Hardware: 8x MI350X (gfx950, 256 CUs, 288 GB HBM3e), AMD EPYC 9575F
Sweep: full scope (35 block_num x 9 warp_per_block = 315 configs/shape)
Configs: `python/mori/ops/tuning_configs/gfx950_mi350x_IntraNodeLL_ep8_{dispatch,combine}.json`
Token counts: 1, 32, 64 (latency-optimal, small batch regime)

## 1. Recommended Configs

### Dispatch — best config per shape

| Tokens | block_num | warp_per_block | BW (GB/s) | Latency |
|--------|-----------|----------------|-----------|---------|
| 1 | 144 | 16 | 3.95 | 22.3 us |
| 32 | 32 | 10 | 95.0 | 25.7 us |
| 64 | 72 | 10 | 148.8 | 32.7 us |

### Combine — best config per shape

| Tokens | block_num | warp_per_block | BW (GB/s) | Latency |
|--------|-----------|----------------|-----------|---------|
| 1 | 32 | 10 | 3.42 | 19.4 us |
| 32 | 56 | 8 | 104.8 | 23.2 us |
| 64 | 112 | 8 | 169.9 | 28.7 us |

## 2. Before/After — MI350X Tuned vs Shipped Baseline

Baseline: MI355X bn/wpb configs measured on MI350X hardware.
Note: MI355X only shipped dispatch configs for IntraNodeLL; combine configs are new.

### Dispatch

| Tokens | Baseline bn/wpb | Baseline BW | Tuned bn/wpb | Tuned BW | Delta |
|--------|-----------------|-------------|--------------|----------|-------|
| 1 | 1/9 | 2.8 | 144/16 | 3.95 | +41% |
| 32 | 32/9 | 79.0 | 32/10 | 95.0 | +20% |
| 64 | 64/9 | 127.9 | 72/10 | 148.8 | +16% |

Significant gains across all token counts. MI355X's wpb=9 is suboptimal on MI350X — wpb=10 or 16 performs better. At 1 token, bn=1 severely underutilizes the GPU; bn=144 allows more warps to overlap memory latency.

### Combine (new — no MI355X baseline existed)

| Tokens | block_num | warp_per_block | BW (GB/s) | Latency |
|--------|-----------|----------------|-----------|---------|
| 1 | 32 | 10 | 3.42 | 19.4 us |
| 32 | 56 | 8 | 104.8 | 23.2 us |
| 64 | 112 | 8 | 169.9 | 28.7 us |

## 3. IntraNodeLL vs IntraNode — EP8 Latency Comparison

| Tokens | LL Dispatch lat | IntraNode Dispatch lat | LL Combine lat | IntraNode Combine lat |
|--------|-----------------|------------------------|----------------|----------------------|
| 1 | 22.3 us | — | 19.4 us | — |
| 32 | 25.7 us | — | 23.2 us | — |
| 64 | 32.7 us | 32.1 us (fp8_e4m3) | 28.7 us | 28.6 us (P2P) |

At 64 tokens, IntraNodeLL latency is comparable to IntraNode. The LL kernel's advantage is at very small token counts (1-32) where its cooperative 1-block-per-token design avoids the IntraNode kernel's fixed block overhead.

## 4. Code Changes

1. **`tests/python/ops/bench_dispatch_combine.py`** — Added `--kernel-type` CLI arg (`IntraNode` or `IntraNodeLL`) to support tuning both kernel types. Threads through to config creation and JSON save.

## 5. Remaining Work

- [x] EP8 IntraNode (all 4 groups)
- [x] EP4 IntraNode (all 4 groups)
- [x] EP2 IntraNode (all 4 groups)
- [x] EP8 IntraNodeLL
- [x] Profiler analysis (static occupancy — see `mi350x_profiler_analysis.md`)
