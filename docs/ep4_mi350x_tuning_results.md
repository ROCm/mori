# EP4 IntraNode Tuning Results — MI350X (gfx950, 256 CUs)

Date: 2026-07-09
Hardware: 8x MI350X (gfx950, 256 CUs, 288 GB HBM3e), AMD EPYC 9575F
Sweep: full scope (35 block_num x 9 warp_per_block = 315 configs/shape)
Configs: `python/mori/ops/tuning_configs/gfx950_mi350x_IntraNode_ep4_{dispatch,combine}.json`

## 1. Recommended Configs

### Dispatch — best config per shape

| Tokens | Dtype | Hidden | block_num | warp_per_block | BW (GB/s) | Latency |
|--------|-------|--------|-----------|----------------|-----------|---------|
| 64 | fp4 | 3584 | 72 | 8 | 26.3 | 31.5 us |
| 128 | fp4 | 3584 | 128 | 8 | 45.9 | 36.6 us |
| 256 | fp4 | 3584 | 256 | 8 | 70.8 | 47.1 us |
| 512 | fp4 | 3584 | 256 | 8 | 96.3 | 68.8 us |
| 1024 | fp4 | 3584 | 256 | 8 | 119.7 | 111.0 us |
| 2048 | fp4 | 3584 | 184 | 15 | 140.1 | 189.6 us |
| 4096 | fp4 | 3584 | 200 | 15 | 157.7 | 336.6 us |
| 64 | fp8_e4m3 | 7168 | 208 | 4 | 55.2 | 30.4 us |
| 128 | fp8_e4m3 | 7168 | 176 | 6 | 85.9 | 39.0 us |
| 256 | fp8_e4m3 | 7168 | 256 | 4 | 117.6 | 56.7 us |
| 512 | fp8_e4m3 | 7168 | 128 | 8 | 142.1 | 93.2 us |
| 1024 | fp8_e4m3 | 7168 | 208 | 8 | 164.1 | 161.8 us |
| 2048 | fp8_e4m3 | 7168 | 228 | 8 | 182.0 | 291.7 us |
| 4096 | fp8_e4m3 | 7168 | 208 | 16 | 192.1 | 552.9 us |

### Combine — best config per shape (winner of P2P vs non-P2P)

| Tokens | Quant | Path | block_num | warp_per_block | BW (GB/s) | Latency |
|--------|-------|------|-----------|----------------|-----------|---------|
| 64 | fp8_direct_cast | non-P2P | 224 | 8 | 108.0 | 30.5 us |
| 128 | fp8_direct_cast | non-P2P | 224 | 8 | 186.2 | 36.0 us |
| 256 | fp8_direct_cast | non-P2P | 256 | 16 | 252.7 | 52.4 us |
| 512 | fp8_direct_cast | non-P2P | 224 | 16 | 309.3 | 85.8 us |
| 1024 | fp8_direct_cast | non-P2P | 256 | 16 | 337.3 | 157.0 us |
| 2048 | fp8_direct_cast | non-P2P | 256 | 16 | 356.4 | 297.0 us |
| 4096 | fp8_direct_cast | non-P2P | 256 | 16 | 369.1 | 575.5 us |
| 64 | none | P2P | 112 | 16 | 104.2 | 32.1 us |
| 128 | none | P2P | 112 | 16 | 142.6 | 46.6 us |
| 256 | none | P2P | 128 | 16 | 173.6 | 76.4 us |
| 512 | none | P2P | 128 | 16 | 195.7 | 135.8 us |
| 1024 | none | P2P | 128 | 16 | 208.0 | 254.9 us |
| 2048 | none | P2P | 256 | 16 | 215.6 | 492.5 us |
| 4096 | none | P2P | 256 | 16 | 215.0 | 988.9 us |

### P2P vs non-P2P decision rule

- **With fp8_direct_cast quant:** use non-P2P at all token counts (even 64 tokens — unlike EP8).
- **Without quant:** use P2P at all token counts.
- **Rule:** quant flips the winner. With quant, halved FP8 read traffic makes the staging buffer copy worthwhile. Without quant, the staging copy is pure overhead.

## 2. Before/After — MI350X Tuned vs Shipped Baseline

Baseline: MI355X bn/wpb configs measured on MI350X hardware (from sweep data).

### Dispatch

| Tokens | Dtype | Baseline bn/wpb | Baseline BW | Tuned bn/wpb | Tuned BW | Delta |
|--------|-------|-----------------|-------------|--------------|----------|-------|
| 64 | fp4 | 128/4 | 26.5 | 72/8 | 26.3 | -0.8% |
| 128 | fp4 | 256/4 | 45.9 | 128/8 | 45.9 | +0.0% |
| 256 | fp4 | 256/8 | 69.5 | 256/8 | 70.8 | +1.8% |
| 512 | fp4 | 256/8 | 94.7 | 256/8 | 96.3 | +1.6% |
| 1024 | fp4 | 256/8 | 118.3 | 256/8 | 119.7 | +1.1% |
| 2048 | fp4 | 256/8 | 137.8 | 184/15 | 140.1 | +1.7% |
| 4096 | fp4 | 256/16 | 153.4 | 200/15 | 157.7 | +2.8% |
| 64 | fp8_e4m3 | 128/4 | 54.2 | 208/4 | 55.2 | +1.8% |
| 128 | fp8_e4m3 | 256/4 | 85.7 | 176/6 | 85.9 | +0.2% |
| 256 | fp8_e4m3 | 256/4 | 115.5 | 256/4 | 117.6 | +1.8% |
| 512 | fp8_e4m3 | 256/8 | 141.0 | 128/8 | 142.1 | +0.8% |
| 1024 | fp8_e4m3 | 256/8 | 162.6 | 208/8 | 164.1 | +0.9% |
| 2048 | fp8_e4m3 | 256/8 | 178.8 | 228/8 | 182.0 | +1.8% |
| 4096 | fp8_e4m3 | 256/8 | 189.8 | 208/16 | 192.1 | +1.2% |

Dispatch: +0-3% gains across the board.

### Combine

| Tokens | Quant | Path | Baseline bn/wpb | Baseline BW | Tuned bn/wpb | Tuned BW | Delta |
|--------|-------|------|-----------------|-------------|--------------|----------|-------|
| 64 | fp8_direct_cast | non-P2P | 256/8 | 106.5 | 224/8 | 108.0 | +1.5% |
| 128 | fp8_direct_cast | non-P2P | 256/8 | 124.6 | 224/8 | 186.2 | +49.4% |
| 256 | fp8_direct_cast | non-P2P | 256/16 | 249.7 | 256/16 | 252.7 | +1.2% |
| 512 | fp8_direct_cast | non-P2P | 256/16 | 305.9 | 224/16 | 309.3 | +1.1% |
| 1024 | fp8_direct_cast | non-P2P | 256/16 | 336.7 | 256/16 | 337.3 | +0.2% |
| 2048 | fp8_direct_cast | non-P2P | 256/16 | 356.1 | 256/16 | 356.4 | +0.1% |
| 4096 | fp8_direct_cast | non-P2P | 256/16 | 369.1 | 256/16 | 369.1 | +0.0% |
| 64 | none | non-P2P | 256/8 | 93.2 | 224/8 | 98.7 | +5.8% |
| 128 | none | non-P2P | 256/8 | 129.7 | 224/16 | 132.8 | +2.4% |
| 256 | none | non-P2P | 256/16 | 162.8 | 224/16 | 163.8 | +0.6% |
| 512 | none | non-P2P | 256/16 | 180.2 | 224/16 | 180.7 | +0.3% |
| 1024 | none | non-P2P | 256/16 | 189.1 | 256/16 | 190.4 | +0.7% |
| 2048 | none | non-P2P | 256/16 | 193.1 | 256/16 | 194.3 | +0.6% |
| 4096 | none | non-P2P | 256/16 | 192.1 | 256/16 | 192.9 | +0.4% |
| 64 | fp8_direct_cast | P2P | 64/16 | 99.0 | 112/16 | 104.4 | +5.4% |
| 128 | fp8_direct_cast | P2P | 128/16 | 141.4 | 112/16 | 142.3 | +0.7% |
| 256 | fp8_direct_cast | P2P | 128/16 | 172.9 | 112/16 | 172.7 | -0.1% |
| 512 | fp8_direct_cast | P2P | 128/16 | 196.7 | 128/16 | 196.8 | +0.0% |
| 1024 | fp8_direct_cast | P2P | 128/16 | 204.6 | 256/16 | 208.1 | +1.7% |
| 2048 | fp8_direct_cast | P2P | 256/16 | 211.2 | 256/16 | 215.1 | +1.9% |
| 4096 | fp8_direct_cast | P2P | 256/16 | 213.6 | 256/16 | 216.0 | +1.1% |
| 64 | none | P2P | 256/16 | 99.2 | 112/16 | 104.2 | +5.0% |
| 128 | none | P2P | 128/16 | 140.8 | 112/16 | 142.6 | +1.3% |
| 256 | none | P2P | 128/16 | 173.0 | 128/16 | 173.6 | +0.3% |
| 512 | none | P2P | 128/16 | 195.6 | 128/16 | 195.7 | +0.0% |
| 1024 | none | P2P | 128/16 | 206.6 | 128/16 | 208.0 | +0.7% |
| 2048 | none | P2P | 128/16 | 213.3 | 256/16 | 215.6 | +1.1% |
| 4096 | none | P2P | 256/16 | 214.2 | 256/16 | 215.0 | +0.4% |

Combine: mostly +0-6% gains. Notable outlier: fp8dc/nP2P at 128 tokens shows **+49%** — bn=256/wpb=8 hits a performance cliff on MI350X (124.6 GB/s), while bn=224/wpb=8 avoids it (186.2 GB/s).

## 3. P2P vs non-P2P Detail

### With fp8_direct_cast quant (hidden=7168)

| Tokens | P2P BW | P2P lat | P2P bn/wpb | non-P2P BW | non-P2P lat | non-P2P bn/wpb | Winner | Uplift |
|--------|--------|---------|-----------|-----------|------------|---------------|--------|--------|
| 64 | 104.4 | 32.1 us | 112/16 | 108.0 | 30.5 us | 224/8 | non-P2P | +3% |
| 128 | 142.3 | 46.7 us | 112/16 | 186.2 | 36.0 us | 224/8 | non-P2P | +31% |
| 256 | 172.7 | 77.1 us | 112/16 | 252.7 | 52.4 us | 256/16 | non-P2P | +46% |
| 512 | 196.8 | 135.0 us | 128/16 | 309.3 | 85.8 us | 224/16 | non-P2P | +57% |
| 1024 | 208.1 | 254.5 us | 256/16 | 337.3 | 157.0 us | 256/16 | non-P2P | +62% |
| 2048 | 215.1 | 492.1 us | 256/16 | 356.4 | 297.0 us | 256/16 | non-P2P | +66% |
| 4096 | 216.0 | 983.9 us | 256/16 | 369.1 | 575.5 us | 256/16 | non-P2P | +71% |

### Without quant (hidden=7168)

| Tokens | P2P BW | P2P lat | P2P bn/wpb | non-P2P BW | non-P2P lat | non-P2P bn/wpb | Winner | Uplift |
|--------|--------|---------|-----------|-----------|------------|---------------|--------|--------|
| 64 | 104.2 | 32.1 us | 112/16 | 98.7 | 33.4 us | 224/8 | P2P | +6% |
| 128 | 142.6 | 46.6 us | 112/16 | 132.8 | 49.5 us | 224/16 | P2P | +7% |
| 256 | 173.6 | 76.4 us | 128/16 | 163.8 | 80.9 us | 224/16 | P2P | +6% |
| 512 | 195.7 | 135.8 us | 128/16 | 180.7 | 146.9 us | 224/16 | P2P | +8% |
| 1024 | 208.0 | 254.9 us | 128/16 | 190.4 | 279.7 us | 256/16 | P2P | +9% |
| 2048 | 215.6 | 492.5 us | 256/16 | 194.3 | 546.6 us | 256/16 | P2P | +11% |
| 4096 | 215.0 | 988.9 us | 256/16 | 192.9 | 1101.6 us | 256/16 | P2P | +11% |

## 4. Dispatch Detail

### fp4 (hidden=3584)

| Tokens | BW (GB/s) | Latency | block_num | warp_per_block |
|--------|-----------|---------|-----------|----------------|
| 64 | 26.3 | 31.5 us | 72 | 8 |
| 128 | 45.9 | 36.6 us | 128 | 8 |
| 256 | 70.8 | 47.1 us | 256 | 8 |
| 512 | 96.3 | 68.8 us | 256 | 8 |
| 1024 | 119.7 | 111.0 us | 256 | 8 |
| 2048 | 140.1 | 189.6 us | 184 | 15 |
| 4096 | 157.7 | 336.6 us | 200 | 15 |

### fp8_e4m3 (hidden=7168)

| Tokens | BW (GB/s) | Latency | block_num | warp_per_block |
|--------|-----------|---------|-----------|----------------|
| 64 | 55.2 | 30.4 us | 208 | 4 |
| 128 | 85.9 | 39.0 us | 176 | 6 |
| 256 | 117.6 | 56.7 us | 256 | 4 |
| 512 | 142.1 | 93.2 us | 128 | 8 |
| 1024 | 164.1 | 161.8 us | 208 | 8 |
| 2048 | 182.0 | 291.7 us | 228 | 8 |
| 4096 | 192.1 | 552.9 us | 208 | 16 |

## 5. Remaining Work

- [x] EP4 IntraNode (all 4 groups)
- [x] EP2 IntraNode
- [x] IntraNodeLL
- [x] Profiler analysis (static occupancy — see `mi350x_profiler_analysis.md`)
