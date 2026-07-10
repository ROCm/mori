# EP2 IntraNode Tuning Results — MI350X (gfx950, 256 CUs)

Date: 2026-07-10
Hardware: 8x MI350X (gfx950, 256 CUs, 288 GB HBM3e), AMD EPYC 9575F
Sweep: full scope (35 block_num x 9 warp_per_block = 315 configs/shape)
Configs: `python/mori/ops/tuning_configs/gfx950_mi350x_IntraNode_ep2_{dispatch,combine}.json`

## 1. Recommended Configs

### Dispatch — best config per shape

| Tokens | Dtype | Hidden | block_num | warp_per_block | BW (GB/s) | Latency |
|--------|-------|--------|-----------|----------------|-----------|---------|
| 64 | fp4 | 3584 | 64 | 8 | 16.0 | 28.6 us |
| 128 | fp4 | 3584 | 184 | 6 | 28.0 | 32.7 us |
| 256 | fp4 | 3584 | 208 | 10 | 43.2 | 42.3 us |
| 512 | fp4 | 3584 | 208 | 10 | 58.8 | 62.4 us |
| 1024 | fp4 | 3584 | 128 | 16 | 70.7 | 103.3 us |
| 2048 | fp4 | 3584 | 256 | 16 | 81.1 | 180.5 us |
| 4096 | fp4 | 3584 | 255 | 16 | 87.9 | 332.5 us |
| 64 | fp8_e4m3 | 7168 | 64 | 8 | 32.7 | 28.0 us |
| 128 | fp8_e4m3 | 7168 | 128 | 8 | 49.6 | 36.9 us |
| 256 | fp8_e4m3 | 7168 | 128 | 8 | 68.0 | 53.9 us |
| 512 | fp8_e4m3 | 7168 | 176 | 8 | 82.4 | 89.0 us |
| 1024 | fp8_e4m3 | 7168 | 208 | 8 | 92.9 | 157.5 us |
| 2048 | fp8_e4m3 | 7168 | 128 | 16 | 99.1 | 295.4 us |
| 4096 | fp8_e4m3 | 7168 | 128 | 16 | 103.2 | 567.1 us |

### Combine — best config per shape (winner of P2P vs non-P2P)

| Tokens | Quant | Path | block_num | warp_per_block | BW (GB/s) | Latency |
|--------|-------|------|-----------|----------------|-----------|---------|
| 64 | fp8_direct_cast | non-P2P | 224 | 4 | 63.4 | 28.8 us |
| 128 | fp8_direct_cast | non-P2P | 224 | 8 | 108.7 | 33.7 us |
| 256 | fp8_direct_cast | non-P2P | 224 | 8 | 144.1 | 50.8 us |
| 512 | fp8_direct_cast | non-P2P | 224 | 16 | 175.7 | 83.4 us |
| 1024 | fp8_direct_cast | non-P2P | 256 | 8 | 188.8 | 155.0 us |
| 2048 | fp8_direct_cast | non-P2P | 256 | 16 | 202.2 | 289.4 us |
| 4096 | fp8_direct_cast | non-P2P | 256 | 16 | 206.8 | 566.1 us |
| 64 | none | P2P | 64 | 8 | 58.7 | 31.2 us |
| 128 | none | P2P | 56 | 16 | 78.2 | 46.9 us |
| 256 | none | P2P | 128 | 16 | 94.7 | 77.4 us |
| 512 | none | P2P | 64 | 16 | 104.9 | 139.2 us |
| 1024 | none | P2P | 80 | 14 | 110.8 | 264.0 us |
| 2048 | none | P2P | 32 | 16 | 113.2 | 516.9 us |
| 4096 | none | P2P | 160 | 16 | 115.0 | 1019.0 us |

### P2P vs non-P2P decision rule

- **With fp8_direct_cast quant:** use non-P2P at all token counts (+8% at 64 tokens, up to +79% at 4096). Stronger effect than EP4/EP8 — fewer peers means even less XGMI contention for the staging path.
- **Without quant:** use P2P at all token counts (+3-9%).

## 2. Before/After — MI350X Tuned vs Shipped Baseline

Baseline: MI355X bn/wpb configs measured on MI350X hardware (from sweep data).

### Dispatch

| Tokens | Dtype | Baseline bn/wpb | Baseline BW | Tuned bn/wpb | Tuned BW | Delta |
|--------|-------|-----------------|-------------|--------------|----------|-------|
| 64 | fp4 | 64/8 | 15.9 | 64/8 | 16.0 | +0.1% |
| 128 | fp4 | 256/4 | 28.3 | 184/6 | 28.0 | -0.8% |
| 256 | fp4 | 256/8 | 6.7 | 208/10 | 43.2 | +543% |
| 512 | fp4 | 256/8 | 58.5 | 208/10 | 58.8 | +0.5% |
| 1024 | fp4 | 128/16 | 70.1 | 128/16 | 70.7 | +0.9% |
| 2048 | fp4 | 256/16 | 80.2 | 256/16 | 81.1 | +1.1% |
| 4096 | fp4 | 256/16 | 85.6 | 255/16 | 87.9 | +2.8% |
| 64 | fp8_e4m3 | 64/8 | 32.2 | 64/8 | 32.7 | +1.6% |
| 128 | fp8_e4m3 | 128/8 | 49.3 | 128/8 | 49.6 | +0.5% |
| 256 | fp8_e4m3 | 128/8 | 67.1 | 128/8 | 68.0 | +1.2% |
| 512 | fp8_e4m3 | 256/8 | 82.4 | 176/8 | 82.4 | +0.0% |
| 1024 | fp8_e4m3 | 256/8 | 93.5 | 208/8 | 92.9 | -0.7% |
| 2048 | fp8_e4m3 | 256/8 | 97.3 | 128/16 | 99.1 | +1.8% |
| 4096 | fp8_e4m3 | 256/8 | 100.4 | 128/16 | 103.2 | +2.8% |

Dispatch: mostly within noise (+/-3%). Major outlier at fp4/256 tokens: MI355X's bn=256/wpb=8 hits a performance cliff on MI350X (6.7 GB/s), tuned config avoids it (+543%).

### Combine

| Tokens | Quant | Path | Baseline bn/wpb | Baseline BW | Tuned bn/wpb | Tuned BW | Delta |
|--------|-------|------|-----------------|-------------|--------------|----------|-------|
| 64 | fp8_direct_cast | non-P2P | 128/8 | 61.4 | 224/4 | 63.4 | +3.3% |
| 128 | fp8_direct_cast | non-P2P | 256/8 | 107.5 | 224/8 | 108.7 | +1.1% |
| 256 | fp8_direct_cast | non-P2P | 256/8 | 143.9 | 224/8 | 144.1 | +0.1% |
| 512 | fp8_direct_cast | non-P2P | 256/16 | 175.2 | 224/16 | 175.7 | +0.3% |
| 1024 | fp8_direct_cast | non-P2P | 128/16 | 186.0 | 256/8 | 188.8 | +1.5% |
| 2048 | fp8_direct_cast | non-P2P | 256/16 | 202.0 | 256/16 | 202.2 | +0.1% |
| 4096 | fp8_direct_cast | non-P2P | 256/16 | 167.2 | 256/16 | 206.8 | +23.7% |
| 64 | none | non-P2P | 128/8 | 54.1 | 224/8 | 57.0 | +5.3% |
| 128 | none | non-P2P | 256/8 | 74.5 | 224/8 | 75.4 | +1.1% |
| 256 | none | non-P2P | 256/8 | 90.6 | 224/8 | 90.1 | -0.5% |
| 512 | none | non-P2P | 256/16 | 99.1 | 224/16 | 98.8 | -0.4% |
| 1024 | none | non-P2P | 256/16 | 103.3 | 256/16 | 103.4 | +0.1% |
| 2048 | none | non-P2P | 128/16 | 103.8 | 128/16 | 103.9 | +0.2% |
| 4096 | none | non-P2P | 128/16 | 104.9 | 256/16 | 106.5 | +1.5% |
| 64 | fp8_direct_cast | P2P | 32/8 | 56.8 | 64/8 | 58.9 | +3.6% |
| 128 | fp8_direct_cast | P2P | 64/8 | 77.2 | 112/8 | 78.6 | +1.8% |
| 256 | fp8_direct_cast | P2P | 64/16 | 94.4 | 64/16 | 94.4 | +0.0% |
| 512 | fp8_direct_cast | P2P | 64/16 | 104.7 | 64/16 | 104.8 | +0.1% |
| 1024 | fp8_direct_cast | P2P | 128/16 | 110.9 | 32/16 | 110.3 | -0.6% |
| 2048 | fp8_direct_cast | P2P | 32/16 | 112.7 | 152/16 | 114.7 | +1.8% |
| 4096 | fp8_direct_cast | P2P | 32/16 | 114.0 | 56/15 | 115.4 | +1.3% |
| 64 | none | P2P | 32/8 | 56.5 | 64/8 | 58.7 | +3.9% |
| 128 | none | P2P | 64/8 | 78.2 | 56/16 | 78.2 | +0.0% |
| 256 | none | P2P | 64/16 | 94.2 | 128/16 | 94.7 | +0.5% |
| 512 | none | P2P | 64/16 | 104.9 | 64/16 | 104.9 | +0.0% |
| 1024 | none | P2P | 128/16 | 110.8 | 80/14 | 110.8 | +0.0% |
| 2048 | none | P2P | 32/16 | 113.1 | 32/16 | 113.2 | +0.1% |
| 4096 | none | P2P | 32/16 | 113.9 | 160/16 | 115.0 | +0.9% |

Combine: mostly +0-5%. Two outliers where MI355X configs hit performance cliffs on MI350X: fp8dc/nP2P at 4096 tokens (+23.7%), and none/nP2P at 64 tokens (+5.3%).

## 3. P2P vs non-P2P Detail

### With fp8_direct_cast quant (hidden=7168)

| Tokens | P2P BW | P2P lat | P2P bn/wpb | non-P2P BW | non-P2P lat | non-P2P bn/wpb | Winner | Uplift |
|--------|--------|---------|-----------|-----------|------------|---------------|--------|--------|
| 64 | 58.9 | 31.1 us | 64/8 | 63.4 | 28.8 us | 224/4 | non-P2P | +8% |
| 128 | 78.6 | 46.6 us | 112/8 | 108.7 | 33.7 us | 224/8 | non-P2P | +38% |
| 256 | 94.4 | 77.4 us | 64/16 | 144.1 | 50.8 us | 224/8 | non-P2P | +53% |
| 512 | 104.8 | 139.5 us | 64/16 | 175.7 | 83.4 us | 224/16 | non-P2P | +68% |
| 1024 | 110.3 | 265.1 us | 32/16 | 188.8 | 155.0 us | 256/8 | non-P2P | +71% |
| 2048 | 114.7 | 509.8 us | 152/16 | 202.2 | 289.4 us | 256/16 | non-P2P | +76% |
| 4096 | 115.4 | 1014.5 us | 56/15 | 206.8 | 566.1 us | 256/16 | non-P2P | +79% |

### Without quant (hidden=7168)

| Tokens | P2P BW | P2P lat | P2P bn/wpb | non-P2P BW | non-P2P lat | non-P2P bn/wpb | Winner | Uplift |
|--------|--------|---------|-----------|-----------|------------|---------------|--------|--------|
| 64 | 58.7 | 31.2 us | 64/8 | 57.0 | 32.3 us | 224/8 | P2P | +3% |
| 128 | 78.2 | 46.9 us | 56/16 | 75.4 | 48.6 us | 224/8 | P2P | +4% |
| 256 | 94.7 | 77.4 us | 128/16 | 90.1 | 81.2 us | 224/8 | P2P | +5% |
| 512 | 104.9 | 139.2 us | 64/16 | 98.8 | 148.2 us | 224/16 | P2P | +6% |
| 1024 | 110.8 | 264.0 us | 80/14 | 103.4 | 282.9 us | 256/16 | P2P | +7% |
| 2048 | 113.2 | 516.9 us | 32/16 | 103.9 | 563.4 us | 128/16 | P2P | +9% |
| 4096 | 115.0 | 1019.0 us | 160/16 | 106.5 | 1098.7 us | 256/16 | P2P | +8% |

## 4. Dispatch Detail

### fp4 (hidden=3584)

| Tokens | BW (GB/s) | Latency | block_num | warp_per_block |
|--------|-----------|---------|-----------|----------------|
| 64 | 16.0 | 28.6 us | 64 | 8 |
| 128 | 28.0 | 32.7 us | 184 | 6 |
| 256 | 43.2 | 42.3 us | 208 | 10 |
| 512 | 58.8 | 62.4 us | 208 | 10 |
| 1024 | 70.7 | 103.3 us | 128 | 16 |
| 2048 | 81.1 | 180.5 us | 256 | 16 |
| 4096 | 87.9 | 332.5 us | 255 | 16 |

### fp8_e4m3 (hidden=7168)

| Tokens | BW (GB/s) | Latency | block_num | warp_per_block |
|--------|-----------|---------|-----------|----------------|
| 64 | 32.7 | 28.0 us | 64 | 8 |
| 128 | 49.6 | 36.9 us | 128 | 8 |
| 256 | 68.0 | 53.9 us | 128 | 8 |
| 512 | 82.4 | 89.0 us | 176 | 8 |
| 1024 | 92.9 | 157.5 us | 208 | 8 |
| 2048 | 99.1 | 295.4 us | 128 | 16 |
| 4096 | 103.2 | 567.1 us | 128 | 16 |

## 5. Remaining Work

- [x] EP8 IntraNode (all 4 groups)
- [x] EP4 IntraNode (all 4 groups)
- [x] EP2 IntraNode (all 4 groups)
- [x] IntraNodeLL tuning
- [x] Profiler analysis (static occupancy — see `mi350x_profiler_analysis.md`)
