# EP8 IntraNode Tuning Results — MI350X (gfx950, 256 CUs)

Date: 2026-07-06
Hardware: 8x MI350X (gfx950, 256 CUs, 288 GB HBM3e), AMD EPYC 9575F
Sweep: full scope (35 block_num x 9 warp_per_block = 315 configs/shape)
Configs: `python/mori/ops/tuning_configs/gfx950_mi350x_IntraNode_ep8_{dispatch,combine}.json`

## 1. Recommended Configs

### Dispatch — best config per shape

| Tokens | Dtype | Hidden | block_num | warp_per_block | BW (GB/s) | Latency |
|--------|-------|--------|-----------|----------------|-----------|---------|
| 64 | fp8_e4m3 | 7168 | 128 | 4 | 76.0 | 32.1 us |
| 128 | fp8_e4m3 | 7168 | 208 | 5 | 121.5 | 39.9 us |
| 256 | fp8_e4m3 | 7168 | 256 | 4 | 166.6 | 58.1 us |
| 512 | fp8_e4m3 | 7168 | 228 | 6 | 221.3 | 87.9 us |
| 1024 | fp8_e4m3 | 7168 | 256 | 8 | 272.0 | 142.6 us |
| 2048 | fp8_e4m3 | 7168 | 208 | 8 | 314.4 | 247.6 us |
| 4096 | fp8_e4m3 | 7168 | 256 | 8 | 342.4 | 454.2 us |
| 64 | fp4 | 3584 | 128 | 4 | 35.7 | 34.1 us |
| 128 | fp4 | 3584 | 112 | 15 | 61.4 | 39.8 us |
| 256 | fp4 | 3584 | 256 | 8 | 93.8 | 51.9 us |
| 512 | fp4 | 3584 | 216 | 10 | 128.8 | 75.6 us |
| 1024 | fp4 | 3584 | 184 | 15 | 165.9 | 117.5 us |
| 2048 | fp4 | 3584 | 228 | 12 | 201.7 | 192.8 us |
| 4096 | fp4 | 3584 | 184 | 15 | 237.4 | 328.0 us |

### Combine — best config per shape (winner of P2P vs non-P2P)

| Tokens | Quant | Path | block_num | warp_per_block | BW (GB/s) | Latency |
|--------|-------|------|-----------|----------------|-----------|---------|
| 64 | fp8_direct_cast | P2P | 56 | 8 | 168.2 | 28.6 us |
| 128 | fp8_direct_cast | non-P2P | 224 | 8 | 285.9 | 34.1 us |
| 256 | fp8_direct_cast | non-P2P | 224 | 16 | 421.3 | 46.3 us |
| 512 | fp8_direct_cast | non-P2P | 224 | 16 | 525.9 | 73.5 us |
| 1024 | fp8_direct_cast | non-P2P | 256 | 8 | 544.5 | 143.1 us |
| 2048 | fp8_direct_cast | non-P2P | 256 | 16 | 639.0 | 242.6 us |
| 4096 | fp8_direct_cast | non-P2P | 256 | 16 | 642.2 | 483.8 us |
| 64 | none | P2P | 64 | 8 | 170.3 | 28.5 us |
| 128 | none | P2P | 64 | 8 | 244.8 | 39.7 us |
| 256 | none | P2P | 64 | 8 | 313.5 | 62.2 us |
| 512 | none | P2P | 64 | 8 | 368.1 | 105.5 us |
| 1024 | none | P2P | 56 | 10 | 402.4 | 193.3 us |
| 2048 | none | P2P | 72 | 10 | 424.3 | 366.4 us |
| 4096 | none | P2P | 56 | 15 | 435.5 | 713.4 us |

### P2P vs non-P2P decision rule

- **With fp8_direct_cast quant:** use P2P only at 64 tokens; use non-P2P at >= 128 tokens (17-51% faster).
- **Without quant:** use P2P at all token counts.

## 2. Before/After — MI350X Tuned vs Shipped Baseline

Baseline: MI355X bn/wpb configs measured on MI350X hardware (from sweep data).

### Dispatch

| Tokens | Dtype | Baseline bn/wpb | Baseline BW | Tuned bn/wpb | Tuned BW | Delta |
|--------|-------|-----------------|-------------|--------------|----------|-------|
| 64 | fp4 | 64/8 | 34.7 | 128/4 | 35.7 | +2.8% |
| 128 | fp4 | 256/4 | 60.6 | 112/15 | 61.4 | +1.3% |
| 256 | fp4 | 128/16 | 91.1 | 256/8 | 93.8 | +3.0% |
| 512 | fp4 | 256/8 | 126.5 | 216/10 | 128.8 | +1.8% |
| 1024 | fp4 | 256/8 | 163.5 | 184/15 | 165.9 | +1.4% |
| 2048 | fp4 | 256/8 | 201.2 | 228/12 | 201.7 | +0.3% |
| 4096 | fp4 | 256/16 | 222.5 | 184/15 | 237.4 | +6.7% |
| 64 | fp8_e4m3 | 128/4 | 75.3 | 128/4 | 76.0 | +0.8% |
| 128 | fp8_e4m3 | 256/4 | 118.1 | 208/5 | 121.5 | +2.9% |
| 256 | fp8_e4m3 | 256/4 | 165.6 | 256/4 | 166.6 | +0.6% |
| 512 | fp8_e4m3 | 256/8 | 219.2 | 228/6 | 221.3 | +1.0% |
| 1024 | fp8_e4m3 | 256/8 | 266.3 | 256/8 | 272.0 | +2.1% |
| 2048 | fp8_e4m3 | 256/8 | 313.6 | 208/8 | 314.4 | +0.3% |
| 4096 | fp8_e4m3 | 256/8 | 342.4 | 256/8 | 342.4 | +0.0% |

Dispatch: +0-7% gains. Largest at fp4/4096 tokens (+6.7%).

### Combine

| Tokens | Quant | Path | Baseline bn/wpb | Baseline BW | Tuned bn/wpb | Tuned BW | Delta |
|--------|-------|------|-----------------|-------------|--------------|----------|-------|
| 64 | fp8_direct_cast | non-P2P | 256/4 | 147.8 | 112/8 | 149.4 | +1.1% |
| 128 | fp8_direct_cast | non-P2P | 256/8 | 284.6 | 224/8 | 285.9 | +0.5% |
| 256 | fp8_direct_cast | non-P2P | 256/16 | 409.3 | 224/16 | 421.3 | +3.0% |
| 512 | fp8_direct_cast | non-P2P | 256/16 | 523.6 | 224/16 | 525.9 | +0.4% |
| 1024 | fp8_direct_cast | non-P2P | 256/8 | 543.0 | 256/8 | 544.5 | +0.3% |
| 2048 | fp8_direct_cast | non-P2P | 256/16 | 635.2 | 256/16 | 639.0 | +0.6% |
| 4096 | fp8_direct_cast | non-P2P | 256/16 | 642.2 | 256/16 | 642.2 | +0.0% |
| 64 | none | non-P2P | 128/8 | 144.0 | 224/8 | 156.5 | +8.7% |
| 128 | none | non-P2P | 256/8 | 216.6 | 224/8 | 222.2 | +2.6% |
| 256 | none | non-P2P | 256/8 | 282.2 | 224/16 | 288.7 | +2.3% |
| 512 | none | non-P2P | 256/16 | 321.3 | 224/16 | 325.7 | +1.4% |
| 1024 | none | non-P2P | 256/8 | 344.1 | 256/8 | 344.8 | +0.2% |
| 2048 | none | non-P2P | 256/16 | 356.9 | 256/16 | 360.5 | +1.0% |
| 4096 | none | non-P2P | 256/16 | 364.2 | 256/16 | 366.6 | +0.7% |
| 64 | fp8_direct_cast | P2P | 64/8 | 165.1 | 56/8 | 168.2 | +1.9% |
| 128 | fp8_direct_cast | P2P | 64/8 | 243.5 | 64/8 | 244.3 | +0.3% |
| 256 | fp8_direct_cast | P2P | 64/8 | 313.1 | 64/8 | 313.9 | +0.2% |
| 512 | fp8_direct_cast | P2P | 64/8 | 367.1 | 64/8 | 367.9 | +0.2% |
| 1024 | fp8_direct_cast | P2P | 64/8 | 397.6 | 64/16 | 401.8 | +1.1% |
| 2048 | fp8_direct_cast | P2P | 64/8 | 420.0 | 88/8 | 423.5 | +0.9% |
| 4096 | fp8_direct_cast | P2P | 64/8 | 430.9 | 56/15 | 434.1 | +0.7% |
| 64 | none | P2P | 64/8 | 166.0 | 64/8 | 170.2 | +2.6% |
| 128 | none | P2P | 64/8 | 242.8 | 64/8 | 244.8 | +0.8% |
| 256 | none | P2P | 64/8 | 312.0 | 64/8 | 313.5 | +0.5% |
| 512 | none | P2P | 64/8 | 365.8 | 64/8 | 368.1 | +0.6% |
| 1024 | none | P2P | 64/8 | 398.4 | 56/10 | 402.4 | +1.0% |
| 2048 | none | P2P | 64/8 | 419.6 | 72/10 | 424.3 | +1.1% |
| 4096 | none | P2P | 128/4 | 433.8 | 56/15 | 435.5 | +0.4% |

Combine: +0-9% gains. Largest at none/nP2P/64 tokens (+8.7%) where bn=128→224 better fills MI350X's 256 CUs.

## 3. P2P vs non-P2P Detail

### With fp8_direct_cast quant (hidden=7168)

| Tokens | P2P BW | P2P lat | P2P bn/wpb | non-P2P BW | non-P2P lat | non-P2P bn/wpb | Winner | Uplift |
|--------|--------|---------|-----------|-----------|------------|---------------|--------|--------|
| 64 | 168.2 | 28.6 us | 56/8 | 149.4 | 32.7 us | 112/8 | P2P | +13% |
| 128 | 244.4 | 39.8 us | 64/8 | 285.9 | 34.1 us | 224/8 | non-P2P | +17% |
| 256 | 313.9 | 62.2 us | 64/8 | 421.3 | 46.3 us | 224/16 | non-P2P | +34% |
| 512 | 367.9 | 105.7 us | 64/8 | 525.9 | 73.5 us | 224/16 | non-P2P | +43% |
| 1024 | 401.8 | 194.1 us | 64/16 | 544.5 | 143.1 us | 256/8 | non-P2P | +36% |
| 2048 | 423.5 | 367.3 us | 88/8 | 639.0 | 242.6 us | 256/16 | non-P2P | +51% |
| 4096 | 434.2 | 717.2 us | 56/15 | 642.2 | 483.8 us | 256/16 | non-P2P | +48% |

### Without quant (hidden=7168)

| Tokens | P2P BW | P2P lat | P2P bn/wpb | non-P2P BW | non-P2P lat | non-P2P bn/wpb | Winner | Uplift |
|--------|--------|---------|-----------|-----------|------------|---------------|--------|--------|
| 64 | 170.3 | 28.5 us | 64/8 | 156.5 | 30.8 us | 224/8 | P2P | +9% |
| 128 | 244.8 | 39.7 us | 64/8 | 222.2 | 43.9 us | 224/8 | P2P | +10% |
| 256 | 313.5 | 62.2 us | 64/8 | 288.7 | 67.6 us | 224/16 | P2P | +9% |
| 512 | 368.1 | 105.5 us | 64/8 | 325.7 | 119.8 us | 224/16 | P2P | +13% |
| 1024 | 402.4 | 193.3 us | 56/10 | 344.8 | 225.0 us | 256/8 | P2P | +17% |
| 2048 | 424.3 | 366.4 us | 72/10 | 360.5 | 431.4 us | 256/16 | P2P | +18% |
| 4096 | 435.5 | 713.4 us | 56/15 | 366.6 | 848.5 us | 256/16 | P2P | +19% |

## 4. Dispatch Detail

### fp8_e4m3 (hidden=7168)

| Tokens | BW (GB/s) | Latency | block_num | warp_per_block |
|--------|-----------|---------|-----------|----------------|
| 64 | 76.0 | 32.1 us | 128 | 4 |
| 128 | 121.5 | 39.9 us | 208 | 5 |
| 256 | 166.6 | 58.1 us | 256 | 4 |
| 512 | 221.3 | 87.9 us | 228 | 6 |
| 1024 | 272.0 | 142.6 us | 256 | 8 |
| 2048 | 314.4 | 247.6 us | 208 | 8 |
| 4096 | 342.4 | 454.2 us | 256 | 8 |

### fp4 (hidden=3584)

| Tokens | BW (GB/s) | Latency | block_num | warp_per_block |
|--------|-----------|---------|-----------|----------------|
| 64 | 35.7 | 34.1 us | 128 | 4 |
| 128 | 61.4 | 39.8 us | 112 | 15 |
| 256 | 93.8 | 51.9 us | 256 | 8 |
| 512 | 128.8 | 75.6 us | 216 | 10 |
| 1024 | 165.9 | 117.5 us | 184 | 15 |
| 2048 | 201.7 | 192.8 us | 228 | 12 |
| 4096 | 237.4 | 328.0 us | 184 | 15 |

## 5. Code Changes

1. **`python/mori/ops/tuning_config.py`** — Added `_ARCH_CU_TO_MODEL` fallback table so `detect_gpu_model()` returns `"mi350x"` when `torch.cuda.get_device_properties(0).name` is empty (as on MI350X). Maps `(gfx950, 256 CUs) -> "mi350x"`.

## 6. Remaining Work

- [x] EP8 IntraNode (all 4 groups)
- [x] EP4 IntraNode (all 4 groups)
- [x] EP2 IntraNode
- [x] IntraNodeLL
- [x] Profiler analysis (static occupancy — see `mi350x_profiler_analysis.md`; trace profiling needs `ENABLE_PROFILER=ON` rebuild)


