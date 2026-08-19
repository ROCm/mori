# EP Communication Library Performance Evaluation SOP

# Background

1.  Communication library: [mori](https://github.com/amd/mori) (ep-tuning branch)

2.  Hardware / NIC used for the test
    - GPU: 8× AMD Instinct MI355X (gfx950, 256 CU)
    - Intra-node interconnect: XGMI (AMD Infinity Fabric)
    - EP mode: IntraNode EP8 (8 GPUs in a single node)
    - FP8 format: fp8_e4m3 (OCP standard)

3.  Test environment
    - Benchmark script: `tools/bench_ep_performance.sh`
    - Usage examples:
      ```bash
      # Small-message latency test (zero-copy combine by default)
      bash tools/bench_ep_performance.sh --tokens "1,2,4,8,16,32,64,128,256,512,768"

      # Large-message bandwidth test (zero-copy combine by default)
      bash tools/bench_ep_performance.sh --tokens "4096,8192,16384,32768,65536,131072,262144,524288"

      # non-zero-copy combine test
      bash tools/bench_ep_performance.sh --zero-copy 0 --dtypes "bf16" --tokens "1,2,4,8,16,32,64,128,256,512,768"
      bash tools/bench_ep_performance.sh --zero-copy 0 --dtypes "bf16" --tokens "4096,8192,16384,32768,65536,131072,262144,524288"

      # Full sweep
      bash tools/bench_ep_performance.sh
      ```
    - The script automatically creates the output directory `bench_results/ep8_<timestamp>/`, containing `raw/` (raw data) and `summary.txt` (best-performance summary)
    - The environment variable `MORI_SHMEM_HEAP_SIZE` is set automatically based on the token count

# Test Goals

1.  Does it reach the officially published performance?

2.  Does it reach the theoretical hardware bandwidth: intra-node XGMI
    MI355X theoretical XGMI bandwidth: 7x 153.6 GB/s (bidirectional) = 76.8 GB/s/link unidirectional

Derivation of the EP8 dispatch theoretical ceiling (measured XGMI is about 80% of theoretical):
- Theoretical XGMI unidirectional bandwidth = 76.8 GB/s/link, effective bandwidth in practice ≈ 60 GB/s/link
- Total unidirectional send bandwidth of a single GPU = 7 × 60 = 420 GB/s
- During EP8 dispatch, 7/8 of the data travels over XGMI (the local 1/8 needs no communication)
- The bw reported by the bench = total_recv_bytes / time (local data included)
- Theoretical ceiling = 420 × 8/7 ≈ **480 GB/s**

| Metric | Theoretical Ceiling | Measured | Utilization |
|---|---|---|---|
| Dispatch FP8 | 480 GB/s | 377 GB/s | 79% |
| Dispatch BF16 | 480 GB/s | 388 GB/s | 81% |
| Combine ZC | 480 GB/s | 436 GB/s | 91% |
| Combine Non-ZC | 480 GB/s | 366 GB/s | 76% |

3.  Performance comparison against DeepEP on NVIDIA GPUs

# Test Plan

- Use `--cmd tuning --tuning-scope quick` to search for the best block_num / warp_per_block configuration
- Search space: block_num ∈ {32, 64, 128, 256}, warp_per_block ∈ {4, 8, 16}, 12 configurations in total
- Each configuration runs 10 rounds; bw takes the round with the maximum avg(per_rank_bw), lat takes the round with the minimum avg(per_rank_duration)
- Dispatch FP8: `--dtype fp8_e4m3`; Dispatch BF16 + Combine BF16: `--dtype bf16`

# Test Data

At a minimum the following data should be included

## Bandwidth test at large message sizes

| Num Tokens | Dispatch FP8 (GB/s) | Dispatch FP8 Latency (us) | Dispatch FP8 Config | Dispatch BF16 (GB/s) | Dispatch BF16 Latency (us) | Dispatch BF16 Config | Combine ZC (GB/s) | Combine ZC Latency (us) | Combine ZC Config | Combine Non-ZC (GB/s) | Combine Non-ZC Latency (us) | Combine Non-ZC Config |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 331.27 | 469.0 | 128,16 | 359.92 | 863.0 | 256,8 | 419.25 | 741.1 | 128,4 | 359.21 | 865.5 | 256,16 |
| 8192 | 349.21 | 888.9 | 256,8 | 369.86 | 1681.7 | 256,16 | 425.26 | 1461.1 | 128,4 | 362.45 | 1716.0 | 256,16 |
| 16384 | 361.31 | 1722.5 | 256,16 | 377.40 | 3298.2 | 256,16 | 429.50 | 2895.5 | 128,4 | 364.50 | 3414.8 | 256,16 |
| 32768 | 366.60 | 3393.1 | 256,16 | 380.18 | 6543.7 | 256,16 | 432.74 | 5749.1 | 128,4 | 364.10 | 6832.3 | 256,16 |
| 65536 | 372.14 | 6681.5 | 256,16 | 383.03 | 12983.1 | 256,16 | 433.93 | 11462.6 | 128,4 | 364.91 | 13627.0 | 256,16 |
| 131072 | 372.78 | 13345.7 | 256,16 | 383.12 | 25971.6 | 256,16 | 435.08 | 22867.7 | 128,4 | 364.70 | 27281.2 | 256,16 |
| 262144 | 374.48 | 26566.9 | 256,16 | 387.79 | 51311.4 | 256,16 | 435.28 | 45713.9 | 128,4 | 362.73 | 54851.7 | 256,16 |
| 524288 | 377.24 | 52745.7 | 256,16 | 388.24 | 102501.6 | 256,16 | 435.83 | 91308.9 | 128,4 | 365.83 | 108780.4 | 256,16 |

## Latency test at small message sizes

| Num Tokens | Dispatch FP8 (GB/s) | Dispatch FP8 Latency (us) | Dispatch FP8 Config | Dispatch BF16 (GB/s) | Dispatch BF16 Latency (us) | Dispatch BF16 Config | Combine ZC (GB/s) | Combine ZC Latency (us) | Combine ZC Config | Combine Non-ZC (GB/s) | Combine Non-ZC Latency (us) | Combine Non-ZC Config |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.44 | 28.6 | 128,16 | 2.54 | 32.4 | 128,4 | 4.26 | 19.4 | 128,4 | 2.63 | 27.9 | 32,4 |
| 2 | 2.82 | 27.4 | 256,4 | 4.87 | 31.7 | 256,4 | 8.11 | 18.8 | 32,8 | 5.38 | 28.3 | 32,4 |
| 4 | 5.61 | 27.9 | 128,4 | 9.75 | 32.2 | 128,4 | 15.84 | 19.8 | 128,4 | 10.52 | 29.1 | 32,4 |
| 8 | 11.33 | 28.0 | 64,4 | 19.38 | 32.7 | 64,4 | 30.31 | 19.9 | 128,8 | 21.47 | 29.5 | 64,4 |
| 16 | 22.19 | 28.1 | 128,4 | 37.79 | 33.0 | 128,4 | 56.05 | 21.8 | 128,16 | 43.40 | 28.8 | 256,8 |
| 32 | 42.81 | 28.5 | 256,4 | 74.64 | 33.0 | 64,4 | 102.00 | 24.2 | 64,4 | 80.63 | 30.3 | 256,4 |
| 64 | 77.58 | 31.3 | 128,4 | 129.60 | 37.4 | 256,4 | 171.21 | 28.3 | 64,8 | 149.48 | 32.5 | 128,8 |
| 128 | 122.95 | 39.5 | 256,4 | 187.12 | 51.9 | 256,4 | 243.91 | 40.1 | 64,8 | 218.09 | 44.4 | 256,8 |
| 256 | 164.74 | 59.0 | 256,4 | 239.64 | 81.2 | 256,4 | 307.94 | 63.2 | 64,8 | 282.49 | 68.5 | 256,8 |
| 512 | 216.08 | 89.9 | 256,8 | 283.19 | 137.2 | 256,8 | 360.48 | 107.8 | 64,8 | 312.47 | 124.2 | 256,16 |
| 768 | 249.58 | 116.8 | 256,8 | 311.60 | 187.0 | 256,8 | 375.92 | 155.3 | 64,16 | 297.46 | 195.8 | 256,16 |

## Comparative Analysis

Add comparative-analysis data according to the test goals

# Conclusions

# Open Issues
