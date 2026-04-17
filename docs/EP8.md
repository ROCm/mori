# EP通信库性能测评 SOP

# 背景

1.  通信库链接: [mori](https://github.com/amd/mori) (ep-tuning 分支)

2.  测试基于的相关硬件/网卡说明
    - GPU: 8× AMD Instinct MI355X (gfx950, 256 CU)
    - 机内互联: XGMI (AMD Infinity Fabric)
    - EP 模式: IntraNode EP8 (单机 8 卡)
    - FP8 格式: fp8_e4m3 (OCP 标准)

3.  测试环境
    - 测试脚本: `tools/bench_ep_performance.sh`
    - 用法示例:
      ```bash
      # 小数据量延迟测试（默认 zero-copy combine）
      bash tools/bench_ep_performance.sh --tokens "1,2,4,8,16,32,64,128,256,512,768"

      # 大数据量带宽测试（默认 zero-copy combine）
      bash tools/bench_ep_performance.sh --tokens "4096,8192,16384,32768,65536,131072,262144,524288"

      # non-zero-copy combine 测试
      bash tools/bench_ep_performance.sh --zero-copy 0 --dtypes "bf16" --tokens "1,2,4,8,16,32,64,128,256,512,768"
      bash tools/bench_ep_performance.sh --zero-copy 0 --dtypes "bf16" --tokens "4096,8192,16384,32768,65536,131072,262144,524288"

      # 全量测试
      bash tools/bench_ep_performance.sh
      ```
    - 脚本自动创建 `bench_results/ep8_<timestamp>/` 输出目录，包含 `raw/`（原始数据）和 `summary.txt`（最佳性能汇总）
    - 环境变量 `MORI_SHMEM_HEAP_SIZE` 根据 token 数自动设置

# 测试目标

1.  是否达到官方给出性能

2.  是否达到硬件理论带宽：机内XGMI
    MI355X 理论XGMI带宽：7x 153.6 GB/s（双向）= 单向 76.8 GB/s/link

EP8 Dispatch 理论上限推导（XGMI 实测约为理论值的 80%）：
- XGMI 理论单向带宽 = 76.8 GB/s/link，实际有效带宽 ≈ 60 GB/s/link
- 单个 GPU 总单向发送带宽 = 7 × 60 = 420 GB/s
- EP8 dispatch 时 7/8 数据走 XGMI（1/8 本地不需通信）
- bench 报告的 bw = total_recv_bytes / time（含本地数据）
- 理论上限 = 420 × 8/7 ≈ **480 GB/s**

| 指标 | 理论上限 | 实测 | 利用率 |
|---|---|---|---|
| Dispatch FP8 | 480 GB/s | 377 GB/s | 79% |
| Dispatch BF16 | 480 GB/s | 388 GB/s | 81% |
| Combine ZC | 480 GB/s | 436 GB/s | 91% |
| Combine Non-ZC | 480 GB/s | 366 GB/s | 76% |

3.  与N卡的DeepEP性能对比

# 测试方案

- 使用 `--cmd tuning --tuning-scope quick` 搜索最佳 block_num / warp_per_block 配置
- 搜索空间: block_num ∈ {32, 64, 128, 256}, warp_per_block ∈ {4, 8, 16}，共 12 种配置
- 每个配置跑 10 轮，bw 取 avg(per_rank_bw) 的最大轮次，lat 取 avg(per_rank_duration) 的最小轮次
- Dispatch FP8: `--dtype fp8_e4m3`; Dispatch BF16 + Combine BF16: `--dtype bf16`

# 测试数据

应至少包括以下数据

## 大数据量下带宽测试

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

## 小数据量下延迟测试

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

## 对比分析

根据测试目标添加对比分析的数据

# 测试结论

# 遗留问题
