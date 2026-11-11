# EP V1 vs V2 Inter-node Performance (DeepSeek-R1 on MI308X, MI300X and MI355X)

This compares the inter-node dispatch/combine performance of MORI EP V1 and V2
for the DeepSeek-R1 shape on three GPUs: MI308X, MI300X and MI355X. Section 1 is
the test setup, section 2 the overall results, and section 3 the per-bucket data
for each GPU.

## 1. Test setup

### 1.1 Common settings

- **Shape**: 256 experts, top-8 per token, hidden 7168. EP16: two nodes × 8 GPUs.
- **Workload**: uniform routing, buffer capacity equal to each call's token
  count, no scales, no quantization. Combine is BF16; dispatch is measured in
  both FP8 and BF16. FP8 is OCP e4m3 on MI355X and e4m3 fnuz on MI308X and
  MI300X. T is the number of tokens per rank per call, from 4 to 4096.
- **Kernels**: for T ≤ 512 the LL kernels are compared (V1 InterNodeV1LL vs V2
  InterNodeV2LL), and for larger T the normal kernels (V1 InterNodeV1 vs V2
  InterNodeV2), matching the switch point of V2's `internode_kernel="auto"`.
  Every row compares kernels of the same family.
- **V1**: SHMEM backend with `num_qp_per_pe` 2. That is the default of the V1
  tuning script (`examples/ops/dispatch_combine/test_dispatch_combine_internode.py`),
  and V1's tuning tables were tuned with it; the V1 API default is 1. Geometry
  is chosen from V1's tuning tables by `MORI_EP_LAUNCH_CONFIG_MODE=AUTO`; the
  rules actually used on each GPU are listed in section 3.
- **V2**: HIP backend with the default `num_qp_per_pe` of 8. The geometry and
  each bucket's active QP count come from the tuning tables in
  `python/mori/ops/dispatch_combine_v2/tuning_configs/`.
- **Timing**: both sides use the same data and correctness checks. Each
  measurement is 3 passes × (20 warmup + 30 timed rounds), dropping the first
  round of each pass, and latency is the mean. V1 and V2 are measured
  alternately three times, and each cell is the median of the three; runs that
  stalled throughout were discarded or re-measured. The D+C change compares the
  medians of each run's D+C latency, not the sum of the D and C medians shown,
  so it can differ from a change computed from those cells; the largest
  difference is MI308X FP8 T4096 (−22.4% here, −25.1% from the D and C cells).
  Every run used here passed the correctness checks.
- **Bandwidth**: as in V1, the mean over ranks and rounds of "bytes / time",
  not the NIC line rate. The two variants differ only in how bytes are counted:
  - **XGMI bandwidth** counts the tokens this rank received.
  - **RDMA bandwidth** counts token × destination node, the local node
    included; for LL every token counts all nodes. With two nodes, about half
    of it actually crosses nodes.
  - Both sides move the same bytes in a cell, so a bandwidth change roughly
    mirrors the latency change. It is not the exact inverse, since bandwidth is
    a mean of per-round bytes / time and each cell is its own median; in two
    MI308X cells (FP8 T4096 combine, BF16 T2048 dispatch) both changes are
    negative, so the latency change favors V2 and the bandwidth change V1. In
    every row the XGMI and RDMA changes differ by less than 1.5 percentage
    points.

### 1.2 Environment per GPU

| | MI308X | MI300X | MI355X |
|---|---|---|---|
| GPU | AMD Instinct MI308X | AMD Instinct MI300X | AMD Instinct MI355X |
| NIC (8 per node, 400 Gb/s) | Broadcom BCM57608, RoCE | NVIDIA ConnectX-7 (MT2910), InfiniBand | AMD Pensando Pollara 400 (ionic), RoCE |
| ROCm | 7.2.1, with the HIP runtime built separately from CLR source | 7.14 (HIP 7.14.60850) | 7.14 (HIP 7.14.60850) |
| NIC firmware | 235.2.40.0 | 28.40.1702 | 1.117.5-a-45 |

The MI308X numbers were measured with a HIP runtime built from CLR source, not
the one ROCm 7.2.1 ships: with the stock 7.0–7.2.x runtime V2 cannot set up
across two nodes (`hipMemSetAccess` fails on its symmetric-memory windows; see
Issue 2 of the known-issues skill, `.claude/skills/known-issues/SKILL.md`).
ROCm 7.14 and later need no rebuild.

## 2. Overall results

**How much lower V2's D+C latency is than V1's:**

| GPU | dispatch | small batch (T4–T32) | medium batch (T64–T512) | large batch (T1024–T4096) |
|---|---|---|---|---|
| MI308X | FP8 | 20%–31% | 11%–18% | 22%–28% |
| MI308X | BF16 | 18%–33% | 10%–16% | 9%–13% |
| MI300X | FP8 | 36%–42% | 12%–22% | 25%–38%; V1 runs borrowed rules, see 3.2 |
| MI300X | BF16 | 35%–43% | 10%–22% | 11%–22% |
| MI355X | FP8 | 23%–30% | 12%–19% | 7%–17% |
| MI355X | BF16 | 21%–30% | 13%–16% | 9%–15% |

**Large-batch (T1024–T4096) bandwidth change** (XGMI; the RDMA figures differ by
less than 1.5 percentage points):

| GPU | dispatch | dispatch bandwidth | combine bandwidth |
|---|---|---|---|
| MI308X | FP8 | +67% to +98% | −4% to +15% |
| MI308X | BF16 | −1% to +27% | −10% to +20% |
| MI300X | FP8 | +89% to +127%; V1 runs borrowed rules, see 3.2 | +4% to +35% |
| MI300X | BF16 | +14% to +20% | +8% to +35% |
| MI355X | FP8 | +11% to +18% | +4% to +20% |
| MI355X | BF16 | +10% to +14% | +7% to +21% |

- **V2 is faster in every bucket on all three GPUs.** No cell has a higher V2
  D+C latency than V1.
- **The gain is largest at small batch.** Small batches are 18%–43% faster;
  medium and large batches usually 7%–22%. FP8 large batches on MI308X and
  MI300X gain more (22%–38%), mostly in dispatch: on MI308X the two tuning
  tables themselves differ that much, and on MI300X V1 runs rules tuned for
  MI308X.
- **Small batches gain mostly in combine.** At large batch it varies by GPU:
  FP8 on MI308X and MI300X gains mostly in dispatch; BF16 on MI300X and both
  dtypes on MI355X gain in both phases, by 8%–27% and 5%–18% respectively;
  MI308X's combine is within 7% of V1 except at T2048.
- **Treat the MI308X small-batch numbers with caution.** At T ≤ 32 V1 runs much
  slower than its own tuning table records; going by the two tables, V1 is the
  faster side at T ≤ 32, with V2's FP8 D+C 3%–8% higher. See 3.1.
- **Bandwidth is listed for large batches only.** Small batches are
  latency-bound, and their bandwidth change only mirrors the latency change.
- **Part of the gap comes from rules V1 never tuned.** V1 has no BF16 dispatch
  rules for H7168 on MI308X and only some on MI355X, and V1's MI300X table has
  only LL rules for T ≤ 64; the MI308X rules the normal kernel borrows are too
  small for MI300X. See section 3.
- **This is a kernel-level comparison, not end-to-end inference.** Only this
  one shape with uniform routing was measured; real workloads route unevenly
  and run expert computation between dispatch and combine.

## 3. Per-GPU data

In the kernel column, LL means V1 InterNodeV1LL vs V2 InterNodeV2LL, and normal
means V1 InterNodeV1 vs V2 InterNodeV2. D is dispatch, C is combine, and T is
the number of tokens per rank per call. Latency is in µs and bandwidth in GB/s;
bandwidth cells read "V1 → V2 (change)". Change = V2 / V1 − 1, so a negative
latency change and a positive bandwidth change mean V2 is better.

### 3.1 MI308X

| dispatch | T | kernel | V1 D / C | V2 D / C | D latency | C latency | D+C latency | XGMI BW D | XGMI BW C | RDMA BW D | RDMA BW C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP8 | 4 | LL | 60.0 / 93.0 | 47.0 / 67.8 | −21.7% | −27.1% | −24.9% | 3.20 → 4.06 (+26.9%) | 4.11 → 5.63 (+37.0%) | 0.96 → 1.22 (+27.1%) | 1.24 → 1.69 (+36.3%) |
| FP8 | 8 | LL | 61.3 / 101.2 | 50.9 / 60.6 | −16.9% | −40.1% | −31.4% | 6.15 → 7.39 (+20.2%) | 7.44 → 12.41 (+66.8%) | 1.88 → 2.26 (+20.2%) | 2.27 → 3.80 (+67.4%) |
| FP8 | 16 | LL | 61.9 / 103.8 | 48.7 / 69.0 | −21.4% | −33.5% | −28.9% | 12.13 → 15.47 (+27.5%) | 14.46 → 21.72 (+50.2%) | 3.72 → 4.74 (+27.4%) | 4.43 → 6.66 (+50.3%) |
| FP8 | 32 | LL | 67.3 / 114.8 | 57.6 / 87.5 | −14.5% | −23.8% | −20.3% | 22.32 → 26.30 (+17.8%) | 26.10 → 34.38 (+31.7%) | 6.85 → 8.07 (+17.8%) | 8.00 → 10.54 (+31.7%) |
| FP8 | 64 | LL | 85.9 / 153.0 | 77.0 / 121.7 | −10.3% | −20.5% | −16.8% | 35.18 → 40.02 (+13.8%) | 39.34 → 49.77 (+26.5%) | 10.74 → 12.21 (+13.7%) | 12.01 → 15.21 (+26.6%) |
| FP8 | 128 | LL | 121.4 / 203.8 | 97.5 / 176.8 | −19.7% | −13.2% | −15.6% | 50.76 → 62.66 (+23.4%) | 59.27 → 68.20 (+15.1%) | 15.59 → 19.23 (+23.3%) | 18.18 → 20.92 (+15.1%) |
| FP8 | 256 | LL | 186.2 / 304.1 | 153.8 / 280.8 | −17.4% | −7.7% | −11.4% | 65.86 → 79.56 (+20.8%) | 79.45 → 85.87 (+8.1%) | 20.17 → 24.36 (+20.8%) | 24.33 → 26.30 (+8.1%) |
| FP8 | 512 | LL | 361.7 / 514.4 | 246.2 / 469.7 | −31.9% | −8.7% | −18.2% | 67.22 → 97.58 (+45.2%) | 93.75 → 101.92 (+8.7%) | 20.63 → 29.96 (+45.2%) | 28.77 → 31.28 (+8.7%) |
| FP8 | 1024 | normal | 826.6 / 913.8 | 414.2 / 853.5 | −49.9% | −6.6% | −27.1% | 60.13 → 118.95 (+97.8%) | 108.08 → 112.84 (+4.4%) | 18.36 → 36.30 (+97.7%) | 32.97 → 34.43 (+4.4%) |
| FP8 | 2048 | normal | 1531.6 / 1549.8 | 911.5 / 1317.7 | −40.5% | −15.0% | −27.7% | 65.01 → 108.71 (+67.2%) | 127.32 → 146.99 (+15.4%) | 19.86 → 33.22 (+67.3%) | 38.89 → 44.90 (+15.5%) |
| FP8 | 4096 | normal | 3264.7 / 3167.7 | 1719.3 / 3098.9 | −47.3% | −2.2% | −22.4% | 62.56 → 112.37 (+79.6%) | 128.47 → 123.82 (−3.6%) | 19.13 → 34.35 (+79.6%) | 39.29 → 37.85 (−3.7%) |
| BF16 | 4 | LL | 59.8 / 93.2 | 47.4 / 66.0 | −20.6% | −29.2% | −25.7% | 6.42 → 8.05 (+25.4%) | 4.10 → 5.78 (+41.0%) | 1.93 → 2.43 (+25.9%) | 1.23 → 1.74 (+41.5%) |
| BF16 | 8 | LL | 62.1 / 101.0 | 49.7 / 60.0 | −19.9% | −40.6% | −32.7% | 12.13 → 15.18 (+25.1%) | 7.44 → 12.53 (+68.4%) | 3.71 → 4.64 (+25.1%) | 2.28 → 3.83 (+68.0%) |
| BF16 | 16 | LL | 66.4 / 102.6 | 52.9 / 68.1 | −20.4% | −33.6% | −28.4% | 22.61 → 28.82 (+27.5%) | 14.62 → 22.20 (+51.8%) | 6.94 → 8.83 (+27.2%) | 4.48 → 6.81 (+52.0%) |
| BF16 | 32 | LL | 78.7 / 114.4 | 70.8 / 87.8 | −10.0% | −23.3% | −17.9% | 38.20 → 43.51 (+13.9%) | 26.21 → 34.70 (+32.4%) | 11.71 → 13.34 (+13.9%) | 8.04 → 10.64 (+32.3%) |
| BF16 | 64 | LL | 109.9 / 152.3 | 99.3 / 121.0 | −9.7% | −20.5% | −16.0% | 55.32 → 61.00 (+10.3%) | 39.71 → 49.94 (+25.8%) | 16.90 → 18.63 (+10.2%) | 12.12 → 15.25 (+25.8%) |
| BF16 | 128 | LL | 168.6 / 204.2 | 144.8 / 177.2 | −14.1% | −13.2% | −13.7% | 72.61 → 83.25 (+14.7%) | 59.60 → 68.27 (+14.5%) | 22.30 → 25.55 (+14.6%) | 18.28 → 20.94 (+14.6%) |
| BF16 | 256 | LL | 269.2 / 311.9 | 239.9 / 283.2 | −10.9% | −9.2% | −10.2% | 91.34 → 100.94 (+10.5%) | 78.07 → 85.22 (+9.2%) | 27.96 → 30.91 (+10.6%) | 23.92 → 26.10 (+9.1%) |
| BF16 | 512 | LL | 497.5 / 507.4 | 418.6 / 469.6 | −15.9% | −7.4% | −11.6% | 97.28 → 114.51 (+17.7%) | 95.23 → 101.94 (+7.0%) | 29.85 → 35.15 (+17.8%) | 29.24 → 31.29 (+7.0%) |
| BF16 | 1024 | normal | 923.9 / 874.8 | 713.1 / 837.8 | −22.8% | −4.2% | −12.7% | 106.49 → 135.41 (+27.2%) | 112.56 → 114.85 (+2.0%) | 32.49 → 41.31 (+27.1%) | 34.35 → 35.05 (+2.0%) |
| BF16 | 2048 | normal | 1622.4 / 1616.5 | 1607.5 / 1321.6 | −0.9% | −18.2% | −9.4% | 121.21 → 120.21 (−0.8%) | 121.53 → 146.26 (+20.3%) | 37.03 → 36.72 (−0.8%) | 37.12 → 44.68 (+20.4%) |
| BF16 | 4096 | normal | 3619.4 / 2957.5 | 2765.5 / 3110.2 | −23.6% | +5.2% | −11.2% | 110.66 → 138.71 (+25.3%) | 136.87 → 123.31 (−9.9%) | 33.83 → 42.41 (+25.4%) | 41.84 → 37.70 (−9.9%) |

- **Where the gain comes from.** Up to T64 the LL kernel gains mostly in
  combine, about 40% at T8; from T128 to T512 dispatch gains as much or more.
  The normal kernel gains mostly in dispatch, which is 40%–50% faster with FP8;
  BF16 T2048 is the exception, with dispatch level and combine 18% faster. The
  normal kernel's combine is clearly faster only at T2048 (15%–18%); at T1024
  and T4096 it is within 7% of V1.
- **V1 has no tuned rules for BF16 dispatch.** V1's tuning tables have no BF16
  dispatch rule for H7168: V1's LL kernel borrows the H6144 (GLM-5) rules by
  relaxed matching (the four for T ≤ 32, with larger T reusing the T32 rule),
  and the normal kernel uses the AUTO default 96/64/8. So part of the BF16 gap
  comes from V1 not being tuned; with FP8 both sides use their own tuned rules.
- **Data and stability.** This is the cleanest of five identical measurements
  taken from 2026-11-03 to 11-05. V1's per-cell spread across the five has a
  median of about 2%; V2 is as stable at T ≤ 512 and in combine. V2's
  large-batch dispatch is not: in the 11-04 run FP8 T1024 and T2048 were
  23%–32% faster than in the other four; FP8 T4096 occasionally has rounds
  whose mean latency is 2–4× higher, and in one of the five runs two of the
  three rounds stalled; BF16 T1024 and T2048 were 8%–17% slower on 11-05 than
  on 11-03 and 11-04. The cause has not been investigated.
- **V1 is slower at small batch than its own tuning table.** At T ≤ 32, V1's
  dispatch is 13%–30% slower than the table records and its combine 37%–67%
  slower, in all five measurements; at T64–T512 it is within ±10% of the table.
  V2 is within 7% of its table at T ≤ 512. Going by the two tables, V2's FP8
  D+C at T ≤ 32 is 3%–8% higher than V1's (V1 has no H7168 rule for BF16
  dispatch), so the small-batch gap here may come entirely from the
  measurement conditions; the cause has not been investigated. These V1
  rules (#689) were measured with V1's own benchmark script, keeping only the
  lowest-latency runs.

### 3.2 MI300X

| dispatch | T | kernel | V1 D / C | V2 D / C | D latency | C latency | D+C latency | XGMI BW D | XGMI BW C | RDMA BW D | RDMA BW C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP8 | 4 | LL | 45.7 / 77.7 | 37.2 / 40.3 | −18.7% | −48.1% | −37.2% | 4.20 → 5.20 (+23.8%) | 4.92 → 9.50 (+93.1%) | 1.26 → 1.57 (+24.6%) | 1.48 → 2.86 (+93.2%) |
| FP8 | 8 | LL | 44.9 / 77.8 | 37.8 / 40.7 | −15.7% | −47.6% | −36.0% | 8.42 → 10.10 (+20.0%) | 9.68 → 18.55 (+91.6%) | 2.58 → 3.09 (+19.8%) | 2.96 → 5.67 (+91.6%) |
| FP8 | 16 | LL | 45.0 / 89.4 | 35.0 / 43.2 | −22.1% | −51.7% | −41.6% | 16.71 → 21.45 (+28.4%) | 16.79 → 34.72 (+106.8%) | 5.13 → 6.57 (+28.1%) | 5.14 → 10.65 (+107.2%) |
| FP8 | 32 | LL | 47.7 / 95.3 | 39.7 / 52.5 | −16.7% | −45.0% | −35.5% | 31.52 → 37.87 (+20.1%) | 31.43 → 57.23 (+82.1%) | 9.67 → 11.61 (+20.1%) | 9.64 → 17.55 (+82.1%) |
| FP8 | 64 | LL | 56.4 / 106.3 | 50.9 / 76.8 | −9.7% | −27.7% | −21.5% | 53.43 → 59.37 (+11.1%) | 56.61 → 78.47 (+38.6%) | 16.32 → 18.14 (+11.2%) | 17.29 → 23.96 (+38.6%) |
| FP8 | 128 | LL | 83.0 / 146.1 | 69.8 / 117.7 | −15.8% | −19.4% | −18.0% | 72.48 → 87.17 (+20.3%) | 82.07 → 102.17 (+24.5%) | 22.22 → 26.73 (+20.3%) | 25.15 → 31.30 (+24.5%) |
| FP8 | 256 | LL | 110.7 / 214.1 | 106.0 / 178.4 | −4.3% | −16.7% | −12.3% | 109.39 → 113.52 (+3.8%) | 112.29 → 134.58 (+19.9%) | 33.48 → 34.74 (+3.8%) | 34.36 → 41.18 (+19.8%) |
| FP8 | 512 | LL | 181.0 / 362.1 | 151.2 / 313.1 | −16.4% | −13.5% | −14.5% | 137.20 → 160.34 (+16.9%) | 133.49 → 153.42 (+14.9%) | 42.00 → 49.09 (+16.9%) | 40.88 → 46.98 (+14.9%) |
| FP8 | 1024 | normal | 576.4 / 606.5 | 251.9 / 492.1 | −56.3% | −18.9% | −37.1% | 83.86 → 190.53 (+127.2%) | 159.15 → 194.91 (+22.5%) | 25.60 → 58.16 (+127.2%) | 48.59 → 59.50 (+22.5%) |
| FP8 | 2048 | normal | 947.3 / 1159.1 | 458.4 / 848.9 | −51.6% | −26.8% | −37.9% | 102.62 → 208.81 (+103.5%) | 166.73 → 225.45 (+35.2%) | 31.39 → 63.87 (+103.5%) | 50.99 → 68.95 (+35.2%) |
| FP8 | 4096 | normal | 1749.4 / 1756.7 | 930.3 / 1683.0 | −46.8% | −4.2% | −25.4% | 109.45 → 206.64 (+88.8%) | 218.15 → 227.76 (+4.4%) | 33.46 → 63.18 (+88.8%) | 66.70 → 69.65 (+4.4%) |
| BF16 | 4 | LL | 62.4 / 78.2 | 42.1 / 40.0 | −32.6% | −48.9% | −41.6% | 6.14 → 9.19 (+49.7%) | 4.89 → 9.60 (+96.3%) | 1.85 → 2.77 (+49.7%) | 1.47 → 2.89 (+96.6%) |
| BF16 | 8 | LL | 60.9 / 78.3 | 41.8 / 39.7 | −31.4% | −49.3% | −41.2% | 12.36 → 18.24 (+47.6%) | 9.62 → 18.99 (+97.4%) | 3.78 → 5.57 (+47.4%) | 2.94 → 5.81 (+97.6%) |
| BF16 | 16 | LL | 62.0 / 88.3 | 42.2 / 43.1 | −31.9% | −51.2% | −43.2% | 24.20 → 35.56 (+46.9%) | 16.98 → 34.84 (+105.2%) | 7.42 → 10.90 (+46.9%) | 5.21 → 10.68 (+105.0%) |
| BF16 | 32 | LL | 65.3 / 92.9 | 50.0 / 52.4 | −23.5% | −43.6% | −35.3% | 45.90 → 60.12 (+31.0%) | 32.23 → 57.35 (+77.9%) | 14.07 → 18.43 (+31.0%) | 9.89 → 17.59 (+77.9%) |
| BF16 | 64 | LL | 81.1 / 106.4 | 68.6 / 77.4 | −15.4% | −27.2% | −22.1% | 74.46 → 87.93 (+18.1%) | 56.73 → 77.89 (+37.3%) | 22.75 → 26.85 (+18.0%) | 17.32 → 23.78 (+37.3%) |
| BF16 | 128 | LL | 115.6 / 148.4 | 97.0 / 118.4 | −16.1% | −20.2% | −18.5% | 104.02 → 124.53 (+19.7%) | 81.00 → 101.77 (+25.6%) | 31.89 → 38.17 (+19.7%) | 24.82 → 31.19 (+25.7%) |
| BF16 | 256 | LL | 167.9 / 218.1 | 145.8 / 177.7 | −13.2% | −18.5% | −16.1% | 143.42 → 164.79 (+14.9%) | 110.16 → 135.06 (+22.6%) | 43.89 → 50.43 (+14.9%) | 33.71 → 41.33 (+22.6%) |
| BF16 | 512 | LL | 268.0 / 354.4 | 246.3 / 314.6 | −8.1% | −11.2% | −9.9% | 179.45 → 195.51 (+8.9%) | 135.50 → 152.75 (+12.7%) | 54.94 → 59.86 (+9.0%) | 41.49 → 46.78 (+12.8%) |
| BF16 | 1024 | normal | 587.2 / 608.8 | 483.6 / 505.7 | −17.6% | −16.9% | −17.3% | 166.27 → 198.61 (+19.5%) | 159.42 → 189.94 (+19.1%) | 50.75 → 60.62 (+19.4%) | 48.67 → 57.99 (+19.1%) |
| BF16 | 2048 | normal | 1020.7 / 1173.4 | 844.1 / 858.8 | −17.3% | −26.8% | −22.3% | 191.29 → 226.85 (+18.6%) | 165.30 → 222.93 (+34.9%) | 58.52 → 69.38 (+18.6%) | 50.55 → 68.18 (+34.9%) |
| BF16 | 4096 | normal | 1951.8 / 1829.6 | 1692.5 / 1674.1 | −13.3% | −8.5% | −10.9% | 198.32 → 226.68 (+14.3%) | 211.64 → 229.08 (+8.2%) | 60.63 → 69.31 (+14.3%) | 64.72 → 70.05 (+8.2%) |

- **Where the gain comes from.** Small batches (LL) gain mostly in combine,
  11%–52% across the buckets; large batches (normal kernel) gain mostly in
  dispatch with FP8 (47%–56%), and in both phases with BF16 (8%–27%).
- **System setting.** NUMA auto-balancing was disabled
  (`kernel.numa_balancing=0`). With it on, the kernel's scans of process memory
  make KFD evict all of a process's GPU queues for about 17 ms, and in EP16
  every rank waits when any one rank is evicted. Both V1 and V2 are affected:
  mean latency rises noticeably and a table can show a false "V2 is slower",
  while bandwidth and medians are less affected.
- **For most buckets V1 has no rules tuned for MI300X.** The configurations
  AUTO actually used:
  - LL FP8 dispatch: T4–T64 use the rules in the MI300X table (16/8/6,
    64/32/4, 64/32/4, 128/64/4, 256/128/4 in order), and T128–T512 reuse the
    T64 rule (256/128/4).
  - LL BF16 dispatch: the table has no BF16 rules, so the AUTO default
    256/128/8 is used.
  - LL combine: T4–T64 use the rules in the MI300X table (304/76/4, 304/76/4,
    128/32/8, 64/42/8, 128/85/4 in order), and T128–T512 reuse the T64 rule
    (128/85/4).
  - Normal kernel (T1024–T4096): MI300X has no InterNodeV1 table, so AUTO
    borrows the table of MI308X, also gfx942. FP8 dispatch uses 32/16/8,
    64/32/4 and 32/16/8 in order, and combine 80/40/8, 64/32/16 and 64/32/8;
    BF16 dispatch has no rules and uses the default 96/64/8. MI308X has only 80
    CUs, so these rules are too small for the 304-CU MI300X: V1's FP8 T1024
    dispatch is 576 µs, against about 355 µs with the same 2 QPs and the
    default geometry 96/64/8. Much of the FP8 normal-kernel dispatch gap comes
    from this.

### 3.3 MI355X

| dispatch | T | kernel | V1 D / C | V2 D / C | D latency | C latency | D+C latency | XGMI BW D | XGMI BW C | RDMA BW D | RDMA BW C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP8 | 4 | LL | 46.4 / 65.9 | 36.2 / 44.3 | −21.9% | −32.8% | −28.1% | 4.12 → 5.33 (+29.4%) | 5.78 → 8.66 (+49.8%) | 1.24 → 1.60 (+29.0%) | 1.74 → 2.61 (+50.0%) |
| FP8 | 8 | LL | 46.5 / 66.0 | 37.5 / 45.4 | −19.3% | −31.3% | −26.6% | 8.10 → 10.20 (+25.9%) | 11.39 → 16.81 (+47.6%) | 2.48 → 3.12 (+25.8%) | 3.48 → 5.15 (+48.0%) |
| FP8 | 16 | LL | 49.5 / 78.8 | 39.8 / 49.6 | −19.5% | −37.1% | −30.2% | 15.19 → 19.13 (+25.9%) | 19.02 → 30.48 (+60.3%) | 4.66 → 5.86 (+25.8%) | 5.83 → 9.35 (+60.4%) |
| FP8 | 32 | LL | 56.1 / 81.8 | 45.0 / 61.8 | −19.7% | −24.5% | −22.6% | 26.86 → 33.70 (+25.5%) | 36.69 → 48.80 (+33.0%) | 8.24 → 10.33 (+25.4%) | 11.25 → 14.97 (+33.1%) |
| FP8 | 64 | LL | 70.5 / 110.2 | 62.6 / 87.1 | −11.3% | −20.9% | −17.2% | 42.78 → 48.53 (+13.4%) | 54.66 → 69.34 (+26.9%) | 13.07 → 14.83 (+13.5%) | 16.69 → 21.17 (+26.8%) |
| FP8 | 128 | LL | 87.3 / 134.8 | 72.2 / 107.4 | −17.4% | −20.3% | −19.1% | 69.92 → 84.37 (+20.7%) | 89.39 → 112.38 (+25.7%) | 21.42 → 25.87 (+20.8%) | 27.41 → 34.44 (+25.6%) |
| FP8 | 256 | LL | 114.4 / 192.1 | 99.3 / 164.0 | −13.2% | −14.6% | −14.1% | 107.50 → 122.58 (+14.0%) | 125.76 → 146.98 (+16.9%) | 32.90 → 37.51 (+14.0%) | 38.48 → 44.98 (+16.9%) |
| FP8 | 512 | LL | 155.6 / 306.2 | 143.5 / 265.3 | −7.8% | −13.4% | −11.5% | 155.43 → 169.23 (+8.9%) | 156.94 → 181.37 (+15.6%) | 47.59 → 51.82 (+8.9%) | 48.06 → 55.54 (+15.6%) |
| FP8 | 1024 | normal | 274.6 / 557.3 | 232.0 / 460.2 | −15.5% | −17.4% | −16.8% | 180.60 → 207.48 (+14.9%) | 173.51 → 208.74 (+20.3%) | 55.07 → 63.27 (+14.9%) | 52.91 → 63.65 (+20.3%) |
| FP8 | 2048 | normal | 501.0 / 918.5 | 411.8 / 829.2 | −17.8% | −9.7% | −12.6% | 197.90 → 233.91 (+18.2%) | 210.40 → 231.07 (+9.8%) | 60.58 → 71.61 (+18.2%) | 64.40 → 70.72 (+9.8%) |
| FP8 | 4096 | normal | 872.8 / 1650.5 | 765.8 / 1573.5 | −12.3% | −4.7% | −7.3% | 227.16 → 251.85 (+10.9%) | 234.25 → 243.84 (+4.1%) | 69.39 → 76.92 (+10.9%) | 71.55 → 74.48 (+4.1%) |
| BF16 | 4 | LL | 53.7 / 66.6 | 40.9 / 43.0 | −23.8% | −35.4% | −30.2% | 7.13 → 9.34 (+31.0%) | 5.73 → 8.88 (+55.0%) | 2.14 → 2.81 (+31.3%) | 1.73 → 2.68 (+54.9%) |
| BF16 | 8 | LL | 55.3 / 64.3 | 42.9 / 44.3 | −22.4% | −31.2% | −27.1% | 13.62 → 17.64 (+29.5%) | 11.68 → 17.05 (+46.0%) | 4.16 → 5.39 (+29.6%) | 3.57 → 5.22 (+46.2%) |
| BF16 | 16 | LL | 58.9 / 77.7 | 48.7 / 49.8 | −17.4% | −35.9% | −27.9% | 25.62 → 31.33 (+22.3%) | 19.34 → 30.55 (+58.0%) | 7.85 → 9.59 (+22.2%) | 5.93 → 9.38 (+58.2%) |
| BF16 | 32 | LL | 71.1 / 80.8 | 58.9 / 60.7 | −17.1% | −24.9% | −21.4% | 42.47 → 51.32 (+20.8%) | 37.28 → 49.75 (+33.4%) | 13.02 → 15.73 (+20.8%) | 11.43 → 15.26 (+33.5%) |
| BF16 | 64 | LL | 95.9 / 109.4 | 86.6 / 86.7 | −9.7% | −20.8% | −15.8% | 63.18 → 70.24 (+11.2%) | 55.25 → 70.37 (+27.4%) | 19.30 → 21.47 (+11.2%) | 16.87 → 21.47 (+27.3%) |
| BF16 | 128 | LL | 108.8 / 134.9 | 101.8 / 107.8 | −6.4% | −20.1% | −14.0% | 112.97 → 119.35 (+5.6%) | 89.90 → 112.48 (+25.1%) | 34.61 → 36.58 (+5.7%) | 27.56 → 34.46 (+25.0%) |
| BF16 | 256 | LL | 183.0 / 193.1 | 152.8 / 164.7 | −16.5% | −14.7% | −15.6% | 132.23 → 157.83 (+19.4%) | 125.25 → 146.44 (+16.9%) | 40.47 → 48.30 (+19.3%) | 38.33 → 44.81 (+16.9%) |
| BF16 | 512 | LL | 264.1 / 301.2 | 230.5 / 263.1 | −12.7% | −12.7% | −12.7% | 182.19 → 209.60 (+15.0%) | 159.71 → 183.24 (+14.7%) | 55.79 → 64.18 (+15.0%) | 48.90 → 56.11 (+14.7%) |
| BF16 | 1024 | normal | 470.4 / 552.1 | 410.3 / 455.2 | −12.8% | −17.6% | −15.4% | 206.32 → 235.02 (+13.9%) | 175.12 → 211.64 (+20.9%) | 62.91 → 71.67 (+13.9%) | 53.40 → 64.54 (+20.9%) |
| BF16 | 2048 | normal | 851.5 / 920.5 | 754.9 / 819.2 | −11.3% | −11.0% | −11.2% | 227.92 → 254.03 (+11.5%) | 210.46 → 234.06 (+11.2%) | 69.76 → 77.76 (+11.5%) | 64.43 → 71.64 (+11.2%) |
| BF16 | 4096 | normal | 1609.2 / 1674.0 | 1447.9 / 1548.4 | −10.0% | −7.5% | −8.7% | 241.96 → 265.72 (+9.8%) | 232.36 → 248.27 (+6.8%) | 73.91 → 81.15 (+9.8%) | 70.96 → 75.83 (+6.9%) |

- **Where the gain comes from.** Small batches (LL) gain mostly in combine,
  13%–37%, with dispatch 6%–24% faster; large batches (normal kernel) gain
  5%–18% in both phases. The gap narrows as T grows; at T4096 D+C is 7%–9%
  faster.
- **For most buckets V1 uses the table's rule for that bucket.** Every bucket
  does for FP8 dispatch and BF16 combine. Of the buckets measured here, BF16
  dispatch has rules only for LL T64 and T128 and normal-kernel T4096, and AUTO
  uses the next larger rule for the rest: LL T4–T32 use T64's 128/64/8, LL T256
  and T512 use the LL table's T2048 rule (256/170/16), and normal-kernel T1024
  and T2048 use T4096's 256/128/16. V1's combine tables also have
  `fp8_direct_cast` rules (combine sent as FP8) for T64–T4096, which are faster
  but do not apply to the BF16 combine here.
- **Both sides' measurements match their own tuning tables.** With the same
  rule, V1's measurement in every bucket is within −6% to +13% of its table
  record, and V2's within −10% to +1%.
