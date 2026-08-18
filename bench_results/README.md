# Dispatch Benchmark Scripts

These scripts run and plot the intra-node dispatch experiments. Run them from
the repository root on a machine with eight available GPUs and the
`mori_dev_bench` container running.

## Build

Build MoRI with profiler support. The same build can run both benchmark and
profile sweeps.

```bash
docker exec mori_dev_bench bash -lc \
  'cd /home/dasidler/mori && ENABLE_PROFILER=ON python3 -m pip install -e .'
```

## Latency And Bandwidth Sweep

`run_dispatch_sweep.sh` runs every configuration with SDMA disabled and
enabled. This example runs the full FP8 dispatch sweep:

```bash
result_dir="bench_results/fp8_dispatch_$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$result_dir" \
DTYPE=fp8_e4m3_fnuz \
COMBINE_DTYPE=bf16 \
BLOCKS='8 16 32 64' \
TOKENS='64 128 256 512 1024 2048 4096' \
bench_results/run_dispatch_sweep.sh
```

For BF16, set both dtype variables to `bf16`. The runner writes raw logs under
`$result_dir/raw/`. Reuse the same `RESULT_DIR` to resume an interrupted sweep;
completed points are skipped.

The plotting scripts consume a `summary.csv` with these columns:

```text
sdma,blocks,max_tokens,dispatch_us_med,dispatch_bw_gbs_med,rounds,log
```

See `docs/sdma-dispatch/FULL_SWEEP_RUNBOOK.md` for the log-to-CSV parser.

```bash
python3 bench_results/plot_dispatch_sweep.py \
  --input "$result_dir/summary.csv" \
  --output "$result_dir/dispatch_latency.png"

python3 bench_results/plot_dispatch_bandwidth_sweep.py \
  --input "$result_dir/summary.csv" \
  --output "$result_dir/dispatch_bandwidth.png"

python3 bench_results/plot_dispatch_bandwidth_sweep.py \
  --input "$result_dir/summary.csv" \
  --output "$result_dir/dispatch_bandwidth_per_cu.png" \
  --per-cu
```

## SDMA Batch-Occupancy Sweep

This sweep measures the average number of token-copy tasks in each SDMA queue
submission. It requires profiler-enabled kernels containing these events:

```text
dispatch_sdma_submit_call
dispatch_sdma_active_count_bit0..5
```

Run the full FP8 profile sweep:

```bash
result_dir="bench_results/fp8_sdma_batch_profile_$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$result_dir" \
DTYPE=fp8_e4m3_fnuz \
COMBINE_DTYPE=bf16 \
BLOCKS='8 16 32 64' \
TOKENS='64 128 256 512 1024 2048 4096' \
CAPTURE_ITERS=3 \
CLEAR_JIT=1 \
bench_results/run_sdma_batch_profile_sweep.sh
```

Reuse the same `RESULT_DIR` to resume. Set `CLEAR_JIT=0` when resuming to avoid
unnecessary recompilation.

Parse traces and produce CSV, Markdown, PNG, and SVG outputs:

```bash
docker exec mori_dev_bench bash -lc \
  "cd /home/dasidler/mori && python3 bench_results/plot_sdma_batch_occupancy.py --input '$result_dir'"
```

## Common Variables

| Variable | Default |
|---|---|
| `DTYPE` | `bf16` for dispatch sweep; `fp8_e4m3_fnuz` for profile sweep |
| `COMBINE_DTYPE` | dispatch dtype; `bf16` for profile sweep |
| `BLOCKS` | `8 16 32 64` |
| `TOKENS` | `64 128 256 512 1024 2048 4096` |
| `WORLD_SIZE` | `8` |
| `HIDDEN_DIM` | `7168` |
| `DISPATCH_WARPS` | `16` for profile sweep |
| `COMBINE_WARPS` | `4` for profile sweep |
| `CAPTURE_ITERS` | `3` |

