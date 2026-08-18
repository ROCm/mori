# SDMA Dispatch Full Sweep Runbook

This runbook reproduces the SDMA-versus-direct-copy dispatch sweep on another
machine. It covers BF16 dispatch and FP8 E4M3 FNUZ dispatch, with 8, 16, 32,
and 64 dispatch blocks across 64 through 4096 tokens.

The sweep runs both paths:

- `MORI_ENABLE_SDMA=0`: normal GPU-thread payload copy.
- `MORI_ENABLE_SDMA=1`: peer-affine SDMA payload copy.

Each point uses EP8, top-8 routing, hidden dimension 7168, zero-copy mode, 16
dispatch waves per block, and 4 combine waves per block. One 1024-thread
dispatch block generally occupies one CU, so the block sweep approximates an
8/16/32/64-CU sweep.

## 1. Reproduce the Same Source State

Record the source state on the original machine:

```bash
cd /home/dasidler/mori
git rev-parse HEAD
git status --short --branch
git diff --check
git diff > /tmp/sdma-subgroup8.patch
```

A plain `git diff` does not include untracked files. The most reliable transfer
is to commit the experiment changes, including these reusable files:

```text
bench_results/run_dispatch_sweep.sh
bench_results/plot_dispatch_sweep.py
bench_results/plot_dispatch_bandwidth_sweep.py
docs/sdma-dispatch/FULL_SWEEP_RUNBOOK.md
```

If committing is not appropriate, copy those files separately in addition to
the tracked-file patch.

The subgroup experiment was developed from commit `0dc78c59` on branch
`dev/dasidler/sdma-dispatch-more-work`, with local changes in
`src/ops/dispatch_combine/intranode.hpp`. Transfer and apply the patch if the
changes have not been committed:

```bash
cd /home/dasidler/mori
git switch --detach 0dc78c59
git apply /path/to/sdma-subgroup8.patch
git diff --check
```

The current subgroup path requires `num_experts_per_token == 8`, which is the
benchmark default.

Before comparing machines, verify that the queue-size experiment was reverted:

```bash
rg 'SDMA_QUEUE_SIZE' include/mori/application/transport/sdma/anvil_device.hpp
```

Expected:

```text
constexpr uint32_t SDMA_QUEUE_SIZE = 256 * 1024;
```

## 2. Record Machine and Software Information

Save this information alongside the results:

```bash
mkdir -p machine_info
git rev-parse HEAD > machine_info/git_head.txt
git status --short --branch > machine_info/git_status.txt
git diff > machine_info/git_diff.patch
rocminfo > machine_info/rocminfo.txt
rocm-smi --showproductname --showserial --showuniqueid --showclocks \
  --showmeminfo vram --showpids > machine_info/rocm_smi.txt
docker version > machine_info/docker_version.txt
uname -a > machine_info/uname.txt
lspci -nn > machine_info/lspci.txt
```

For a useful machine-dependence comparison:

- ensure all eight GPUs are idle before starting;
- stop unrelated GPU, PCIe, profiling, and monitoring workloads;
- use the same ROCm/PyTorch image and source revision;
- use the same GPU clock/power policy where possible;
- do not run BF16 and FP8 sweeps concurrently;
- retain every raw log, including slow/outlier rounds.

## 3. Build and Start the Container

The tested base image is:

```text
rocm/pytorch:rocm7.2.4_ubuntu22.04_py3.10_pytorch_release_2.8.0
```

Build the development image:

```bash
cd /home/dasidler/mori
docker pull rocm/pytorch:rocm7.2.4_ubuntu22.04_py3.10_pytorch_release_2.8.0
docker build --network=host -t rocm/mori:sdma-dev -f docker/Dockerfile.dev .
```

Create the persistent benchmark container:

```bash
docker run -d \
  --name mori_dev_bench \
  --group-add video \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --privileged \
  --ulimit nproc=100000:100000 \
  --pids-limit=-1 \
  -v /home/dasidler/mori:/home/dasidler/mori \
  -w /home/dasidler/mori \
  rocm/mori:sdma-dev sleep infinity
```

The scripts currently assume both the host checkout and container checkout are
`/home/dasidler/mori`, and that the container is named `mori_dev_bench`. Adjust
`bench_results/run_dispatch_sweep.sh` if the other machine uses different
paths.

Install MoRI with profiler support. System CMake is used intentionally because
pip build isolation may install a CMake version that fails to pair the
`/opt/venv` interpreter with Ubuntu's Python development files.

```bash
docker exec mori_dev_bench bash -lc \
  'python3 -m pip install "setuptools_scm[toml]>=6.2" wheel ninja pybind11'

docker exec mori_dev_bench bash -lc \
  'cd /home/dasidler/mori && \
   ENABLE_PROFILER=ON python3 -m pip install --no-build-isolation -e .'
```

Verify the runtime:

```bash
docker exec mori_dev_bench python3 -c \
  'import torch, mori; print(mori.__version__); print(torch.__version__); print(torch.cuda.device_count())'
```

The GPU count must be 8.

## 4. Shared Sweep Script

The reusable runner is:

```text
bench_results/run_dispatch_sweep.sh
```

It accepts configuration through environment variables:

| Variable | Default |
|---|---|
| `RESULT_DIR` | `bench_results/dispatch_sweep_<timestamp>` |
| `DTYPE` | `bf16` |
| `COMBINE_DTYPE` | same as `DTYPE` |
| `BLOCKS` | `8 16 32 64` |
| `TOKENS` | `64 128 256 512 1024 2048 4096` |
| `WORLD_SIZE` | `8` |
| `HIDDEN_DIM` | `7168` |

It writes one raw log per point and skips completed points when restarted. It
does not clear the JIT cache before every point; MoRI's source/configuration
hash selects the correct kernel.

## 5. Run the BF16 Sweep

```bash
cd /home/dasidler/mori
stamp=$(date +%Y%m%d_%H%M%S)
result_dir="bench_results/bf16_sdma_vs_nosdma_blocks_tokens_${stamp}"

RESULT_DIR="${result_dir}" \
DTYPE=bf16 \
COMBINE_DTYPE=bf16 \
BLOCKS='8 16 32 64' \
TOKENS='64 128 256 512 1024 2048 4096' \
bench_results/run_dispatch_sweep.sh
```

This runs 56 points: 2 copy modes x 4 block counts x 7 token counts.

To resume, run the same command with the same `RESULT_DIR`.

## 6. Run the FP8 Dispatch Sweep

The repository's canonical cross-type configuration is FP8 E4M3 FNUZ
dispatch with BF16 combine and `quant_type=none`. The inputs are already FP8;
this is not BF16-to-FP8 blockwise quantization.

```bash
cd /home/dasidler/mori
stamp=$(date +%Y%m%d_%H%M%S)
result_dir="bench_results/fp8_sdma_vs_nosdma_blocks_tokens_${stamp}"

RESULT_DIR="${result_dir}" \
DTYPE=fp8_e4m3_fnuz \
COMBINE_DTYPE=bf16 \
BLOCKS='8 16 32 64' \
TOKENS='64 128 256 512 1024 2048 4096' \
bench_results/run_dispatch_sweep.sh
```

Cross-type FP8-dispatch/BF16-combine runs intentionally report:

```text
End-to-end result: skipped (cross-type)
```

Dispatch and combine timings are still complete.

## 7. Generate `summary.csv`

Run the parser below after either sweep. Set `RESULT_DIR` to the completed
directory.

```bash
cd /home/dasidler/mori
RESULT_DIR=bench_results/<completed-result-directory> python3 - <<'PY'
import csv
import os
import pathlib
import re
import statistics

root = pathlib.Path(os.environ["RESULT_DIR"]) / "raw"
out = root.parent / "summary.csv"
rows = []
missing = []

for sdma in (0, 1):
    for blocks in (8, 16, 32, 64):
        for tokens in (64, 128, 256, 512, 1024, 2048, 4096):
            p = root / f"sdma_{sdma}_blocks_{blocks}_tokens_{tokens}.log"
            if not p.exists():
                missing.append((sdma, blocks, tokens, "missing"))
                continue

            text = p.read_text(errors="ignore")
            match = re.search(
                r"Dispatch result:\n(.*?)(?:\n\nCombine result:)", text, re.S
            )
            latencies = []
            bandwidths = []
            if match:
                for item in re.finditer(
                    r"Round\s+(\d+)\s+duration\(us\).*?"
                    r"\blat\s+([0-9.]+)\s+bw\s+([0-9.]+)",
                    match.group(1),
                ):
                    if int(item.group(1)) == 0:
                        continue
                    latencies.append(float(item.group(2)))
                    bandwidths.append(float(item.group(3)))

            if not latencies:
                missing.append((sdma, blocks, tokens, "incomplete"))

            rows.append(
                {
                    "sdma": sdma,
                    "blocks": blocks,
                    "max_tokens": tokens,
                    "dispatch_us_med": (
                        round(statistics.median(latencies), 1) if latencies else ""
                    ),
                    "dispatch_bw_gbs_med": (
                        round(statistics.median(bandwidths), 2) if bandwidths else ""
                    ),
                    "rounds": len(latencies),
                    "log": str(p),
                }
            )

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("wrote", out, "rows", len(rows), "missing", missing)
PY
```

Expected result for a complete sweep:

```text
rows 56 missing []
```

Round 0 is excluded from the summary, matching the existing experiment
convention. The remaining nine round averages are summarized with the median.

## 8. Generate Dtype-Labeled Plots

Reusable plotters are stored at:

```text
bench_results/plot_dispatch_sweep.py
bench_results/plot_dispatch_bandwidth_sweep.py
```

They infer the dtype from `<result-dir>/metadata.txt`. The bandwidth plot uses
a linear y-axis anchored at zero.

Run plotting inside `mori_dev_bench`, which contains Matplotlib:

```bash
cd /home/dasidler/mori
result_dir=/home/dasidler/mori/bench_results/<completed-result-directory>

docker exec mori_dev_bench bash -lc "
  cd /home/dasidler/mori &&
  python3 bench_results/plot_dispatch_sweep.py \
    --input ${result_dir}/summary.csv \
    --output ${result_dir}/dispatch_vs_max_tokens.png &&
  python3 bench_results/plot_dispatch_sweep.py \
    --input ${result_dir}/summary.csv \
    --output ${result_dir}/dispatch_vs_max_tokens.svg --format svg &&
  python3 bench_results/plot_dispatch_bandwidth_sweep.py \
    --input ${result_dir}/summary.csv \
    --output ${result_dir}/dispatch_bandwidth_vs_max_tokens.png &&
  python3 bench_results/plot_dispatch_bandwidth_sweep.py \
    --input ${result_dir}/summary.csv \
    --output ${result_dir}/dispatch_bandwidth_vs_max_tokens.svg --format svg
"
```

## 9. Machine-Dependence Check for the 4096-Token Stall

The observed issue is bimodal at low block counts: normal dispatch is around
0.95 ms, while some rounds take several milliseconds. Test the suspect points
separately after the full sweep:

```bash
for blocks in 8 16 32; do
  for run in $(seq 1 5); do
    echo "blocks=${blocks} run=${run}"
    docker exec \
      -e MORI_ENABLE_SDMA=1 \
      -e ENABLE_PROFILER=0 \
      mori_dev_bench \
      bash -lc "cd /home/dasidler/mori && \
        PYTHONPATH=/home/dasidler/mori/python:/home/dasidler/mori \
        MORI_OPS_LOG_LEVEL=INFO MORI_SHMEM_HEAP_SIZE=6G timeout 360 \
        python3 tests/python/ops/bench_dispatch_combine.py \
          --cmd bench --world-size 8 --max-tokens 4096 \
          --dtype bf16 --hidden-dim 7168 --zero-copy 1 \
          --dispatch-block-num ${blocks} --dispatch-warp-per-block 16 \
          --combine-block-num ${blocks} --combine-warp-per-block 4"
  done
done
```

Compare at least these statistics between machines:

- fast-mode latency;
- median latency;
- p90 and maximum latency;
- number of rounds above 1.5 ms and 2 ms;
- whether slow rounds affect all ranks or only selected ranks;
- whether combine slows during the same rounds.

The routing and tensor data are deterministic. Each worker seeds its device RNG
with 123, and one `test_data` allocation is reused for all rounds and graph
replays in a benchmark process. Timing differences between rounds are therefore
not caused by changing input or routing distributions.
