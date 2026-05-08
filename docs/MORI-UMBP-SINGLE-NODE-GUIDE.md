# MORI UMBP Single-Node Smoke Test

This guide documents how to run the single-node UMBP + SGLang correctness smoke test that ships with Mori. The flow mirrors the `umbp-single-node-launcher` skill and launches the [`run_umbp_single_node_hicache.sh`](../src/umbp/scripts/run_umbp_single_node_hicache.sh) helper script inside a ROCm container.

## Overview

The script spins up the Docker image, rebuilds Mori with UMBP enabled, launches a single-node SGLang server backed by UMBP hierarchical cache, issues a probe completion, and collects logs under `~/umbp_single_node_results`. It is intended as a fast health check for DP+EP configurations.

## Prerequisites

- Access to the target node (for example `banff-ccs-aus-p20-38.cs-aus.dcgpu`) and an active Conductor reservation.
- Mori checkout at `/nfs/users/<user>/mori` and SGLang checkout at `/nfs/users/<user>/sglang`.
- Real checkpoints at `/nfs/data/Deepseek-R1` (or adjust `MODEL_PATH` accordingly).
- Docker privileges on the node.

## Required Inputs

Before launching the script, collect:

| Input | Value / Default | Notes |
| ----- | --------------- | ----- |
| Dummy weights | `--real-weights` (default) | Pass `--use-dummy-weights` to skip checkpoint validation. |
| Parallelism mode | `ENABLE_DP=true`, `DP_SIZE`, `EP_SIZE`, `TP_SIZE` | For DP+EP runs, export all three sizes (defaults are 8/8/8). Set `ENABLE_DP=false` for TP-only. |
| Repository paths | `SGLANG_REPO`, `MORI_REPO` | Default to `/nfs/users/<user>/sglang` and `/nfs/users/<user>/mori`. |
| Model path | `MODEL_PATH=/nfs/data/Deepseek-R1` | Required when using real weights. |
| Branch overrides | Optional | The script uses whatever branch is already checked out unless `SGLANG_BRANCH` or `MORI_BRANCH` are provided. |

## Launch Steps

1. **SSH to the node** (example):
   ```bash
   ssh -o StrictHostKeyChecking=no banff-ccs-aus-p20-38.cs-aus.dcgpu
   ```

2. **Change into the script directory**:
   ```bash
   cd /nfs/users/<user>/mori/src/umbp/scripts
   ```

3. **Export environment variables** (adjust values as needed):
   ```bash
   export ENABLE_DP=true
   export DP_SIZE=8
   export EP_SIZE=8
   export TP_SIZE=8
   export START_UMBP_MASTER=true
   export MODEL_PATH=/nfs/data/Deepseek-R1
   export SGLANG_REPO=/nfs/users/<user>/sglang
   export MORI_REPO=/nfs/users/<user>/mori
   export USE_DUMMY_WEIGHTS=false
   ```

4. **Run the script**:
   ```bash
   bash run_umbp_single_node_hicache.sh --real-weights
   ```

   The script will:
   - Start or recycle the `umbp-single-node` container (`rocm/sgl-dev:v0.5.9-rocm700-mi30x-20260316`).
   - Rebuild Mori with `BUILD_UMBP=ON`.
   - Compute network interface settings for NCCL/Gloo.
   - Launch the SGLang server with the following command (formatted for readability):
     ```
     python -m sglang.launch_server \
       --enable-cache-report \
       --enable-metrics \
       --model-path "${MODEL_PATH}" \
       --tp-size "${TP_SIZE}" \
       --dp-size "${DP_SIZE}" \
       --ep-size "${EP_SIZE}" \
       --moe-a2a-backend mori \
       --deepep-mode normal \
       --enable-dp-attention \
       --enable-dp-lm-head \
       --moe-dense-tp-size 1 \
       --decode-log-interval 1 \
       --trust-remote-code \
       --watchdog-timeout 1000000 \
       --chunked-prefill-size 65536 \
       --attention-backend aiter \
       --kv-cache-dtype fp8_e4m3 \
       --max-total-tokens 1024 \
       --mem-fraction-static "${MEM_FRACTION_STATIC:-0.7}" \
       --enable-hierarchical-cache \
       --hicache-write-policy write_through \
       --hicache-mem-layout page_first \
       --hicache-ratio 5.0
     ```

5. **Probe and cleanup**:
   - The script waits up to one hour for `http://127.0.0.1:30000/health` to report ready.
   - It then issues a probe request:
     ```
     curl -sf --max-time "${PROBE_MAX_TIME:-300}" \
       -X POST http://127.0.0.1:30000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{"model":"deepseek-v3","prompt":"Say hello in one word.","max_tokens":8,"stream":false}' \
       | tee ${RESULTS_DIR}/probe_response.json
     ```
   - On success, the script terminates the SGLang server and UMBP master, leaving artifacts in `~/umbp_single_node_results`.

## Core Dumps

By default the container routes crashes through Apport (`/usr/share/apport/apport`). If raw core files are required, disable Apport inside the container and set:
```bash
sysctl -w kernel.core_pattern=/nfs/users/<user>/cores/core.%e.%p
ulimit -c unlimited
```
Make sure the directory exists and has enough space.

## Troubleshooting

- **Container launch**: `docker ps` should show `umbp-single-node`. If it exits immediately, check `docker logs`.
- **Server logs**: Available under `~/umbp_single_node_results/server_*.log`.
- **UMBP master logs**: Saved alongside server logs when `START_UMBP_MASTER=true`.
- **Model validation**: The script validates that `MODEL_PATH` contains `model-*.safetensors` when running with real weights.
- **Probe failures**: Inspect the tail of the server log; the script prints it automatically on errors.

Refer back to this document whenever the single-node smoke test needs to be run or automated. For the original skill instructions see `/home/ditian12/.codex/skills/umbp-single-node-launcher/SKILL.md`.
