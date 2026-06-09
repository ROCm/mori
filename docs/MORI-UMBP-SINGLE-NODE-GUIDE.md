# MORI UMBP Single-Node Smoke Test

This guide documents how to run the single-node UMBP + SGLang correctness smoke test that ships with Mori. The flow mirrors the `umbp-single-node-launcher` skill and launches the [`run_umbp_single_node_hicache.sh`](../src/umbp/scripts/run_umbp_single_node_hicache.sh) helper script inside a ROCm container.

Use the placeholders in this document to map the workflow to any target node. Every environment-dependent value is surfaced as an argument or environment variable; update those inputs and the script can run unchanged.

For the underlying architecture see
[`src/umbp/doc/design-master-control-plane.md`](../src/umbp/doc/design-master-control-plane.md);
for the full env-var inventory see
[`src/umbp/doc/runtime-env-vars.md`](../src/umbp/doc/runtime-env-vars.md).

## Overview

The script spins up the Docker image, rebuilds Mori with UMBP enabled, launches a single-node SGLang server backed by UMBP hierarchical cache, issues a probe completion, and collects logs under `~/umbp_single_node_results`. It is intended as a fast health check for DP+EP configurations.

## Prerequisites

- Access to the target node (for example `<node-name>`), including an active reservation if required by the cluster scheduler.
- Mori and SGLang checkouts on the node.
- Access to a checkpoint directory when running with real weights.
- Docker privileges on the node.

## Configuration Inputs

### Script constants

The script declares a few host-specific defaults near the top. Update them before running if your environment differs.

| Variable | Purpose | Typical override |
| --- | --- | --- |
| `USER_HOME` | Home directory mounted into the container. | Set to your `${HOME}` path on the node. |
| `NFS_BASE` | Base path containing the repositories. | Point to the shared filesystem path that holds Mori and SGLang. |
| `EXTRA_MOUNTS` | Additional `docker run` bind mounts. | Remove entries that do not exist or add the host paths you require. |
| `DOCKER_IMAGE` | ROCm container tag. | Change if you need a different toolchain. |

### Runtime environment variables

Export the variables below to adapt the run to your environment. The script reads these values at launch time.

| Variable | Default in script | When to change | Example |
| --- | --- | --- | --- |
| `MODEL_PATH` | `/apps/data/models/DeepSeek-V3-0324` | Real checkpoint location. | `/apps/data/models/DeepSeek-V3-0324` |
| `SGLANG_REPO` | `${NFS_BASE}/sglang` | Local SGLang checkout. | `/apps/ditian12/sglang` |
| `MORI_REPO` | `${NFS_BASE}/mori` | Local Mori checkout. | `/apps/ditian12/mori` |
| `RESULTS_DIR` | `${USER_HOME}/umbp_single_node_results` | Where to store logs. | `/tmp/umbp_results` |
| `ENABLE_DP` | `false` | Enable DP+EP mode. | `true` |
| `DP_SIZE` | `8` | Data parallel group size. | `4` |
| `EP_SIZE` | `8` | Expert parallel group size. | `2` |
| `TP_SIZE` | `8` | Tensor parallel group size. | `8` |
| `START_UMBP_MASTER` | `true` | Disable if you already manage the master. | `false` |
| `UMBP_MASTER_ADDRESS` | `127.0.0.1:15558` | Master listener address. | `<host-ip>:15558` |
| `UMBP_NODE_ADDRESS` | `127.0.0.1` | IP advertised to clients. | `<node-ip>` |
| `UMBP_IO_ENGINE_PORT` | `16000` | Hicache IO engine port. | `16010` |
| `UMBP_PEER_SERVICE_PORT` | `17000` | Peer service port. | `17010` |
| `UMBP_SSD_STAGING_BYTES` | `268435456` (256 MiB) | Size of the dedicated remote-SSD read staging buffer. Allocated **only** when SSD is enabled (DRAM-only runs allocate nothing and are unaffected). Raise for large-KV models so each read slot fits one key's value. | `268435456` |
| `USE_DUMMY_WEIGHTS` | `false` | Skip checkpoint validation. | `true` |
| `MEM_FRACTION_STATIC` | `0.7` | Fraction reserved for static allocations. | `0.6` |
| `PROBE_MAX_TIME` | `300` | Timeout for probe request in seconds. | `600` |

The script also respects `MORI_BRANCH`, `SGLANG_BRANCH`, and `MORI_PIP_FLAGS` if they are supplied.

## Launch Steps

1. **SSH to the target node**:
   ```bash
   ssh -o StrictHostKeyChecking=no <node-name>
   ```

2. **Review script constants**:
   ```bash
   sed -n '1,120p' run_umbp_single_node_hicache.sh
   ```
   Update `USER_HOME`, `NFS_BASE`, `EXTRA_MOUNTS`, and `DOCKER_IMAGE` if the defaults do not match your environment.

3. **Export runtime variables** (fill placeholders with your values):
   ```bash
   export MODEL_PATH=<path-to-model-checkpoints>
   export SGLANG_REPO=<path-to-sglang>
   export MORI_REPO=<path-to-mori>
   export RESULTS_DIR=<path-for-results>
   export ENABLE_DP=<true|false>
   export DP_SIZE=<dp-size>
   export EP_SIZE=<ep-size>
   export TP_SIZE=<tp-size>
   export START_UMBP_MASTER=true
   export UMBP_NODE_ADDRESS=$(hostname -I | awk '{print $1}')
   export USE_DUMMY_WEIGHTS=false
   ```
   Add any other overrides from the table above as needed.

4. **Run the script** with the desired weight mode:
   ```bash
   bash run_umbp_single_node_hicache.sh --real-weights
   # or
   bash run_umbp_single_node_hicache.sh --use-dummy-weights
   ```

   The script will:
   - Start or recycle the `umbp-single-node` container.
   - Rebuild Mori with `BUILD_UMBP=ON` via `pip install`.
   - Compute network interface settings for NCCL/Gloo.
   - Launch the SGLang server with the configured parallelism values and hicache settings.

5. **Monitor the probe and cleanup**:
   - The script waits up to one hour for `http://127.0.0.1:30000/health` to report ready.
   - It issues a probe request and stores the output in `${RESULTS_DIR}/probe_response.json`.
   - On success, the script terminates the SGLang server and UMBP master, leaving artifacts in `${RESULTS_DIR}`.

## Core Dumps

By default the container routes crashes through Apport (`/usr/share/apport/apport`). If raw core files are required, disable Apport inside the container and set:
```bash
sysctl -w kernel.core_pattern=/nfs/users/<user>/cores/core.%e.%p
ulimit -c unlimited
```
Make sure the directory exists and has enough space.

## Troubleshooting

- **Container launch**: `docker ps` should show `umbp-single-node`. If it exits immediately, check `docker logs`.
- **Server logs**: Available under `${RESULTS_DIR}/server_*.log`.
- **UMBP master logs**: Saved alongside server logs when `START_UMBP_MASTER=true`.
- **Model validation**: The script validates that `MODEL_PATH` contains `model-*.safetensors` when running with real weights.
- **Probe failures**: Inspect the tail of the server log; the script prints it automatically on errors.
- **Remote SSD reads silently recompute (`size_too_large`)**: A remote SSD read must fit a key's whole value into one slot, where `per_slot = ssd_staging_buffer_size / ssd_staging_buffer_slots` (defaults 256 MiB / 16 = 16 MiB). The single-key page KV for an MLA model ≈ `page_size × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes × num_layers` (DeepSeek-V3 5-layer ≈ 360 KB; full R1/V3 61-layer ≈ 4.5 MB). If `per_slot` is smaller, the read is dropped and the block is recomputed — requests don't error, but `mori_umbp_ssd_read_total{status="size_too_large"}` climbs and hit rate/throughput fall. Fix: raise `UMBP_SSD_STAGING_BYTES` or lower `ssd_staging_buffer_slots` so `per_slot ≥` the single-key page KV. (`ssd_staging_buffer_size` is allocated only when SSD is enabled, separate from the general `staging_buffer_size`.)

Refer back to this document whenever the single-node smoke test needs to be run or automated. For the original skill instructions see `/home/ditian12/.codex/skills/umbp-single-node-launcher/SKILL.md`.
