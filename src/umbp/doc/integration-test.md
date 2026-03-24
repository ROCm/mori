# UMBP Integration Test

## Usage

```bash
bash src/umbp/scripts/test_umbp_integration.sh [branch] [storage_mode] [sglang_branch]
```

- `branch` — mori git branch to build (default: `main`)
- `storage_mode` — `local` (default) or `distributed`
- `sglang_branch` — sglang git branch to use (default: `main`)

Examples:

```bash
# Local mode (default)
bash src/umbp/scripts/test_umbp_integration.sh                    # main, local, sglang main
bash src/umbp/scripts/test_umbp_integration.sh feat_umbp_pool     # feat_umbp_pool, local, sglang main

# Distributed mode — requires mori branch feat_umbp_pool and sglang branch feat/umbp-monitoring-dist
bash src/umbp/scripts/test_umbp_integration.sh feat_umbp_pool distributed feat/umbp-monitoring-dist
```

Single command, non-interactive, no manual steps needed.

## How It Works

The test is split into two scripts:

- `test_umbp_integration.sh` -- runs on the host, launches a Docker container and invokes the inner script
- `test_umbp_inner.sh` -- runs inside the container, performs the actual test

Steps executed inside the container:

1. **Update sglang** -- pulls latest from `/sgl-workspace/sglang/`
2. **Build mori** -- checks out the specified branch (default `main`), builds with `BUILD_UMBP=ON BUILD_TESTS=ON`
3. **Start UMBP Master** (distributed mode only) -- launches `umbp_master` process, verifies it is alive
4. **Run hicache benchmark** -- starts an SGLang server with UMBP-backed hierarchical cache, waits for health check, then runs 2 rounds of GSM8K benchmark (200 questions each)

The server (and UMBP master, if running) is automatically shut down after benchmarks complete or on failure. Server logs are written to `server_hicache_<timestamp>.log`; master logs to `umbp_master_<timestamp>.log`.

## Distributed Mode

Distributed mode requires specific branches:
- **mori**: `feat_umbp_pool`
- **sglang**: `feat/umbp-monitoring-dist`

When `storage_mode` is set to `distributed`, the test:

- Locates the `umbp_master` binary from the mori build directory
- Starts the UMBP master process (listening on `UMBP_MASTER_LISTEN`, default `0.0.0.0:50051`)
- Configures the SGLang storage backend with `master_address`, `node_address`, and `io_engine_port`

The following environment variables can be used to override distributed defaults:

| Variable | Default | Description |
|---|---|---|
| `UMBP_MASTER_LISTEN` | `0.0.0.0:50051` | Address the master binds to |
| `UMBP_MASTER_ADDRESS` | `localhost:50051` | Address clients connect to |
| `UMBP_NODE_ADDRESS` | `localhost` | This node's address for the cluster |
| `UMBP_IO_ENGINE_PORTS` | `50100,50101,...,50107` | Comma-separated IO engine ports |

## Expected Output

On success, the two benchmark rounds should each report **Accuracy >= 0.95**:

```
100%|██████████| 200/200 [01:34<00:00,  2.13it/s]
Accuracy: 0.980
Invalid: 0.000
Latency: 94.085 s
Output throughput: 192.029 token/s
=== Benchmark run 2/2 ===
100%|██████████| 200/200 [01:29<00:00,  2.24it/s]
Accuracy: 0.970
Invalid: 0.000
Latency: 89.176 s
Output throughput: 202.677 token/s
=== Both benchmark runs complete ===
```
