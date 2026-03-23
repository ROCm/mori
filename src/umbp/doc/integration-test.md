# UMBP Integration Test

## Usage

```bash
bash src/umbp/scripts/test_umbp_integration.sh [branch]
```

Examples:

```bash
bash src/umbp/scripts/test_umbp_integration.sh                  # uses main (default)
bash src/umbp/scripts/test_umbp_integration.sh feat_ump_dist    # uses feat_ump_dist
```

Single command, non-interactive, no manual steps needed.

## How It Works

The test is split into two scripts:

- `test_umbp_integration.sh` -- runs on the host, launches a Docker container and invokes the inner script
- `test_umbp_inner.sh` -- runs inside the container, performs the actual test

Steps executed inside the container:

1. **Update sglang** -- pulls latest from `/sgl-workspace/sglang/`
2. **Build mori** -- checks out the specified branch (default `main`), builds with `BUILD_UMBP=ON BUILD_TESTS=ON`
3. **Run hicache benchmark** -- starts an SGLang server in `dp_ep` mode (DP=8, EP=8, TP=8) with UMBP-backed hierarchical cache, waits for health check, then runs 2 rounds of GSM8K benchmark (200 questions each)

The server is automatically shut down after benchmarks complete or on failure. Server logs are written to `server_hicache_<timestamp>.log`.

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
