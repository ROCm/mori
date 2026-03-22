# UMBP Integration Test

## Usage

```bash
bash src/umbp/scripts/test_umbp_integration.sh
```

Single command, non-interactive, no manual steps needed.

## How It Works

The test is split into two scripts:

- `test_umbp_integration.sh` -- runs on the host, launches a Docker container and invokes the inner script
- `test_umbp_inner.sh` -- runs inside the container, performs the actual test

Steps executed inside the container:

1. **Update sglang** -- pulls latest from `/sgl-workspace/sglang/`
2. **Build mori** -- checks out `feat_ump_dist`, builds with `BUILD_UMBP=ON BUILD_TESTS=ON`
3. **Run hicache benchmark** -- starts an SGLang server in `dp_ep` mode (DP=8, EP=8, TP=8) with UMBP-backed hierarchical cache, waits for health check, then runs 2 rounds of GSM8K benchmark (200 questions each)

The server is automatically shut down after benchmarks complete or on failure. Server logs are written to `server_hicache_<timestamp>.log`.
