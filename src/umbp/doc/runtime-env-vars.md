# UMBP Runtime-Tunable Environment Variables

Single source of truth for every `UMBP_*` env var consumed by the
Mori UMBP stack at runtime — both the in-process timing/retry knobs
parsed by the C++ library and the deployment knobs read by Python
launcher scripts and the SGLang/hicache integration layer.

See also:
- [`design-master-control-plane.md`](./design-master-control-plane.md)
  — what each knob actually affects.
- `src/umbp/include/umbp/common/env_time.h` — parser helpers
  (`GetEnvSeconds / GetEnvMilliseconds / GetEnvMicroseconds /
  GetEnvUint32`).
- `src/umbp/include/umbp/common/config.h::UMBPConfig::FromEnvironment`
  — the env overlay applied to per-process `UMBPConfig` defaults.

---

## Resolution semantics

- Unset or empty env value → default is used, no log.
- Non-numeric value, negative number, trailing garbage (e.g. `"10abc"`),
  `uint32` overflow, or a value below the parameter's `min_allowed`
  threshold → default is used, and **one WARN per env name per process**
  is emitted on stderr via `UMBP_LOG_WARN`.
- Parsing uses `std::strtoll` with base 10. This means:
  - Leading whitespace is skipped (`"  123"` parses as `123`).
  - An explicit sign prefix is accepted (`"+123"` OK; `"-5"` fails the
    non-negative check on all current params and falls back).
  - Trailing whitespace or any non-digit suffix (`"123 "`, `"0x10"`) is
    rejected, falling back to the default.
- Every production call site caches the resolved value in a
  function-local `static const auto` on first use (distributed/SPDK-proxy
  consumers). `ClientRegistryConfig::FromEnvironment()` /
  `EvictionConfig::FromEnvironment()` themselves do not cache, but the
  `MasterServerConfig` they produce is built once at master startup, so
  the net effect is the same: env changes after first touch have **no
  effect within the same process**. To exercise a different value, fork
  a fresh binary.
- `std::getenv` and the logger are NOT async-signal-safe. First use must
  happen on a normal thread, not inside a signal handler.

When master starts, `bin/master_main.cpp` prints one line
`[Master] Resolved timing: ...` after `MasterServerConfig::FromEnvironment()`
so operators can audit the effective values.

---

## Master / client registry

Read by the **master process** (`bin/master_main.cpp` via
`MasterServerConfig::FromEnvironment()`).

| Env var | Default | Unit | Description |
|---|---|---|---|
| `UMBP_HEARTBEAT_TTL_SEC` | `10` | sec | Registry entry TTL; client is evicted if no heartbeat arrives within `heartbeat_ttl × max_missed_heartbeats`. |
| `UMBP_REAPER_INTERVAL_SEC` | `5` | sec | Reaper wake-up period inside `ClientRegistry`. |
| `UMBP_ALLOCATION_TTL_SEC` | `30` | sec | Legacy: pending allocation TTL on master. Unused on the live path (pending TTL now lives on `PeerDramAllocator`). Retained for back-compat. |
| `UMBP_FINALIZED_RECORD_TTL_SEC` | `120` | sec | Legacy: finalized-allocation idempotency window on master. Unused on the live path. |
| `UMBP_MAX_MISSED_HEARTBEATS` | `3` | count | Consecutive misses before a client is considered dead. |
| `UMBP_EVICTION_CHECK_INTERVAL_SEC` | `5` | sec | `EvictionManager` loop period. |
| `UMBP_LEASE_DURATION_SEC` | `10` | sec | Master-side read-lease length granted by `Router::RouteGet` to keep a key alive across the writer's RDMA round trip. Distinct from the peer's `read_lease_ttl_` (~500 ms by default), which protects against concurrent eviction during a single `ResolveKey`. |
| `UMBP_HEARTBEAT_INTERVAL_DIVISOR` | `2` | count | Recommended client heartbeat interval = `heartbeat_ttl / divisor`. `min_allowed=1` guards against div-by-zero. Read by the master and echoed in `RegisterClientResponse.heartbeat_interval_ms`. |
| `UMBP_EVICTKEY_DEADLINE_MS` | `1000` | ms | Per-call gRPC deadline applied to outbound `EvictKey` RPCs from `MasterPeerStubPool`. |
| `UMBP_HIT_INDEX_TTL_SEC` | `7200` | sec | External KV hit-count entry TTL. A hash with no counted match for longer than this is removed from the hit index. |
| `UMBP_HIT_INDEX_GC_INTERVAL_SEC` | `60` | sec | External KV hit-count GC sweep interval. |
| `UMBP_HIT_QUERY_MAX_BATCH` | `4096` | count | Maximum hashes accepted by one `GetExternalKvHitCounts` request. Oversized requests return gRPC `INVALID_ARGUMENT`; the server does not truncate. |

## Peer / pool client

Read by each **pool client** process (typically an SGLang/vLLM worker
that has loaded `libmori_pybinds.so`).

| Env var | Default | Unit | Description |
|---|---|---|---|
| `UMBP_RPC_SHUTDOWN_TIMEOUT_MS` | `3000` | ms | Deadline for `UnregisterClient` and the last `Heartbeat` in `~MasterClient`. Bounds `~MasterClient` worst-case at ≤ 2 × this value. |
| `UMBP_GRPC_SHUTDOWN_DEADLINE_SEC` | `3` | sec | `server_->Shutdown(deadline)` budget, shared by master and peer service. |
| `UMBP_METRICS_REPORT_INTERVAL_MS` | `1000` | ms | Cadence at which the pool client's `MasterClient` flushes buffered counters/gauges/histograms via `ReportMetrics`. |
| `UMBP_RELEASE_LEASE_MAX_RETRIES` | `2` | count | `ReleaseSsdLease` RPC attempt cap on the SSD read path. `min_allowed=1`. |

## SPDK proxy

Read by the **spdk_proxy daemon** (for intervals it emits) and by the
**pool client process** via `SpdkProxyTier` (for stale / poll checks).

| Env var | Default | Unit | Description |
|---|---|---|---|
| `UMBP_SPDK_PROXY_HEARTBEAT_STALE_MS` | `5000` | ms | Threshold after which the SHM-header heartbeat is considered stale. Consumed independently by proxy daemon, `SpdkProxyTier`, and probe code in `spdk_proxy_shm.cpp`. |
| `UMBP_SPDK_PROXY_HEARTBEAT_INTERVAL_MS` | `500` | ms | How often the proxy daemon's `PollLoop` writes `proxy_heartbeat_ms`. |
| `UMBP_SPDK_PROXY_REAP_INTERVAL_SEC` | `5` | sec | Period of dead-channel reap + `SyncTelemetry` in `PollLoop`. |
| `UMBP_SPDK_PROXY_POLL_INTERVAL_MS` | `100` | ms | `SpdkProxyTier::WaitForProxy` poll step. |
| `UMBP_SPDK_PROXY_INIT_FAIL_SLEEP_SEC` | `2` | sec | Sleep before detach when `SpdkEnv::Init` fails during daemon startup. |
| `UMBP_SPDK_PROXY_BUSY_YIELD_MS` | `1` | ms | Yield step used by writeback / batch-drain busy waits. |
| `UMBP_SPDK_PROXY_TIMEOUT_MS` | `30000` | ms | Max time `SpdkProxyTier` waits for the proxy to reach `READY`. |
| `UMBP_SPDK_PROXY_IDLE_EXIT_TIMEOUT_MS` | `30000` | ms | Daemon self-exits after this much idle time with zero active sessions. |
| `UMBP_SPDK_PROXY_TENANT_GRACE_MS` | `30000` | ms | Grace period before forcibly reaping an inactive tenant. |
| `UMBP_SPDK_PROXY_WRITE_BACK` | `0` | bool | Set non-zero to enable proxy write-back caching. |
| `UMBP_SPDK_PROXY_DEFAULT_TENANT_QUOTA_BYTES` | `0` | bytes | Per-tenant SHM data-region quota. `0` = no per-tenant cap. |
| `UMBP_SPDK_PROXY_CACHE_MB` / `UMBP_SPDK_RING_MB` | — | MB | SPDK ring buffer size in MB. `UMBP_SPDK_RING_MB` is the canonical name; `UMBP_SPDK_PROXY_CACHE_MB` is the legacy alias. |
| `UMBP_SPDK_RAID_STRIP_KB` | `128` | KB | RAID strip size when constructing a SPDK RAID bdev across multiple NVMe controllers. |

---

## UMBPConfig overlay (FromEnvironment)

`UMBPConfig::FromEnvironment()` overlays these on top of the struct
defaults. Set them before constructing the C++ client (or letting the
Python wrapper construct one) — they are read once.

| Env var | Default | Description |
|---|---|---|
| `UMBP_DRAM_CAPACITY` | 4 GiB | `dram.capacity_bytes`. |
| `UMBP_DRAM_HIGH_WM` / `UMBP_DRAM_LOW_WM` | `0.9` / `0.7` | DRAM tier eviction watermarks. |
| `UMBP_SSD_ENABLED` | `1` | `0` to disable the SSD tier entirely. |
| `UMBP_SSD_DIR` | `/tmp/umbp_ssd` | POSIX backend root. |
| `UMBP_SSD_CAPACITY` | 32 GiB | `ssd.capacity_bytes`. |
| `UMBP_SSD_BACKEND` | `posix` | `posix` or `spdk`. Implicitly upgraded to `spdk` if `UMBP_SPDK_NVME_PCI` is set. |
| `UMBP_EVICTION_POLICY` | `lru` | Forwarded to `eviction.policy`. |
| `UMBP_ROLE` | (empty) | `leader` / `follower` / `standalone`. If unset, falls back to `LOCAL_RANK` / `OMPI_COMM_WORLD_LOCAL_RANK` / `SLURM_LOCALID` / `MPI_LOCALRANKID`: rank 0 → leader, others → follower. |
| `UMBP_SPDK_BDEV` | (empty) | SPDK bdev name (e.g. `Malloc0`, `NVMe0n1`). |
| `UMBP_SPDK_REACTOR_MASK` | `0x1` | SPDK reactor CPU mask. |
| `UMBP_SPDK_MEM_MB` | `256` | DPDK hugepage limit (MB). |
| `UMBP_SPDK_NVME_PCI` | (empty) | NVMe PCI BDF (e.g. `0000:47:00.0`). |
| `UMBP_SPDK_NVME_CTRL` | `NVMe0` | SPDK NVMe controller name. |
| `UMBP_SPDK_IO_WORKERS` | `4` | Internal I/O worker threads for `SpdkSsdTier` batch ops. |
| `UMBP_SPDK_PROXY_SHM` | `/umbp_spdk_proxy` | SHM segment name. |
| `UMBP_SPDK_PROXY_TENANT_ID` | `0` | Tenant id for this client. |
| `UMBP_SPDK_PROXY_TENANT_QUOTA_BYTES` | `0` | Per-tenant quota, `0` = unlimited. |
| `UMBP_SPDK_PROXY_MAX_CHANNELS` (alias `UMBP_SPDK_PROXY_MAX_RANKS`) | `8` | Channel count. |
| `UMBP_SPDK_PROXY_DATA_PER_CHANNEL_MB` (alias `UMBP_SPDK_PROXY_DATA_MB`) | `32` | MB of SHM data region per channel. |
| `UMBP_SPDK_PROXY_BIN` | (auto) | Path to the `spdk_proxy` binary. The Python `mori.umbp` package auto-fills this from the packaged binary. |
| `UMBP_SPDK_PROXY_AUTO_START` | `1` | Auto-spawn the proxy daemon if not already running. |
| `UMBP_SPDK_PROXY_ALLOW_BORROW` | `0` | Allow tenants to borrow capacity from the shared pool. |
| `UMBP_SPDK_PROXY_RESERVED_SHARED_BYTES` | `0` | Reserved shared bytes that cannot be borrowed. |

---

## Deployment / launcher env vars

Not parsed by the C++ library directly. These are consumed by the
SGLang / hicache wrappers, `src/umbp/scripts/run_umbp_single_node_hicache.sh`,
and `src/umbp/scripts/test_umbp_inner.sh` to construct the
`UMBPDistributedConfig` plumbed into the C++ side. Listed here so
operators can find them in one place.

| Env var | Description |
|---|---|
| `UMBP_MASTER_ADDRESS` | `host:port` of the master to connect to (e.g. `10.0.0.1:15558`). |
| `UMBP_MASTER_LISTEN` | `host:port` the master should listen on (when starting it locally). |
| `UMBP_MASTER_AUTO_START` | `true`/`false`: auto-spawn `umbp_master` on this node before connecting. |
| `UMBP_MASTER_BIN` | Path to the `umbp_master` binary. The Python `mori.umbp` package auto-fills this from the packaged binary; override to point at a custom build. |
| `UMBP_NODE_ADDRESS` | This node's address as advertised to peers. Must be reachable from every other node. |
| `UMBP_IO_ENGINE_HOST` | `mori::io::IOEngine` listener host (typically `127.0.0.1`). |
| `UMBP_IO_ENGINE_PORT` / `UMBP_IO_ENGINE_PORTS` | IO engine port (single port, or comma-separated list for multi-engine deployments). |
| `UMBP_PEER_SERVICE_PORT` | Port `PeerServiceServer` should bind. |
| `UMBP_CACHE_REMOTE_FETCHES` | `true`/`false`: locally re-cache blocks fetched from a remote peer. Set to `false` for clean throughput benchmarks where you want to measure raw remote-fetch cost. |

---

## Pre-existing / unrelated knobs

| Env var | Default | Description |
|---|---|---|
| `UMBP_LOG_LEVEL` | `1` (WARN) | `0=INFO`, `1=WARN`, `2=ERROR`; see `umbp/common/log.h`. Both `MORI_UMBP_LOG_LEVEL=DEBUG` and `UMBP_LOG_LEVEL=0` route through the same logger. |

`MORI_IO_SQ_BACKOFF_TIMEOUT_US` is **not** in the UMBP namespace; it is
owned by MORI-IO (`src/io/rdma/common.cpp`).

---

## Testing

- `tests/cpp/umbp/distributed/test_env_time.cpp` covers the parser
  helpers (default / valid / empty / non-numeric / trailing garbage /
  negative / below-min / zero-when-allowed / uint32 overflow / multiple
  independent names).
- Business-path tests that require exercising multiple values of the
  same env within one test suite must `fork` — the function-local
  `static const` caches cannot be reset mid-process.
- CI environments that export any `UMBP_*` globally must strip those
  variables before running UMBP test targets, otherwise the first test
  to touch a given name will freeze the CI-injected value for the
  entire process.
