# UMBP Runtime-Tunable Environment Variables

Single source of truth for all `UMBP_*` timing / retry knobs that the
distributed master, pool client, peer service, and SPDK proxy read at
runtime. Covers both "plain" timing (TTL, heartbeat, poll, sleep,
shutdown deadlines) and bounded retry counts.

See also:
- [`distributed-known-issues.md`](./distributed-known-issues.md) —
  parameter invariants and dangerous combinations.
- `src/umbp/include/umbp/common/env_time.h` — parser helpers
  (`GetEnvSeconds / GetEnvMilliseconds / GetEnvMicroseconds / GetEnvUint32`).

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

## Variables

### Master / client registry

| Env var | Default | Unit | Description |
|---------|---------|------|-------------|
| `UMBP_HEARTBEAT_TTL_SEC` | `10` | sec | Registry entry TTL; client is evicted if no heartbeat arrives within this window. |
| `UMBP_REAPER_INTERVAL_SEC` | `5` | sec | Reaper wake-up period inside `ClientRegistry`. |
| `UMBP_ALLOCATION_TTL_SEC` | `30` | sec | Pending allocation TTL before master reclaims capacity. |
| `UMBP_FINALIZED_RECORD_TTL_SEC` | `120` | sec | How long finalized records are retained for idempotency checks. |
| `UMBP_MAX_MISSED_HEARTBEATS` | `3` | count | Consecutive misses before a client is considered dead. |
| `UMBP_EVICTION_CHECK_INTERVAL_SEC` | `5` | sec | `EvictionManager` loop period. |
| `UMBP_LEASE_DURATION_SEC` | `10` | sec | Storage lease length granted by `EvictionManager`. |
| `UMBP_HEARTBEAT_INTERVAL_DIVISOR` | `2` | count | Recommended client heartbeat interval = `heartbeat_ttl / divisor`. `min_allowed=1` guards against div-by-zero. |

Read by the **master process** (`bin/master_main.cpp` via
`MasterServerConfig::FromEnvironment()`).

### RPC / shutdown

| Env var | Default | Unit | Description |
|---------|---------|------|-------------|
| `UMBP_RPC_SHUTDOWN_TIMEOUT_MS` | `3000` | ms | Deadline for `UnregisterClient` and the last `Heartbeat` in `~MasterClient`. Bounds `~MasterClient` worst-case at ≤ 2 × this value. |
| `UMBP_GRPC_SHUTDOWN_DEADLINE_SEC` | `3` | sec | `server_->Shutdown(deadline)` budget, shared by master and peer service. |

Read by the **master client** (pool client process) and by the
**master server** / **peer service** (master / pool client processes).

### Pool client

| Env var | Default | Unit | Description |
|---------|---------|------|-------------|
| `UMBP_BATCH_PUT_WARN_INTERVAL_SEC` | `60` | sec | Minimum gap between two batch-level `"src not registered"` WARNs from the same `PoolClient`. |
| `UMBP_RELEASE_LEASE_MAX_RETRIES` | `2` | count | `ReleaseSsdLease` RPC attempt cap on the SSD read path. `min_allowed=1`. |

Read by each **pool client** process.

### SPDK proxy

| Env var | Default | Unit | Description |
|---------|---------|------|-------------|
| `UMBP_SPDK_PROXY_HEARTBEAT_STALE_MS` | `5000` | ms | Threshold after which the SHM-header heartbeat is considered stale. Consumed independently by proxy daemon, `SpdkProxyTier`, and probe code in `spdk_proxy_shm.cpp`. |
| `UMBP_SPDK_PROXY_HEARTBEAT_INTERVAL_MS` | `500` | ms | How often the proxy daemon's `PollLoop` writes `proxy_heartbeat_ms`. |
| `UMBP_SPDK_PROXY_REAP_INTERVAL_SEC` | `5` | sec | Period of dead-channel reap + `SyncTelemetry` in `PollLoop`. |
| `UMBP_SPDK_PROXY_POLL_INTERVAL_MS` | `100` | ms | `SpdkProxyTier::WaitForProxy` poll step. |
| `UMBP_SPDK_PROXY_INIT_FAIL_SLEEP_SEC` | `2` | sec | Sleep before detach when `SpdkEnv::Init` fails during daemon startup. |
| `UMBP_SPDK_PROXY_BUSY_YIELD_MS` | `1` | ms | Yield step used by writeback / batch-drain busy waits. |

Read by the **spdk_proxy daemon** (for intervals it emits) and by the
**pool client process** via `SpdkProxyTier` (for stale / poll checks).

---

## Pre-existing env vars (kept unchanged)

These were introduced earlier and retain their original semantics.
Listed for completeness so the `UMBP_*` namespace is documented in one
place:

| Env var | Default | Description |
|---------|---------|-------------|
| `UMBP_SPDK_PROXY_TIMEOUT_MS` | `30000` | Max time `SpdkProxyTier` waits for the proxy to reach `READY`. |
| `UMBP_SPDK_PROXY_IDLE_EXIT_TIMEOUT_MS` | `30000` | Daemon self-exits after this much idle time with zero active sessions. |
| `UMBP_SPDK_PROXY_TENANT_GRACE_MS` | `30000` | Grace period before forcibly reaping an inactive tenant. |
| `UMBP_LOG_LEVEL` | `1` (WARN) | `0=INFO`, `1=WARN`, `2=ERROR`; see `umbp/common/log.h`. |

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
