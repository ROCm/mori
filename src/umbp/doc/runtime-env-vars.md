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
| `UMBP_MAX_MISSED_HEARTBEATS` | `3` | count | Consecutive misses before a client is considered dead. |
| `UMBP_EVICTION_CHECK_INTERVAL_SEC` | `5` | sec | `EvictionManager` loop period. |
| `UMBP_LEASE_DURATION_SEC` | `2` | sec | Master-side read-lease length granted by `Router::RouteGet`: `IsLeased()` keys are skipped by the eviction scan, keeping a key alive from the moment the master returns its location until the reader connects to the owning peer. Only needs to cover the master→reader gRPC round trip + reach the peer (the actual RDMA transfer is covered peer-side by `UMBP_DRAM_READ_LEASE_MS`), so seconds is already generous; larger values pin actively-read (hot) keys against eviction. |
| `UMBP_HEARTBEAT_INTERVAL_DIVISOR` | `2` | count | Recommended client heartbeat interval = `heartbeat_ttl / divisor`. `min_allowed=1` guards against div-by-zero. Read by the master and echoed in `RegisterClientResponse.heartbeat_interval_ms`. |
| `UMBP_EVICTKEY_DEADLINE_MS` | `1000` | ms | Per-call gRPC deadline applied to outbound `EvictKey` RPCs from `MasterPeerStubPool`. |
| `UMBP_HIT_INDEX_TTL_SEC` | `7200` | sec | External KV hit-count entry TTL. A hash with no counted match for longer than this is removed from the hit index. |
| `UMBP_HIT_INDEX_GC_INTERVAL_SEC` | `60` | sec | External KV hit-count GC sweep interval. |
| `UMBP_HIT_QUERY_MAX_BATCH` | `4096` | count | Maximum hashes accepted by one `GetExternalKvHitCounts` request. Oversized requests return gRPC `INVALID_ARGUMENT`; the server does not truncate. |
| `UMBP_ROUTE_PUT_SELECT_ALGO` | `most_available` | enum | Base RoutePut placement algorithm over eligible nodes (tier order HBM → DRAM → SSD, the SSD step gated by `UMBP_ROUTE_PUT_SSD_MODE`). `most_available` = pick the node with the most projected free space; `random` = capacity-weighted random (probability proportional to projected `available_bytes`, never picks a node that cannot fit). Unknown value → default + one WARN. **On a pure-SSD deployment use `random`**: projected deductions are discarded at the end of each batch, so `most_available` sends every batch between two heartbeats to the same "emptiest" node and caps aggregate NVMe bandwidth at one node's drives — see [pure-ssd-mode.md](pure-ssd-mode.md). |
| `UMBP_ROUTE_PUT_NODE_AFFINITY` | `none` | enum | Node-affinity bias layered on top of the base algorithm. `none` = pure base algorithm; `same` = try to place the whole batch on one node that fits the non-dedup total, else per-key sticky to the first picked node; `local` = per-key prefer the requester's local node. All three fall back to the base algorithm so affinity never makes a key fail that the base algorithm could route. Unknown value → default + one WARN. |
| `UMBP_PUT_DIST_LOG` | `1` | count | **The call site is commented out** (once per batch is too chatty to ship on); uncomment `ConfigurableRoutePutStrategy::LogBatchDistribution`'s `MORI_UMBP_INFO` and rebuild to use it. How often to emit the RoutePut fan-out line `[RoutePutStrategy] batch_dist`: `0` = off, `N` = one line every Nth batch (cumulative totals are folded in on every batch regardless, so a sampled line still reflects all of them). The line reports, per `node/TIER`, the keys/bytes this batch placed and the running totals since master start, plus `batch_max_share` / `cumulative_max_share` (100/N% = perfectly even over N targets, 100% = everything on one node). This is the number to watch on a pure-SSD deployment, where an uneven split directly costs aggregate NVMe bandwidth. The same variable also gates the writer-side `[PoolClient] put_dist` line — see the pool-client table. |
| `UMBP_MASTER_INDEX_SHARDS` | `32` | count | Number of independently-locked, key-hashed shards backing the block-location index inside `InMemoryMasterMetadataStore` (the block lock domain, separate from the single `meta_mutex_` that guards client records + external-KV). A heartbeat's event batch only takes the exclusive lock on the shards its keys hash into, so unrelated `RoutePut` / `BatchLookup` readers on other shards don't block behind a large apply; full-sync likewise becomes N small critical sections instead of one giant one. Read once at store construction via `std::strtol`; unset / unparseable / `< 1` → default `32` (a WARN is logged on unparseable input), clamped to a max of `4096`. `1` reproduces the old single-lock block index. Production guidance for heavy heartbeat fan-out (hundreds of clients): `64`. |

## Peer / pool client

Read by each **pool client** process (typically an SGLang/vLLM worker
that has loaded `libmori_pybinds.so`).

| Env var | Default | Unit | Description |
|---|---|---|---|
| `UMBP_DRAM_READ_LEASE_MS` | `500` | ms | Peer-side DRAM/HBM read lease: how long a single `PeerDramAllocator::Resolve` protects its key's pages from concurrent local `Evict`, covering one RDMA read of those pages. Only needs to exceed one DRAM RDMA round trip (sub-ms), so 500 ms is ~100x margin. Read once at `PoolClient::Init`; `min_allowed=1`. |
| `UMBP_SSD_READ_LEASE_MS` | `3000` | ms | Peer-side SSD read-staging slot lease: how long a claimed staging slot is reserved before the peer reclaims it by TTL (the fallback when the reader's best-effort `ReleaseSsdLease` is lost), and, echoed back in `PrepareSsdReadResponse.lease_ttl_ms`, the reader's validity window anchored at `t_send`. Must exceed one SSD read + RDMA (slower than DRAM), but too long pins one of only ~16 slots on a lost release. Also the fallback for the `PrepareSsdRead` RPC deadline when `UMBP_SSD_PREPARE_TIMEOUT_MS` is unset. Read once at `PoolClient::Init`; `min_allowed=1`. |
| `UMBP_RPC_SHUTDOWN_TIMEOUT_MS` | `3000` | ms | Deadline for `UnregisterClient` and the last `Heartbeat` in `~MasterClient`. Bounds `~MasterClient` worst-case at ≤ 2 × this value. |
| `UMBP_GRPC_SHUTDOWN_DEADLINE_SEC` | `3` | sec | `server_->Shutdown(deadline)` budget, shared by master and peer service. |
| `UMBP_METRICS_REPORT_INTERVAL_MS` | `1000` | ms | Cadence at which the pool client's `MasterClient` flushes buffered counters/gauges/histograms via `ReportMetrics`. |
| `UMBP_RELEASE_LEASE_MAX_RETRIES` | `2` | count | `ReleaseSsdLease` RPC attempt cap on the SSD read path. `min_allowed=1`. |
| `UMBP_SSD_GET_MAX_ATTEMPTS` | `1` | count | Total remote SSD get attempts per key. `1` = no retry. Only NO_SLOT and a reader-local lease expiry retry; rpc failure / NOT_FOUND do not. Raise to absorb staging-slot contention. `min_allowed=1`. |
| `UMBP_SSD_GET_RETRY_BACKOFF_MS` | `2` | ms | Sleep between remote SSD get retries (only applied when another attempt follows). `min_allowed=1`. |
| `UMBP_RELEASE_LEASE_TIMEOUT_MS` | `1000` | ms | Per-attempt gRPC deadline for the best-effort `ReleaseSsdLease` RPC so a slow peer can't stall the reader. `min_allowed=1`. |
| `UMBP_SSD_PREPARE_TIMEOUT_MS` | `0` | ms | Per-call gRPC deadline for `PrepareSsdRead` so a hung/slow peer can't stall the serial batch. `0` = fall back to `UMBP_SSD_READ_LEASE_MS` (cluster-homogeneous). A timed-out / failed prepare is a hard not-served outcome (NOT retried, and never a miss). `min_allowed=0`. |
| `UMBP_PUSH_TARGET_TTL_MS` | `30000` | ms | How long a client's SSD read fan-out **push registration** (its engine desc + destination buffers, learned from its `GetPeerInfo`) stays usable on a peer without being refreshed. Every `BatchPrepareSsdRead` from that client refreshes its own entry, so an active client never ages out; only one that has gone silent (crashed, killed, restarted) does. This bound exists because the push path makes the peer an RDMA *initiator* against the client role — the short-lived, restart-prone side — and a client that restarts with the same `node_id` reuses the same per-process `MemoryUniqueId` sequence. Expiry fails safe: the key falls back to the classic staging + client-pull path and the response sets `push_registration_stale`, which makes the client re-run the handshake. Read once at `PeerServiceServer` construction; `min_allowed=1000`. |
| `UMBP_AUTO_FLUSH_EVENT_THRESHOLD` | `128` | count | Peer-side unshipped `KvEvent` outbox size at which a completed batch of puts auto-triggers a heartbeat flush (`FlushHeartbeat`), so the ADDs become visible at the master without waiting for the heartbeat interval or an explicit `Flush()`. Counted on `PeerDramAllocator` only (SSD events still wait for the interval). Parsed via `std::strtoull` (no WARN on bad input); unset / unparseable -> default `128`; `0` disables size-based auto-flush entirely (ADDs then ship only on the heartbeat interval or an explicit `Flush()`); set to a very large value to keep auto-flush armed but effectively never fire on size. Cached on first use in `MasterClient::SetPeerDramAllocator`. |

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
| `UMBP_SSD_DIR` | `/tmp/umbp_ssd` | POSIX backend root(s).  **Comma-separated for multi-drive**, one directory per physical drive (e.g. `/mnt/nvme0,/mnt/nvme1`).  More than one turns the tier into a `ShardedSsdTier`: keys are placed on the drive with the most free space and batch IO runs on every drive at once, so the tier delivers their aggregate bandwidth. |
| `UMBP_SSD_CAPACITY` | 32 GiB | `ssd.capacity_bytes`.  With multiple `UMBP_SSD_DIR` entries this is the TOTAL budget, split evenly across the drives. |
| `UMBP_SSD_SHARD_IO_THREADS` | `0` | Worker threads for the multi-drive batch paths.  `0` = one per drive (what saturates N drives).  Ignored with a single directory. |
| `UMBP_SSD_TIER_IO_THREADS` | `4` | `ssd.tier_io_threads` — fans the tier's CPU phases (checksum verify on read, record assembly on write) across workers **within one drive's batch**.  Orthogonal to `UMBP_SSD_SHARD_IO_THREADS`, which fans out across drives.  Defaults to 4 to match `DRAMTier::read_threads_` so a DRAM-vs-SSD comparison is not 4 threads against 1. |
| `UMBP_SSD_DIRECT_IO` | `0` | `ssd.direct_io` — `1` opens segments `O_DIRECT`, bypassing the page cache.  Buffered (the default) can serve an entire working set from page cache and move **zero** bytes to the device, which makes drive-count and DRAM-vs-SSD comparisons meaningless.  Set it for any pure-SSD measurement.  Requires record format v3 (already unconditional), so a directory reads back either way. |
| `UMBP_SSD_VERIFY_CRC` | `1` | `ssd.verify_crc` — `0` skips checksum verification, which is 51–67% of GET time and has no `DRAMTier` equivalent.  Records written with checksums off carry `kFlagNoCrc` and stay readable by a process with verification on. |
| `UMBP_SSD_BACKEND` | `file` | `file` or `spdk`. Implicitly upgraded to `spdk` if `UMBP_SPDK_NVME_PCI` is set. |
| `UMBP_EVICTION_POLICY` | `lru` | Forwarded to `eviction.policy`. |
| `UMBP_ROUTE_PUT_SSD_MODE` | `auto` | Whether RoutePut may target a node's SSD tier directly.  `auto` = only on **pure-SSD nodes** (nodes reporting no DRAM/HBM capacity at all), so mixed deployments are unchanged; `always` = SSD considered on every node after HBM and DRAM, turning a full-DRAM node into a spill target instead of failing the put; `never` = legacy HBM→DRAM only. |
| `UMBP_PUT_DIST_LOG` | `1` | **Call site commented out** — uncomment `PoolClient::LogBatchPutDistribution`'s `MORI_UMBP_INFO` and rebuild to use it. Writer-side fan-out line `[PoolClient] put_dist`: `0` = off, `N` = every Nth `BatchPut`. Same shape as the master's `batch_dist` line but from one client's point of view, with self-target keys tagged `<node>(local)`, so a client that keeps landing on a single node is visible without cross-referencing the master log. `remote_peers` is the number of peers this batch actually writes to — the put path runs one concurrent worker per remote SSD peer, so it is also the write concurrency. |
| `UMBP_GET_DIST_LOG` | `1` | **Call site commented out** — uncomment the log-only tail of `PoolClient::LogBatchGetDistribution` (the straggler block *and* the `MORI_UMBP_INFO`) and rebuild to use it. Read-side counterpart, `[PoolClient] get_dist`, emitted after the batch completes: `0` = off, `N` = every Nth `BatchGet`. Adds `remote_ssd_workers` (the number of concurrently-read owning nodes, one thread each), the batch wall time and the implied aggregate GB/s, so read fan-out and overlap can be checked rather than assumed. |
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
