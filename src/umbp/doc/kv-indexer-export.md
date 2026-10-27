# Export local UMBP DRAM placements to the SGLang KV Indexer

UMBP can optionally publish committed DRAM residency directly to SGLang's
`kv_indexer.v1.KVIndexer/ApplyExternalKvBatch` gRPC API. This enables the
Indexer's existing placement and component-aware prefix queries to see UMBP
objects. No SGLang engine, Router, or ZMQ bridge code changes are required.

This is a polling exporter inside **distributed PoolClient**, including the
standalone server that owns a PoolClient. It is not the existing SGLang KV-event
subscriber that feeds the UMBP master. It does not add a data transfer path.
The worker associated with a placement must already be able to load the reported
UMBP objects.

## Required logical metadata

UMBP stores physical object keys. It cannot derive the Router's canonical block
hash, parent hash, token count, or component/shard layout from those keys.
The cache integration must supply an explicit manifest. Enabling this exporter
alone does not discover that metadata or automatically instrument an existing
SGLang storage connector.

Each logical block describes **all** physical objects needed for each declared
component, including required layers and TP shards. All listed objects must be
committed in local DRAM before the component is reported. Objects can be spread
across the PoolClient's local DRAM backends; duplicate copies count once.
Remote-only objects, SSD, and HBM do not count. A rank that sees only its own
shard must not advertise a complete worker's block. Use a single pool owner
with visibility of all required objects, or add a separate aggregation layer.

Hashes must exactly match the querying Router's hashes, including any existing
LoRA/cache-salt namespacing and signed int64 representation. Token count is not
object byte size. Supply the real parent for every non-root and retain ancestry
entries even when their objects are absent. A hash's parent and token count
must remain stable; the exporter rejects changes during its lifetime. The
same canonical-identity rule applies across restarts and to other publishers.

## Configuration

Set this only on the process that owns the exported pool, before initialization:

```bash
export UMBP_KV_INDEXER_CONFIG=/etc/mori/kv-indexer.json
```

Example FULL-only manifest, for two 16-token blocks with two physical shards:

```json
{
  "endpoint": "127.0.0.1:50051",
  "worker_id": "sglang-worker-0",
  "worker_address": "http://worker-0:30000",
  "cache_spec": {
    "version": 1,
    "components": 1,
    "full_tier_mask": 6
  },
  "poll_interval_ms": 1000,
  "rpc_timeout_ms": 200,
  "full_sync_interval_ms": 30000,
  "blocks": [
    {
      "hash": "-9007199254740993",
      "block_size": 16,
      "components": [{"component": 1, "keys": ["block-a/rank0", "block-a/rank1"]}]
    },
    {
      "hash": "42",
      "parent_hash": "-9007199254740993",
      "block_size": 16,
      "components": [{"component": 1, "keys": ["block-b/rank0", "block-b/rank1"]}]
    }
  ]
}
```

These hashes and keys are illustrative; replace them with actual cache metadata.
64-bit hashes should be decimal JSON strings to avoid precision loss in tools.
Unknown JSON fields are rejected. An absent parent means a root; zero is a valid
hash, not an absent-parent sentinel. The gRPC target is a host:port or resolver
target, not an HTTP URL. This first version uses insecure gRPC, matching the
upstream server's local/trusted-network deployment.

`worker_address` must byte-for-byte match the worker URL registered with the
Router. `worker_id`, address, and cache spec must agree with any existing HBM
publisher for that worker: the Indexer replaces worker metadata on every batch.
Tier masks are `(1 << tier)`: HBM=2, DRAM=4, HBM+DRAM=6. FULL=1, SWA=2,
MAMBA=4; for a hybrid cache declare `components: 7`, a positive
`swa_window_tokens`, DRAM-enabled masks for each component, and three component
entries per block. The exported mask reflects current complete components and
is replaced when residency changes. SWA/MAMBA-only blocks are not reported
without FULL, as required by the Indexer's stored-block contract.

Exactly **one exporter must own `(worker_id, DRAM)`**. Initial/recovery/periodic
sync and clean shutdown issue `CLEAR_ALL_AT_TIER(DRAM)`. Do not simultaneously
publish HiCache DRAM and UMBP DRAM under the same worker through independent
publishers. If retaining the upstream ZMQ bridge for GPU events, configure it
to own HBM only, with the same worker identity and cache spec. Indexer `seq` is
observability only; it provides no fencing between writers.

Replace the configuration file atomically (write a temporary file, then rename)
to add/remove blocks or change physical mappings. Only `blocks` can change
live. Identity, cache spec, endpoint, and timing options require restart. A
malformed reload or snapshot exception triggers withdrawal of the old DRAM
bucket and logs the error; retries resume after the configuration is repaired.
If the Indexer is unreachable, withdrawal cannot take effect until reconnect.

## Consistency and recovery

The exporter takes `MediumBackend::SnapshotOwnedKeys()` snapshots. It never
drains `DrainPendingEvents()` or acknowledges/clears the master's full-sync
outbox. Allocated-but-uncommitted writes are excluded. Backend locks are
released before any network RPC. Nothing is added to PUT/GET's synchronous
network path when the exporter is enabled; when disabled, no exporter thread
or Indexer connection is created.

Normal polls send REPORT/REVOKE differences. Each REPORT carries a replacement
component mask, token count, and parent. Every batch includes address and cache
spec. Batches contain at most 256 actions and one hash per report/revoke, below
the upstream 16,384-hash limit. After an RPC error, the exporter samples again
and clears/reports the current snapshot, because a failed response may still
have committed on the server. It never retries an old report in isolation.

Periodic full sync repairs silent Indexer restarts. Default poll/RPC/full-sync
intervals are 1000/200/30000 ms; zero selects those defaults. RPC deadlines are
capped at 60 seconds. Stop joins the poller before backend destruction, checks
for shutdown between batches, and makes one bounded withdrawal attempt. A
slow snapshot must still finish before join returns.

This is eventually consistent metadata. Snapshots across backends are not
atomic, changes during a poll appear in a later poll, and full resync can briefly
under-report between clear and report batches. At healthy steady state, changes
usually appear within a poll plus its snapshot/RPC time; this is not a hard
latency bound. The manifest and known logical identities are kept in memory;
this first version scans the manifest each poll and does not implement an
incremental metadata ingestion API.

The Indexer has no persistence or worker TTL. A crash or unsuccessful shutdown
withdrawal can leave stale placements until the same owner reconnects/resyncs,
or the deployment clears that worker's DRAM tier. Router health checks and
normal cache-miss fallback remain necessary. This exporter does not claim
remote/shared-pool-wide coverage or HA writer coordination.

## Build and validation

The exporter builds with the existing `BUILD_UMBP=ON` gRPC/protobuf dependencies;
there are no new runtime libraries. The vendored wire proto is pinned and
attributed in [`../distributed/proto/README.sgl-kv-indexer.md`](../distributed/proto/README.sgl-kv-indexer.md).

```bash
cmake --build build --target test_kv_indexer_exporter
ctest --test-dir build -R '^umbp_kv_indexer_exporter$' --output-on-failure
```

The regular suite uses real PageBackend commit/evict/clear operations and a
loopback gRPC receiver with failures after commit. It also exercises PoolClient
CPU PUT/GET, clear, shutdown/reinitialization, invalid configuration cleanup,
and the disabled path. The optional interoperability
case queries an unmodified SGLang Rust Indexer, exercising a hybrid
FULL/SWA/MAMBA prefix, component eviction, reinsertion, resync, and withdrawal.
Run it only against an isolated test server:

```bash
# In the pinned SGLang sgl-router/sgl-kv-indexer crate:
KV_INDEXER_LISTEN_ADDR=127.0.0.1:55179 cargo run --bin kv-indexer-server

# In MoRI, after building the test:
UMBP_TEST_KV_INDEXER_ENDPOINT=127.0.0.1:55179 \
  build/tests/cpp/umbp/distributed/test_kv_indexer_exporter \
  --gtest_filter=KvIndexerInterop.*
```

Without `UMBP_TEST_KV_INDEXER_ENDPOINT`, that interoperability case is skipped.
These tests validate metadata behavior without a GPU/model; they do not measure
Router scheduling performance or exercise a model's end-to-end offload/load.
