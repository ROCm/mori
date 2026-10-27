# UMBP → SGLang KV Indexer

Configure an existing DRAM `UMBPClient` once, then register canonical block
metadata supplied by the integration:

```python
client.configure_kv_indexer("127.0.0.1:50051", "worker-0", "http://worker-0:30000")
client.register_kv_block(101, None, 16, {1: ["block101-k", "block101-v"]})
client.put_from_ptr("block101-k", k_ptr, nbytes)
client.put_from_ptr("block101-v", v_ptr, nbytes)  # Automatically reports FULL.
# Eviction, clear(), and storage-owner shutdown automatically withdraw residency.
client.register_kv_block(101, None, 0, {})  # Unregister metadata when no longer needed.
```

`component_keys` maps FULL=1, SWA=2, MAMBA=4 to all physical keys required for
that component, including layers and shards. Supply the true signed int64 hash,
parent (`None` only for roots), and block size in tokens. Physical keys must
identify immutable content. MoRI cannot derive this metadata from physical keys.
Registration can precede or follow writes; it never advertises uncommitted data.
An empty component map unregisters the block. Otherwise, metadata must describe
all configured components; an existing hash cannot change parent or token count.
For hybrid caches pass `components` and `swa_window_tokens` to configuration.
A component becomes resident only when all its keys have committed locally;
reports require FULL and replace the component mask when residency changes.

Commit, eviction and clear hooks wake a background worker. It reconciles the
metadata against committed DRAM keys across the owner's local backends and sends
only changed placements. Batch/ranged writes, peer writes and internal evictions
use the same hooks. RPCs run outside the storage path: Indexer failures are logged
without changing PUT success. `flush()` synchronizes the current snapshot and
returns false if reporting fails. As with other asynchronous indexes, concurrent
mutations can briefly lag; a match is not a storage lease.

Standalone-process clients forward configuration and registration to the storage
owner. Configure once per shared pool, not once per forwarding client. Closing a
forwarder leaves the owner's reporting active. For distributed deployments,
register metadata at the actual owner: remote-only objects and components split
across independent owners are not aggregated by this API.

Use exactly one reporter for a worker's DRAM tier. The worker identity, address
and cache spec must match any HBM publisher, and the address must match the
Router's registered worker URL. Configuration clears that DRAM bucket before
rebuilding it. After RPC errors the worker retries from a fresh snapshot (checked
every second, also on changes); a full rebuild every approximately 30 seconds
repairs quiet Indexer restarts. Rebuilds can briefly reduce hits. Graceful owner
shutdown attempts to clear DRAM; crashes and unavailable shutdown RPCs cannot
withdraw immediately because the Indexer protocol has no TTL or fencing. HBM
placements are not cleared.

This is opt-in: merging/installing MoRI alone does not supply an endpoint or
logical metadata. The integration calls the two APIs above; no manual event
reporting or SGLang source changes are required by MoRI. The existing
`report_external_kv_blocks` API targets the UMBP master and is unrelated.

For integrations that already own event timing and recovery, the small explicit
`mori.umbp.UMBPIndexerReporter` remains available with `report(hashes,
parent_hash, block_size, component_mask=1)`, `revoke(hashes)` and `clear()`.
Do not use it alongside automatic reporting for the same DRAM bucket.

All RPCs use insecure gRPC with a per-RPC `timeout_ms` (default 200). Python
methods release the GIL. Invalid input/lifecycle use raises `ValueError` or
`RuntimeError`; the explicit reporter also raises on RPC failures,
which may follow a server commit. Build with `BUILD_UMBP=ON`, `BUILD_PYBINDS=ON`.
Run `ctest -R '^umbp_kv_indexer_reporter$'` for wire/error, automatic residency,
recovery and standalone forwarding tests.
