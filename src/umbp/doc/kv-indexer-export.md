# UMBP → SGLang KV Indexer

Enable reporting through `UMBPConfig.kv_indexer`. Existing Python and C++ writes
keep their original signatures; there is no Python `register_kv_block` call:

```python
from mori.umbp import UMBPClient, KvIndexerConfig, KvBlockMetadata

# Connect this callback to the integration's existing logical-block metadata.
# Called once per write/batch with the physical keys; return [] to skip them.
def metadata_for_keys(keys):
    hashes = {key_to_hash[key] for key in keys if key in key_to_hash}
    return [metadata_by_hash[h] for h in hashes]

# Each metadata_by_hash value describes the COMPLETE logical block, for example:
# KvBlockMetadata(block_hash, parent_hash, token_count, {1: [k_key, v_key]})
config.kv_indexer = KvIndexerConfig(
    "127.0.0.1:50051", "worker-0", "http://worker-0:30000",
    metadata_provider=metadata_for_keys,
)
client = UMBPClient(config)  # Existing DRAM configuration.
client.put_from_ptr(key, ptr, nbytes)
client.batch_put_from_ptr([k_key, v_key], [k_ptr, v_ptr], [k_nbytes, v_nbytes])
```

Leave `config.kv_indexer = None` (the default) to disable reporting at owner
creation. Configuring an endpoint alone cannot recover logical metadata from
arbitrary physical keys: the integration supplies the callback, including its
lookup state. It also works with depth-aware and ranged batch writes. The
callback runs synchronously before storage writes, outside storage locks; Python
callbacks acquire the GIL. Invalid metadata or callback errors fail the call
before writing. Actual commit/eviction hooks, not callback execution, determine
residency. Indexer RPC failures do not fail successful storage writes.
The callback must be callable (`None` is rejected). Python bound methods retain
their owner weakly to avoid an owner/client reference cycle; keep that owner
alive while using the client. Other callable objects/closures must not own the
client themselves. Repeating identical block metadata is a no-op.

Configure once before issuing writes. Changing the config after construction
does not toggle the client. Metadata for blocks written before enabling is not
automatically recovered; subsequent writes invoke the callback normally.

The integration supplies signed int64 hashes, true parent (`None` for roots),
block size in tokens, and all physical keys per component: FULL=1, SWA=2, MAMBA=4.
Keys must identify immutable content. For hybrid caches configure `components`
and `swa_window_tokens`; metadata must describe every configured component.
Registered parent links must be acyclic. Each component is reported only when
all its keys are committed; FULL is required. Eviction updates/revokes the mask.

Commit and eviction hooks identify affected blocks by key; clear checks all
registered blocks. Residency checks and RPCs run outside the metadata lock, and
the UMBP master's event queue is untouched. REPORT actions pack parent-child
chains, with at most 256 actions and 16384 hashes per RPC. `flush()` synchronizes
pending changes and returns false on RPC failure. Background retries wait one
second even during writes, resample current state and retain ambiguous reports
for withdrawal. Every 30 seconds the reporter checks and re-reports all known
live blocks to repair quiet Indexer restarts, without clearing DRAM first.

Promotion into DRAM uses the same commit hook and reports known blocks when
complete; moving the last DRAM copy out revokes it. Copying to another tier while
retaining DRAM leaves its residency intact. HBM/SSD residency is not published by
this reporter. Promotion of a block with no known metadata cannot advertise it.

Configure once per storage owner. Standalone clients run their configured
callback and send metadata with each batch write; the owner validates it before
writing. New forwarders require an owner advertising batched metadata support.
Closing a forwarder does not
withdraw shared data. Supply metadata where
all required keys reside: reporting covers local DRAM backends, not cross-owner
aggregation. Use one writer per worker's DRAM bucket and match the Router worker
URL and any HBM publisher's identity/cache spec. Startup and graceful owner
shutdown clear DRAM only. The protocol has no TTL/fencing for crash-time cleanup.
An engine bridge sharing this identity must only mutate HBM. The upstream bridge
also maps `CPU_PINNED` to DRAM and clears all tiers by default. Set its
`KV_INDEXER_CLEAR_TIERS=HBM` and exclude non-HBM store/remove events before they
reach the bridge. Tier eligibility masks do not filter these mutations.

Shared DRAM across multiple Indexer worker identities requires an upstream pool
identity protocol and is not supported here. Metadata is retained after eviction
to support SSD promotion; automatic reclamation remains open. Incremental work
avoids scanning this history on every write, but memory use and the periodic
repair still grow with registered metadata.

Reporting is opt-in and requires the endpoint and metadata above. No manual
report/revoke calls are needed; wiring the metadata callback is the integration's
responsibility. The existing
`report_external_kv_blocks` API targets the UMBP master and is unrelated.
RPCs use insecure gRPC with `timeout_ms=200` by default; Python calls release the
GIL. Build with `BUILD_UMBP=ON`, `BUILD_PYBINDS=ON`, and run
`ctest -R '^umbp_kv_indexer_reporter$'` for the automatic reporting tests.
