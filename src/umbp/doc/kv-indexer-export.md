# UMBP → SGLang KV Indexer

Set `UMBPConfig.kv_indexer` before creating a client to report committed local
DRAM automatically. The default `None` disables reporting; write APIs are unchanged.

```python
from mori.umbp import KvIndexerConfig, KvBlockMetadata, UMBPClient

# The integration supplies complete canonical metadata for each logical block.
def metadata_for_keys(keys):
    return [metadata_by_hash[h] for h in {key_to_hash[k] for k in keys}]

config.kv_indexer = KvIndexerConfig(
    "indexer:50051", dram_pool_id="model/node-0/dram",
    metadata_provider=metadata_for_keys,
)
client = UMBPClient(config)
```

Each metadata record is `KvBlockMetadata(hash, parent_hash, token_count,
component_keys)`: signed int64 hash, `None` for a root, and all physical keys for
FULL=1/SWA=2/MAMBA=4. Records are immutable while tracked; repeats reuse metadata.
The callback runs before writes, outside storage locks. Indexer outages do not
fail successful storage writes. Bound
Python methods are held weakly: keep the adapter alive while using its client.

Pool mode requires [SGLang's shared-pool protocol](../distributed/proto/README.sgl-kv-indexer.md).
Use one owner per pool and compatible model/page/key layouts. All forwarders use
the same pool configuration; each HBM bridge sets `KV_INDEXER_DRAM_POOL_ID` and
its own component/window settings. The owner reports DRAM once and the Indexer
combines it with each member's HBM. Closing a forwarder preserves the pool.

Commit/eviction hooks send deltas; metadata travels with standalone batch writes.
`flush()` awaits reporting. RPCs default to a 2s `timeout_ms`; periodic repair uses
`repair_interval_ms` (default 30000, must be positive) and re-reports in bounded
batches without clearing first. Longer intervals reduce work but delay idle
Indexer restart recovery. Repeated sync-failure warnings are limited to once per
minute. Metadata is reclaimed after its last
local physical copy and in-flight write are gone and withdrawal succeeds; SSD
copies retain metadata for promotion. C++ callers hold the `RegisterKvBlock`
return value through their write. Graceful owner shutdown withdraws DRAM.
There is no crash-time TTL/fencing or HBM/SSD publication.
