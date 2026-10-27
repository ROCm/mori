# Report UMBP DRAM blocks to the SGLang KV Indexer

`UMBPIndexerReporter` sends `report`, `revoke`, and `clear` directly to the
existing Indexer gRPC API. It has no background thread or automatic hooks.
Call it from the cache integration after successful commit, eviction, or clear:

```python
from mori.umbp import UMBPIndexerReporter

reporter = UMBPIndexerReporter("127.0.0.1:50051", "worker-0", "http://worker-0:30000")
reporter.clear()  # Explicitly reset this worker's DRAM at startup, if appropriate.
# After all required objects/shards have committed: consecutive logical blocks.
reporter.report([101, 102], parent_hash=None, block_size=16)
reporter.revoke([102])  # After eviction.
reporter.clear()        # After clearing the cache, or before a clean shutdown.
```

The caller supplies canonical signed int64 hashes (including LoRA/cache-salt
namespacing), the true parent of the first hash (`None` only for a root), and
block size in **tokens**. Physical UMBP keys cannot supply this metadata. Report
only complete, usable components; a single TP rank cannot claim missing shards.
Each call accepts at most 16,384 hashes; split larger chains at the caller,
passing the preceding hash as the next call's parent.

For hybrid caches set constructor `components` (FULL=1, SWA=2, MAMBA=4) and
`swa_window_tokens`, then pass the currently resident `component_mask` to
`report`. Masks replace the existing placement; FULL must be present. All
components permit HBM+DRAM in prefix queries. Only DRAM placements are mutated.
One reporter must own the worker's DRAM tier; any HBM publisher must use the
same worker ID, router-facing address, and cache spec. The address must exactly
match the URL registered with the Router.

Calls are synchronous, serialized, and release Python's GIL. Invalid input
raises `ValueError`; RPC failures raise `RuntimeError` with the gRPC code and
message. `timeout_ms` defaults to 200. A failure may follow a server commit;
reconcile current state instead of blindly replaying an old event. No retry,
replay, worker TTL, or restart recovery is added. The transport is insecure
gRPC, matching the upstream trusted-network server. No SGLang code changes
are needed, but the caller must supply metadata and invoke these methods.

Build with `BUILD_UMBP=ON` and `BUILD_PYBINDS=ON`. Native API:
`mori::umbp::KvIndexerReporter`. Run `ctest -R '^umbp_kv_indexer_reporter$'`;
set `UMBP_TEST_KV_INDEXER_ENDPOINT` to an isolated real Indexer to enable the
optional hybrid prefix interoperability case. The pinned, unmodified wire
proto and its Apache-2.0 attribution are under `../distributed/proto/`.
