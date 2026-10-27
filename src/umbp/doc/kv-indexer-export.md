# UMBP → SGLang KV Indexer

```python
from mori.umbp import UMBPIndexerReporter
r = UMBPIndexerReporter("127.0.0.1:50051", "worker-0", "http://worker-0:30000")
r.report([101, 102], parent_hash=None, block_size=16)  # After all shards commit.
r.revoke([102])                                      # After eviction.
r.clear()                                           # This worker's DRAM only.
```

Supply canonical int64 hashes, true parent (`None` only for roots), and token
count. One call describes a consecutive chain of at most 16,384 blocks. The
caller owns event timing, complete shard metadata, and recovery. For hybrid
caches set constructor `components` (FULL=1, SWA=2, MAMBA=4), `swa_window_tokens`,
and report's `component_mask` (replacement, including FULL). Prefix queries
permit HBM+DRAM. Use one DRAM writer and the same identity/address/cache spec as
any HBM publisher; address must match the Router's registered worker URL.

Calls are synchronous, release the GIL, and use insecure gRPC. `timeout_ms`
defaults to 200. Invalid input raises `ValueError`; RPC failure raises
`RuntimeError` and may follow a server commit. Reconcile current state instead
of blindly replaying old events. There are no automatic hooks or retries.
Build with `BUILD_UMBP=ON`, `BUILD_PYBINDS=ON`; run
`ctest -R '^umbp_kv_indexer_reporter$'` for wire/error tests.
