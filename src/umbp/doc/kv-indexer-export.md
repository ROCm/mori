# UMBP → SGLang KV Indexer

Configure an existing DRAM `UMBPClient` once and register canonical metadata:

```python
client.configure_kv_indexer("127.0.0.1:50051", "worker-0", "http://worker-0:30000")
client.register_kv_block(101, None, 16, {1: ["block101-k", "block101-v"]})
client.put_from_ptr("block101-k", k_ptr, nbytes)
client.put_from_ptr("block101-v", v_ptr, nbytes)  # Automatically reports FULL.
client.register_kv_block(101, None, 0, {})  # Unregister when no longer needed.
```

The integration supplies signed int64 hashes, true parent (`None` for roots),
block size in tokens, and all physical keys per component: FULL=1, SWA=2, MAMBA=4.
Keys must identify immutable content. For hybrid caches configure `components`
and `swa_window_tokens`; metadata must describe every configured component.
Registered parent links must be acyclic. Registration may precede or follow writes. Each component is reported only when
all its keys are committed; FULL is required. Eviction updates/revokes the mask.

Commit, eviction and clear hooks wake one background reporter, including batch,
ranged and peer operations. It batches changes into at most 256 actions per RPC
and leaves the UMBP master's event queue untouched. RPC failures do not change
PUT success. `flush()` synchronizes current residency and returns false on RPC
failure. Background retries wait one second even during writes and resample
current state. A rebuild every 30 seconds also repairs quiet Indexer restarts;
rebuilds may briefly reduce hits.

Configure once per storage owner. Standalone clients forward metadata to that
owner; closing a forwarder does not withdraw shared data. Register metadata where
all required keys reside: reporting covers local DRAM backends, not cross-owner
aggregation. Use one writer per worker's DRAM bucket and match the Router worker
URL and any HBM publisher's identity/cache spec. Startup and graceful owner
shutdown clear DRAM only. The protocol has no TTL/fencing for crash-time cleanup.

Reporting is opt-in and requires the endpoint and metadata above. No SGLang
changes or manual report/revoke calls are needed. The existing
`report_external_kv_blocks` API targets the UMBP master and is unrelated.
RPCs use insecure gRPC with `timeout_ms=200` by default; Python calls release the
GIL. Build with `BUILD_UMBP=ON`, `BUILD_PYBINDS=ON`, and run
`ctest -R '^umbp_kv_indexer_reporter$'` for the automatic reporting tests.
