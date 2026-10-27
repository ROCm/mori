# Asynchronous prefetch to local DRAM

This extends #823's SSD copies and batched peer pulls with a bounded FIFO for
Engine hints, Router requests and reactive ranged-read prefetch.

- `BatchPrefetch(keys)` / Python `batch_prefetch(keys)` returns the number of
  input keys admitted. Admission does not mean the data is ready.
- C++ `SubmitPrefetch(keys, options, request_id)` returns immediately;
  `GetPrefetchStatus(id)` reports a per-input `ok` vector, completion count,
  state and timing. Repeating an ID with the same keys and normalized options
  is idempotent; a different payload is rejected.

One worker executes requests in order, in batches of up to 128 unique keys.
It first uses #823's SSD-to-DRAM copy, then routes nonlocal keys and pulls them
into the configured local medium when it is DRAM. Remote destination slots
are bounded to 128 MiB per batch. A failed allocation or transfer leaves that
key unprepared; the ordinary GET path remains available. No prefetch-specific
source retry or fallback to another DRAM backend is added.

Duplicate inputs retain their result positions and copy once per request.
Later requests recheck the copies already in DRAM. The worker validates and
leases committed DRAM copies after the whole request, so an SSD copy removed
by the prefetch budget does not report success. SSD sources remain intact;
remote copies use ordinary resident eviction policy. Prefetch stops at local
DRAM; the Engine's normal LOAD transfers data into its GPU buffer.

## Router contract

The deployed experimental Router keeps these RPCs and wire fields:

- `/umbp.UMBPStandalone/SubmitPrefetch`: keys=1, lease_ttl_ms=2, timeout_ms=3,
  request_id=4; response fields 1–3 are unchanged.
- `/umbp.UMBPStandalone/GetPrefetchStatus`: status fields 1–8 are unchanged.
- States: UNKNOWN=0, QUEUED=1, RUNNING=2, READY=3, PARTIAL=4, FAILED=5, EXPIRED=6.
  Results are published when the worker finishes. READY means every input
  was prepared; PARTIAL means some succeeded; EXPIRED gives no lease guarantee.

`UMBP_PREFETCH_CONTROL_ADDRESS=host:port` enables the optional TCP control
listener. Router worker/key configuration and the Engine's UDS/FD handoff
remain compatible. `Ping` advertises async support and count-hint version 1.
The older synchronous `PrefetchRequest -> BatchBoolResponse` is incompatible
with #823's count hint: identifiable legacy requests are rejected; keys-only
legacy clients must migrate to Submit/Status.

## Bounds and lifetime

Timeout and lease default to 5 s and 500 ms; zero selects these defaults and
each is capped at 5 minutes. Timeout includes FIFO queue time. Expiry never
releases a live I/O buffer. Leases begin at final validation; a retained READY
record does not promise permanent residency.

The queue retains up to 1024 tasks, 65536 input positions and 64 MiB of key
strings. Each request allows 4096 keys; keys and IDs are capped at 4096 and
256 bytes. Completed tracked records are retained for 5 minutes; completed
hints are discarded. Clear cancels queued work, drains the active request and
then clears storage. Shutdown joins the worker before destroying its backends.

GET/ranged GET wait only for relevant pending keys, within
`UMBP_EXPLICIT_PREFETCH_WAIT_TIMEOUT_MS` (default 5000, maximum 300000; zero
disables waiting). `prefetch_budget_bytes=0` disables explicit hints/tasks;
a positive value keeps #823's soft SSD-copy budget. Reactive ranged prefetch
retains its independent switch. Remote whole-object destinations must be
contiguous within one MR; unsupported shapes are aborted.

CPU tests cover admission, timeout/drain, actual SSD files, remote destination
commit/abort with simulated transport, and the deployed Router wire contract.
Transport-seam tests require `MORI_UMBP_TESTING=ON` for all linked objects.
Real two-node RDMA and SGLang performance still require hardware validation.
