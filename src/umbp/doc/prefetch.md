# Async prefetch

A single FIFO worker reuses #823's SSD-to-DRAM copies and whole-object peer
pulls to prepare keys in local DRAM before an Engine read.

- `BatchPrefetch(keys)` / Python `batch_prefetch(keys)` returns the number of
  inputs admitted, not completed.
- `SubmitPrefetch(keys, options, request_id)` returns an ID for
  `GetPrefetchStatus(id)`. Results preserve input order; READY means all keys
  were committed and leased in DRAM, PARTIAL means some succeeded.
- Repeating an ID with the same keys/options is idempotent. Duplicate keys
  copy once per request; later requests recheck existing local copies.

The deployed Router's Submit/Status RPCs and wire fields remain compatible.
`UMBP_PREFETCH_CONTROL_ADDRESS=host:port` enables TCP control; Engine I/O keeps
UDS/FD handoff. Older synchronous prefetch clients must migrate to Submit/Status.

Timeout defaults to 5 s (including queue time), lease to 500 ms; zero selects
the default and both are capped at 5 minutes. Completed records last 5 minutes;
READY does not imply permanent residency. Clear drains active I/O before
clearing storage, and Shutdown joins the worker.

`prefetch_budget_bytes=0` disables explicit prefetch; a positive value keeps
#823's soft SSD-copy budget. GET waits for relevant pending keys up to
`UMBP_EXPLICIT_PREFETCH_WAIT_TIMEOUT_MS` (default 5000, zero disables waiting).
Remote destinations must be DRAM and contiguous within one registered buffer.
Allocation/transfer failures fall back to ordinary GET; no source retry or
alternate remote destination is added.
