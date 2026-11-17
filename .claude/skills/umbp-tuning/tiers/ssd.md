# SSD tier module

SSD-specific mechanics for the stages in `../SKILL.md`. Read that first; this file only adds what the SSD medium changes.

Defaults below are from `src/umbp/include/umbp/common/config.h` and `src/umbp/doc/runtime-env-vars.md` at `7e23fa2ae`. Re-check them in the build under test. `src/umbp/doc/pure-ssd-mode.md` is the deployment guide.

## How the tier behaves (the facts tuning rests on)

- **Segmented log.** Objects are appended to segment files. Disk comes back only when a whole segment is empty and its file is deleted; eviction hands out whole segments, emptiest and coldest first. A segment's recency is its most recent key's.
- **One open file per live segment.** Capacity ÷ segment size must stay under the process's open-file limit.
- **Direct I/O by default.** The page cache is bypassed; the tier falls back to buffered with a warning on filesystems that reject it.
- **Staging arena.** Reads and remote writes cross it (drive ↔ arena, arena ↔ wire or GPU); local writes do not. A read holds its pages under a lease.
- **Capacity is charged in padded on-disk bytes**, and "used" includes dead records until their segment empties.
- **Tiers are declared in a policy file** (`UMBP_BACKEND_POLICY`): backends (`type`, `capacity`, `path`, `staging_slots`, `min_object_size`) and tiers (`backends` weights, `offload_to`, `offload_trigger` on_evict | watermark, `high_watermark`, `low_watermark`, `promote_trigger` never | on_read | on_hits, `promote_hits`, `promote_mode` copy | move).

## Stage 0 additions

| Check | Where to look |
|---|---|
| SSD settings reached the tier | startup line `[SSDTier] <dir>: direct_io= verify_crc= tier_io_threads= io_backend=`. Policy-declared SSD tiers ignored `UMBP_SSD_*` in builds before #967. |
| Direct I/O is really on | same line; and the drives show real utilization under read load (`iostat -x 1`). Near-idle drives with high bandwidth means page cache. |
| Each directory is its own drive | two directories on one filesystem double-count free space and share one device's bandwidth. |
| Free space | shared drives change under you; leave a reserve per filesystem. |
| The tier filled and evicted in the window | SSD line of the activity summary: `resident` near its watermark and `evict` non-zero. |
| Device baseline exists | a raw-device read/write test at the object size, including sustained writes. Without it a device limit cannot be told from a UMBP limit. |

## Stage 2A — hit-loss classes on SSD

| Class | SSD mechanism | Signal | Lever |
|---|---|---|---|
| Partly present, unusable | Segment-granular reclaim: a segment holds objects of several prefixes, so an idle prefix loses scattered objects and the rest cannot be loaded as a run. DRAM evicts per key and does not show this. | Lookups find many of a prefix's objects but a short or zero usable run; the same workload on DRAM shows none of this class. | Segment size close to one object. Cost: many files, slower pool I/O, open-file limit. |
| Evicted whole | Tier smaller than what is demoted between reuses. | `evict` on the SSD summary line during the window; `mori_umbp_client_local_evict_total`. | Capacity (larger tier, more drives, pooling another node's drives through the master); or cut inflow (2C). |
| Short-lived state gone | Data the engine reuses within seconds to minutes is pushed out of the DRAM entry tier before that reuse. | Loss concentrated in requests that branch from a recent prompt; entry-tier turnover time shorter than the reuse gap. | Larger entry tier; its watermarks; per-object eviction classes, where the build has them. |
| Present, not served | Staging stayed full past the busy-retry deadline, a batch's own working set exceeds the arena, or a read lease expired before the copy. | `mori_umbp_backend_medium_events_total{event="staging_slot_full_reject"}`, `{event="staging_expired_reclaim"}`; failed loads in the engine. | A larger staging arena (scale with concurrent readers and largest batch), `UMBP_RESOLVE_BUSY_TIMEOUT_MS`, `UMBP_SSD_READ_LEASE_MS`. |
| Never stored | Entry tier at its watermark when a burst arrives; staging full on a remote put (not retried); object below `min_object_size`. | `[PageBackend] Allocate reason=NO_SPACE`; `{event="staging_put_full_reject"}`. | Headroom between entry-tier watermark and capacity; staging size. |
| Promotion churn | Copy-promotion on read leaves the SSD copy looking cold, so it is evicted and later rewritten. | Promote and demote volumes both large in the summary; writes far above live data. | `promote_trigger` never or on_hits; `promote_mode` move. |

## Stage 2B — load latency on SSD

Turn on `UMBP_SSD_TIMING=1` for a stage-by-stage `[SsdPerf/tier]`, `[SsdPerf/shard]`, `[SsdPerf/peer]` line per batch (index lookup, wait for a ring, device I/O, checksum, fan-out, implied GB/s). `mori_umbp_backend_op_seconds_total{tier="SSD",op=...}` gives the same split over time.

| Dominant stage | Lever |
|---|---|
| Waiting for a ring or staging | `UMBP_SSD_READ_RINGS`, staging arena size, peer poller threads (`UMBP_PEER_MIN_POLLERS`) |
| Device read | more drives (one directory per drive), faster drives, `UMBP_SSD_SHARD_IO_THREADS`; check the drive baseline |
| Checksum / CPU | `UMBP_SSD_VERIFY_CRC`, `UMBP_SSD_TIER_IO_THREADS` |
| Staging copy | hugepage-backed staging (`UMBP_DISTRIBUTED_SSD_STAGING_USE_HUGEPAGES`); coalescing concurrent reads of one key (`UMBP_SSD_SINGLE_FLIGHT`) |
| Reading more bytes than needed | ranged reads and extent staging (`UMBP_SSD_EXTENT_STAGING`, `UMBP_DISTRIBUTED_RANGED_SCRATCH_BYTES`). Compare drive bytes read with KV bytes the engine needed; a ratio well above 1 means whole-object reads. An engine layout whose first range spans an object's head and tail defeats ranged reads (#920). |
| Sync on write | `UMBP_SSD_DURABILITY` |

Moving data ahead of need (SSD → DRAM prefetch, promotion) is a 2B lever only. Use it when the load is shown to sit on the critical path; with asynchronous loads behind a queue it usually is not.

## Stage 2C — write pressure on SSD

| Quantity | How to get it | What it tells you |
|---|---|---|
| Demote rate into SSD | `move-write` on the SSD summary line; `mori_umbp_client_tier_move_write_bytes_total` | sizing input, and bytes written per day against the drive's endurance rating |
| Read-back share | `load` on the SSD line ÷ `move-write` | low share means most demoted data is never reused |
| Refused puts | `NO_SPACE`, `staging_put_full_reject` | entry tier or staging cannot absorb bursts |
| Live vs used | `mori_umbp_backend_medium_state{state="live_bytes"}` vs used; `{event="segment_deleted"}` should grow while evicting | dead data pinned by partly live segments |

Levers: do not demote short-lived classes; retire data the engine knows is superseded; store one copy where ranks hold identical data; `min_object_size` to keep small objects off SSD; a larger entry tier so less is demoted at all.

## Knobs by lever family

| Family | Knob (default) |
|---|---|
| Capacity | backend `capacity` in the policy, or `UMBP_SSD_CAPACITY` (32 GiB) with `UMBP_SSD_DIR` (comma-separated, one per drive) |
| Eviction unit and order | `UMBP_SSD_SEGMENT_SIZE` (4 GiB; multiple of 4096); tier `high_watermark` / `low_watermark`, or `UMBP_SSD_HIGH_WM` / `UMBP_SSD_LOW_WM` (0.9 / 0.7); engine recency hints (touch) reaching the pool |
| Residence time | entry tier `UMBP_DRAM_CAPACITY`, `UMBP_DRAM_HIGH_WM` / `UMBP_DRAM_LOW_WM` (0.95 / 0.9), or the tier's watermarks in the policy |
| Placement and promotion | `promote_trigger`, `promote_hits`, `promote_mode`, `offload_trigger`, `min_object_size`; with a master, `UMBP_ROUTE_PUT_SELECT_ALGO` (set on the master process, not the engine) |
| Read path | `UMBP_SSD_READ_RINGS` (4), `UMBP_SSD_TIER_IO_THREADS` (4), `UMBP_SSD_SHARD_IO_THREADS` (0 = one per drive), `UMBP_SSD_SINGLE_FLIGHT` (1), policy `staging_slots` or `UMBP_DISTRIBUTED_SSD_STAGING_BUFFER_SLOTS` (0 = 4 GiB worth), `UMBP_SSD_EXTENT_STAGING`, `UMBP_RESOLVE_BUSY_TIMEOUT_MS` (30000), `UMBP_SSD_READ_LEASE_MS` (3000), `UMBP_SSD_BACKEND` (file; spdk, spdk_proxy) |
| Integrity overhead | `UMBP_SSD_VERIFY_CRC` (1), `UMBP_SSD_DURABILITY` (strict) |
| Host resources | `UMBP_DISTRIBUTED_SSD_STAGING_USE_HUGEPAGES` (0), `UMBP_SSD_DIRECT_IO` (1; leave on) |
| Diagnostics | `UMBP_SSD_TIMING`, `UMBP_SUMMARY_INTERVAL_MS` (60000) |

Engine-side load settings (load concurrency, load window, prefetch, async vs in-step) belong to the engine integration, not mori; find them in the connector or store class of the engine build under test.

## Sizing on SSD

- **Capacity** ≥ demote rate × the longest idle gap a session must survive. Measure the demote rate on the workload at the target load.
- **Segment size:** start just above the largest object if the "partly present" class is material; otherwise larger segments mean fewer files and cheaper I/O. Check capacity ÷ segment size against the open-file limit.
- **Staging:** the arena must hold the largest single batch, and grows with concurrent readers per node, not with data volume.
- **Entry tier:** capacity × high watermark ÷ inflow rate must exceed the reuse gap of data that is never demoted, with headroom above the watermark for bursts.
- **Drives:** one directory per physical drive; split by usable free space; keep a reserve per filesystem.

## Traps

- A setting made in the wrong process does nothing (engine vs pool server vs master). Confirm from that process's startup lines.
- Knobs differ between builds; a variable that an older or patched image honoured may be absent on `main`.
- On-disk format changes drop older records at startup, so a node can come up with an empty cache after an upgrade.
- A tier that looks full early is often padding or dead records, not live data.
- Many small segment files trade hit rate for slower loads and stores; report both sides.
