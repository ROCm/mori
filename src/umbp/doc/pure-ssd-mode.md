# Pure-SSD mode and multi-drive SSD tiers

Run a UMBP node with **no host-DRAM tier**, SSD carrying the whole cache, with
one node's SSD tier spanning **several drives** at once.

The point of the mode is aggregate NVMe bandwidth: N nodes × M drives all
writing and reading in parallel. Most of this page is about not accidentally
giving that up.

## Quick start

Two pieces: the `extra_config` your engine passes to `UMBPStore`, and a handful
of env vars. This is a validated 2-drive-per-node deployment under SGLang
HiCache.

**`extra_config`, per engine:**

```json
{
  "dram_capacity_bytes": 0,
  "ssd_enabled": true,
  "ssd_storage_dir": "/umbp_ssd/drive1,/umbp_ssd/drive2",
  "ssd_capacity_bytes": 68719476736,
  "ssd_backend": "file",
  "ssd_io_backend": "io_uring",

  "master_address": "<UMBP_MASTER>",
  "node_address": "<NODE_IP>",
  "io_engine_port": "19600",
  "peer_service_port": "19700",
  "cache_remote_fetches": false,
  "kv_events_subscriber": true,
  "kv_events_endpoint": "tcp://localhost:6557",

  "staging_buffer_size": 2147483648,
  "ssd_staging_buffer_slots": 512,
  "ssd_staging_buffer_size": 4294967296,
  "ssd_write_staging_slots": 512,
  "ssd_write_staging_size": 4294967296,
  "ssd_staging_use_hugepages": true,
  "ssd_staging_hugepage_size": 2097152
}
```

**Env vars, on each engine:**

```bash
export UMBP_SSD_DIRECT_IO=1          # bypass the page cache -- see "Direct I/O"
export UMBP_SSD_VERIFY_CRC=0         # skip checksums, matching the DRAM tier
export UMBP_SSD_TIER_IO_THREADS=4
export UMBP_SSD_DURABILITY=strict    # ~free once direct I/O is on
export UMBP_SSD_READ_LEASE_MS=30000
export UMBP_SSD_TIMING=1             # per-phase timing; off by default
export UMBP_DRAM_USE_HUGEPAGES=1
export UMBP_DISTRIBUTED_SSD_STAGING_USE_HUGEPAGES=1
```

**Env var, on `umbp_master` — not the engine:**

```bash
export UMBP_ROUTE_PUT_SELECT_ALGO=random   # see "Spreading writes"
```

## Enabling the mode

`dram_capacity_bytes: 0` plus an enabled SSD tier **is** the switch. Nothing
else needs setting: `UMBP_ROUTE_PUT_SSD_MODE` stays at its `auto` default, which
recognises a node with no DRAM and starts routing puts to its SSD, without
changing behaviour on any other node in the pool. Direct-SSD write staging turns
itself on too.

A zero DRAM pool with **no** SSD tier is rejected at startup — that would leave
no tier able to accept a put.

## Spreading writes

**Set `UMBP_ROUTE_PUT_SELECT_ALGO=random`.** This is the single most important
knob in the mode, and the default is wrong for it.

The default, `most_available`, sends each key to the emptiest node. Node capacity
only refreshes on a heartbeat, so every batch in between sees the same "emptiest"
node and piles onto it — and concurrent writers herd onto it together. Aggregate
write bandwidth collapses to one node's drives. `random` is capacity-weighted
(a node that cannot fit a block is never chosen), so it still fills evenly
without the herding.

Leave `UMBP_ROUTE_PUT_NODE_AFFINITY` at `none`. `same` and `local` trade fan-out
for locality, which is the opposite of what this mode is for.

Both are read by the **master process**. Setting them on an engine does nothing.
The master logs its resolved strategy at startup — check that line if a change
seems to have no effect.

## Sizing

| Setting | Meaning |
|---|---|
| `ssd_storage_dir` | Comma-separated, **one directory per physical drive**. More than one turns on multi-drive: keys are balanced across drives and batch I/O runs on all of them at once. |
| `ssd_capacity_bytes` | The **total** across every drive, split evenly — not per drive. |
| `ssd_staging_buffer_slots` / `_size` | Read staging. Per-slot size (`size / slots`) must stay **≥ your largest single value**. 512 slots over 4 GiB = 8 MiB/slot. |
| `ssd_write_staging_slots` / `_size` | Same, for writes. `-1` slots = auto (on in pure-SSD mode). |

Too few staging slots and a large batch spends rounds waiting for one. With the
default `UMBP_SSD_GET_MAX_ATTEMPTS=1` that surfaces as a **cache miss**, not a
retry — raise the slot count, or that setting, if your read staging is small
relative to your batches.

Each key lives whole on one drive, so a single-key read costs one drive's I/O.
Capacity is charged in **padded on-disk bytes**: values are padded to 4 KiB, so
a 512 B value occupies 8 KiB. Negligible for multi-MiB KV pages, but size
`ssd_capacity_bytes` with your real value size in mind if yours are small.

## Direct I/O

**Set `UMBP_SSD_DIRECT_IO=1` for any deployment whose numbers you intend to
believe.** Without it the tier opens buffered, and reads are served from the
page cache — a whole working set can be served while moving *zero* bytes to the
device, which makes drive-count and DRAM-vs-SSD comparisons meaningless.

Pair it with `ssd_io_backend: io_uring`; the posix driver on the direct path is
far slower, and the tier warns about it.

| Env var | Default | Meaning |
|---|---|---|
| `UMBP_SSD_DIRECT_IO` | `0` | `1` = `O_DIRECT`, bypassing the page cache |
| `UMBP_SSD_VERIFY_CRC` | `1` | `0` = skip checksum verification. Checksumming is 51–67% of GET time and the DRAM tier does none, so turn it off to compare the two without that confound. |
| `UMBP_SSD_TIER_IO_THREADS` | `4` | Parallelism **within one drive's** batch. Worth ~42% on reads versus 1. |
| `UMBP_SSD_SHARD_IO_THREADS` | `0` | Parallelism **across drives**. `0` = one per drive; leave it there unless CPU-bound. |
| `UMBP_SSD_DURABILITY` | `strict` | With direct I/O on, strict costs ~1.4% instead of 42%. Little reason to relax it. |

Measured on ext4/nvme1n1, 64 × 4 MiB, io_uring (MB/s):

| Config | Write | ReadBatch | Device r/w (MiB) |
|---|---|---|---|
| buffered / crc-on / 4t | 1551 | 7503 | **0 / 0** |
| direct / crc-on / 4t | 1954 | 5356 | 5633 / 6150 |
| direct / crc-off / 4t | 2004 | 6002 | — |
| direct / crc-on / 1t | 2063 | 3772 | — |

**Confirm it is on**: run `iostat -x 1` during a read-heavy phase. Buffered shows
~1% device utilisation while reporting absurd bandwidth; direct shows the drives
busy at something near their rating.

## Checking your fan-out

`UMBP_SSD_TIMING=1` prints per-phase timing (`[SsdPerf/*]`: device read/write,
CRC, RDMA, staging) — the first thing to reach for when throughput is low.
`UMBP_LOCAL_COPY_TIMING=1` does the same for node-local traffic.

There is also per-node/tier placement accounting behind `UMBP_PUT_DIST_LOG` /
`UMBP_GET_DIST_LOG`, which answers "are writes actually spreading?" directly.
Its log lines are **commented out in the source** — one line per batch is too
chatty to ship — so using it means uncommenting the `MORI_UMBP_INFO` in
`LogBatchPutDistribution` / `LogBatchGetDistribution` /
`ConfigurableRoutePutStrategy::LogBatchDistribution` and rebuilding. The number
to watch is `cumulative_max_share`: `100/N%` over N targets is a perfect spread,
100% means everything landed on one node.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Absurd read bandwidth, `iostat` shows ~1% device utilisation | `UMBP_SSD_DIRECT_IO` not set — you are reading the page cache |
| Adding drives or nodes doesn't add bandwidth | Writes are skewed. Set `UMBP_ROUTE_PUT_SELECT_ALGO=random`, check `UMBP_ROUTE_PUT_NODE_AFFINITY=none` |
| `random` set but still skewed | It was set on the engine instead of `umbp_master`. Check the master's startup strategy line |
| Reads report misses under load | Read staging slots exhausted. Raise `ssd_staging_buffer_slots` or `UMBP_SSD_GET_MAX_ATTEMPTS` |
| Puts fail on a pure-SSD node | That node advertises no write-staging slots — `ssd_write_staging_slots` forced to `0`, or it is not actually in pure-SSD mode |
| Node comes up with an empty cache after an upgrade | On-disk record format changed; older records are dropped on version mismatch |
| Tier fills far earlier than expected | Capacity is charged in padded bytes — small values pay up to 16× |
| Startup rejects `dram_capacity_bytes: 0` | No SSD tier configured, so no tier could accept a put |
| A config key seems to do nothing | `extra_config` is an allow-list; unknown or wrong-mode keys are ignored with a warning. Grep the engine log for `UMBPStore:` |

## Notes

- **Scope traps.** `UMBP_ROUTE_PUT_SELECT_ALGO` is read by the master.
  `UMBP_DISTRIBUTED_SSD_STAGING_USE_HUGEPAGES` is read only by the standalone
  server binary — on the SGLang path the `ssd_staging_use_hugepages` JSON key is
  the one that takes effect; setting both covers either launch mode.
  `UMBP_DRAM_USE_HUGEPAGES` does nothing when `dram_capacity_bytes` is 0.
- `cache_remote_fetches: false` keeps remote-fetch cost honest by not re-caching
  locally. Turn it on for a production hit-rate setup.
- **SPDK.** `UMBP_SPDK_REACTOR_MASK` defaults to `0x1` — a single core polling
  all drives. Give it at least one core per drive (`0x3`, `0xf`, …) before
  expecting multi-drive scaling on that path. Multi-drive tiers and SPDK RAID0
  compose; a drive in the list may itself be a RAID0 bdev.
- Every setting above is also a plain field on `UMBPConfig` / `UMBPDistributedConfig`
  for callers driving the client directly, without SGLang.

Full env-var reference: [runtime-env-vars.md](runtime-env-vars.md).
