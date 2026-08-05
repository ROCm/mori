# Pure-SSD mode and multi-drive SSD tiers

Branch: `feat/umbp-pure-ssd-multidrive`.

Three changes that compose:

1. **Pure-SSD mode** — a node can run with *no* host-DRAM tier, SSD carrying the
   whole cache. SSD becomes a routable put target instead of a place bytes only
   ever arrive at asynchronously.
2. **Multi-drive tiers** — one node's SSD tier can span several drives, with
   keys spread across them and IO issued to all of them concurrently.
3. **A device-bound SSD tier** — direct I/O, optional checksumming, and per-tier
   IO threads, so the drives are actually the thing being measured.

## TL;DR — recommended configuration

This is a validated 2-drive-per-node pure-SSD deployment under SGLang HiCache.
Two pieces: the `UMBPStore` `extra_config` JSON on each engine, and the
SSD-tier / routing env vars.

**`extra_config` (per engine):**

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

**Env vars — on each engine (SSD tier, direct I/O):**

```bash
export UMBP_SSD_DIRECT_IO=1          # bypass the page cache — see "device-bound"
export UMBP_SSD_VERIFY_CRC=0         # match DRAMTier: no checksums
export UMBP_SSD_TIER_IO_THREADS=4    # CPU fan-out within one drive's batch
export UMBP_SSD_DURABILITY=strict    # ~free once direct I/O is on
export UMBP_SSD_READ_LEASE_MS=30000  # generous lease; 512 slots × large batches
export UMBP_SSD_TIMING=1             # per-phase device timing
export UMBP_DRAM_USE_HUGEPAGES=1
export UMBP_DISTRIBUTED_SSD_STAGING_USE_HUGEPAGES=1
```

**Env var — on `umbp_master`, *not* the engine:**

```bash
export UMBP_ROUTE_PUT_SELECT_ALGO=random   # RECOMMENDED — see below
```

Notes on the combo:

- `dram_capacity_bytes: 0` + an enabled SSD tier **is** the pure-SSD switch.
  `UMBP_ROUTE_PUT_SSD_MODE` can stay at its `auto` default: `auto` keys off
  exactly this "no DRAM/HBM total bytes" signal, so SSD becomes a routable put
  target on these nodes without changing behaviour anywhere else in the pool.
- `ssd_capacity_bytes` is the **total across both drives** (64 GiB → 32 GiB per
  drive), not per drive.
- `ssd_io_backend: "io_uring"` matters with `UMBP_SSD_DIRECT_IO=1`; the posix
  driver on the direct path is far slower and the tier logs a warning about it.
- **Staging slots are generous on purpose.** 512 read slots over a 4 GiB region
  is 8 MiB/slot, which must stay ≥ your largest single key; too few slots and a
  large batch burns rounds on `NO_SLOT`, which with the default
  `UMBP_SSD_GET_MAX_ATTEMPTS=1` surfaces as a *miss*, not a retry.
- **Scope traps.** `UMBP_ROUTE_PUT_SELECT_ALGO` is read by the master — setting
  it on the engine does nothing. `UMBP_DISTRIBUTED_SSD_STAGING_USE_HUGEPAGES` is
  read only by the standalone-server binary; on the SGLang path the JSON key
  `ssd_staging_use_hugepages` is the one that lands. Setting both covers either
  launch mode. `UMBP_DRAM_USE_HUGEPAGES` is inert here — with
  `dram_capacity_bytes: 0` no DRAM pool is allocated at all — but harmless, and
  it is what you want if a node later runs with DRAM.
- `cache_remote_fetches: false` keeps remote-fetch cost honest by not
  re-caching locally; turn it on for a production hit-rate setup.

**Use `random`, not the `most_available` default, for the RoutePut algorithm.**
`most_available` picks the single emptiest node per key, and projected capacity
deductions are discarded at the end of each batch — every batch restarts from
the same heartbeat-aged capacity snapshot. So a stream of `BatchPut`s between two
heartbeats all see the same "emptiest" node and pile onto it, and with several
clients writing concurrently they herd onto it *together*. On a DRAM deployment
that costs little. On a pure-SSD deployment it directly costs aggregate NVMe
bandwidth: the whole point is to have N nodes × M drives writing at once, and a
skewed batch collapses that to one node's drives.

`random` is **capacity-weighted** random — the probability of a node is
proportional to its projected `available_bytes`, and a node that cannot fit the
block is never picked. So it still respects capacity and still drains toward
balance, but without the stale-snapshot herding. Confirm the effect with
`cumulative_max_share` in the `batch_dist` / `put_dist` lines (see
[Observability](#observability)): 100/N% over N targets is a perfect split.

## Why SSD was not a put target before

`RoutePut` only ever considered HBM and DRAM. SSD was filled by exactly one
mechanism: `SsdCopyPipeline`, an async copy-on-commit mirror of a DRAM commit
that already landed on that same node. That is why a node with no DRAM tier had
nowhere for a put to land, and why SSD content was a strict function of DRAM
placement rather than something the router could balance.

## Enabling pure-SSD mode

Set the DRAM pool to zero and give the SSD tier a capacity:

```bash
UMBP_DRAM_CAPACITY=0 \
UMBP_SSD_ENABLED=1 \
UMBP_SSD_CAPACITY=$((2048 * 1024 * 1024 * 1024)) \
UMBP_SSD_DIR=/mnt/nvme0,/mnt/nvme1,/mnt/nvme2
```

What that does:

- `DistributedClient` allocates **no** host DRAM pool at all.
- The node advertises **no** `TierType::DRAM` entry in `tier_capacities` — the
  entry is omitted, not reported as zero. `RoutePut`'s `auto` SSD mode keys off
  exactly this "no DRAM/HBM total bytes" signal to identify a pure-SSD node.
- Direct-SSD write staging is enabled automatically (see below), because a peer
  with no DRAM tier *must* be able to accept a direct SSD write.

`UMBPConfig::Validate` allows `dram.capacity_bytes == 0` only in this
configuration; a zero DRAM pool with no SSD tier is still an error, since it
would leave no tier that can accept a put.

## How keys are distributed

Unchanged in shape, extended in reach: the client asks the master for a routing
advisory (`BatchRoutePut`), and `ConfigurableRoutePutStrategy` picks a node per
key. Tier order is HBM → DRAM → SSD, with the SSD step gated by
`UMBP_ROUTE_PUT_SSD_MODE`:

| Mode | SSD eligible when |
|---|---|
| `auto` (default) | the node reports no DRAM/HBM capacity at all (pure-SSD node) |
| `always` | always, after HBM and DRAM — a full-DRAM node becomes a spill target |
| `never` | never (the previous behavior) |

`auto` is deliberately conservative: a node whose DRAM merely happens to be
*full* right now still routes elsewhere rather than demoting the put to its own
cold tier. Use `always` only when you want a DRAM node to spill to its own SSD
instead of failing a put; it changes behaviour on every node in the pool.

The existing `UMBP_ROUTE_PUT_SELECT_ALGO` and `UMBP_ROUTE_PUT_NODE_AFFINITY`
knobs apply to SSD placement exactly as they do to DRAM:

| `UMBP_ROUTE_PUT_SELECT_ALGO` | Behaviour | Pure-SSD verdict |
|---|---|---|
| `random` | capacity-weighted random over the nodes that fit | **recommended** — spreads writes over every node's drives |
| `most_available` (default) | always the emptiest node | herds between heartbeats; caps aggregate write BW at one node |

Within a *single* batch both algorithms deduct projected capacity as they go, so
a large batch spreads either way. The difference is *across* batches, which is
where a real workload lives — hence the `random` recommendation above.

`UMBP_ROUTE_PUT_NODE_AFFINITY` should stay `none` for a pure-SSD pool. `same`
pins a whole batch to one node and `local` prefers the requester's own node;
both deliberately trade fan-out for locality, which is the opposite of what an
aggregate-bandwidth deployment wants.

## The put path

**Self-target** — one `PeerSsdManager::WriteBatch` for the whole local group:
one dedup pass, one batched device write, one recording pass.

**Remote** — a three-step flow mirroring the DRAM put, re-introducing RPCs an
earlier redesign had removed:

```
BatchAllocateSsdWriteSlots   → per-key staging offset + lease
   (writer RDMA-writes into those offsets, one batched transfer)
BatchCommitSsdWrites         → peer runs ONE PeerSsdManager::WriteBatch
                               and emits ADD SSD for every key that landed
BatchAbortSsdWriteSlots      → on transfer failure, hand the slots straight back
```

Per-key allocate outcomes are typed: `NO_SLOT` is transient (the writer retries
just those keys next round), `NO_SPACE` means the tier is full so the writer
should retry `RoutePut` with the node excluded, and `ALREADY_EXISTS` is a
peer-side dedup hit needing no transfer at all.

Different peers own different drives, so their allocate/RDMA/commit round trips
run **concurrently** — that is where a multi-node pure-SSD pool's aggregate
write bandwidth comes from.

### Write staging

Direct-SSD put needs a staging region on the *owner* for the writer to RDMA
into. It is allocated **on top of** the read region, never carved out of it, so
enabling it never shrinks a read slot (and never turns a working read into
`SSD_READ_SIZE_TOO_LARGE`).

| Setting | Default | Meaning |
|---|---|---|
| `ssd_write_staging_slots` | `-1` | `-1` = auto (on in pure-SSD mode, off otherwise); `0` = force off; `>0` = force on with that many slots |
| `ssd_write_staging_size` | 256 MiB | total write region; per-slot = this / slots, and it bounds the largest single direct SSD put |

A peer advertises its slot count and slot size via `GetPeerInfo`. Zero slots
means "I do not accept direct SSD writes", and a writer routed to that node's
SSD tier fails the key loudly rather than RDMAing into a region that isn't there.

Both staging regions can be hugepage-backed, which matters here because these
regions are RDMA-registered and get re-registered as the deployment grows:

| Setting | Default | Meaning |
|---|---|---|
| `staging_buffer_use_hugepages` | `true` | hugepages for the RDMA staging buffer |
| `ssd_staging_use_hugepages` | `false` | hugepages for the SSD staging region |
| `*_hugepage_size` | 2 MiB | hugepage size for the corresponding region |

Both fall back to ordinary anonymous pages automatically when hugepages are not
available, so turning them on is safe on a node without a hugetlb pool
configured — it just does nothing.

## The get path

Both the local and remote SSD reads are now batched.

**Self-target** — one `PeerSsdManager::PrepareReadBatch` straight into the user
buffers.

**Remote** — `BatchPrepareSsdRead` claims as many staging slots as are free,
serves them all with one batched device read, and returns every offset at once;
the reader issues **one** batched RDMA and one `BatchReleaseSsdLeases`.

This replaces a per-key `PrepareSsdRead` → device read → RDMA → release round
trip that ran strictly one key at a time. Additionally, the per-owner groups of
a `BatchGet` now run in **parallel** workers; previously they were a serial
`for` loop over nodes, which capped a multi-node SSD get at one node's (in fact
one drive's) bandwidth.

Lease semantics are unchanged and still conservative: the reader anchors its
deadline before sending, re-checks it after the transfers, and treats a late
arrival as a transient retry rather than a served key. A `NO_SLOT` is never
surfaced as a cache miss.

## Multi-drive tiers

`UMBP_SSD_DIR` accepts a comma-separated list — the same convention
`UMBP_SPDK_NVME_PCI` already used. More than one directory builds a
`ShardedSsdTier` over one `SSDTier` per drive.

- **Placement is balanced, not hashed.** Each write goes to the drive with the
  most free space, which degenerates to exact round-robin for uniform-size
  values on uniform drives (1000 keys over 2 drives → 500/500) and still fills
  correctly when drives or values differ in size. The chosen shard is recorded,
  so a re-put of a live key lands back on its own drive and is never duplicated.
- **Batch IO is parallel.** `BatchWrite` / `BatchReadIntoPtr` bucket keys by
  drive and hand each bucket to its own worker (`UMBP_SSD_SHARD_IO_THREADS`,
  default one per drive).
- **`capacity_bytes` is the total**, split evenly across the directories.
- **Each key stays whole on one drive**, so a single-key read costs one drive's
  IO rather than a strip-split across several.

### Relationship to SPDK RAID0

`SpdkEnv` already builds a RAID0 bdev when `UMBP_SPDK_NVME_PCI` names more than
one controller. The two mechanisms compose and solve different problems:

| | SPDK RAID0 | `ShardedSsdTier` |
|---|---|---|
| Layer | block | tier |
| Backend | SPDK only | works with the plain `file` backend on ordinary mounts |
| Drive sizes | uniform | capacity-aware, may differ |
| Single-key read | split across strips | one drive |
| Membership change | device-wide rebuild | config edit |

A shard may itself be a RAID0 bdev.

## Making the tier device-bound

By default the `file` backend opens its segments **buffered**, so reads are
served out of the page cache and the drive is barely touched. That makes
pure-SSD mode behave like a DRAM tier with a filesystem in front of it, and it
makes drive-count and DRAM-vs-SSD comparisons meaningless — a 1-NVMe vs 2-NVMe
A/B differed by 15% because neither configuration was drive-limited.

| Env var | Default | Meaning |
|---|---|---|
| `UMBP_SSD_DIRECT_IO` | `0` | `1` = `O_DIRECT`, bypassing the page cache. **Set this for any measurement you intend to believe.** |
| `UMBP_SSD_VERIFY_CRC` | `1` | `0` = skip checksum verification. `DRAMTier` does no checksumming at all, while `SSDTier` checksums every byte — 51–67% of GET time. Turn it off to compare tiers without that confound. |
| `UMBP_SSD_TIER_IO_THREADS` | `4` | Fans the tier's CPU phases (checksum verify on read, record assembly on write) across workers **within one drive's batch**. Distinct from `UMBP_SSD_SHARD_IO_THREADS`, which fans out **across drives**. Defaults to 4 to match `DRAMTier::read_threads_`. |
| `UMBP_SSD_DURABILITY` | `strict` | `strict` / `relaxed`. With direct I/O on, the strict-write penalty drops from 42% to 1.4% — there is no writeback left to defer — so `strict` is cheap and there is little reason to relax it. |

Measured on ext4/nvme1n1, 64 × 4 MiB, io_uring (MB/s):

| Config | Write | ReadBatch | Device r/w (MiB) |
|---|---|---|---|
| buffered / crc-on / 4t | 1551 | 7503 | **0 / 0** |
| direct / crc-on / 4t | 1954 | 5356 | 5633 / 6150 |
| direct / crc-off / 4t | 2004 | 6002 | — |
| direct / crc-on / 1t | 2063 | 3772 | — |

Buffered moved *zero* bytes to the device across the whole workload. Direct
writes are 26% faster than buffered (no page-cache memcpy, no writeback), and
thread fan-out is worth 42% on `ReadBatch`.

### On-disk format and capacity accounting

`O_DIRECT` requires buffer, file offset and length to all be alignment
multiples, so this needed an on-disk layout change. **Record format v3** pads to
a fixed `kRecordAlign` of 4096: `[header|key|pad][value|pad]`, with both the
record and the value starting on a boundary. Two consequences:

- The padding is **unconditional**, not gated on `direct_io`, so one directory
  reads back either way. `kRecordAlign` is a fixed constant rather than the
  device's reported DIO alignment, so the layout cannot change meaning when a
  directory moves between drives.
- **v2 records are dropped by the scanner** on version mismatch (as v1 was
  before them). A directory written by an older build comes up empty — wipe it,
  or set `UMBP_SSD_STARTUP_DISCARD` and let it rebuild.

Capacity is charged in **padded on-disk bytes** (`KeyMeta::disk_bytes`), not raw
value bytes. Without this a store of small values overruns `capacity_bytes` on
disk: a 512 B value occupies 8192 B, 16× its size. For the multi-MiB KV pages
this tier normally holds the overhead is under 0.1% — but size your
`UMBP_SSD_CAPACITY` with the real value size in mind if yours are small.

Records written with checksums off carry `kFlagNoCrc`, so they stay readable by
a process running with verification on.

## Observability

Three log lines answer "did the writes actually spread, and did the reads
actually overlap" without attaching a profiler.

> **These three are commented out in the source.** Each fires once per batch,
> which is far too chatty to ship on. The env vars and the cumulative accounting
> are still there, so re-enabling one is a matter of uncommenting its
> `MORI_UMBP_INFO` (for `get_dist`, the log-only straggler block just above it
> too) and rebuilding. `UMBP_PUT_DIST_LOG` / `UMBP_GET_DIST_LOG` then still work
> as documented: `0` to disable, `N` to print every Nth batch.

| Env var | Line | Call site | Emitted by |
|---|---|---|---|
| `UMBP_PUT_DIST_LOG` | `[RoutePutStrategy] batch_dist` | `routing/route_put_strategy.cpp` | master — where the router *decided* to put keys |
| `UMBP_PUT_DIST_LOG` | `[PoolClient] put_dist` | `distributed/pool_client.cpp` | writer — where one client's keys *actually went*, self-target tagged `<node>(local)` |
| `UMBP_GET_DIST_LOG` | `[PoolClient] get_dist` | `distributed/pool_client.cpp` | reader — after the batch completes |

Each line reports the per-`node/TIER` key and byte split for this batch plus the
process-cumulative split. The number to watch is **`cumulative_max_share`**:
`100/N%` over N targets is a perfect spread, 100% means everything landed on one
node. Cumulative totals are folded in on *every* batch even when the line is
sampled, so a sampled line still reflects all preceding batches.

Beyond the split:

- `put_dist` reports `remote_peers` — the put path runs one concurrent worker
  per remote SSD peer, so this is also the write concurrency.
- `get_dist` reports `remote_ssd_workers` (concurrently-read owning nodes, one
  thread each), the batch wall time, the implied aggregate `GB_s`, and a
  `straggler` breakdown: every remote worker starts together and the batch
  blocks on the last one, so `tail_ms` is what the batch spent waiting after
  half its targets were already done.

`UMBP_SSD_TIMING=1` adds per-phase device timing (the `[SsdPerf/tier]`,
`[SsdPerf/shard]`, `[SsdPerf/peer]` and `[SsdPerf/remote]` lines: prepare RPC,
device read/write, CRC, RDMA, release). Unlike the three above these are **live
in the source** — they simply print nothing unless the env var is set — so this
is the diagnostic to reach for first, and it is where to look when the split is
even but `GB_s` is low. `UMBP_LOCAL_COPY_TIMING` does the same for the
self-target `[LocalCopy]` path.

**Sanity check that direct I/O is really on**: run `iostat -x 1` on the target
node during a GET-heavy phase. Buffered mode shows ~1% device utilisation while
reporting absurd read bandwidth; direct mode shows the drives busy and a
`dev_GB_s` that matches the drive's rating.

## Configuring from Python directly

The TL;DR config goes through SGLang's `UMBPStore` `extra_config`, which is an
explicit allow-list; an unknown key is *ignored with a warning*, and the store
also warns when a key is standalone-only or distributed-only for the mode you
are in. Watch the log for those the first time you change the config.

When driving the client directly (no SGLang), the same settings are plain fields
on the config objects. The environment overlay (`UMBPConfig::FromEnvironment()`)
is applied first, so env vars are the defaults and explicit assignment wins:

```python
cfg.ssd.enabled          = True
cfg.ssd.storage_dir      = "/mnt/nvme0,/mnt/nvme1,/mnt/nvme2"
cfg.ssd.capacity_bytes   = 6144 * (1 << 30)
cfg.ssd.direct_io        = True
cfg.ssd.verify_crc       = False
cfg.ssd.tier_io_threads  = 4
cfg.ssd.shard_io_threads = 0
cfg.dram.capacity_bytes  = 0          # pure-SSD

dist.staging_buffer_size       = 2 << 30
dist.ssd_staging_buffer_slots  = 512  # read staging
dist.ssd_staging_buffer_size   = 4 << 30
dist.ssd_write_staging_slots   = 512  # -1 = auto (on in pure-SSD mode)
dist.ssd_write_staging_size    = 4 << 30
dist.ssd_staging_use_hugepages = True
```

`direct_io` / `verify_crc` / `tier_io_threads` were env-only until `b48f04cb`;
on an older build a Python caller setting them gets an `AttributeError`.

## Tuning for aggregate bandwidth

- **`UMBP_ROUTE_PUT_SELECT_ALGO=random`** — the first thing to set, and the one
  that most often explains a pool that scales with drive count on paper but not
  in practice. Verify with `cumulative_max_share`.
- `UMBP_SSD_DIRECT_IO=1` — the second thing to set. Without it you are
  benchmarking the page cache and every other number here is meaningless. Pair
  it with `ssd_io_backend: io_uring`; posix on the direct path is far slower.
- `UMBP_SSD_TIER_IO_THREADS` — 4 is a reasonable default; worth 42% on
  `ReadBatch` versus 1. Raise only if a single drive's batch is CPU-bound.
- `UMBP_SSD_SHARD_IO_THREADS` — leave at `0` (one per drive) unless CPU-bound.
- `ssd_staging_buffer_slots` — bounds how many keys one `BatchPrepareSsdRead`
  can claim; too few and a large batch spends rounds on `NO_SLOT`. Per-slot size
  (`ssd_staging_buffer_size / slots`) must stay ≥ the largest single key.
  512 slots over 4 GiB (8 MiB/slot) is the validated setting above.
- `ssd_write_staging_slots` — same trade-off on the put side; the validated
  config sets it explicitly to 512 rather than leaving it at `-1` (auto).
- `UMBP_SSD_READ_LEASE_MS` — raise it (30 s in the config above) when batches
  are large; a lease that expires mid-batch is treated as a transient retry and
  costs a round trip.
- `UMBP_SSD_GET_MAX_ATTEMPTS` defaults to `1` (no retry). Under staging-slot
  contention a `NO_SLOT` then surfaces as a miss rather than a retry; raise it
  if the read staging region is small relative to the batch.
- On the SPDK path, `UMBP_SPDK_REACTOR_MASK` defaults to `0x1`, i.e. a **single**
  reactor core polling for all drives. Give it at least one core per drive
  (`0x3`, `0xf`, …) before expecting multi-drive scaling there.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Absurd read bandwidth, `iostat` shows ~1% device utilisation | `UMBP_SSD_DIRECT_IO` not set — you are reading the page cache |
| Adding drives / nodes doesn't add bandwidth | `cumulative_max_share` near 100% → switch `UMBP_ROUTE_PUT_SELECT_ALGO` to `random`, and check `UMBP_ROUTE_PUT_NODE_AFFINITY` is `none` |
| `random` set but the split is still skewed | `UMBP_ROUTE_PUT_SELECT_ALGO` was set on the engine, not on `umbp_master`. The master logs its resolved strategy at startup (`most_available/none/ssd:auto`) — check that line |
| A config key seems to have no effect | `extra_config` is an allow-list: unknown keys are ignored with a warning, and keys outside the current mode's scope warn too. Grep the engine log for `UMBPStore:` |
| Puts fail on a pure-SSD node | That peer advertises zero write-staging slots — `ssd_write_staging_slots` was forced to `0`, or the node isn't actually in pure-SSD mode so `auto` didn't enable staging |
| Node comes up with an empty cache after upgrading | Record format v3: v2 records are dropped on version mismatch |
| Tier hits `capacity_bytes` far earlier than expected | Capacity is charged in padded on-disk bytes; small values pay up to 16× (512 B → 8192 B) |
| Startup rejects `dram.capacity_bytes == 0` | No SSD tier configured — a zero DRAM pool with no SSD leaves no tier that can accept a put |
| Even split, low `GB_s` | Not a routing problem — use `UMBP_SSD_TIMING=1` and the `get_dist` `straggler` breakdown |
