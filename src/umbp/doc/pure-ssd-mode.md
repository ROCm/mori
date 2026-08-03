# Pure-SSD mode and multi-drive SSD tiers

Two changes that compose:

1. **Pure-SSD mode** — a node can run with *no* host-DRAM tier, SSD carrying the
   whole cache. SSD becomes a routable put target instead of a place bytes only
   ever arrive at asynchronously.
2. **Multi-drive tiers** — one node's SSD tier can span several drives, with
   keys spread across them and IO issued to all of them concurrently.

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
cold tier. The existing `UMBP_ROUTE_PUT_SELECT_ALGO` and
`UMBP_ROUTE_PUT_NODE_AFFINITY` knobs apply to SSD placement exactly as they do
to DRAM, so `most_available` spreads a batch across a pool of pure-SSD nodes
with projected-capacity deduction inside the batch.

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

## Tuning for aggregate bandwidth

- `UMBP_SSD_SHARD_IO_THREADS` — leave at `0` (one per drive) unless CPU-bound.
- `ssd_staging_buffer_slots` — bounds how many keys one `BatchPrepareSsdRead`
  can claim; too few and a large batch spends rounds on `NO_SLOT`. Per-slot size
  (`ssd_staging_buffer_size / slots`) must stay ≥ the largest single key.
- `ssd_write_staging_slots` — same trade-off on the put side.
- On the SPDK path, `UMBP_SPDK_REACTOR_MASK` defaults to `0x1`, i.e. a **single**
  reactor core polling for all drives. Give it at least one core per drive
  (`0x3`, `0xf`, …) before expecting multi-drive scaling there.
