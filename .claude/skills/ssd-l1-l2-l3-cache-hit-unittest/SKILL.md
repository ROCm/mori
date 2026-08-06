---
name: ssd-l1-l2-l3-cache-hit-unittest
description: Deterministic unit test that isolates and measures L1 (GPU device radix cache), L2 (host DRAM chunk cache), and L3 (UMBP SSD/distributed storage) hit latency for sglang's HiCache, versus a genuine cold recompute — precise token-length control via raw input_ids, precise L1/L2 sizing via --max-total-tokens/--hicache-ratio, /metrics-based verification, and scalable 1/2/4/8-NVMe topologies via standalone UMBP storage workers across up to two nodes. Use when asked to benchmark or unit-test HiCache L1/L2/L3 hit cost, SSD read performance in isolation, or "cold vs cache-hit" TTFT for a fixed-length request.
allowed-tools: Bash(ssh *), Bash(docker *), Read, Edit, Write, AskUserQuestion
---

# SSD/L2/L1 cache-hit unit test (non-PD, pure TP8, 1-8 NVMe drives)

Isolates ONE specific cache-tier hit (L1, L2, or L3) for a fixed-length request, cleanly
separated from a genuine cold recompute — by controlling exactly which tier holds the data via
precise `--max-total-tokens` / `--hicache-ratio` sizing, dedicated standalone storage workers,
and the real flush/clear HTTP + signal-based mechanisms, not by inference from noisy end-to-end
benchmark numbers.

## Step 0 — ask the user before touching any node

This test needs a node reserved via slurm, and — once you scale past 2 NVMe drives — a *second*
node too. Before doing anything else, ask the user:

1. **Which tier(s) to test — L1, L2, L3, or some combination.** Each has a different launch-config
   requirement (see the Sizing section below) and a different recipe, so knowing this up front
   determines how you size `--max-total-tokens`/`--hicache-ratio` and whether you need standalone
   workers at all (L1/L2 never touch them; only L3 does). Don't default to testing all three
   unless asked — that's 3x the setup/runtime cost.
2. **What context length(s) to test — default to 32K (32768 tokens) if the user doesn't say.**
   Sizing (settle delay, and for L2 tests whether `max-total-tokens` is even correctly forcing
   eviction) depends on the size actually tested, so get this before building the launch config,
   not after. If testing L3, the settle-delay table in this doc is keyed by size — confirm the
   size before picking a settle value.
3. **Which node(s) to use for the engine** (compute) **and, if going beyond 2 drives, for
   storage.** Don't assume — a node holding someone else's work should never be silently reused.
4. **New slurm allocation, or reuse an existing one?** If reusing, confirm the job is still
   `R`(unning) via `squeue -u $USER` before building anything on top of it — allocations lapse
   silently on a contended cluster, and slurm-guard kills containers on nodes that fall out of
   your allocation without warning (verified painfully in practice: an engine can sit healthy for
   hours after `squeue` already shows nothing for it).
5. **If scaling past 2 NVMe drives (4nvme/8nvme topologies), tell the user explicitly you will
   use `n06-21` as the second/storage node** — it has 6 physical NVMe drives available (2×
   Micron 7450 3.5TB already conventionally used as `drive3`/`drive4`, plus 3 more KIOXIA 14TB
   drives usable as `drive5`-`drive7`, plus the OS root disk itself usable as an 8th "drive" via
   a plain directory if the user accepts sharing it — see the Topology reference below). Don't
   silently pick a different node; n06-21 is the established, validated choice with known
   RDMA/driver characteristics (see Node/cluster setup notes).
6. **Always use the latest available docker image, not one named in this doc.** Across this
   investigation the image changed constantly as RDMA/caching optimizations were layered in
   (`directio` → `singleflight` → `sf-rrput` → `pollers` → `getdist` → `getdist-fanout` →
   `rdmapush`, and more almost certainly exist by the time you read this) — each new tag
   consistently carried the best performance found so far, since they're incremental builds on
   top of each other. Tags encode a build date (`...-mori-local-YYYYMMDD-<commit>-<feature>`) —
   when the user gives you an image, or when you're picking one up from where a prior session
   left off, prefer the **newest date/latest-named tag you can find** (ask the user to confirm
   if more than one plausible candidate exists, e.g. check what's already pulled on the target
   nodes with `docker images` and go with the most recently built one). Don't default back to an
   older tag named in this doc's findings tables — those are dated snapshots for reference, not a
   recommendation. Once you have a tag, verify it's actually present on both target nodes with
   `docker image inspect <image> --format '{{.Id}}'` before building anything on top of it (see
   the Docker Hub rate-limiting bug below if a pull unexpectedly fails).

## Topology reference: 1 / 2 / 4 / 8 NVMe

| Name | Drives | Nodes involved | Notes |
|---|---|---|---|
| **1nvme** | 1 (e.g. `drive2`, on the engine node) | 1 (engine node only) | Simplest baseline. |
| **2nvme** | 2 (`drive1`+`drive2`, both local to the engine node) | 1 (engine node only) | One of the two "drives" is often actually the OS root disk (`/tmp/...` on a node with no separate data partition) — check with `df` before assuming true physical separation; this is fine and was explicitly accepted as OK in practice, just be aware. |
| **4nvme** | 2 local (engine node) + 2 remote (n06-21) | 2 | The first cross-node config. Needs one standalone worker on n06-21 covering `drive3`+`drive4`. |
| **8nvme** | 2 local (engine node) + 6 remote (n06-21) | 2 | n06-21's remote contribution is 3 dedicated standalone workers, each covering 2 drives (`drive3`+`4`, `drive5`+`6`, `drive7`+`8`). `drive8` is a plain directory on n06-21's root disk (`/data/umbp_ssd_ditian12_drive8`) — using it means sharing I/O with the OS and any other users' containers on that shared node; get explicit confirmation before wiping/using it (it held someone else's `gds_test` scratch data once — always check `df`/`ls` first). |

Going from 4nvme→8nvme only changes n06-21's side (2 drives → 6 drives, 1 worker → 3 workers) —
the engine node's own 2 local drives never change.

## Standalone UMBP worker architecture — the preferred approach for 4nvme/8nvme

**Do not rely on the engine's own embedded per-TP-rank UMBPStore clients to serve as the real L3
storage backend once you're testing more than 1 node's worth of drives.** In TP8 mode the engine
already spins up 8 per-TP-rank UMBPStore clients that each register their own local SSD capacity
with the master — this is fine for 1nvme/2nvme (single node), but for cross-node topologies it
creates a "self-dispatch" problem: the engine's own local drives remain valid capacity-weighted
`random`-routing targets alongside the remote nodes', so a meaningful fraction of "cross-node"
traffic silently gets served from local disk instead, contaminating the measurement of what
you're actually trying to test.

**Fix: run dedicated standalone worker processes as the real storage backend, and shrink the
engine's own embedded SSD capacity to something unusably tiny (1 MB) so it can never be
capacity-selected as a put target itself** (`random` routing skips any node that "cannot fit a
block"). The engine still needs `ssd_enabled: true` / `dram_capacity_bytes: 0` in its own
`hicache-storage-backend-extra-config` for HiCache's L3 path to initialize at all — just set
`"ssd_capacity_bytes": 1048576` instead of a real value. This is a big lever: it was the single
biggest performance win found across many rounds of testing in this investigation, bigger than
any image-level RDMA/singleflight optimization tried on top of it.

### The worker script (`umbp_ssd_worker.py`, in this skill directory)

A passive pure-SSD UMBP client: `dram_capacity_bytes=0`, its own real SSD tier (2 drives per
worker, matching the engine's own local-drive-pair convention), joins the same master, then idles
forever except for a `SIGUSR1` handler that calls `client.clear()` on demand (see the clear/flush
section below — this is *required* for multi-round testing, not optional).

```
python3 /tmp/umbp_ssd_worker.py <master_ip:51051> <drive_dir1,drive_dir2> <ssd_capacity_bytes> [staging_slots] [staging_size_bytes]
```

`staging_slots`/`staging_size_bytes` are optional (default 512 slots / 4GiB) — **you almost
certainly need to raise them**, see the staging-buffer bug below.

### Worker count and capacity — validated configs

| Topology | Workers on remote node | Capacity per worker | Staging per worker |
|---|---|---|---|
| 4nvme | 1 worker, 2 drives (`drive3`+`4`) | 64GB (mirrors one TP-rank's share) OR 512GB (whole-node) — both validated, see caveats below | 512 slots/4GiB (default) is fine for 1 worker |
| 8nvme | 3 workers, 2 drives each | 512GB each | **4096 slots / 32GiB each** — see staging-buffer bug below, mandatory at this worker count |

Two validated sub-variants for "how many worker processes per node," both correct once sized
right, with different tradeoffs:
- **8 workers × 64GB, one per TP-rank** (mirrors the engine's own 8 identities 1:1) — best
  balance/routing-fairness against the engine's 8 tiny (1MB) identities; needed no staging-buffer
  tuning at the default settings. Slightly better at 16K in one comparison, slightly worse at
  32K/128K than the 1-worker alternative below.
- **1 worker × full node capacity (512GB)** — fewer processes, but its staging buffer must be
  scaled up ~8x (to `4096` slots / `32GiB`) to handle the same aggregate concurrent load that 8
  separate processes used to split between them, or it fails almost every round with
  `NO_SLOT/lease-expired` transient misses (confirmed via the exact log line — see bugs section).
  Once fixed, this config won 32K/128K, lost 16K, by single-digit percentages either way — not a
  clear overall winner, but simpler to operate (fewer processes to track/clear).

Either is fine; **always scale staging capacity to the actual number of TP ranks that can
concurrently hit one worker process**, not just "big enough in isolation."

## Precise request-length control

Do **not** use aiperf/trace-replay (real trace data has variable token counts). Instead:
load the model tokenizer directly (`AutoTokenizer.from_pretrained(MODEL_PATH,
trust_remote_code=True)`), tokenize repeated filler text, slice to the exact desired length, and
POST to `/generate` with the raw `input_ids` field (never `text`) — this bypasses any
client/server tokenizer mismatch entirely. `sampling_params={"max_new_tokens": 1, "temperature":
0}` keeps decode/OSL from adding noise, and means the measured `wall_time`/`e2e_latency` **is**
TTFT — there is no separate decode phase to speak of with a 1-token generation. See `build_ids()`
in the scripts here.

Use two *disjoint* filler strings for "A" (small, fixed, the thing under test) and "B" (large,
the eviction driver) so their token sequences never accidentally share a prefix.

**Prefix-sharing contamination gotcha:** `build_ids()` slices a *repeated* filler string to the
exact desired length — which means a 4096-token "A" and a 16384-token "A" (or 32K/128K) share an
**identical prefix** (the first 4096 tokens are the same sequence either way). If you test
multiple sizes back-to-back on the same long-lived engine, a *smaller* size's "cold" round can
silently prefetch-hit against a *larger* size's leftover L3 data from an earlier test, since the
radix tree does prefix matching, not whole-request matching. This bit us directly: a genuinely
fresh-looking "cold" round showed `cached_tokens>0` on round 1 because an earlier, larger-size L3
test had already written that shared prefix. Always clear all tiers (including remote workers,
see below) before switching test sizes on the same engine, not just between rounds of the same
size.

## Sizing L1 (device) and L2 (host) precisely

- `--max-total-tokens N` sets the device KV pool size in tokens (this IS the L1 tier size).
- `--hicache-ratio R` sets L2 capacity = `R × max-total-tokens` tokens.
- **Hard constraint, easy to miss:** a single request's context must fit **entirely** within
  `max-total-tokens` — chunked prefill (`--chunked-prefill-size`) only paces the *compute*, it does
  not let a request's resident KV exceed the pool. Concretely: `max_total_tokens < N` fails with
  `Input length (N tokens) exceeds the maximum allowed length (max_total_tokens-6 tokens)`.

### For an L3-hit test (A evicted from BOTH L1 and L2)
`max-total-tokens` just needs to comfortably hold your largest test size — sizing doesn't need to
force eviction here at all, since the L3-hit recipe clears L1+L2 directly via `/flush_cache`
rather than relying on a "B" eviction request (see the recipe below). `148480` (validated up to
131072-token requests) with `--hicache-ratio 2.2` works fine.

### For an L2-hit test (A evicted from L1 ONLY, survives in L2) — **the sizing bug that silently breaks this test**
This recipe genuinely needs `max-total-tokens` sized **small enough that A_len + B_len exceeds
it**, so sending B actually evicts A from L1. **If you reuse an engine sized for the L3-hit
recipe above (e.g. `148480`), B (65536 tokens) will NOT evict A — A just sits in L1 the whole
time, and what you measure is actually a mislabeled L1 hit, not an L2 hit, even though
`cached_tokens` looks identical either way.** This happened in practice and produced numbers that
looked plausible (similar magnitude to genuine L2 hit) right up until checking
`sglang:cache_hit_tokens_l2_total` via `/metrics` and finding it stayed at `0` the whole test.
**Always verify `l2_total` actually incremented — never trust `cached_tokens` alone for this
recipe.** Validated correct sizing: `A_len ∈ {4096, 16384}`, `B_len=65536`, `--max-total-tokens
66048` (= B_len + 512 margin, so A_len+B_len > 66048 and eviction is forced), `--hicache-ratio
2.2` (L2 = 145305 tokens, comfortably holds A+B without spilling to L3).

### For an L1-hit test (A never leaves the device tier)
No sizing tricks needed — any launch config works (ratio doesn't matter since nothing is
evicted). Just don't send any evicting request between the cold send and the resend: flush all 3
tiers, send A once (cold), then immediately resend A with **no** intervening request. It stays
resident in the device radix tree, so the resend is a pure in-GPU-memory lookup.

Launch scripts here: `launch_ssd_l3_unittest_1nvme.sh` / `_2nvme.sh` (single-node only,
`--hicache-ratio 0.5`, sized to force L3 eviction the old B-driven way) and
`launch_ssd_l2_unittest_1nvme.sh` (`--hicache-ratio 2.0`, correctly sized for genuine L2 testing,
also reusable for L1-hit tests). For 4nvme/8nvme cross-node topologies, start from
`launch_ssd_l3_tinyssd_crossnode_TEMPLATE.sh` in this skill directory — it's the **exact,
complete env-var set** (~40 vars) validated across the whole cross-node investigation, not a
partial summary; just fill in `<DRIVE1_DIR>,<DRIVE2_DIR>` for the engine node's own local drives
and export `UMBP_MASTER=<master_ip>:51051` before invoking it. Don't drop vars from it without
re-validating — several (the aiter/NCCL/CPU-affinity block) look unrelated to SSD/L3 at a glance
but were present in every launch that produced a clean result.

**Master launch** (positional args, not flags — `<listen_addr:port> <metrics_port>`):
```bash
docker exec -d <container> bash -c \
  'cd /sgl-workspace/mori && UMBP_ROUTE_PUT_SELECT_ALGO=random \
   LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mori \
   ./build/src/umbp/umbp_master 0.0.0.0:51051 51052 > <master_log> 2>&1'
```

**Standalone worker launch** (one per drive-pair, see worker-count table above):
```bash
docker exec -d <container> bash -c '\
export UMBP_SSD_DIRECT_IO=1; export UMBP_SSD_VERIFY_CRC=0; export UMBP_SSD_TIER_IO_THREADS=4; \
export UMBP_SSD_DURABILITY=strict; export UMBP_SSD_READ_LEASE_MS=30000; export UMBP_SSD_TIMING=1; \
export UMBP_DRAM_USE_HUGEPAGES=1; export UMBP_DISTRIBUTED_SSD_STAGING_USE_HUGEPAGES=1; \
export MORI_RDMA_SL=3; export MORI_IO_SL=3; export MORI_UMBP_LOG_LEVEL=info; export HICACHE_UMBP=on; \
python3 /tmp/umbp_ssd_worker.py <master_ip:51051> <drive1_dir,drive2_dir> <capacity_bytes> [staging_slots] [staging_size_bytes] > <worker_log> 2>&1'
```

**Known bug, already fixed in the bundled `.sh` files:** earlier revisions had an accidental
`--port 30001` baked in. If a stale copy elsewhere fails health checks on port 30000 despite the
log saying "fired up and ready to roll," check for this.

## The write-through async-ack race (settle delay is NOT optional, and must scale with size AND worker count)

`hicache_write_policy=write_through` pushes a completed request's KV to L2/L3 immediately — but
the *acknowledgment* that marks the tree node `backuped=True` drains asynchronously, and the
underlying write to the storage backend takes real time. If a second request/flush arrives
**before** that write genuinely completes, the node gets dropped via `_evict_regular()` (full
loss, no L3 trace) instead of `_evict_backuped()` (keeps a shadow pointer that enables future
prefetch) — the resend then silently full-recomputes with `cached_tokens=0`, not the L3 hit you
were expecting. Root-caused by reading `hiradix_cache.py`'s
`_evict_backuped`/`_evict_regular`/`match_prefix`.

**The settle delay must scale with both request size and how much write parallelism is
available** (fewer/larger workers need more settle time than many small ones for the same data
volume, since fewer workers means slower aggregate write completion):

| A length | Settle needed (validated) |
|---|---|
| 4096 | 3.0s (occasional single-round misses even so — budget for retries) |
| 16384 | 3.0s–5.0s |
| 32768 | 8.0s–10.0s minimum; a 3.0s settle at this size failed **8/10 rounds** in one run |
| 131072 | 40.0s |

Don't assume a fixed settle works across sizes — always re-validate when scaling up, and check
`cached_tokens` on the "L3 hit" step across all rounds for silent misses even after picking a
settle value that mostly works.

## The real flush/clear endpoints — and why they are NOT enough for multi-node tests

- `POST /flush_cache` — clears L1 **and** L2, on the **engine only**.
- `POST /clear_hicache_storage_backend` (aka `/hicache/storage-backend/clear`) — clears L3, but
  **also only the engine's own local view/data**. Directly `rm`-ing the SSD backing file does
  NOT work either (POSIX unlink-while-open: the process keeps serving the "deleted" data through
  its existing fd) — always use the real endpoint, never delete files directly.
- **Critical, easy to miss: neither endpoint reaches remote standalone worker processes.** Traced
  this down to the actual implementation: `clear()` on the UMBP client calls
  `PoolClient::Clear()`, which does `peer_alloc_->ClearLocal()` + `peer_ssd_->ClearLocal()` (both
  scoped to the calling process only) plus `master_client_->ClearFullSync()` (a *sync/consistency*
  signal to the master about the caller's own state, not a "everyone clear yourselves" broadcast
  — there is no such broadcast in this protocol). **If you're running standalone workers (4nvme/
  8nvme), you must clear each of them independently — e.g. via the `SIGUSR1` handler built into
  `umbp_ssd_worker.py` here — or leftover data from earlier rounds/sizes will silently persist and
  contaminate later "cold" measurements** (confirmed directly: cold rounds started returning
  nonzero `cached_tokens` once enough prior rounds had pushed data to a worker whose own state was
  never cleared).
- Both endpoints can 400 with `"Cache not flushed because there are pending requests"` even when
  `#queue-req: 0, #running-req: 0` — same async-ack-drain race as above, just on the flush side.
  Retry with ~1s backoff (see `_post_with_retry()` in the scripts); reliably succeeds by the 2nd
  attempt.

### The multi-node clear pattern (required for 4nvme/8nvme)

Per round, on top of the engine's own `/flush_cache` + `/clear_hicache_storage_backend` (already
inside the driver scripts), also send `SIGUSR1` to **every** standalone worker process before
each round:

```bash
for pid in $ALL_WORKER_PIDS; do
  ssh <worker-host> "docker exec <worker-container> kill -USR1 $pid"
done
sleep 1
# then run one round of the test driver with --rounds 1
```

Since the test drivers here (`ssd_l1/l2/l3_unittest.py`) don't know about remote workers
themselves, drive them with `--rounds 1` in an external bash loop that does the `SIGUSR1` fan-out
before each invocation, rather than trusting their own internal `--rounds N` loop for multi-round
multi-node tests.

## Other bugs found and fixed along the way — check these before assuming a test result is real

- **Master registry never expires dead clients.** Repeatedly killing/relaunching worker processes
  without restarting the master leaves stale, still-capacity-weighted entries behind — `random`
  routing can still select them, causing genuine (not transient) misses once the live:stale ratio
  gets bad enough. Symptom: almost every round shows `cached_tokens=0` on the resend despite
  correct settle/clearing. **Fix: whenever worker topology changes (count, capacity, drives), do
  a full clean restart — master, engine, and all workers together — not just the piece that
  changed.** Verify via the master's own metrics port (`curl :51052/metrics | grep
  capacity_total_bytes`) that the registered node count matches exactly what you expect before
  trusting any test result.
- **Staging buffer must scale with aggregate concurrent load, not just "big enough."** See the
  worker-count table above — this caused near-100%-miss (`NO_SLOT/lease-expired` in the engine
  log) when consolidating from 8 small workers to 1 big one without also scaling
  `ssd_staging_buffer_slots`/`ssd_write_staging_slots` proportionally.
- **Don't mount an empty jit-cache directory over `/sgl-workspace/aiter/aiter/jit`.** A fresh
  empty host directory bind-mounted there shadows the image's built-in aiter JIT module entirely,
  causing `ImportError: cannot import name 'core' from 'aiter.jit'` at engine startup. Reuse an
  already-populated jit cache directory from a prior successful launch of the *same image* (JIT
  artifacts are portable across nodes with the same GPU arch via shared NFS) instead of creating
  a fresh empty one per node.
- **Docker Hub rate-limiting (`toomanyrequests: too many failed login attempts`) is common and
  usually transient**, not a sign the image is genuinely unpullable — even `-local-`-tagged
  images often do have a real registry digest. Retry the plain `docker pull` a few minutes later
  before falling back to a node-to-node transfer (`docker save <image> | ssh <node> docker load`,
  from any node that already has it) — check `docker image inspect <image> --format
  '{{.RepoDigests}}'` on a node that has it to confirm it's not actually local-only first.

## Verification — don't trust `cached_tokens` alone

There are four levels of authority, weakest to strongest:

1. **`meta_info.cached_tokens`** (client response) — the weakest signal, unreliable for L3 hits
   specifically and provably fooled by the L2-sizing bug above (looks identical for a genuine L2
   hit and a mislabeled L1-still-resident "hit"). Fine as a quick sanity check, never as sole proof.
2. **Raw scheduler `Prefill batch` log lines** (`#new-token`, `#cached-token`) — exact ground
   truth at the per-chunk level.
3. **`grep -aoE 'SsdPerf/[a-z]+\] (GET|PUT)' <sglang_log> | sort | uniq -c`** — confirms whether
   L3 was physically touched. An L3 hit shows many `SsdPerf/remote` `GET` lines (real RDMA hops,
   even single-node) and zero new `PUT`s. An L2 hit shows zero new GET lines. An L1 hit shows
   neither.
4. **`GET /metrics` — the most authoritative, use this to settle any doubt:**
   - `sglang:cache_hit_tokens_l1_total` / `_l2_total` / `_l3_total` — exact cumulative per-tier
     hit-token counters, **cumulative for the engine process's whole life** — diff against a
     pre-test baseline, don't read the raw number. An L1-hit round increments only `_l1_total`; an
     L2-hit round increments only `_l2_total`; an L3-hit round increments only `_l3_total`.
   - `sglang:load_back_tokens_total` — raw, summed across all 8 TP ranks (~8× the per-rank-0
     hit-token counters above).
   - `sglang:load_back_duration_seconds_{sum,count}` — isolates the actual CUDA-event-timed
     host→device transfer cost specifically, separate from everything else (SSD promotion,
     scheduling overhead). Useful for genuinely surprising results: a matched-size, same-engine,
     back-to-back comparison found L3-sourced `load_back` events ~3x *faster per token* than
     L2-sourced ones at the same size, even though L3 must do this same step *plus* an earlier
     SSD→host promotion step — meaning the per-token host→device copy path itself differs
     between L2-origin and L3-origin data (likely: L3 lands in a dedicated, hugepage-backed,
     RDMA-tuned staging buffer, while genuinely-resident L2 data sits in a more general-purpose
     pool). Don't assume "L3 = L2 + one extra step, so L3 latency ≥ L2 latency" holds for
     overall wall-clock without checking — it can go either way depending on size (confirmed: L3
     beat L2 end-to-end at 4K, lost to L2 at 16K, on the *same* engine, matched size, back-to-back).
   - `sglang:load_back_bandwidth_gb_s_sum` — sanity-checks the transfer is a genuine PCIe
     host-to-device copy, not some cached/no-op path.

## Single-node RDMA loopback quirk (inflates round-1, sometimes several rounds)

Even on one node, each TP rank runs its own UMBPStore client/peer-service, and
`UMBP_ROUTE_PUT_SELECT_ALGO=random` spreads pages across all of them — so a rank often fetches
data another rank wrote, causing genuine RDMA hops through the NIC back to itself. First-time
GETs after a fresh master/engine start can take 300-1000ms+ each; once warm, drops to a few ms.
**Warmup can take more than one round** — on one node the first 5 rounds of a fresh L3-hit test
were all noisy outliers before settling; on another node it settled by round 2. Don't assume
"round 1 is the only outlier" — run at least 10 rounds and eyeball where numbers actually flatten
before trusting an average, especially on a genuinely fresh node/topology.

## Recipe: N-round L3-hit test

Per round — **no engine/master restart needed within a fixed topology**, single long-lived
engine + workers:
```
1. Flush all 3 tiers: POST /flush_cache, then POST /clear_hicache_storage_backend on the engine.
   For 4nvme/8nvme: also SIGUSR1 every standalone worker process.
2. Send A -> expect cached_tokens=0 (genuine cold recompute) -- this is your baseline
3. sleep(settle)  # scale with size + worker count, see table above
4. POST /flush_cache (L1+L2 only, leave L3 populated)
5. Resend A -> expect an L3 hit (verify via /metrics, not cached_tokens alone)
-> loop to step 1
```
Driver: `python3 ssd_l3_unittest.py --endpoint http://<ip>:<port>/generate --a-len <N>
--settle-secs <S> --rounds 1` (run with `--rounds 1` in an external bash loop for multi-node
tests, so you can interleave the remote-worker `SIGUSR1` clear between rounds — see the pattern
above). Run it **inside the engine's own container** (`docker cp` the script in) — it needs
`transformers` and hits `localhost:<port>` directly.

**Validated findings, single-node (1-NVMe, 4096-token A):** cold ≈0.18-0.19s, L3 hit ≈0.44-0.45s
once warmed up. Recompute is ~2.4x faster than an L3 round-trip at this size and topology — the
RDMA-loopback fetch cost dominates over redoing the (kernel-warm) attention compute.

**Validated findings, cross-node (4nvme→8nvme progression, standalone-worker architecture,
tiny-engine-SSD-capacity, best config found — rdmapush-family RDMA image):**

| A length | cold | L3 hit (best found) | speedup vs cold |
|---|---|---|---|
| 4096   | ≈0.18-0.19s | ≈0.137-0.147s | ~1.3x faster |
| 16384  | ≈0.68-0.73s | ≈0.184-0.219s | ~3.7x faster |
| 32768  | ≈1.40-1.46s | ≈0.26-0.28s (steady rounds) | ~5.2x faster |
| 131072 | ≈7.42-7.58s | ≈0.818-0.862s | **~8.85x faster** |

This is a dramatic reversal from the earliest cross-node baseline (before the standalone-worker
architecture existed), where L3 was actually **slower than cold** at 4K (~1.7x) and only ~2.4x
faster at 128K. The single biggest lever, in order of impact: (1) standalone workers + tiny
engine SSD capacity (self-dispatch fix), (2) more physical drives/workers on the remote node
(4nvme→8nvme), (3) RDMA-path image optimizations (singleflight/pollers/rdmapush family) — each
compounded on top of the previous. **Numbers above are the best found on this specific
cluster/hardware at investigation time — always re-validate rather than assuming they transfer
to a different node/image/drive-count.**

## Recipe: N-round L2-hit test

Requires a launch config sized per the "L2-hit sizing" section above (`max-total-tokens=66048`,
`hicache-ratio=2.2` — **do not reuse an L3-sized 148480-token engine**, see the sizing bug):
```
1. Flush all 3 tiers (+ SIGUSR1 remote workers if any are on this engine's topology)
2. Send A -> cold recompute baseline
3. sleep(3.0)
4. Send B (65536 tokens) -- forces A out of L1 device pool; L2 absorbs both
5. Resend A -> expect an L2 hit: verify via /metrics l2_total, not cached_tokens alone
-> loop to step 1
```
Driver: `python3 ssd_l2_unittest.py --endpoint http://<ip>:<port>/generate --a-len <N> --b-len
65536 --settle-secs 3.0 --rounds 1` (external loop for multi-node topologies, same as L3).

**Validated crossover finding (single-node) — L2-hit cost is essentially FLAT regardless of
request size, while cold recompute scales ~linearly:**

| A length | cold recompute | L2 hit | ratio |
|---|---|---|---|
| 4096   | ≈0.18-0.21s | ≈0.198-0.207s | ~1.0x — wash, not a win |
| 16384  | ≈0.68-0.83s | ≈0.199-0.210s | ~3.4-3.7x faster |
| 32768  | ≈1.38-1.55s | ≈0.20-0.22s | ~7x faster |
| 131072 | ≈7.58-8.11s | ≈0.44-0.45s | ~17x faster |

At 4096 tokens, caching in L2 buys essentially nothing over just recomputing. The benefit only
shows up once recompute cost grows past L2's flat overhead floor.

## Recipe: N-round L1-hit test

Sizing doesn't matter — any launch config works (nothing gets evicted):
```
1. Flush all 3 tiers (+ remote workers if applicable, though L1 never touches them anyway)
2. Send A -> genuine cold recompute
3. Immediately resend A -- NO settle, NO evicting request -- pure device radix-tree hit
-> loop to step 1
```
Driver: `python3 ssd_l1_unittest.py --endpoint http://<ip>:<port>/generate --a-len <N> --rounds
10`. No `--settle-secs`/`--b-len` args.

**Validated finding — L1 is the fastest tier at every size, growing only slightly with request
length (pure radix-tree pointer walk + in-GPU copy, no host/network transfer):**

| A length | cold recompute | L1 hit |
|---|---|---|
| 4096   | ≈0.19s | ≈0.098-0.102s |
| 16384  | ≈0.72s | ≈0.116-0.124s |
| 32768  | ≈1.45s | ≈0.14s |
| 131072 | ≈7.6-8.8s | ≈0.29s |

## Full comparison table (reference — re-validate before quoting as current)

| A length | cold | L1 hit | L2 hit | L3 hit (best cross-node config found) |
|---|---|---|---|---|
| 4096   | ≈0.18-0.19s | ≈0.10s | ≈0.202s | ≈0.137-0.147s |
| 16384  | ≈0.68-0.78s | ≈0.12s | ≈0.204s | ≈0.184-0.219s |
| 32768  | ≈1.38-1.46s | ≈0.14s | ≈0.20-0.22s | ≈0.26-0.28s |
| 131072 | ≈7.4-8.1s | ≈0.29s | ≈0.44-0.45s | ≈0.818-0.862s |

Ordering: **L1 is unconditionally fastest at every size.** L2 vs L3 is size-dependent and has
flipped direction multiple times across configs in this investigation (L3 lost badly to L2 in the
earliest baseline, caught up to roughly tied, then beat L2 outright at small sizes with the best
cross-node config while L2 kept the edge at larger sizes) — **don't assume a fixed L2-vs-L3
ordering without checking the current config at the size you actually care about.**

## Node/cluster setup notes (generic — adjust per session)

- Needs its own node(s) via slurm `salloc` — see Step 0. Re-verify allocation periodically on a
  contended cluster; silent lapses happen and can leave a healthy-looking engine running on a
  node slurm-guard is about to reap.
- Reserve hugepages on ANY newly-salloc'd node before launching: `sudo sysctl -w
  vm.nr_hugepages=350000` — skipping this crashes the engine with `RegisterRdmaMemoryRegionAuto
  failed ... errno:12 (Cannot allocate memory)` partway through startup.
- Container recipe: privileged, `--network host`, NVMe device passthrough (`--device
  /dev/nvmeXn1` per drive used), ionic/libibverbs bind-mounts (host's own `libionic.so*` — driver
  versions differ *per node*, always bind-mount the *host's own* copy, never reuse another node's
  container's paths blindly), jit-cache mounts (reuse a pre-populated one, see the empty-mount
  bug above). Watch for the `ENTRYPOINT ["bash"]` gotcha (`sleep infinity` as CMD needs
  `--entrypoint bash -c "sleep infinity"`, not a bare CMD).
- `umbp_master` binary path inside the image: `/sgl-workspace/mori/build/src/umbp/umbp_master`,
  needs `LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mori` and takes **positional**
  args (`<listen_addr:port> <metrics_port>`), not flags. Env var `UMBP_ROUTE_PUT_SELECT_ALGO=random`
  must be set **on the master process**, not the engine — it's a no-op if set on the wrong side.
- n06-21 specifics: 6 NVMe drives (`nvme0n1`/`nvme1n1`/`nvme4n1` = KIOXIA 14TB, unformatted by
  default — format with `mkfs.ext4` before first use, and note this is a destructive action a
  permission classifier may block; ask the user to run it themselves via the `!` prefix if so;
  `nvme2n1`/`nvme3n1` = Micron 7450 3.5TB, conventionally `drive3`/`drive4`; `nvme5n1` is the OS
  root disk, don't touch its partitions). ionic driver version differs from other nodes in this
  cluster — always bind-mount n06-21's *own* `libionic.so*` files, don't reuse another node's.
  n06-21 does not need its own slurm allocation for the worker-container role (it's typically a
  shared, always-on node) — but it IS shared with other users' containers, so avoid disturbing
  anything not yours (check `docker ps`, `df`, `ls` on any drive before formatting/reusing it).
- Port collisions: if colocating with someone else's container, sglang's default port 30000 may
  already be taken — pick a free one (`ss -ltnp | grep :30000`) and use it consistently.
