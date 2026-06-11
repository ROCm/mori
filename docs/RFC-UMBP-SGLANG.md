# [RFC] MORI-UMBP: A Scheduler-Codesigned, Policy-Pluggable Multi-Tier KV Cache Backend for SGLang

- **Status:** Draft
- **Author(s):** Di Tian (MORI / AMD)
- **Tracking:** mori `src/umbp/`, sglang `python/sglang/srt/mem_cache/storage/umbp/`

---

## 1. Summary

This RFC proposes integrating **MORI-UMBP** (Unified Memory & Bandwidth Pool —
the tiered-storage, distributed KV component of AMD's MORI library) into SGLang
as a HiCache L3 storage backend **plus** a KV-placement control plane that the
request scheduler/router can consult.

Existing L3 backends (file, Mooncake, NIXL, HF3FS, …) are passive byte stores:
SGLang pushes pages down and pulls pages up, and everything the system knows
about cache placement, temperature, and access economics is invisible to the
layer that needs it most — the scheduler. UMBP differs in three ways:

1. **Scheduler-friendly, co-designed with mori-sched.** The UMBP master is a
   cluster-wide KV placement directory covering *all* tiers (engine HBM, host
   DRAM, UMBP DRAM pool, SSD). Beyond placement, it exposes per-key historical
   access counts, routing hit counters, last-access time, per-node
   capacity/utilization, access bandwidth histograms, and RPC latency — so a
   router can build a real *cost model*: not just "who has the prefix" but
   "who can serve it fastest".

2. **Customizable KV cache management policy.** Offload (put-routing), load
   (get-routing), eviction, and (planned) replication are pluggable strategy
   interfaces. Policy authors get access to the same rich metrics, so cache
   management becomes a developer surface instead of a hard-coded LRU.

3. **(WIP) Agent-aware hints.** TTL, session pinning, and priority hints flow
   from the serving API through HiCache into UMBP, so agentic workloads with
   known revisit patterns can shape cache retention explicitly.

The data plane already runs daily in our PD-disaggregation benchmarks
(`--hicache-storage-backend mori`). This RFC documents the design and proposes
the remaining integration work in phases.

![MORI UMBP concept](mori_umbp_concept.jpg)

---

## 2. Design

### Architecture

```
  request (+ agent hints: ttl / session_id / pin / priority)        ◄── Pillar 3 (WIP)
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│            Router (mori-sched / sgl-model-gateway)               │
│                                                                  │ ◄── Pillar 1
│  score(worker) = α·Σ_tier(matched_tokens[tier]·tier_speedup)     │     scheduler
│                − β·queue_tokens − γ·tier_pressure − δ·fetch_eta  │     co-design
└──────────────┬───────────────────────────────────────────────────┘
               │ MatchExternalKv (per-node, per-tier prefix match;
               │   count_as_hit trains the hit index)
               │ GetExternalKvHitCounts / capacity & bw snapshots
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       UMBP Master (gRPC)                         │
│                                                                  │
│  Placement & history (scheduler cost-model inputs)    ◄── Pillar 1
│   ├─ ExternalKvBlockIndex   engine L1 HBM / L2 DRAM placement    │
│   ├─ GlobalBlockIndex       UMBP DRAM/HBM/SSD placement          │
│   │                         + BlockMetrics{access_cnt, recency}  │
│   ├─ ExternalKvHitIndex     historical per-hash hit counters     │
│   └─ Prometheus             per-node capacity/utilization,       │
│                             route counters, bandwidth, latency   │
│                                                                  │
│  Pluggable policies (PolicyContext = metrics above)   ◄── Pillar 2
│   ├─ RouteGetStrategy       load routing  (default: HBM>DRAM>SSD)│
│   ├─ RoutePutStrategy       offload routing (default: most-avail)│
│   ├─ EvictionManager        eviction (default: lease-aware LRU;  │
│   │                         radix depth via BatchPutWithDepth)   │
│   └─ ReplicationPolicy      (planned: hit-count-driven copies)   │
│                                                                  │
│  Hint-aware retention                                 ◄── Pillar 3 (WIP)
│   └─ TTL / session pin / priority consumed by eviction &         │
│      replication; session close → bulk revoke                    │
└───────────▲──────────────▲─────────────────────▲─────────────────┘
  heartbeat │   KV events  │ report/revoke       │ heartbeat
┌───────────┴───┐      ┌───┴───────────┐     ┌───┴───────────┐
│ SGLang engine │      │ SGLang engine │     │ SGLang engine │
│ L1 HBM radix  │      │      ...      │     │      ...      │
│ L2 host pool ←┼── UMBPHostTensorAllocator (hugepage/NUMA)  │
│ L3 UMBPStore ─┼── IUMBPClient ── peer DRAM pool + SSD tier │
└───────────────┘        RDMA data plane (zero-copy)         │
```

The L3 data path (`UMBPStore` ↔ `IUMBPClient`, zero-copy RDMA) and the control
path (KV events ↔ master ↔ router) are decoupled: each is useful alone, and
together they close the loop — engine reports placement → master indexes and
accumulates history → router routes on the cost model (Pillar 1) → routing
trains the hit history → placement / eviction / replication policies consume
it (Pillar 2) → agent hints override where the workload knows better
(Pillar 3). A prefix that fell out of every engine's HBM but survives in the
UMBP pool remains routable and is fetched from the nearest replica over RDMA
instead of being recomputed.

---

## 3. Implementation details

Current state of the integration (working in our fork, exercised daily by the
PD-disaggregation benchmarks):

- **L3 backend** — `UMBPStore` (`mem_cache/storage/umbp/umbp_store.py`),
  registered as `mori` in `StorageBackendFactory`, selected via
  `--hicache-storage-backend mori`. Implements the zero-copy v1
  `HiCacheStorage` interface: `batch_exists` → `BatchExistsConsecutive`
  (prefetch probe), `batch_get_v1` / `batch_set_v1` → pointer-based
  `BatchGet`/`BatchPut` directly against host KV page buffers. Key scheme and
  MHA/MLA/split-heads handling mirror MooncakeStore; radix depth from
  `extra_info.prefix_keys` is forwarded via `BatchPutWithDepth` for
  depth-aware eviction. Existing write/prefetch policy knobs are unchanged.
- **Modes** — standalone (default): per-rank local DRAM+SSD with automatic
  rank isolation (per-rank SSD dirs, MLA+TP shared-SSD leader/follower,
  DP+SPDK tenant quotas). Distributed (`master_address` set): each rank joins
  the master-led pool with per-rank identities, the host KV buffer registered
  once for RDMA zero-copy (staging-buffer fallback for capped-MR NICs),
  master `dram_page_size` auto-derived so one Put/Get = one page = one RDMA
  op, and cross-DP-rank duplicate puts deduped by the master.
- **L2 host allocator** — opt-in `UMBPHostTensorAllocator` hook in
  `memory_pool_host.py`: hugepage + NUMA-bound + prefaulted host KV tensor,
  which also makes the one-shot RDMA registration reliable on AINIC/ROCm.
- **Control plane** — `kv_events_subscriber=true` mirrors L1/L2
  `BlockStored`/`BlockRemoved` events (via SGLang's existing
  `ZmqEventPublisher`, no engine changes) into the master's external-KV
  index; mori-sched queries it per request to implement the
  `umbp_cache_aware` routing policy — closing the Pillar-1 loop.
- **Config & failure handling** — all knobs via
  `--hicache-storage-backend-extra-config` JSON (allow-listed,
  scope-checked) with `UMBP_*` env fallbacks. Flag-gated and import-guarded;
  master outage degrades to local-tier serving.

**Not yet in SGLang:** the `PolicyContext` plugin surface and replication
policy (Pillar 2, mori-side work), and the `cache_hints` extension to
`HiCacheStorageExtraInfo` for TTL/session/priority (Pillar 3).
