UMBP Master Client
==================

``UMBPMasterClient`` is a lightweight Python **control-plane** client for the
UMBP master.  It can register a node, report/revoke externally-managed KV-cache
blocks, and query which nodes hold a given set of blocks — enabling cross-node
KV-cache-aware scheduling for externally-managed L1/L2/L3 caches such as
SGLang HiCache (GPU HBM, pinned host DRAM, and storage-backed L3).

It is *not* the full UMBP data-plane client. Hot-path Put/Get with RDMA / MORI-IO
goes through the C++ ``IUMBPClient`` (``mori.cpp.UMBPClient`` in Python) backed by a
``DistributedClient`` + ``PoolClient``. ``UMBPMasterClient`` only speaks to the master
control plane and never registers a peer service or starts a heartbeat thread.
Schedulers should usually use it for read-only queries such as
``match_external_kv()`` and ``get_client_transfer_rates()``. SGLang HiCache
event forwarding normally uses the distributed UMBP client owned by
``UMBPStore``.

For the full architecture see ``src/umbp/doc/design-master-control-plane.md``.

----

Starting the Master Server
--------------------------

The master server is a standalone binary built alongside the mori wheel.

**Binary location:**

.. code-block:: text

   # After cmake build
   build/src/umbp/umbp_master

   # Inside an installed wheel (auto-detected by UMBPMasterClient)
   python/mori/umbp_master

   # Override at runtime
   export UMBP_MASTER_BIN=/path/to/umbp_master

**Usage:**

.. code-block:: bash

   umbp_master [listen_address] [metrics_port]

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``listen_address``
     - ``0.0.0.0:50051``
     - gRPC listen address in ``host:port`` format
   * - ``metrics_port``
     - ``9091``
     - Prometheus metrics HTTP port

All ``UMBP_*`` timing knobs (``UMBP_HEARTBEAT_TTL_SEC``, ``UMBP_REAPER_INTERVAL_SEC``,
``UMBP_LEASE_DURATION_SEC``, ...) are honored at master startup and printed once as
``[Master] Resolved timing: ...``. See
`runtime-env-vars.md <../../src/umbp/doc/runtime-env-vars.md>`_ for the full list.

**Examples:**

.. code-block:: bash

   # Defaults: gRPC on 0.0.0.0:50051, metrics on 9091
   ./build/src/umbp/umbp_master

   # Custom gRPC port (matches the SGLang/hicache default), default metrics port
   ./build/src/umbp/umbp_master 127.0.0.1:15558

   # Both custom
   ./build/src/umbp/umbp_master 127.0.0.1:15558 9099

   # With debug logging
   MORI_UMBP_LOG_LEVEL=DEBUG ./build/src/umbp/umbp_master 127.0.0.1:15558

The server exits cleanly on ``SIGINT`` / ``SIGTERM`` (e.g. ``Ctrl-C`` or ``kill``).

**Building the binary:**

.. code-block:: bash

   mkdir -p build && cd build
   cmake .. -DUMBP=ON
   make -j$(nproc) umbp_master

The Python ``mori.umbp`` package auto-detects ``umbp_master`` packaged inside the
wheel and exports ``UMBP_MASTER_BIN`` to that path (see
``mori/python/mori/umbp/__init__.py::_configure_packaged_umbp_master``). Set
``UMBP_MASTER_BIN`` explicitly to point at a custom build.

----

**Imports:**

All symbols documented here are bound in ``mori.cpp`` (the underlying pybind
module). Some data types are also re-exported from ``mori.umbp`` for
convenience, but ``UMBPMasterClient`` and ``UMBPExternalKvNodeMatch`` are only
available from ``mori.cpp``.

.. code-block:: python

   from mori.cpp import (
       UMBPMasterClient,
       UMBPTierType,
       UMBPExternalKvNodeMatch,
       UMBPHiCacheTransfer,
       UMBPHiCacheTransferRate,
       UMBPClientTransferRates,
   )

----

UMBPTierType
------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``UMBPTierType.Unknown``
     - Unknown / unspecified tier
   * - ``UMBPTierType.HBM``
     - High-bandwidth memory (on-device)
   * - ``UMBPTierType.DRAM``
     - Host DRAM
   * - ``UMBPTierType.SSD``
     - Solid-state drive

----

UMBPExternalKvNodeMatch
-----------------------

Returned by ``match_external_kv()``. Each instance describes one node that holds
a subset of the queried KV blocks, grouped by every tier each block lives on
for that node.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute / method
     - Description
   * - ``node_id: str``
     - Identifier of the node holding the blocks
   * - ``peer_address: str``
     - PeerService gRPC address of the node, when the node registered one.
       This is used by UMBP data-plane clients for direct transfer.  It may be
       empty for lightweight ``UMBPMasterClient.register_self()`` examples or
       for schedulers that only need ``node_id`` for routing decisions.
   * - ``hashes_by_tier: dict[UMBPTierType, list[str]]``
     - Matched hashes grouped by every tier they currently live on for this
       node.  A single block held on multiple tiers (e.g. write_through has
       created a CPU mirror while the GPU copy is still alive) appears in
       **every** tier bucket it physically resides on — bucket sizes do not
       sum to the distinct count.  Iterating the dict yields tiers in
       sorted ``UMBPTierType`` order, so the first non-empty bucket is the
       fastest tier currently available on this node.
   * - ``matched_hash_count() -> int``
     - Number of *distinct* matched hashes (size of the union across tiers).
       A hash on HBM+DRAM still counts once.  This is the right value to
       feed into "how much of the prompt does this worker have cached?"
       routing decisions; use ``hashes_by_tier`` for per-tier cost models.

----

UMBPHiCacheTransfer
-------------------

``UMBPHiCacheTransfer`` identifies a directional transfer between HiCache tiers.
The API reports transfer **rates**, so the direction matters.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``UMBPHiCacheTransfer.Unknown``
     - Unknown / invalid direction. This value is never returned by
       ``get_client_transfer_rates()``.
   * - ``UMBPHiCacheTransfer.L1ToL2``
     - HBM/GPU KV -> host DRAM KV. Emitted when HiRadixCache consumes
       write-through or write-back acknowledgements.
   * - ``UMBPHiCacheTransfer.L2ToL1``
     - Host DRAM KV -> HBM/GPU KV. Emitted when load-back acknowledgements
       complete.
   * - ``UMBPHiCacheTransfer.L2ToL3``
     - Host DRAM KV -> storage-backed L3. Emitted when storage backup
       operations complete.
   * - ``UMBPHiCacheTransfer.L3ToL2``
     - Storage-backed L3 -> host DRAM KV. Emitted when storage prefetch
       operations complete.

All byte counts use **logical KV bytes**: ``num_tokens`` multiplied by the
HiCache host KV pool's bytes-per-token value. They intentionally do not expose
backend-specific RDMA or NVMe byte counts.

----

UMBPHiCacheTransferRate
-----------------------

Returned inside ``UMBPClientTransferRates.rates``. Each instance describes one
fresh, valid transfer-rate estimate for one direction on one worker.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``direction: UMBPHiCacheTransfer``
     - Direction this rate describes.
   * - ``bytes_per_sec: float``
     - Short-window EWMA in logical KV bytes per second. Always non-negative.
   * - ``rate_age_ms: int``
     - Age of the latest rate-state update. A real transfer sample and an idle
       heartbeat tick both refresh this value. It is a freshness indicator for
       the rate value, not proof that recent transfer traffic occurred.
   * - ``window_ms: int``
     - Window width used by the latest real-sample EWMA update. Idle heartbeat
       ticks do not change it.

----

UMBPClientTransferRates
-----------------------

Returned by ``get_client_transfer_rates()``. Each instance describes one alive
worker and a sparse list of currently fresh transfer-rate entries.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``node_id: str``
     - Worker node id. This is the same namespace used by
       ``UMBPExternalKvNodeMatch.node_id``.
   * - ``peer_address: str``
     - PeerService gRPC address registered by the worker. This is useful when
       joining transfer rates with ``match_external_kv()`` results.
   * - ``tags: list[str]``
     - Opaque tags registered by the worker, for example
       ``"sgl_role=prefill"`` or ``"dp_rank=0"``.
   * - ``rates: list[UMBPHiCacheTransferRate]``
     - Sparse list with 0 to 4 entries. Missing directions mean ``unknown``,
       not zero.

``rates`` is intentionally sparse:

* A direction missing from ``rates`` means the signal is unknown. Common causes
  are cold start, stale samples, non-UMBPStore storage backends, standalone
  UMBP mode, master restart, or an offline node.
* A direction present with ``bytes_per_sec == 0`` and a fresh ``rate_age_ms``
  means the worker is alive, the direction is known, and the current smoothed
  rate has decayed below any meaningful scheduling threshold.
* A direction present with ``bytes_per_sec > 0`` is a usable pressure signal.

Schedulers should treat missing and zero as different facts.

----

UMBPMasterClient
----------------

**Constructor:**

.. code-block:: python

   UMBPMasterClient(
       master_address: str,
       node_id: str = "",
       node_address: str = "",
   )

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``master_address``
     - Address of the UMBP master server, e.g. ``"127.0.0.1:15558"``
   * - ``node_id``
     - Identifier for this node (required for registration)
   * - ``node_address``
     - This node's own gRPC address, used by peers to connect back

Construction is non-blocking — the gRPC channel is created lazily and will not
raise even if the master is unreachable. ``auto_heartbeat`` is forced to ``False``
on this client (no heartbeat thread is started); ``UMBPMasterClient`` is intended
for one-shot or short-lived lookups, not for long-running peer membership.

**Methods:**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``register_self(tier_capacities)``
     - Register this node with the master. ``tier_capacities`` is a ``dict[UMBPTierType, tuple[int, int]]`` mapping each tier to ``(total_bytes, available_bytes)``. The default is an empty dict, which registers membership only without capacity. Raises ``RuntimeError`` on failure.
   * - ``unregister_self()``
     - Unregister this node. Idempotent: calling it on a never-registered or already-unregistered client is a silent no-op. Raises ``RuntimeError`` only on RPC failure.
   * - ``is_registered() -> bool``
     - Return ``True`` if this node is currently registered.
   * - ``report_external_kv_blocks(node_id, hashes, tier)``
     - **Additive**: announce that ``node_id`` now holds ``hashes`` at
       ``tier``.  Existing tier buckets for the same hashes are untouched
       (a re-report at the same tier is a no-op; reporting at a new tier
       adds a bucket without removing previously reported ones).  Raises
       ``RuntimeError`` if ``hashes`` is empty or the call fails.  ``node_id``
       must already be registered and alive; reports for unknown or expired
       nodes are ignored by the master.
   * - ``revoke_external_kv_blocks(node_id, hashes, tier)``
     - Remove ``hashes`` from a single tier on this node.  Other tier
       buckets for the same hashes are untouched.  No-op for hashes that
       were never reported at ``tier``.  Raises ``RuntimeError`` if
       ``hashes`` is empty.
   * - ``revoke_all_external_kv_blocks_at_tier(node_id, tier)``
     - **Bulk**: revoke every hash currently registered by ``node_id`` at
       ``tier``.  Used when an entire tier is wiped (storage backend clear
       or detach, host-pool reset).  Other tier buckets are untouched.
   * - ``match_external_kv(hashes) -> list[UMBPExternalKvNodeMatch]``
     - Query the master for nodes that hold any of the requested ``hashes``. Returns an empty list when no matches exist or ``hashes`` is empty. Raises ``RuntimeError`` on connection failure.
   * - ``get_client_transfer_rates(node_ids=[]) -> list[UMBPClientTransferRates]``
     - Query the master for fresh HiCache transfer-rate estimates. Pass an
       empty list to return all alive clients, or pass specific ``node_id``
       values to restrict the response. Unknown or expired node ids are
       silently omitted. Raises ``RuntimeError`` when the RPC fails, including
       when the request exceeds ``UMBP_TRANSFER_RATE_QUERY_MAX_BATCH``.

**Protocol notes for non-Python clients:**

``MatchExternalKv`` returns one ``ExternalKvNodeMatch`` per node.  Each match
contains ``repeated TierHashes hashes_by_tier`` rather than the legacy
``matched_hashes + tier`` shape.  A single hash may appear in multiple tier
buckets for the same node, so consumers must de-duplicate by hash before using a
match count for routing.

``GetClientTransferRates`` returns one ``ClientTransferRates`` per alive client.
The response order is not guaranteed to match request order; build a
``node_id -> UMBPClientTransferRates`` map before joining it with scheduler
candidate lists.

----

Usage Examples
--------------

**Basic node registration:**

.. code-block:: python

   from mori.cpp import UMBPMasterClient, UMBPTierType

   _1GB = 1 * 1024 * 1024 * 1024

   client = UMBPMasterClient(
       "127.0.0.1:15558",
       node_id="worker-0",
       node_address="worker-0:8080",
   )

   # Register with available DRAM capacity
   client.register_self({UMBPTierType.DRAM: (_1GB, _1GB)})
   assert client.is_registered()

   # ... do work ...

   client.unregister_self()

**Reporting and matching KV blocks:**

.. code-block:: python

   from mori.cpp import UMBPMasterClient, UMBPTierType

   _1GB = 1 * 1024 * 1024 * 1024
   master = "127.0.0.1:15558"

   # Node A reports that it holds some KV blocks in DRAM
   node_a = UMBPMasterClient(master, node_id="node-a", node_address="node-a:8080")
   node_a.register_self({UMBPTierType.DRAM: (_1GB, _1GB)})

   hashes = ["sha256-abc", "sha256-def", "sha256-ghi"]
   node_a.report_external_kv_blocks("node-a", hashes, UMBPTierType.DRAM)

   # Node B queries which nodes hold these blocks
   node_b = UMBPMasterClient(master)
   matches = node_b.match_external_kv(hashes)

   for m in matches:
       per_tier = {t.name: len(hs) for t, hs in m.hashes_by_tier.items()}
       peer = m.peer_address or "<no PeerService>"
       print(f"node {m.node_id} @ {peer} has {m.matched_hash_count()} blocks: {per_tier}")
       # → "node node-a @ <no PeerService> has 3 blocks: {'DRAM': 3}"

**Revoking blocks when one tier is evicted:**

.. code-block:: python

   # GPU was evicted but the host (DRAM) mirror is still alive — drop only
   # the HBM bucket; node-a stays in the index with its DRAM bucket intact.
   evicted = ["sha256-abc", "sha256-def"]
   node_a.revoke_external_kv_blocks("node-a", evicted, UMBPTierType.HBM)

**Bulk revoke when an entire tier is wiped:**

.. code-block:: python

   # Storage backend was cleared — drop every SSD bucket this node has in
   # one RPC.  HBM and DRAM buckets are untouched.
   node_a.revoke_all_external_kv_blocks_at_tier("node-a", UMBPTierType.SSD)

**Same node holding the same blocks on multiple tiers:**

.. code-block:: python

   # write_through created a CPU mirror while the GPU copy is still alive
   # — report both tiers; the master keeps both buckets.
   hashes = ["sha256-prefix-0", "sha256-prefix-1"]
   node_a.report_external_kv_blocks("node-a", hashes, UMBPTierType.HBM)
   node_a.report_external_kv_blocks("node-a", hashes, UMBPTierType.DRAM)

   matches = node_a.match_external_kv(hashes)
   m = matches[0]
   # m.hashes_by_tier == {UMBPTierType.HBM: [...], UMBPTierType.DRAM: [...]}
   # m.matched_hash_count() == 2  (distinct, NOT 4 — same hash on two tiers)

**Multiple nodes holding the same blocks (different tiers):**

.. code-block:: python

   _1GB = 1 * 1024 * 1024 * 1024
   hashes = ["sha256-shared-0", "sha256-shared-1"]

   node_a = UMBPMasterClient(master, node_id="node-a", node_address="node-a:8080")
   node_a.register_self({UMBPTierType.DRAM: (_1GB, _1GB)})
   node_a.report_external_kv_blocks("node-a", hashes, UMBPTierType.DRAM)

   node_b = UMBPMasterClient(master, node_id="node-b", node_address="node-b:8080")
   node_b.register_self({UMBPTierType.HBM: (_1GB, _1GB)})
   node_b.report_external_kv_blocks("node-b", hashes, UMBPTierType.HBM)

   # match_external_kv returns one entry per node; each entry breaks the
   # matched hashes down by every tier they live on for that node.
   matches = node_a.match_external_kv(hashes)
   matched_nodes = {m.node_id: list(m.hashes_by_tier.keys()) for m in matches}
   # → {"node-a": [UMBPTierType.DRAM], "node-b": [UMBPTierType.HBM]}

**Using ``match_external_kv`` for KV-cache-aware scheduling:**

``match_external_kv`` is intentionally grouped by node, then by tier.  A
scheduler such as mori-scheduler can derive both a per-worker cache-hit score
and per-hash source locations from this response.

.. code-block:: python

   from collections import defaultdict
   from mori.cpp import UMBPMasterClient, UMBPTierType

   master = "127.0.0.1:15558"
   query_client = UMBPMasterClient(master)

   query_hashes = [
       "sha256-prefix-0",
       "sha256-prefix-1",
       "sha256-prefix-2",
       "sha256-prefix-3",
   ]
   matches = query_client.match_external_kv(query_hashes)

   # hash -> list of candidate locations.  Useful for building a future
   # prefetch hint or for explaining why a request was routed to a worker.
   locations_by_hash = defaultdict(list)
   for m in matches:
       for tier, hashes in m.hashes_by_tier.items():
           for h in hashes:
               locations_by_hash[h].append(
                   {
                       "node_id": m.node_id,
                       "peer_address": m.peer_address,
                       "tier": tier,
                   }
               )

   # Per-node summaries for routing.  Do NOT sum bucket sizes to get a hit
   # count: the same hash can appear in HBM+DRAM+SSD on the same node.
   summaries = []
   for m in matches:
       best_tier_by_hash = {}
       for tier in sorted(m.hashes_by_tier):
           for h in m.hashes_by_tier[tier]:
               best_tier_by_hash.setdefault(h, tier)

       summaries.append(
           {
               "node_id": m.node_id,
               "matched_blocks": len(best_tier_by_hash),  # distinct hashes
               "per_tier_blocks": {
                   tier.name: len(set(hashes))
                   for tier, hashes in m.hashes_by_tier.items()
               },
               # Fastest tier for each matched hash on this node.
               "best_tier_by_hash": best_tier_by_hash,
           }
       )

   not_found = set(query_hashes) - set(locations_by_hash)
   best_node = max(summaries, key=lambda s: s["matched_blocks"], default=None)

   # Example policy sketch:
   # - HBM hits are best routed to the same worker/rank.
   # - DRAM hits are cheaper than recompute but require H2D load-back.
   # - SSD hits are L3/storage hits and should carry a higher fetch cost.
   tier_cost = {
       UMBPTierType.HBM: 0,
       UMBPTierType.DRAM: 1,
       UMBPTierType.SSD: 3,
   }

   def estimated_fetch_cost(summary):
       return sum(
           tier_cost[tier]
           for tier in summary["best_tier_by_hash"].values()
       )

   cost_aware_node = min(summaries, key=estimated_fetch_cost, default=None)
   # Production policies should combine this tier cost with recompute cost for
   # `not_found`, queue depth, and worker health/load signals.

For Rust/tonic consumers, mirror the current proto shape:

.. code-block:: rust

   use std::collections::{HashMap, HashSet};

   pub struct NodeMatch {
       pub node_id: String,
       pub peer_address: String,
       pub hashes_by_tier: HashMap<i32, Vec<String>>,
   }

   impl NodeMatch {
       pub fn matched_hash_count(&self) -> usize {
           let mut seen = HashSet::new();
           for hashes in self.hashes_by_tier.values() {
               for h in hashes {
                   seen.insert(h);
               }
           }
           seen.len()
       }
   }

Do not keep using a legacy ``matched_hashes: Vec<String>, tier: i32`` wrapper:
it cannot represent one block living on multiple HiCache tiers and will either
lose tier information or double-count hits.

**Using ``get_client_transfer_rates`` for prefill-pressure-aware scheduling:**

``get_client_transfer_rates`` exposes a short-window pressure signal derived
from SGLang HiRadixCache completion points. It is designed to be joined with
``match_external_kv`` candidates by ``node_id``. A typical prefill scheduler can
first find workers that hold prompt KV blocks, then use transfer rates as a
tie-breaker among otherwise comparable workers.

.. code-block:: python

   from mori.cpp import (
       UMBPMasterClient,
       UMBPTierType,
       UMBPHiCacheTransfer,
   )

   master = "127.0.0.1:15558"
   query_client = UMBPMasterClient(master)

   query_hashes = [
       "sha256-prefix-0",
       "sha256-prefix-1",
       "sha256-prefix-2",
   ]

   matches = query_client.match_external_kv(query_hashes)
   candidate_ids = [m.node_id for m in matches]
   rates_by_node = {
       c.node_id: c
       for c in query_client.get_client_transfer_rates(candidate_ids)
   }

   def rate_map(node_id):
       c = rates_by_node.get(node_id)
       if c is None:
           return {}
       return {r.direction: r.bytes_per_sec for r in c.rates}

   def best_tier(match):
       # Tiers are ordered Unknown=0, HBM=1, DRAM=2, SSD=3. Lower is faster.
       # Exclude Unknown defensively; the master should not return it.
       tiers = [
           tier for tier, hashes in match.hashes_by_tier.items()
           if hashes and tier != UMBPTierType.Unknown
       ]
       return min(tiers, default=None)

   def pressure_for_prefill(match):
       rates = rate_map(match.node_id)
       tier = best_tier(match)
       if tier == UMBPTierType.HBM:
           # GPU-resident prompt KV generally does not require HiCache transfer.
           return 0.0
       if tier == UMBPTierType.DRAM:
           # DRAM hit needs host -> GPU load-back.
           return rates.get(UMBPHiCacheTransfer.L2ToL1)
       if tier == UMBPTierType.SSD:
           # SSD hit needs storage -> host and then host -> GPU. Treat a
           # missing direction as unknown, not zero.
           l3_to_l2 = rates.get(UMBPHiCacheTransfer.L3ToL2)
           l2_to_l1 = rates.get(UMBPHiCacheTransfer.L2ToL1)
           if l3_to_l2 is None or l2_to_l1 is None:
               return None
           return max(l3_to_l2, l2_to_l1)
       return None

   scored = []
   fallback = []
   for match in matches:
       pressure = pressure_for_prefill(match)
       if pressure is None:
           fallback.append(match)
           continue
       scored.append((pressure, match))

   # Among candidates with complete signals, prefer lower transfer pressure.
   scored.sort(key=lambda item: item[0])
   chosen = scored[0][1] if scored else None

   # If all candidates have missing / stale transfer signals, fall back to the
   # existing affinity, cache-hit, queue-depth, or load-guard policy.
   if chosen is None:
       chosen = fallback[0] if fallback else None

Important handling rules:

* Missing directions are ``unknown``. Do not treat them as zero pressure.
* Present directions with ``bytes_per_sec == 0`` and a fresh ``rate_age_ms``
  mean the smoothed rate has decayed below any meaningful scheduling threshold
  (the EWMA never reaches floating-point zero in practice; the scheduler should
  treat this as a usable "near-idle" signal, not as missing data).
* Empty ``node_ids`` queries return all alive clients, including clients with
  ``rates=[]``. This is useful for debugging; scheduling hot paths should pass
  only candidate node ids.
* ``peer_address`` and ``tags`` are copied from registration metadata and are
  safe to use for joins, filtering, and debugging.

The API only reports UMBPStore traffic in distributed UMBP mode. Workers using
other HiCache storage backends, or UMBP standalone mode, will not produce
transfer-rate entries for this API.

**Transfer-rate timing knobs:**

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Environment variable
     - Default
     - Description
   * - ``UMBP_TRANSFER_RATE_MIN_SAMPLE_GAP_MS``
     - ``200``
     - Minimum wall-clock gap between accepted samples before computing an
       instantaneous rate. Earlier samples are accumulated.
   * - ``UMBP_TRANSFER_RATE_MAX_SAMPLE_AGE_MS``
     - ``15000``
     - Directions older than this are omitted from ``rates``.
   * - ``UMBP_TRANSFER_RATE_TICK_MIN_GAP_MS``
     - ``2000``
     - Minimum gap used by idle decay and heartbeat anti-burst guards.
   * - ``UMBP_TRANSFER_RATE_EWMA_ALPHA_PERMILLE``
     - ``300``
     - EWMA alpha expressed as an integer in permille. ``300`` means
       ``alpha = 0.3``.
   * - ``UMBP_TRANSFER_RATE_QUERY_MAX_BATCH``
     - ``4096``
     - Maximum number of node ids accepted by one query.

The reporting cadence reuses the existing client metrics flush interval
``UMBP_METRICS_REPORT_INTERVAL_MS``. There is no separate transfer-rate
reporter thread.

**Context-manager pattern for automatic cleanup:**

.. code-block:: python

   import contextlib

   @contextlib.contextmanager
   def registered_client(master_address, node_id, tier_caps):
       client = UMBPMasterClient(master_address, node_id=node_id, node_address=node_id)
       client.register_self(tier_caps)
       try:
           yield client
       finally:
           with contextlib.suppress(Exception):
               client.unregister_self()

   _1GB = 1 * 1024 * 1024 * 1024
   with registered_client("127.0.0.1:15558", "worker-0", {UMBPTierType.DRAM: (_1GB, _1GB)}) as c:
       c.report_external_kv_blocks("worker-0", ["sha256-abc"], UMBPTierType.DRAM)
       matches = c.match_external_kv(["sha256-abc"])

----

End-to-End Example
------------------

``examples/umbp/umbp_master_client_demo.py`` is a self-contained script that
starts the master binary as a subprocess, runs a multi-tier
report/match/revoke scenario, then shuts everything down cleanly.

.. code-block:: bash

   # From the repo root — binary auto-detected from build/
   python examples/umbp/umbp_master_client_demo.py

   # Point at a specific binary
   UMBP_MASTER_BIN=/path/to/umbp_master python examples/umbp/umbp_master_client_demo.py

The script is reproduced in full at
`examples/umbp/umbp_master_client_demo.py <../../examples/umbp/umbp_master_client_demo.py>`_.
