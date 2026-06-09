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
Schedulers and sidecars can use it for synchronous external-KV
``report``/``revoke`` calls on behalf of a registered worker; SGLang HiCache
event forwarding normally uses the distributed UMBP client owned by
``UMBPStore`` so high-rate writes are batched through heartbeats.

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

.. code-block:: python

   from mori.cpp import (
       UMBPMasterClient,
       UMBPTierType,
       UMBPExternalKvNodeMatch,
       UMBPExternalKvHitCountEntry,
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
     - Register this node with the master. ``tier_capacities`` is a ``dict[UMBPTierType, tuple[int, int]]`` mapping each tier to ``(total_bytes, available_bytes)``. Raises ``RuntimeError`` on failure.
   * - ``unregister_self()``
     - Unregister this node. Raises ``RuntimeError`` if the node is not registered or the call fails.
   * - ``is_registered() -> bool``
     - Return ``True`` if this node is currently registered.
   * - ``report_external_kv_blocks(node_id, hashes, tier)``
     - **Additive**: announce that ``node_id`` now holds ``hashes`` at
       ``tier``.  Existing tier buckets for the same hashes are untouched
       (a re-report at the same tier is a no-op; reporting at a new tier
       adds a bucket without removing previously reported ones).  Raises
       ``RuntimeError`` if ``node_id`` or ``hashes`` is empty, or the call
       fails.  ``node_id`` must already be registered and alive; reports for
       unknown or expired nodes are ignored by the master.
   * - ``revoke_external_kv_blocks(node_id, hashes, tier)``
     - Remove ``hashes`` from a single tier on this node.  Other tier
       buckets for the same hashes are untouched.  No-op for hashes that
       were never reported at ``tier``.  Unlike report, revoke does not
       require ``node_id`` to be alive.  Raises ``RuntimeError`` if
       ``node_id`` or ``hashes`` is empty.
   * - ``revoke_all_external_kv_blocks_at_tier(node_id, tier)``
     - **Bulk**: revoke every hash currently registered by ``node_id`` at
       ``tier``.  Used when an entire tier is wiped (storage backend clear
       or detach, host-pool reset).  Other tier buckets are untouched.
       Unlike report, revoke does not require ``node_id`` to be alive.
   * - ``match_external_kv(hashes, count_as_hit=False) -> list[UMBPExternalKvNodeMatch]``
     - Query the master for nodes that hold any of the requested ``hashes``.
       Returns an empty list when no matches exist or ``hashes`` is empty. Set
       ``count_as_hit=True`` **only** on the real user-request routing path —
       this is what feeds the hit-tracking index (see
       `Cache Hit Tracking`_).  Diagnostic, health-check, dashboard, or
       speculative-probe callers must leave the default ``False`` to avoid
       polluting the counters.  Raises ``RuntimeError`` on connection failure.
   * - ``get_external_kv_hit_counts(hashes) -> list[UMBPExternalKvHitCountEntry]``
     - Sparse lookup of cumulative per-hash routing-hit counters maintained by
       the master.  Hashes that have never been counted, or whose entry was
       garbage-collected after ``UMBP_HIT_INDEX_TTL_SEC`` of inactivity, are
       omitted from the response (not returned with ``hit_count_total=0``).
       Duplicate request hashes are de-duplicated by the server and yield at
       most one entry each. Response order is not guaranteed to match request
       order — build a ``{hash: entry}`` map if you need alignment.  Requests
       larger than ``UMBP_HIT_QUERY_MAX_BATCH`` (default ``4096``) fail with
       ``RuntimeError`` carrying ``UMBP_HIT_QUERY_MAX_BATCH`` in the message;
       the server does **not** silently truncate.

**Protocol notes for non-Python clients:**

``MatchExternalKv`` returns one ``ExternalKvNodeMatch`` per node.  Each match
contains ``repeated TierHashes hashes_by_tier`` rather than the legacy
``matched_hashes + tier`` shape.  A single hash may appear in multiple tier
buckets for the same node, so consumers must de-duplicate by hash before using a
match count for routing.

``MatchExternalKvRequest.count_as_hit`` is optional and defaults to ``false``.
When set to ``true``, the master increments ``hit_count_total`` once for every
unique queried hash that is actually matched in the external KV placement
index. Missing hashes are not counted, and the number of nodes or tiers holding
the same hash does not multiply the increment.

``GetExternalKvHitCounts`` returns ``HitCountEntry`` values sparsely: absent
hashes are omitted rather than returned with zero. Response ordering is not
guaranteed to match request ordering, so callers should build a ``hash -> entry``
map when alignment matters.

----

Cache Hit Tracking
------------------

**Purpose — hot KV block awareness.** This API tells you, for any KV block
hash you care about, **how many real-request routing lookups have matched
it** in the external KV placement index.  That is the basic hotness signal
you need to identify hot blocks; what you do with that signal — pin them
to a faster tier, replicate them to more workers, bias the scheduler away
from evicting them, expose a "hottest prompts" operator dashboard, … — is
intentionally left to the caller.

The counter records *lookup matches*, not *cache reads*: the master
increments it the moment a ``count_as_hit=True`` query hits a hash in the
placement index, with no further visibility into whether the caller went
on to dispatch the request, whether the routed worker actually served
from cache, or whether the request was later canceled.  In practice this
is close enough to traffic hotness as long as ``count_as_hit=True`` is
restricted to the real routing path (see below); it is **not** a "this
block was definitely served" log.

UMBP itself does **not** change where blocks live, what gets evicted, or how
routing decisions are made based on hotness.  It exposes the counter as a
primitive; downstream policies are user-built on top.

The mechanism is a single in-memory per-hash counter on the master:

* **Increment** on the real user-request routing path by calling
  ``match_external_kv(hashes, count_as_hit=True)``.
* **Read** the counter from any process by calling
  ``get_external_kv_hit_counts(hashes)``.

Keeping ``count_as_hit=True`` out of diagnostic / probe / dashboard / health-
check callers is what gives the counter its meaning as a *real-routing-path*
signal; otherwise it degenerates into "how many times did anyone ask about
this hash".

Finding hot KV blocks (the typical workflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the main flow this API is designed for. Four steps:

1. **Wire ``count_as_hit=True`` into the routing path, and only there.**
   The component that does the placement lookup for an incoming user
   request should call ``match_external_kv(hashes, count_as_hit=True)``.
   No other caller should set this flag.  This is what keeps the counter
   meaningful as a routing-path signal instead of degenerating into
   "anyone who asked".  (The master only sees the lookup itself, not
   whether the routing decision was followed through — keeping the flag
   off auxiliary callers is what makes the signal close to real traffic in
   practice.)

2. **Maintain the hash universe you want to observe in the caller.**
   There is no scan, no iteration, no "give me the top-K hottest hashes" RPC
   — ``get_external_kv_hit_counts`` is a sparse point lookup keyed by the
   hashes you pass in.  Track the set of hashes you care about yourself
   (e.g. the union of hashes your workers have reported via
   ``report_external_kv_blocks``, or the prefix hashes your scheduler has
   seen).  Split large universes into ``UMBP_HIT_QUERY_MAX_BATCH`` (default
   ``4096``) chunks per RPC.

3. **Periodically read the counters and rank them.** Any process with
   network access to the master can do this — your scheduler, a sidecar
   analyzer, an operator script, a Prometheus exporter.  ``hit_count_total``
   is comparable across hashes, so a simple ``sorted(..., reverse=True)``
   on the returned list gives you a hotness ranking.

4. **Plug the ranking into your own policy.**  Examples of what users
   typically do once they know which blocks are hot — *none of these are
   implemented by UMBP*; they are the kinds of things this API is meant to
   enable:

   * Pin the top-K blocks to a faster tier (HBM) and stop evicting them.
   * Replicate hot blocks to additional workers for more read concurrency.
   * Skip the recompute fallback for hot prefixes even on a partial cache
     miss.
   * Bias the scheduler toward workers that already hold hot blocks, to
     compound locality.
   * Expose a "hottest prompts" dashboard to operators.

Lifetime cumulative — not a rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hit_count_total`` is a **lifetime cumulative** counter, not a rate, not
a sliding window, not exponentially decayed.  It only goes up while the
entry exists, and is reset only when the entry disappears (master restart,
or ``UMBP_HIT_INDEX_TTL_SEC`` of inactivity).

For "recent" hotness — i.e. a current-QPS-style ranking instead of an all-
time ranking — snapshot the counter at two points and diff:

.. code-block:: python

   import time

   def snapshot_counts(client, hashes, batch=4096):
       counts = {}
       for i in range(0, len(hashes), batch):
           for entry in client.get_external_kv_hit_counts(hashes[i : i + batch]):
               counts[entry.hash] = entry.hit_count_total
       return counts

   before = snapshot_counts(monitor, hash_universe)
   time.sleep(300)  # 5 minutes
   after = snapshot_counts(monitor, hash_universe)

   delta = {
       h: after[h] - before.get(h, 0)
       for h in after
       if after[h] - before.get(h, 0) > 0
   }
   recent_hottest = sorted(delta.items(), key=lambda kv: kv[1], reverse=True)[:32]
   # delta[h] is "hits during the 5-minute window".  Hashes absent from
   # `after` either had zero traffic in the window or were GC'd from the
   # hit index after UMBP_HIT_INDEX_TTL_SEC of inactivity.

The raw ``hit_count_total`` itself is most useful as an "ever was hot"
signal (e.g. when deciding whether to re-warm a block that has just been
evicted from every tier).

**Counting rules at a glance:**

* Increments happen on the master, not on the client. There is no client-side
  buffering or batching.
* Per call, each unique input hash is incremented **at most once**, regardless
  of how many ``ExternalKvNodeMatch`` entries reference it (same hash on
  HBM+DRAM+SSD on the same node still counts once; the same hash held by
  N nodes still counts once).
* A query that does **not** match anything in the placement index increments
  nothing — even with ``count_as_hit=True``.
* ``revoke_external_kv_blocks`` and ``revoke_all_external_kv_blocks_at_tier``
  do **not** clear the corresponding counters; a block can carry a non-zero
  lifetime count after its last replica has been dropped.  This is the basis
  for "this block was hot and just got evicted — should I re-warm it?"
  policies.
* The hit index is in-memory only. Counters are lost when the master
  restarts.
* Each entry has its own TTL (``UMBP_HIT_INDEX_TTL_SEC``).  An entry is
  dropped if no ``count_as_hit=True`` call has touched it for that long; the
  master sweeps for expired entries every ``UMBP_HIT_INDEX_GC_INTERVAL_SEC``.
  Absent ≠ ``hit_count_total == 0`` — treat absent as "no recent real
  traffic / no signal".

**Tuning knobs (master process):**

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Env var
     - Default
     - Description
   * - ``UMBP_HIT_INDEX_TTL_SEC``
     - ``7200``
     - Per-entry inactivity TTL. A hash with no counted match for longer
       than this is removed from the hit index.
   * - ``UMBP_HIT_INDEX_GC_INTERVAL_SEC``
     - ``60``
     - Background GC sweep period.
   * - ``UMBP_HIT_QUERY_MAX_BATCH``
     - ``4096``
     - Maximum hashes accepted by one ``get_external_kv_hit_counts``
       call. Oversized requests fail with ``RuntimeError`` (gRPC
       ``INVALID_ARGUMENT``).  The server does not truncate; split your
       request batch on the client side.

See the
`master env-var reference <../../src/umbp/doc/runtime-env-vars.md>`_ for the
full master env-var list.

**Where to call from.**  External-KV query methods are reachable through two
Python clients.  Both ultimately go to the same master and read the same
placement and hit-count indexes, but their **error semantics differ** — pick
the one whose behaviour matches what you want:

* ``UMBPMasterClient`` — the lightweight control-plane client documented on
  this page.  Use it from schedulers, controllers, dashboards, or any
  process that does not need the RDMA / MORI-IO data plane.  Failures
  (connection refused, oversized batch, master returning ``INVALID_ARGUMENT``,
  …) surface as Python ``RuntimeError``.
* ``mori.cpp.UMBPClient`` — the distributed data-plane client backed by
  ``DistributedClient`` + ``PoolClient``.  HiCache / data-plane integrations
  can call ``match_external_kv(hashes, count_as_hit=...)`` and
  ``get_external_kv_hit_counts(hashes)`` on the same client they already
  use for Put/Get, avoiding a second connection.  **Caveat:** the
  data-plane wrapper currently swallows RPC failures and returns an
  **empty list** rather than raising — an empty response from this client
  means "no matches **or** the underlying RPC failed".  If you need to
  distinguish the two (e.g. surface oversized-batch errors), call through
  ``UMBPMasterClient`` instead.

Both methods are also defined on ``StandaloneClient``, but the standalone
implementation is a stub that always returns an empty list and never
contacts a master — it exists only so the ``IUMBPClient`` interface stays
uniform for non-distributed deployments.

For placement writes, use ``UMBPMasterClient.report_external_kv_blocks(node_id,
hashes, tier)`` when a scheduler or sidecar reports on behalf of a registered
worker.  In-process distributed clients use the two-argument synchronous
methods ``mori.cpp.UMBPClient.report_external_kv_blocks(hashes, tier)``,
``revoke_external_kv_blocks(hashes, tier)``, and
``revoke_all_external_kv_blocks_at_tier(tier)`` for their own node.

----

UMBPExternalKvHitCountEntry
---------------------------

Returned by ``get_external_kv_hit_counts()``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``hash: str``
     - KV block hash.
   * - ``hit_count_total: int``
     - Lifetime cumulative count of ``match_external_kv(..., count_as_hit=True)``
       calls in which this hash was actually matched. The count is dropped if
       the entry is garbage-collected after ``UMBP_HIT_INDEX_TTL_SEC`` of
       inactivity or if the master restarts.

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

**Identifying hot KV blocks (end-to-end):**

The hot-block use case has two pieces, usually but not necessarily in
different processes:

*Piece 1 — On the request-routing path,* call ``match_external_kv``
with ``count_as_hit=True`` so each unique hash matched on this lookup
gets one increment on the master:

.. code-block:: python

   from mori.cpp import UMBPMasterClient

   router = UMBPMasterClient("127.0.0.1:15558")

   def route_request(prefix_hashes):
       # count_as_hit=True is the ONLY thing that feeds the hit index.
       # Each unique hash in `prefix_hashes` that actually matches in the
       # placement index is incremented exactly once on the master.
       matches = router.match_external_kv(prefix_hashes, count_as_hit=True)
       if not matches:
           return None  # cache miss — fall back to a recompute worker
       return pick_worker_from_matches(matches)  # caller-defined policy

   # Anywhere else (dashboards, probes, health checks, debug tools) that
   # also wants the placement of these hashes must use count_as_hit=False
   # so it does not pollute the counter.

*Piece 2 — From anywhere with access to the master,* read the
counters back to rank your hashes by hotness.  Maintain the hash
universe in your caller; the master has no scan / top-N RPC.

.. code-block:: python

   import time
   from mori.cpp import UMBPMasterClient

   monitor = UMBPMasterClient("127.0.0.1:15558")

   # The set of hashes you want to observe — typically the union of
   # hashes your workers have reported, or the prefix hashes your
   # scheduler has seen.  You own this set; the master does not enumerate
   # it for you.
   tracked_hashes: list[str] = load_tracked_prefix_hashes()

   # Split into UMBP_HIT_QUERY_MAX_BATCH (default 4096) chunks to avoid
   # RuntimeError on oversized requests.
   def fetch_counts(client, hashes, batch=4096):
       counts: dict[str, int] = {}
       for i in range(0, len(hashes), batch):
           for entry in client.get_external_kv_hit_counts(hashes[i : i + batch]):
               counts[entry.hash] = entry.hit_count_total
       return counts

   while True:
       counts = fetch_counts(monitor, tracked_hashes)
       # Hashes with no recorded routing hit are absent — that is NOT the
       # same as hit_count_total == 0.  Treat absent as "no counted
       # routing lookup yet" / "GC'd after UMBP_HIT_INDEX_TTL_SEC of
       # inactivity".
       hottest = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:32]

       # What you do with `hottest` is entirely up to you — UMBP does not
       # change placement or eviction based on this signal.  Examples:
       #
       #   pin_to_hbm([h for h, _ in hottest])
       #   replicate_to_more_workers([h for h, _ in hottest])
       #   export_to_prometheus(hottest)
       #
       time.sleep(30)

If you need *recent* hotness rather than lifetime hotness, replace the
single ``fetch_counts`` call with the snapshot-and-diff pattern shown in
the `Cache Hit Tracking`_ section above.

**A few subtle properties worth remembering:**

* A counter survives ``revoke_external_kv_blocks`` /
  ``revoke_all_external_kv_blocks_at_tier`` — the placement index and the
  hit index are decoupled.  This is what lets the caller say "this block
  is no longer cached anywhere but it was hot, let's re-warm it".
* A counter does **not** survive a master restart or
  ``UMBP_HIT_INDEX_TTL_SEC`` of inactivity.  Long-tail hashes will
  eventually fall out of the index; only persistently hot blocks stay.
* The two pieces above can live in the same process or in different
  processes on different hosts — both work.  They can use either
  ``UMBPMasterClient`` or the data-plane ``mori.cpp.UMBPClient``; both
  share the same hit index on the master.

**For Rust / tonic consumers**, mirror the current proto shape.
``MatchExternalKv``:

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

``GetExternalKvHitCounts``:

.. code-block:: rust

   pub struct HitCountEntry {
       pub hash: String,
       pub hit_count_total: u64,
   }

   // Request: GetExternalKvHitCountsRequest { hashes: Vec<String> }
   // Response: GetExternalKvHitCountsResponse { entries: Vec<HitCountEntry> }
   //
   // - Entries are sparse: hashes that have never been counted, or that
   //   expired after UMBP_HIT_INDEX_TTL_SEC, are simply omitted.
   // - Response order is not guaranteed to follow request order; build a
   //   HashMap<String, u64> keyed by hash.
   // - Batches larger than UMBP_HIT_QUERY_MAX_BATCH (default 4096) return
   //   gRPC INVALID_ARGUMENT — split client-side.

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
