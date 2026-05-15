UMBP Master Client
==================

``UMBPMasterClient`` is the **read-only** Python query client for the UMBP master.
It lets nodes register themselves, report which KV-cache blocks they hold, and query
which nodes hold a given set of blocks — enabling cross-node KV-cache reuse for
externally-managed L1/L2 caches (e.g. SGLang's host-mem KV cache).

It is *not* the full UMBP data-plane client. Hot-path Put/Get with RDMA / MORI-IO
goes through the C++ ``IUMBPClient`` (``mori.cpp.UMBPClient`` in Python) backed by a
``DistributedClient`` + ``PoolClient``. ``UMBPMasterClient`` only speaks to the master
control plane and never registers a peer service or starts a heartbeat thread.

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

   from mori.cpp import UMBPMasterClient, UMBPTierType, UMBPExternalKvNodeMatch

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
     - gRPC address of the node (for direct transfer)
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
       ``RuntimeError`` if ``hashes`` is empty or the call fails.
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
       print(f"node {m.node_id} @ {m.peer_address} has {m.matched_hash_count()} blocks: {per_tier}")
       # → "node node-a @ node-a:8080 has 3 blocks: {'DRAM': 3}"

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
