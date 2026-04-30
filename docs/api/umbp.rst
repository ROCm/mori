UMBP Master Client
==================

``UMBPMasterClient`` is the Python client for interacting with the UMBP master server.
It lets nodes register themselves, report which KV-cache blocks they hold, and query which
nodes hold a given set of blocks — enabling cross-node KV-cache reuse.

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

**Examples:**

.. code-block:: bash

   # Defaults: gRPC on 0.0.0.0:50051, metrics on 9091
   ./build/src/umbp/umbp_master

   # Custom gRPC port, default metrics port
   ./build/src/umbp/umbp_master localhost:5000

   # Both custom
   ./build/src/umbp/umbp_master localhost:5000 9099

   # With debug logging
   MORI_UMBP_LOG_LEVEL=DEBUG ./build/src/umbp/umbp_master localhost:5000

The server exits cleanly on ``SIGINT`` / ``SIGTERM`` (e.g. ``Ctrl-C`` or ``kill``).

**Building the binary:**

.. code-block:: bash

   mkdir -p build && cd build
   cmake .. -DUMBP=ON
   make -j$(nproc) umbp_master

See the ``UMBP_MASTER_BIN`` environment variable to point the Python client at a
non-default binary location.

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
a subset of the queried KV blocks.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``node_id: str``
     - Identifier of the node holding the blocks
   * - ``peer_address: str``
     - gRPC address of the node (for direct transfer)
   * - ``matched_hashes: list[str]``
     - Subset of queried hashes found on this node
   * - ``tier: UMBPTierType``
     - Storage tier where the blocks reside

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
     - Address of the UMBP master server, e.g. ``"localhost:5000"``
   * - ``node_id``
     - Identifier for this node (required for registration)
   * - ``node_address``
     - This node's own gRPC address, used by peers to connect back

Construction is non-blocking — the gRPC channel is created lazily and will not
raise even if the master is unreachable.

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
     - Announce that ``node_id`` holds the KV blocks identified by ``hashes`` at ``tier``. Raises ``RuntimeError`` if ``hashes`` is empty or the call fails.
   * - ``revoke_external_kv_blocks(node_id, hashes)``
     - Remove previously reported blocks from the master index. No-op if the hashes were never reported. Raises ``RuntimeError`` if ``hashes`` is empty.
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
       "localhost:5000",
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
   master = "localhost:5000"

   # Node A reports that it holds some KV blocks in DRAM
   node_a = UMBPMasterClient(master, node_id="node-a", node_address="node-a:8080")
   node_a.register_self({UMBPTierType.DRAM: (_1GB, _1GB)})

   hashes = ["sha256-abc", "sha256-def", "sha256-ghi"]
   node_a.report_external_kv_blocks("node-a", hashes, UMBPTierType.DRAM)

   # Node B queries which nodes hold these blocks
   node_b = UMBPMasterClient(master)
   matches = node_b.match_external_kv(hashes)

   for m in matches:
       print(f"node {m.node_id} @ {m.peer_address} has {len(m.matched_hashes)} blocks on {m.tier}")
       # → "node node-a @ node-a:8080 has 3 blocks on UMBPTierType.DRAM"

**Revoking blocks when they are evicted:**

.. code-block:: python

   # After evicting some blocks from cache, revoke them so other nodes stop routing to us
   evicted = ["sha256-abc", "sha256-def"]
   node_a.revoke_external_kv_blocks("node-a", evicted)

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

   # match_external_kv returns one entry per node
   matches = node_a.match_external_kv(hashes)
   matched_nodes = {m.node_id: m.tier for m in matches}
   # → {"node-a": UMBPTierType.DRAM, "node-b": UMBPTierType.HBM}

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
   with registered_client("localhost:5000", "worker-0", {UMBPTierType.DRAM: (_1GB, _1GB)}) as c:
       c.report_external_kv_blocks("worker-0", ["sha256-abc"], UMBPTierType.DRAM)
       matches = c.match_external_kv(["sha256-abc"])

----

End-to-End Example
------------------

``examples/umbp/umbp_master_client_demo.py`` is a self-contained script that
starts the master binary as a subprocess, runs a two-node report/match/revoke
scenario, then shuts everything down cleanly.

.. code-block:: bash

   # From the repo root — binary auto-detected from build/
   python examples/umbp/umbp_master_client_demo.py

   # Point at a specific binary
   UMBP_MASTER_BIN=/path/to/umbp_master python examples/umbp/umbp_master_client_demo.py

The script is reproduced in full at
`examples/umbp/umbp_master_client_demo.py <../../examples/umbp/umbp_master_client_demo.py>`_.
