Communication APIs
==================

MORI provides three communication layers, each targeting different use cases.

MORI-EP (Expert Parallelism)
-----------------------------

Dispatch and combine kernels for MoE expert parallelism.

**Imports:**

.. code-block:: python

   from mori.ops import (
       EpDispatchCombineConfig,
       EpDispatchCombineOp,
       EpDispatchCombineKernelType,
   )

**Kernel Types:**

.. list-table::
   :header-rows: 1

   * - Type
     - Value
     - Use Case
   * - ``IntraNode``
     - 0
     - Single-node EP via XGMI
   * - ``InterNode``
     - 1
     - Multi-node baseline
   * - ``InterNodeV1``
     - 2
     - Multi-node optimized bandwidth
   * - ``InterNodeV1LL``
     - 3
     - Multi-node low-latency
   * - ``AsyncLL``
     - 4
     - Async low-latency with pipelining

**EpDispatchCombineOp methods:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``dispatch(input, weights, scales, indices)``
     - Route tokens to expert ranks. Returns (output, weights, scales, indices, recv_count).
   * - ``combine(input, weights, indices)``
     - Combine expert outputs back. Returns (output, weights).
   * - ``dispatch_send() / dispatch_recv()``
     - Split dispatch for overlapping communication with computation.
   * - ``combine_send() / combine_recv()``
     - Split combine for overlapping communication with computation.
   * - ``reset()``
     - Reset internal state between iterations.
   * - ``get_dispatch_src_token_pos()``
     - Get source positions of dispatched tokens (for verification).
   * - ``get_registered_combine_input_buffer(dtype)``
     - Get pre-registered zero-copy combine buffer.

See :doc:`MORI-EP Guide <../MORI-EP-GUIDE>` for full API reference.

AsyncLL v2 SDMA overlap
~~~~~~~~~~~~~~~~~~~~~~~

The HIP backend of ``mori.ops.dispatch_combine_v2`` supports
``async_ll=True, async_ll_transport="sdma"``. Run independent computation
between ``dispatch_send()`` and ``dispatch_recv()``, and between
``combine_send()`` and ``combine_recv()``. Receive data and counts are ready
after the corresponding receive phase completes on the stream.

On a single gfx1250 node, the send phase now posts each payload followed by
its ready notification on the same SDMA queue. The receive phase still waits
for outgoing copies before the source buffers can be reused. Zero-token peers
also exchange notifications. The redundant dispatch receive-count reset is
omitted because the receive copy writes the total, including zero.

These options are read once when an op is constructed:

.. list-table::
   :header-rows: 1

   * - Environment variable
     - Default and effect
   * - ``MORI_ASYNCLL_EARLY_NOTIFY``
     - ``1`` on single-node gfx1250 SDMA; ``0`` elsewhere. Posts notifications
       in the send phase. Other architectures can opt in with ``1``.
   * - ``MORI_ASYNCLL_SKIP_RESET``
     - No longer has an effect. It skipped the host-side ``total_recv`` reset
       before each dispatch; that reset has been removed for every kernel
       family (the dispatch kernels now produce the count themselves), and
       AsyncLL's receive copy always assigned it. Still accepted, so existing
       settings do not fail.
   * - ``MORI_ASYNCLL_FUSED_RECV``
     - ``0``. Set ``1`` to fold receive waits into the copy passes. One wave
       per CTA waits, then an acquire and CTA barrier release the copy waves.

All three variables accept ``0`` or ``1``. Skipping the reset or fusing receive
requires early notification. Set all three to ``0`` for the legacy sequence.
LSA and multi-node ops retain their existing protocol and ignore these SDMA
options. The resolved notification mode is part of every native Plan's
configuration and generated source, so different ops can coexist without
changing the protocol of an already constructed Plan.

The default targets single-node gfx1250 overlap. Receive fusion remains opt-in
because its additional benefit depends on the token count and compute window.

MORI Shmem (Symmetric Memory)
-------------------------------

OpenSHMEM-style APIs for GPU memory management and RDMA.

**Imports:**

.. code-block:: python

   import mori.shmem

**Initialization:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``shmem_torch_process_group_init(group_name)``
     - Init from PyTorch process group (recommended)
   * - ``shmem_init_attr(flags, rank, nranks, unique_id)``
     - Init with broadcast unique ID
   * - ``shmem_finalize()``
     - Cleanup shmem resources

**Query:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``shmem_mype()``
     - Get current PE (rank) ID
   * - ``shmem_npes()``
     - Get total number of PEs
   * - ``shmem_num_qp_per_pe()``
     - Get RDMA queue pairs per PE

**Memory Management:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``shmem_malloc(size)``
     - Allocate symmetric GPU memory
   * - ``shmem_malloc_align(alignment, size)``
     - Aligned symmetric allocation
   * - ``shmem_free(ptr)``
     - Free symmetric memory
   * - ``shmem_buffer_register(ptr, size)``
     - Register existing buffer for RDMA
   * - ``shmem_buffer_deregister(ptr, size)``
     - Deregister buffer

**Communication:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``shmem_ptr_p2p(ptr, my_pe, dest_pe)``
     - Translate pointer to P2P address on remote PE
   * - ``shmem_barrier_all()``
     - Global barrier across all PEs

See :doc:`Shmem Guide <../MORI-SHMEM-GUIDE>` for full API reference.

MORI-IO (Point-to-Point I/O)
------------------------------

RDMA-based P2P communication for KVCache transfer.

**Imports:**

.. code-block:: python

   from mori.io import (
       IOEngine, IOEngineSession, IOEngineConfig,
       BackendType, MemoryLocationType, StatusCode, PollCqMode,
       RdmaBackendConfig, XgmiBackendConfig,
       EngineDesc, MemoryDesc,
       set_log_level,
   )

**IOEngine methods:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Description
   * - ``get_engine_desc()``
     - Get engine descriptor for remote registration
   * - ``create_backend(type, config)``
     - Create RDMA/XGMI/TCP backend
   * - ``register_remote_engine(desc)``
     - Register a remote engine
   * - ``register_torch_tensor(tensor)``
     - Register a PyTorch tensor for transfers
   * - ``read(local, l_off, remote, r_off, size, uid)``
     - One-sided read (remote → local)
   * - ``write(local, l_off, remote, r_off, size, uid)``
     - One-sided write (local → remote)
   * - ``create_session(local_mem, remote_mem)``
     - Create reusable session for repeated transfers

**Enums:**

- ``BackendType``: ``Unknown``, ``XGMI``, ``RDMA``, ``TCP``
- ``StatusCode``: ``SUCCESS``, ``INIT``, ``IN_PROGRESS``, ``ERR_INVALID_ARGS``, ``ERR_NOT_FOUND``, ``ERR_RDMA_OP``, ``ERR_BAD_STATE``, ``ERR_GPU_OP``

See :doc:`MORI-IO Guide <../MORI-IO-GUIDE>` for full API reference.

MORI-IR (Device Bitcode Integration)
--------------------------------------

Framework-agnostic device bitcode and ABI metadata for integrating MORI shmem
into GPU kernel frameworks (Triton, FlyDSL, MLIR, custom HIP, etc.).

**Imports:**

.. code-block:: python

   from mori.ir import (
       find_bitcode,
       get_bitcode_path,
       MORI_DEVICE_FUNCTIONS,
       SIGNAL_SET,
       SIGNAL_ADD,
   )

   # Triton-specific (reference backend)
   from mori.ir.triton import get_extern_libs, install_hook

**Host-side API (framework-agnostic):**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function / Constant
     - Description
   * - ``find_bitcode()``
     - Locate ``libmori_shmem_device.bc`` (auto JIT-compiled for current GPU + NIC)
   * - ``MORI_DEVICE_FUNCTIONS``
     - Dict of all device function ABI metadata (symbol, args, ret types)
   * - ``SIGNAL_SET`` / ``SIGNAL_ADD``
     - Signal operation constants for put-with-signal (values 9, 10)

**Triton-specific API (reference backend):**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``get_extern_libs()``
     - Returns dict for Triton ``extern_libs`` parameter
   * - ``install_hook()``
     - Install Triton compilation hook for automatic bitcode linking

**Key device functions** (available as ``extern "C"`` symbols in bitcode):

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Category
     - Functions
     - Description
   * - Query
     - ``my_pe()``, ``n_pes()``
     - Get PE rank and total PE count
   * - P2P
     - ``ptr_p2p()``
     - Translate pointers across PEs
   * - Put
     - ``putmem_nbi_{thread,warp,block}``
     - Non-blocking put at thread/warp/block scope
   * - Put + Signal
     - ``putmem_nbi_signal_{thread,warp,block}``
     - Put with signal notification
   * - Atomics
     - ``uint{32,64}_atomic_{add,fetch_add}_thread``
     - Atomic operations on remote memory
   * - Wait
     - ``uint{32,64}_wait_until_{equals,greater_than}``
     - Spin-wait on remote memory values
   * - Sync
     - ``quiet_thread()``, ``fence_thread()``, ``barrier_all_block()``
     - Ordering and synchronization

See :doc:`MORI-IR Guide <../MORI-IR-GUIDE>` for full device function table and integration examples.
