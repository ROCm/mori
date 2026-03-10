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

See `MORI-EP Guide <../MORI-EP-GUIDE.md>`_ for full API reference.

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

See `Shmem Guide <../MORI-SHMEM-GUIDE.md>`_ for full API reference.

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

See `MORI-IO Introduction <../MORI-IO-INTRO.md>`_ for full API reference.
