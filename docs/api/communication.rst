Communication APIs
==================

MORI's communication layers, each targeting a different use case.

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
   * - ``IntraNodeLL``
     - 5
     - Single-node low-latency, for small token counts

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

SDMA queues are shared per GPU pair by all ops in a process. When two ops
interleave on one stream (single-stream two-batch overlap), set
``async_ll_own_sdma_queue=True`` on both so a receive does not wait for the
other op's send. When the ops run on separate streams the option is not a
clear win: a two-op microbenchmark gets slower, while DeepSeek-V4 prefill with
two-stream two-batch overlap got faster. It is off by default; measure your
workload before turning it on there. Only the ``"sdma"`` transport uses it.

On a single gfx950 or gfx1250 node, the send phase now posts each payload followed by
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
     - ``1`` on single-node gfx950 and gfx1250 SDMA; ``0`` elsewhere. Posts
       notifications in the send phase. Other architectures can opt in with
       ``1``.
   * - ``MORI_ASYNCLL_SKIP_RESET``
     - No longer has an effect. It skipped the host-side ``total_recv`` reset
       before each dispatch; that reset has been removed for every kernel
       family (the dispatch kernels now produce the count themselves), and
       AsyncLL's receive copy always assigned it. Still accepted, so existing
       settings do not fail.
   * - ``MORI_ASYNCLL_FUSED_RECV``
     - ``0``. Set ``1`` to fold receive waits into the copy passes. One wave
       per block waits, then an acquire and block barrier release the copy waves.
   * - ``MORI_ASYNCLL_FUSED_COMBINE_RECV``
     - ``1``. With ``MORI_ASYNCLL_FUSED_RECV=1``, ``0`` keeps combine's receive
       in two passes, a small wait pass and then the copy on the whole device,
       while dispatch's receive stays fused.
   * - ``MORI_ASYNCLL_FUSED_COMBINE_RECV_TOKENS``
     - ``0``. With ``MORI_ASYNCLL_FUSED_COMBINE_RECV=0``, combine calls of at
       most this many tokens still take the fused receive.
   * - ``MORI_ASYNCLL_FUSED_RECV_BLOCKS``
     - ``0``, meaning half the compute units this process may use: the
       device's, or its share under ``HSA_CU_MASK``. Grid ceiling for the
       fused receive copies; ignored unless ``MORI_ASYNCLL_FUSED_RECV=1``.
       The former name, ``MORI_ASYNCLL_FUSED_RECV_CTAS``, is still accepted.
   * - ``MORI_ASYNCLL_DIRECT``
     - Unset: on for an op on gfx950 without a quantized combine (the
       segmented layout directly, the compact one through
       ``MORI_ASYNCLL_COMPACT_RESERVE``) and for a segmented op on gfx1250, off
       otherwise. ``1`` turns it on
       (also on gfx942), ``0`` off, which also turns compact reserve off. Direct sources have the copy engines read the tokens where the
       caller left them: dispatch posts each peer's rows from the input, one
       packet per run of adjacent source tokens (and, in the segmented layout,
       one more for the run's scale rows), and combine posts each peer's rows
       from the expert output, so neither send phase gathers them first. The
       dispatch input and scales and the combine input are read until the
       matching ``dispatch_recv`` / ``combine_recv``: do not write them in between,
       and do not pass a tensor this op returns from a receive as the dispatch
       input. The op's arena then leaves out the send staging the engines no
       longer read, about a third of it, so every rank must use the same
       value. Needs early notification; ignored
       without it. ``1`` is refused with a quantized combine (``quant_type``),
       which is packed on the send side. gfx942, gfx950 and gfx1250 only.
   * - ``MORI_ASYNCLL_FUSE_META``
     - ``1``. With direct sources (``MORI_ASYNCLL_DIRECT``) on an op whose
       ``async_ll_recv_layout`` resolves to ``"segmented"``, slot assignment
       also writes the metadata rows, so a segmented round's ``dispatch_send``
       runs two passes instead of three; compact reserve needs it. ``0`` keeps
       the separate metadata pass, and on a compact op also turns compact
       reserve off, which changes the arena, so every rank must use the same
       value. Otherwise ignored without direct sources.
   * - ``MORI_ASYNCLL_COMPACT_RESERVE``
     - Applies to a compact op with early notification (single-node SDMA);
       segmented, LSA and multi-node ops ignore it. Unset: on for such an op on
       gfx950 without a ``quant_type`` unless ``MORI_ASYNCLL_DIRECT`` or
       ``MORI_ASYNCLL_FUSE_META`` is ``0``; ``0`` turns it off; ``1`` requires
       it, and is refused there on another arch, with a ``quant_type``, or with
       either of those set to ``0``. Each source reserves its rows'
       range in every receiver's recv views with one remote atomic per peer,
       and the copy engines place rows, scales and metadata there, so
       ``dispatch_recv`` copies nothing. Direct sources and fused metadata
       come with it. The views keep the compact contract, a dense prefix of
       ``total_recv`` rows located by the reverse map, but sources appear in
       arrival order (each source's rows contiguous, in slot order), and since
       peers write them the views are valid only until this rank's
       ``combine_send``. A combine input inside the recv views (an expert run
       in place on them) is first copied to ``combine_in_view()``; any other
       tensor, or ``combine_in_view()`` itself, is sent in place. The arena
       leaves out the dispatch landing, so every rank must use the same value.
   * - ``MORI_ASYNCLL_SEND_BLOCKS``
     - ``0``, meaning the whole device. Grid ceiling for the send-side copy
       passes: slot assignment and the dispatch and combine gathers (not the
       LSA transport's fused dispatch gather). A positive value is rounded down
       to a multiple of the world size, and up to the world size if smaller; it
       never widens a pass's default grid. With direct sources and
       fused metadata (``MORI_ASYNCLL_FUSE_META``, on by default) in the
       segmented layout, or with compact reserve, it also
       sets the send transfer's blocks per peer,
       ceiling / world size (one when 0), up to the device; on gfx1250 the
       transfer keeps one block per peer.
   * - ``MORI_ASYNCLL_RECV_BLOCKS``
     - ``0``, meaning the whole device. Grid ceiling for the receive-side copy
       passes: the dispatch and combine receive copies, in either receive
       layout. Rounded like ``MORI_ASYNCLL_SEND_BLOCKS``. The segmented
       ``dispatch_recv``, which copies no tokens, already runs
       on at most two rows per thread of its largest round, world size times
       the largest segment stride (16 blocks of 1024 threads at EP8 and 4096
       tokens per rank); the ceiling only lowers that. With
       ``MORI_ASYNCLL_FUSED_RECV=1`` the lower of this and the fused receive
       ceiling (``MORI_ASYNCLL_FUSED_RECV_BLOCKS``, half the usable compute units
       by default) applies.
   * - ``MORI_ASYNCLL_SEND_BLOCKS_SCHEDULE``,
       ``MORI_ASYNCLL_RECV_BLOCKS_SCHEDULE``
     - Unset: one ceiling per op, from the two variables above. Set to
       ``tokens:blocks`` pairs, for example ``128:64,0:128``, to pick the
       ceiling per call: the first pair with T <= ``tokens`` applies,
       ``tokens`` 0 matches any T, and a T past every bound keeps the single
       ceiling. The send schedule uses the call's token count; the receive
       schedule uses the matching send's token count, or the segment stride
       for a segmented ``dispatch_recv``. ``blocks`` follows the rounding above
       and 0 means no ceiling. All entries share one compiled module.

``MORI_ASYNCLL_EARLY_NOTIFY``, ``MORI_ASYNCLL_SKIP_RESET``,
``MORI_ASYNCLL_FUSED_RECV``, ``MORI_ASYNCLL_FUSED_COMBINE_RECV``,
``MORI_ASYNCLL_DIRECT``, ``MORI_ASYNCLL_FUSE_META`` and
``MORI_ASYNCLL_COMPACT_RESERVE`` accept ``0`` or ``1``; the ``*_BLOCKS``
variables and ``MORI_ASYNCLL_FUSED_COMBINE_RECV_TOKENS`` accept a non-negative
integer and the schedules a comma-separated list of ``tokens:blocks`` pairs.
Skipping the reset or fusing receive requires early notification. Set the first
three to ``0`` for the legacy sequence.
LSA and multi-node ops retain their existing protocol and ignore the SDMA
options; ``MORI_ASYNCLL_SEND_BLOCKS`` and ``MORI_ASYNCLL_RECV_BLOCKS`` apply to
every AsyncLL op. The resolved notification mode is part of every native Plan's
configuration and generated source, so different ops can coexist without
changing the protocol of an already constructed Plan.

For two-batch overlap on the fewest compute units on gfx950, either receive
layout works: the segmented one turns on direct sources by default, and the
compact one compact reserve, which runs the same send and combine passes and
keeps the dense compact views (on gfx942 use the segmented layout with
``MORI_ASYNCLL_DIRECT=1``). Add ``MORI_ASYNCLL_FUSED_RECV=1`` when
both micro-batches run on one stream (see below), with schedules such as
``MORI_ASYNCLL_SEND_BLOCKS_SCHEDULE=1024:8,0:32`` and
``MORI_ASYNCLL_RECV_BLOCKS_SCHEDULE=128:64,0:128``. Both send phases then run on
eight blocks, the dispatch receive only marks the unused rows of each segment
on its own small grid (compact reserve: one block, nothing to mark), and the
combine receive, the one phase whose time follows
its block count, is sized by the token count. The send schedule assumes direct
sources: without them the send passes gather the token rows and need the whole
device.

On gfx1250 use the segmented layout, which turns on direct sources, with
``MORI_ASYNCLL_SEND_BLOCKS_SCHEDULE=128:4,1024:32,4096:64,0:0``: slot assignment
needs the whole device only past 4096 tokens. When both micro-batches run on one
stream, add ``MORI_ASYNCLL_FUSED_RECV=1``, ``MORI_ASYNCLL_FUSED_COMBINE_RECV=0`` and
``MORI_ASYNCLL_FUSED_COMBINE_RECV_TOKENS=256``. Past a few hundred tokens the
combine receive needs the whole device there, its reduce running at the memory rate
on every compute unit, so it stays unfused; below that one pass is faster.

The default targets single-node gfx950 and gfx1250 overlap. On MI355X (gfx950,
EP8, hidden 7168, top-6), early notification beat the legacy sequence at every
token count from 128 to 4096: by 2-5% on one stream and 3-11% with two
overlapping streams. Receive fusion remains opt-in because its additional
benefit depends on the token count and compute window.

Receive fusion is a single-stream optimisation. A fused receive copy waits for
its peers inside every block, so its whole grid stays resident for the wait. When
two micro-batches overlap on two streams (two-batch overlap), that grid can
crowd out the send pass the other stream must run before any peer's wait can
end, and the ranks deadlock. The grid ceiling makes this unlikely but cannot
rule it out: it is sized from the compute units the process may use, not from
those another stream, kernel or process is occupying at the time.

Do not combine receive fusion with micro-batches on two streams, especially
when the device may be shared. On MI355X, two streams gained nothing from
fusion over early notification, so dual-stream callers should use early
notification alone, which is the gfx950 default.

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
     - Create an RDMA, XGMI, FABRIC or OFI backend (TCP only in simulation mode)
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

- ``BackendType``: ``Unknown``, ``XGMI``, ``RDMA``, ``TCP`` (simulation mode only),
  ``FABRIC``, ``OFI`` (needs MORI built with libfabric)
- ``StatusCode``: ``SUCCESS``, ``INIT``, ``IN_PROGRESS``, ``ERR_INVALID_ARGS``, ``ERR_NOT_FOUND``, ``ERR_RDMA_OP``, ``ERR_BAD_STATE``, ``ERR_GPU_OP``

See :doc:`MORI-IO Guide <../MORI-IO-GUIDE>` for full API reference.

Device Bitcode Integration (mori.ir)
-------------------------------------

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

See the :doc:`Shmem Guide <../MORI-SHMEM-GUIDE>` for the full device function table and integration examples.
