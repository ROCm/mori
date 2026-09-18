Choosing an EP API and backend
==============================

MORI exposes the existing SHMEM-based EP API and the newer CCO-based EPv2 API.
They have different configuration objects and initialization requirements.
Keep an existing deployment on its tested API unless you are explicitly
evaluating a migration.

.. list-table:: EP paths in the development source
   :header-rows: 1
   :widths: 20 20 30 30

   * - Path
     - Import / selection
     - Topology
     - Scope and restrictions
   * - Existing EP
     - ``mori.ops``
     - Intra-node and inter-node kernel types
     - SHMEM initialization; use :doc:`MORI-EP-GUIDE` for configuration,
       dispatch/combine, tuning and supported kernel types.
   * - EPv2, FlyDSL
     - ``mori.ops.dispatch_combine_v2``; ``kernel_backend="flydsl"``
     - Intra-node only
     - Gather/scatter combine, quantization, standard-MoE conversion and routing
       replay, subject to dtype and architecture restrictions below.
   * - EPv2, HIP
     - Same EPv2 import; ``kernel_backend="hip"``
     - Intra-node
     - Gather combine in BF16/FP32. Dispatch can carry already-quantized FP8/FP4
       payloads. Does not require FlyDSL.
   * - EPv2, HIP internode preview
     - HIP backend with ``gpu_per_node < world_size``
     - Multiple physical nodes over CCO/GDA
     - BF16/FP32/FP8 dispatch; no FP4, quantization, standard-MoE conversion or
       scatter combine. The communicator's node grouping must match the config.

This table describes development source, not a guarantee that all paths are
available in every published wheel. Check the installed release and its
`release notes <https://github.com/ROCm/mori/releases>`_.

EPv2 configuration and lifecycle
--------------------------------

Import the EPv2 classes explicitly:

.. code-block:: python

   from mori.ops.dispatch_combine_v2 import (
       EpDispatchCombineConfig,
       EpDispatchCombineOp,
   )

Select the backend with ``config.kernel_backend`` or
``MORI_V2_KERNEL_BACKEND``. The FlyDSL backend is the default. Importing the
package does not load either backend; construction loads the selected one.
EPv2 uses an explicit CCO communicator, described in :doc:`MORI-CCO-GUIDE`.
It does not inherit the legacy API's process-global SHMEM bootstrap contract.

Read the maintained `EPv2 package guide
<https://github.com/ROCm/mori/blob/main/python/mori/ops/dispatch_combine_v2/README.md>`_
for configuration, initialization, teardown and complete commands. It documents
the following important restrictions:

* FP4 conversion kernels require gfx950 or gfx1250; gfx942 lacks the conversion
  intrinsics. The two supported targets use different instructions (gfx950 the
  ``cvt_scalef32_pk_*_fp4`` pair-at-a-time family, gfx1250 a pack-of-8), so this
  is a per-target port rather than one shared path.
* FP8 representation differs by target: OCP E4M3 on gfx950 and E4M3FNUZ on gfx942.
* On the internode path, ``quant_type`` must be ``"none"``. The config rejects
  anything else whenever ``world_size > gpu_per_node``, independently of the
  backend -- the quantized-combine staging path there is incomplete. HIP is
  simply the only backend that has an internode path today.
* ``gpu_per_node`` determines physical node grouping. A single host cannot
  emulate the two-node internode test by changing world size alone.
* For internode, ``internode_kernel`` chooses ``"v2"``, ``"v2_ll"`` or
  ``"auto"``. Auto chooses by the current call's token count, not buffer capacity.

Validation before tuning
------------------------

Run the allocated-node checks in :doc:`quickstart` before measuring performance.
The source package also provides:

* ``tests/python/ops/dispatch_combine_v2/test_ep_backend_parity.py`` for comparing
  HIP and FlyDSL on the same inputs.
* ``tests/python/ops/dispatch_combine_v2/test_graph_capture.py`` for graph replay.
* ``tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py``
  for real two-node correctness, benchmarking, tuning and stress runs.
* ``tests/python/ops/dispatch_combine_v2/bench_ep.py``, whose benchmark points
  include an identity-expert correctness check by default.

Record the MORI/FlyDSL revisions, backend, GPU, NIC, transport, dtype, quantization
mode, shape and graph mode alongside results. A successful intranode benchmark
does not validate an internode configuration.
