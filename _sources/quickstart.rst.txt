Quickstart
==========

Start with :doc:`installation` and choose an API using :doc:`ep_backends`.
The snippets below explain the API flow; variables such as ``input_tokens``,
``weights`` and peer descriptors are application-provided, so these are not
standalone programs.

Runnable checks
---------------

From a source checkout with MORI installed, use the maintained examples and tests:

.. code-block:: bash

   # Legacy EP correctness; requires eight local GPUs and pytest.
   pytest tests/python/ops/test_dispatch_combine_intranode.py -q

   # EPv2 identity-expert correctness; eight GPUs, HIP backend (no FlyDSL needed).
   MORI_V2_KERNEL_BACKEND=hip pytest -q \
       'tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_intranode.py::test_dispatch_combine_v2_intranode[bf16-gather]'

   # SHMEM device integration; two GPUs and a compatible Triton installation.
   torchrun --standalone --nproc_per_node=2 examples/shmem/ir/test_triton_shmem.py

The test programs construct inputs, initialize ranks, check results and clean up.
The EPv2 pytest wrapper also checks the runner's printed PASS/FAIL results; a zero
exit code from the underlying ``test_op.py`` alone is not a correctness gate.
Run them on an allocated node without competing jobs. For two-node EPv2,
including its preview restrictions, see :doc:`ep_backends`.

API sketches
------------

MORI-EP: Dispatch and Combine
------------------------------

Route tokens to MoE experts and combine results back:

.. code-block:: python

   import os, torch, torch.distributed as dist
   import mori

   os.environ["MORI_SHMEM_HEAP_SIZE"] = "6G"

   def run_ep(rank, world_size):
       torch.cuda.set_device(rank)
       dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
       world_group = dist.group.WORLD
       torch._C._distributed_c10d._register_process_group("default", world_group)
       mori.shmem.shmem_torch_process_group_init("default")

       config = mori.ops.EpDispatchCombineConfig(
           data_type=torch.bfloat16, rank=rank, world_size=world_size,
           hidden_dim=7168, scale_dim=0,
           scale_type_size=torch.tensor([], dtype=torch.float8_e4m3fnuz).element_size(),
           max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
           max_num_inp_token_per_rank=4096,
           num_experts_per_rank=32, num_experts_per_token=8,
       )

       op = mori.ops.EpDispatchCombineOp(config)

       # Dispatch tokens to experts
       dispatch_out, dispatch_w, dispatch_s, dispatch_idx, recv_count = \
           op.dispatch(input_tokens, weights, scales, expert_indices)

       # ... run expert computation ...

       # Combine results back
       combine_out, combine_w = op.combine(expert_output, dispatch_w, expert_indices, call_reset=True)

       mori.shmem.shmem_finalize()
       dist.destroy_process_group()

See the :doc:`MORI-EP Guide <MORI-EP-GUIDE>` for the full API reference.

MORI-IO: Point-to-Point Transfers
-----------------------------------

Transfer GPU memory between nodes via RDMA:

.. code-block:: python

   from mori.io import IOEngine, IOEngineConfig, BackendType, RdmaBackendConfig

   config = IOEngineConfig(host="10.0.0.1", port=8080)
   engine = IOEngine(key="node0", config=config)
   engine.create_backend(BackendType.RDMA, RdmaBackendConfig(qp_per_transfer=1))

   # Register memory
   mem_desc = engine.register_torch_tensor(gpu_tensor)

   # Exchange engine descriptors between nodes, then transfer
   uid = engine.allocate_transfer_uid()
   status = engine.write(local_mem, 0, remote_mem, 0, size, uid)
   status.Wait()

See :doc:`MORI-IO Guide <MORI-IO-GUIDE>` for architecture and full API.

MORI Shmem: Symmetric Memory
------------------------------

Allocate GPU memory accessible across all ranks:

.. code-block:: python

   import mori

   mori.shmem.shmem_torch_process_group_init("default")

   my_pe = mori.shmem.shmem_mype()
   ptr = mori.shmem.shmem_malloc(1024 * 1024)
   remote_ptr = mori.shmem.shmem_ptr_p2p(ptr, my_pe, dest_pe=1)

   mori.shmem.shmem_barrier_all()
   mori.shmem.shmem_free(ptr)
   mori.shmem.shmem_finalize()

See the :doc:`Shmem Guide <MORI-SHMEM-GUIDE>` for full API reference.

MORI-IR: Device Bitcode for GPU Kernels
-----------------------------------------

Use MORI shmem device functions inside Triton (or any LLVM-based) kernels:

.. code-block:: python

   from mori.ir import find_bitcode, MORI_DEVICE_FUNCTIONS
   from mori.ir.triton import get_extern_libs, install_hook
   import mori.ir.triton as mori_shmem_device

   # Locate bitcode (auto JIT-compiled for current GPU + NIC)
   bc_path = find_bitcode()

   # Triton: install hook and use device functions in kernels
   install_hook()

   @triton.jit
   def my_kernel(buf_ptr, BLOCK: tl.constexpr):
       pe = mori_shmem_device.my_pe()
       remote = mori_shmem_device.ptr_p2p(
           buf_ptr.to(tl.uint64, bitcast=True), pe, (pe + 1) % mori_shmem_device.n_pes()
       )
       # ... read/write remote memory ...

   my_kernel[(grid,)](buf, BLOCK=1024, extern_libs=get_extern_libs())

See :doc:`MORI-IR Guide <MORI-IR-GUIDE>` for full device function table and non-Triton integration.

Profiling with MORI-VIZ
-------------------------

Capture warp-level kernel traces (build with ``ENABLE_PROFILER=ON``):

.. code-block:: python

   from mori.kernel_profiler import export_to_perfetto

   # After running dispatch/combine:
   trace_buffer = mori.cpp.get_debug_time_buf(op._handle)
   export_to_perfetto(trace_buffer, "ep_trace.json")

   # Visualize at https://ui.perfetto.dev/

See :doc:`Profiler docs <PROFILER>` for details.

Next Steps
----------

* :doc:`MORI-EP Guide <MORI-EP-GUIDE>` — Full EP API reference and examples
* :doc:`Shmem Guide <MORI-SHMEM-GUIDE>` — Symmetric memory concepts and APIs
* :doc:`MORI-IR Guide <MORI-IR-GUIDE>` — Device bitcode integration for Triton
* :doc:`MORI-IO Guide <MORI-IO-GUIDE>` — IO architecture and Python API
* :doc:`Profiler <PROFILER>` — Warp-level kernel profiler
