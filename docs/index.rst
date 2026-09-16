MORI Documentation
==================

**MORI** (**Mo**\dular **R**\DMA **I**\nterface) is a bottom-up, modular, and composable framework for building high-performance communication applications with a strong focus on RDMA + GPU integration.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   ep_backends

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   MORI-EP-GUIDE
   MORI-SHMEM-GUIDE
   MORI-CCO-GUIDE
   MORI-IR-GUIDE
   MORI-IO-GUIDE
   MORI-UMBP-SINGLE-NODE-GUIDE
   PROFILER

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   MORI-EP-BENCHMARK
   MORI-IO-BENCHMARK
   MORI-UMBP-PD-BENCHMARK
   UMBP-TIER-MANAGEMENT-BENCHMARK
   rdma_bandwidth_utilization

.. toctree::
   :maxdepth: 1
   :caption: Developer Guides

   MORI_IR_INTEGRATION
   MORI_JIT_ARCHITECTURE
   MORI_JIT_V2_DESIGN

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/communication
   api/profiler
   api/umbp

Components
----------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Component
     - Description
   * - **MORI-EP**
     - Intra and inter-node dispatch/combine kernels for MoE Expert Parallelism
   * - **MORI-IO**
     - Point-to-point communication library for KVCache transfer via RDMA/XGMI
   * - **MORI-CCL**
     - Collective communication, including SDMA implementations; availability depends on the installed bindings
   * - **MORI-SHMEM**
     - OpenSHMEM-style symmetric memory APIs for GPU memory and RDMA
   * - **MORI-CCO**
     - Collective Communication Object: explicit-handle GPU communication (LSA / GDA / SDMA)
   * - **MORI-UMBP**
     - Unified memory and bandwidth pool with tiered storage and distributed KV access
   * - **MORI-VIZ**
     - Warp-level kernel profiler with Perfetto integration

Supported Hardware
------------------

**GPUs:** MI308X, MI300X, MI325X, MI355X (MI450X under development)

**NICs:** AMD Pollara (AINIC), Mellanox ConnectX-7, Broadcom Thor2 (Volcano under development)

Documentation versions
----------------------

The page title identifies an exact release tag or a development source commit.
Development documentation can describe features that are not in the latest
stable package. Check the `release notes <https://github.com/ROCm/mori/releases>`_
and the backend restrictions before choosing a deployment configuration.

Quick Links
-----------

* **GitHub**: https://github.com/ROCm/mori
* **Issues**: https://github.com/ROCm/mori/issues

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
