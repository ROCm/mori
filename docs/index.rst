MORI Documentation
==================

**MORI** (**Mo**\dular **R**\DMA **I**\nterface) is a bottom-up, modular, and composable framework for building high-performance communication applications with a strong focus on RDMA + GPU integration.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   MORI-EP-GUIDE
   MORI-SHMEM-GUIDE
   MORI-IR-GUIDE
   MORI-IO-GUIDE
   PROFILER

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   MORI-EP-BENCHMARK
   MORI-IO-BENCHMARK

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/communication
   api/profiler

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
     - Lightweight collective communication for latency-sensitive environments (coming soon)
   * - **MORI-SHMEM**
     - OpenSHMEM-style symmetric memory APIs for GPU memory and RDMA
   * - **MORI-VIZ**
     - Warp-level kernel profiler with Perfetto integration

Supported Hardware
------------------

**GPUs:** MI308X, MI300X, MI325X, MI355X (MI450X under development)

**NICs:** AMD Pollara (AINIC), Mellanox ConnectX-7, Broadcom Thor2 (Volcano under development)

Quick Links
-----------

* **GitHub**: https://github.com/ROCm/mori
* **Issues**: https://github.com/ROCm/mori/issues

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
