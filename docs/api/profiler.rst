MORI-VIZ Kernel Profiler
========================

MORI-VIZ is a warp-level GPU kernel profiler with Perfetto integration. It captures fine-grained timing data from dispatch/combine kernels.

Requirements
------------

Build MORI with profiling enabled, and keep ``ENABLE_PROFILER=ON`` set when you
run, so the JIT compiles the instrumented kernels:

.. code-block:: bash

   ENABLE_PROFILER=ON pip install .

Usage
-----

**Capture and export traces:**

.. code-block:: python

   import mori
   from mori.kernel_profiler import export_to_perfetto

   # Run dispatch/combine operations first, then:
   if hasattr(mori.cpp, "get_debug_time_buf"):
       trace_buffer = op.get_debug_time_buf()
       export_to_perfetto(trace_buffer, "ep_trace.json")

**Visualize:** Open the JSON file at `ui.perfetto.dev <https://ui.perfetto.dev/>`_.

export_to_perfetto
------------------

.. code-block:: python

   export_to_perfetto(
       trace_buffer,
       filename="trace.json",
       slot_map=None,
       gpu_freq_ghz=None,
       validate_pairs=True,
       sanitize_orphans=True,
   )

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``trace_buffer``
     -
     - The trace buffer tensor from ``op.get_debug_time_buf()``
   * - ``filename``
     - ``"trace.json"``
     - Output file path (Perfetto JSON format)
   * - ``slot_map``
     - ``None``
     - Custom slot name mapping (uses built-in profiler slots if None)
   * - ``gpu_freq_ghz``
     - ``None``
     - GPU clock frequency; auto-detected via ``get_cur_device_wall_clock_freq_mhz()`` if None
   * - ``validate_pairs``
     - ``True``
     - Validate begin/end event pairs
   * - ``sanitize_orphans``
     - ``True``
     - Remove unpaired events

C++ Profiler APIs
-----------------

Available via ``mori.cpp`` when built with ``ENABLE_PROFILER=ON``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``get_debug_time_buf(handle)``
     - ``(pointer, length)`` of an EP handle's trace buffer. Use
       ``op.get_debug_time_buf()`` instead, which wraps it as a ``torch.Tensor``
   * - ``get_debug_time_offset(handle)``
     - ``(pointer, length)`` of the per-warp write offsets. Use
       ``op.get_debug_time_offset()`` instead, which wraps it as a ``torch.Tensor``
   * - ``get_cur_device_wall_clock_freq_mhz()``
     - Get GPU wall clock frequency in MHz

See :doc:`PROFILER.md <../PROFILER>` for full profiler documentation.
