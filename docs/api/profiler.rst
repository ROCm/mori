MORI-VIZ Kernel Profiler
========================

MORI-VIZ is a warp-level GPU kernel profiler with Perfetto integration. It captures fine-grained timing data from dispatch/combine kernels.

Requirements
------------

Build MORI with profiling enabled:

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
       trace_buffer = mori.cpp.get_debug_time_buf(op._handle)
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
     - Raw trace buffer from ``mori.cpp.get_debug_time_buf(handle)``
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
     - Get raw trace buffer from an EP handle
   * - ``get_debug_time_offset(handle)``
     - Get current write offset in the trace buffer
   * - ``get_cur_device_wall_clock_freq_mhz()``
     - Get GPU wall clock frequency in MHz

See `PROFILER.md <../PROFILER.md>`_ for full profiler documentation.
