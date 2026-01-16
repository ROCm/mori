# Mori Kernel Profiler

Mori provides a high-performance, low-overhead kernel profiler designed to trace GPU execution at the warp level. It allows developers to instrument C++ kernel code with minimal impact on performance and visualize the execution timelines using [Perfetto](https://ui.perfetto.dev/).

## Overview

The profiler works by logging events (Begin/End/Instant) into a per-warp circular buffer in GPU global memory. These events are then copied to the host, parsed, and converted into the Chrome Trace Event format for visualization.

Key features:
- **Low Overhead**: Writes are buffered in registers/shared memory or written directly to global memory with minimal synchronization.
- **Warp-Level Granularity**: Traces execution per warp, allowing detailed analysis of divergence and latency.
- **Automatic Binding Generation**: A helper script scans your C++ code for profiler macros and automatically generates the necessary C++ enums and Python bindings.
- **Perfetto Integration**: Tools to export traces directly to JSON for Perfetto.

## Instrumentation

### 1. Include Headers

Ensure you have the profiler headers available. The main entry point is typically:

```cpp
#include "mori/core/profiler/kernel_profiler.hpp"
```

### 2. Define Profiler Context

In your kernel or device function, you need to initialize the profiler context. It is recommended to use the generated `<FILENAME>_PROFILER_INIT_CONTEXT` macro, which handles wrapping and namespacing for you.

**Naming Convention**: All generated names are derived from the source filename (without extension). For example, if your file is `my_kernel.cpp`:

| Generated Item | Naming Pattern | Example |
|---------------|----------------|---------|
| C++ Macro | `<UPPERCASE_FILENAME>_PROFILER_INIT_CONTEXT` | `MY_KERNEL_PROFILER_INIT_CONTEXT` |
| Python Module | `mori.cpp.<CamelCaseFilename>Slots` | `mori.cpp.MyKernelSlots` |

> **Note**: For filenames with underscores like `internode_v1.cpp`, the CamelCase conversion produces `InternodeV1`, so the Python module would be `mori.cpp.InternodeV1Slots`.

```cpp
// Example: in a file named 'my_kernel.cpp'
template <typename T>
__global__ void MyKernelLaunch(MyKernelArgs<T> args) {
  // Calculate warp and lane IDs
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = globalThdId / warpSize;
  int laneId = threadIdx.x % warpSize;

  // Initialize profiler context using the auto-generated macro
  // Macro naming: <FILENAME>_PROFILER_INIT_CONTEXT
  // IMPORTANT: Must wrap with IF_ENABLE_PROFILER to avoid parameter evaluation
  // when profiling is disabled (args.profilerConfig may not exist)
  IF_ENABLE_PROFILER(
      MY_KERNEL_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId)
  )
  // Arguments:
  //   1. profiler - variable name for the profiler instance
  //   2. profilerConfig - ProfilerConfig object from kernel args
  //   3. globalWarpId - global warp ID
  //   4. laneId - lane ID within warp

  // Now you can use profiler in trace macros
  MORI_TRACE_SPAN(profiler, Slot::KernelMain);
  // ... kernel code ...
}
```

**Why `IF_ENABLE_PROFILER` is needed**: When `ENABLE_PROFILER=OFF`, the `profilerConfig` member may not exist in your args struct. Without the wrapper, the compiler would still try to evaluate `args.profilerConfig` before macro expansion, causing a compilation error. `IF_ENABLE_PROFILER` removes the entire call at preprocessor stage.

### 3. Add Trace Points

Use the `MORI_TRACE_*` macros to instrument your code. The slot names (e.g., `Slot::Compute`, `Slot::MemoryWait`) are automatically detected and generated.

**Scoped Span (RAII):**
Records `BEGIN` when created and `END` when it goes out of scope.
```cpp
{
  MORI_TRACE_SPAN(profiler, Slot::Compute);
  // ... compute intensive code ...
} // END event logged here
```

**Sequential Phases:**
Useful for loops or state machines where one phase immediately follows another.
```cpp
MORI_TRACE_SEQ(seq, profiler);

MORI_TRACE_NEXT(seq, Slot::Phase1);
// ... phase 1 code ...

MORI_TRACE_NEXT(seq, Slot::Phase2);
// ... phase 2 code ...
// Previous phase ends, new phase begins automatically
```

**Instant Events:**
Log a single point in time.
```cpp
MORI_TRACE_INSTANT(profiler, Slot::Checkpoint);
```

**Note on Slot Names**: You do not need to define `Slot::Compute` manually. The build system's code generator will find `Slot::Compute` in your usage and generate the enum for you.

## Build & Code Generation

### Enabling the Profiler

To enable profiling, set the `ENABLE_PROFILER` option before building:

```bash
# Using setup.py
ENABLE_PROFILER=ON pip install -e .

# Using CMake directly
cmake -DENABLE_PROFILER=ON -B build -S .
cmake --build build
```

### Code Generation

The profiler relies on a code generation script `tools/generate_profiler_bindings.py`. This script:
1.  Scans `src/` for `MORI_TRACE_*` usage.
2.  Extracts unique slot names.
3.  Generates C++ headers into the build directory (e.g., `build/generated/include/mori/profiler/.../slots.hpp`).
4.  Generates Python bindings to map enum values to strings.

This is integrated into CMake and runs automatically at configure time. If you add new slots, simply rebuild the project, and the new slots will be available in Python.

## Python Analysis

After running your kernel, use the Python API to export the trace.

```python
import mori
from mori.kernel_profiler import export_to_perfetto

# Create and run your operation
op = mori.ops.EpDispatchCombineOp(config)
# ... run dispatch/combine ...

# Get debug buffer from handle (only available when ENABLE_PROFILER=ON)
if hasattr(mori.cpp, "get_debug_time_buf"):
    trace_buffer = mori.cpp.get_debug_time_buf(op._handle)

    # Option 1: Auto-discover all slots (simplest, recommended)
    export_to_perfetto(trace_buffer, "trace.json")

    # Option 2: Use the merged ALL_PROFILER_SLOTS (explicit)
    export_to_perfetto(trace_buffer, "trace.json", mori.cpp.ALL_PROFILER_SLOTS)

    # Option 3: Use specific module slots (if you know which module)
    # Module name is derived from filename: internode_v1.cpp -> InternodeV1Slots
    export_to_perfetto(trace_buffer, "trace.json", mori.cpp.InternodeV1Slots)
```

The first option is recommended as it automatically discovers all profiler slots from your build.

### GPU Frequency Configuration

The `gpu_freq_ghz` parameter converts raw GPU clock cycles to microseconds. The default `1.7` is approximate for MI300X.

To get accurate timing:
1. Query the actual SCLK frequency:
   ```bash
   rocm-smi --showclocks
   ```
2. Pass the value to `export_to_perfetto`:
   ```python
   export_to_perfetto(trace_buffer, "trace.json", gpu_freq_ghz=1.9)
   ```

### Viewing the Trace

1.  Open [ui.perfetto.dev](https://ui.perfetto.dev/) in Chrome.
2.  Click "Open trace file" and select your generated `trace.json`.
3.  You will see a timeline view with rows for each Warp, showing the instrumented spans.

## Best Practices

-   **Minimize Scope**: Keep profiled regions granular but not too small (overhead vs visibility).
-   **Conditional Compilation**: The `ENABLE_PROFILER` macro controls whether profiling code is compiled. In production builds, this is typically disabled to ensure zero overhead.
-   **Buffer Limits**: Each warp can store up to 16384 events in a circular buffer. The system supports up to 512 warps per rank by default. If profiling long-running kernels or multiple iterations, clear the buffer between runs to avoid data overlap:
    ```python
    trace_buffer = mori.cpp.get_debug_time_buf(handle)
    trace_buffer.zero_()  # Clear before next profiled iteration

    # Also clear the offset buffer if needed
    if hasattr(mori.cpp, "get_debug_time_offset"):
        offset_buffer = mori.cpp.get_debug_time_offset(handle)
        offset_buffer.zero_()
    ```
-   **Single Iteration for Profiling**: When profiling, consider running only one iteration to get a clean trace without buffer wraparound.
