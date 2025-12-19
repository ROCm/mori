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

The macro naming follows the pattern: `<UPPERCASE_FILENAME>_PROFILER_INIT_CONTEXT`, where `FILENAME` is the base name of your source file (e.g., `my_kernel.cpp` → `MY_KERNEL_PROFILER_INIT_CONTEXT`).

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

The profiler relies on a code generation script `tools/generate_profiler_bindings.py`. This script:
1.  Scans `src/` for `MORI_TRACE_*` usage.
2.  Extracts unique slot names.
3.  Generates C++ headers (e.g., `include/mori/profiler/.../slots.hpp`).
4.  Generates Python bindings to map enum values to strings.

This is typically integrated into your `CMakeLists.txt`. If you add new slots, simply rebuild the project, and the new slots will be available in Python.

## Python Analysis

After running your kernel and copying the debug buffer back to the host, use the Python API to export the trace.

```python
import torch
import mori
from mori.kernel_profiler import export_to_perfetto

# ... run kernel ...
# Copy debug buffer from device to host
# debug_buffer_gpu comes from your ProfilerConfig
debug_buffer_cpu = debug_buffer_gpu.cpu()

# Option 1: Auto-discover all slots (simplest, recommended)
export_to_perfetto(debug_buffer_cpu, "trace.json")

# Option 2: Use the merged ALL_PROFILER_SLOTS (explicit)
export_to_perfetto(debug_buffer_cpu, "trace.json", mori.cpp.ALL_PROFILER_SLOTS)

# Option 3: Use specific module slots (if you know which module)
export_to_perfetto(debug_buffer_cpu, "trace.json", mori.cpp.InternodeV1Slots)
```

The first option is recommended as it automatically discovers all profiler slots from your build.

### Viewing the Trace

1.  Open [ui.perfetto.dev](https://ui.perfetto.dev/) in Chrome.
2.  Click "Open trace file" and select your generated `trace.json`.
3.  You will see a timeline view with rows for each Warp, showing the instrumented spans.

## Best Practices

-   **Minimize Scope**: Keep profiled regions granular but not too small (overhead vs visibility).
-   **Conditional Compilation**: The `ENABLE_PROFILER` macro controls whether profiling code is compiled. In production builds, this is typically disabled to ensure zero overhead.
