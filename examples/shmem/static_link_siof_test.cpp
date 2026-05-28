// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// ============================================================================
// Static-link SIOF reproducer for mori_shmem's GpuStatesAddrProvider registry.
// ============================================================================

#include <cstdio>
#include <cstdlib>

// Pulls in `__device__ globalGpuStates`, `_getGpuStatesAddr`, and
// `_s_gpuStatesRegistrar` from the static-init block at shmem.hpp:62-76.
// Must be compiled as HIP (the static-init block is gated on __HIPCC__ /
// __HIP__ / __CUDACC__) — see the LANGUAGE HIP property on this source file
// in examples/CMakeLists.txt.
#include "mori/shmem/shmem.hpp"

// Diagnostic accessor for the GpuStates addr-provider registry. Used by
// examples/shmem/static_link_siof_test.cpp to detect a Static Initialization
// Order Fiasco between user-TU `_s_gpuStatesRegistrar` ctors (emitted by
// `mori/shmem/shmem.hpp`) and the registry's own backing storage in
// `src/shmem/runtime.cpp`. Always safe to call; returns 0 if no consumer has
// registered yet.
namespace mori::shmem {
size_t GetGpuStatesAddrProviderCount();
}

int main(int /*argc*/, char* /*argv*/[]) {
  fprintf(stderr, "[SIOF-TEST] static_link_siof_test starting\n");

  // Address of the registrar emitted in *this* TU. Printing it confirms the
  // static-init block was compiled (i.e. the file was processed as HIP).
  fprintf(stderr, "[SIOF-TEST] this TU's _s_gpuStatesRegistrar = %p (ctor should have run)\n",
          static_cast<const void*>(&mori::shmem::_static_init::_s_gpuStatesRegistrar));

  const size_t count = mori::shmem::GetGpuStatesAddrProviderCount();
  fprintf(stderr, "[SIOF-TEST] GpuStatesAddrProvider registry size = %zu\n", count);

  if (count == 0) {
    fprintf(stderr, "\n");
    fprintf(stderr, "[SIOF-TEST] FAIL: registry is empty.\n");
    fprintf(stderr, "\n");
    fprintf(stderr,
            "  This TU's `_s_gpuStatesRegistrar` ctor either did not run, or ran\n"
            "  BEFORE the registry's std::vector ctor in runtime.cpp and its push\n"
            "  was silently erased — the classic Static Initialization Order Fiasco.\n"
            "\n"
            "  Downstream symptom: `CopyGpuStatesToDevice` iterates an empty list,\n"
            "  never populates device-side `globalGpuStates`, and the first kernel\n"
            "  that dereferences its fields faults with GPU illegal memory access.\n"
            "\n"
            "  Fix: wrap the registry in a Meyer's singleton in src/shmem/runtime.cpp\n"
            "  (`GpuStatesProviders()` function-local static). See the header comment\n"
            "  of this file for the full chain.\n");
    return 1;
  }

  fprintf(stderr, "\n[SIOF-TEST] PASS: registry contains %zu provider(s); SIOF not present.\n",
          count);
  return 0;
}
