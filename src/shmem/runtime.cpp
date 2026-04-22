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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// Shmem runtime APIs: module management, barriers, query functions,
// and GpuStates management (shared with init.cpp).

#include <cassert>

#include "hip/hip_runtime_api.h"
#include "mori/shmem/internal.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "mori/utils/hip_helper.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                  JIT Module & GpuStates Management                            */
/* ---------------------------------------------------------------------------------------------- */

using GpuStatesAddrProvider = void* (*)();
static GpuStatesAddrProvider s_gpuStatesAddrProvider = nullptr;

using BarrierLauncher = void (*)(hipStream_t);
static BarrierLauncher s_staticBarrierLauncher = nullptr;

void RegisterGpuStatesAddrProvider(GpuStatesAddrProvider provider) {
  s_gpuStatesAddrProvider = provider;
}

void RegisterBarrierLauncher(BarrierLauncher launcher) { s_staticBarrierLauncher = launcher; }

int LoadShmemModule(const char* hsaco_path) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  ModuleStates& ms = states->moduleStates;

  if (ms.module != nullptr) return 0;
  hipError_t err = hipModuleLoad(&ms.module, hsaco_path);
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("Failed to load shmem module from {}: {}", hsaco_path, hipGetErrorString(err));
    return -1;
  }
  err = hipModuleGetGlobal(reinterpret_cast<hipDeviceptr_t*>(&ms.gpuStatesPtr), nullptr, ms.module,
                           "_ZN4mori5shmem15globalGpuStatesE");
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("globalGpuStates symbol not found in shmem module: {}",
                     hipGetErrorString(err));
    return -1;
  }
  err = hipModuleGetFunction(&ms.barrierFunc, ms.module, "mori_shmem_barrier_all_block");
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("mori_shmem_barrier_all_block not found in shmem module: {}",
                     hipGetErrorString(err));
    return -1;
  }
  MORI_SHMEM_TRACE("Loaded shmem JIT module: globalGpuStates={:p}, barrier={:p}",
                   (void*)ms.gpuStatesPtr, (void*)ms.barrierFunc);
  return 0;
}

void CopyGpuStatesToDevice(ShmemStates* states) {
  const GpuStates* gpuStates = &states->gpuStates;
  ModuleStates& ms = states->moduleStates;

  if (ms.gpuStatesPtr != nullptr) {
    MORI_SHMEM_TRACE("Copying GpuStates to JIT module globalGpuStates ({:p})",
                     (void*)ms.gpuStatesPtr);
    HIP_RUNTIME_CHECK(
        hipMemcpy(ms.gpuStatesPtr, gpuStates, sizeof(GpuStates), hipMemcpyHostToDevice));
  }

  if (s_gpuStatesAddrProvider != nullptr) {
    void* staticAddr = s_gpuStatesAddrProvider();
    if (staticAddr != nullptr) {
      MORI_SHMEM_TRACE("Copying GpuStates to static globalGpuStates ({:p})", staticAddr);
      HIP_RUNTIME_CHECK(hipMemcpy(staticAddr, gpuStates, sizeof(GpuStates), hipMemcpyHostToDevice));
    }
  }

  MORI_SHMEM_TRACE("Successfully copied GpuStates to device (rank={}, worldSize={})",
                   gpuStates->rank, gpuStates->worldSize);
}

void FinalizeRuntime(ShmemStates* states) {
  ModuleStates& ms = states->moduleStates;
  if (ms.module != nullptr) {
    hipModuleUnload(ms.module);
    ms.module = nullptr;
    ms.gpuStatesPtr = nullptr;
    ms.barrierFunc = nullptr;
  }
  states->gpuStates = {};
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Module Initialization                                    */
/* ---------------------------------------------------------------------------------------------- */

int ShmemModuleInit(void* hipModule) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  hipModule_t module = static_cast<hipModule_t>(hipModule);
  GpuStates* moduleGlobalGpuStatesAddr = nullptr;

  hipError_t err = hipModuleGetGlobal(reinterpret_cast<hipDeviceptr_t*>(&moduleGlobalGpuStatesAddr),
                                      nullptr, module, "_ZN4mori5shmem15globalGpuStatesE");

  if (err != hipSuccess) {
    (void)hipGetLastError();
    MORI_SHMEM_TRACE("Module does not contain globalGpuStates symbol ({}), skipping init",
                     hipGetErrorString(err));
    return -1;
  }

  MORI_SHMEM_TRACE("Module globalGpuStates address: {:p} (JIT module address: {:p})",
                   (void*)moduleGlobalGpuStatesAddr, (void*)states->moduleStates.gpuStatesPtr);

  HIP_RUNTIME_CHECK(hipMemcpy(moduleGlobalGpuStatesAddr, &states->gpuStates, sizeof(GpuStates),
                              hipMemcpyHostToDevice));

  MORI_SHMEM_TRACE("Successfully initialized globalGpuStates in module (rank={}, worldSize={})",
                   states->gpuStates.rank, states->gpuStates.worldSize);

  return 0;
}

int CopyGpuStatesToSymbol(void* deviceSymbolAddr) {
  if (deviceSymbolAddr == nullptr) return -1;
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  HIP_RUNTIME_CHECK(
      hipMemcpy(deviceSymbolAddr, &states->gpuStates, sizeof(GpuStates), hipMemcpyHostToDevice));
  return 0;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Query APIs                                               */
/* ---------------------------------------------------------------------------------------------- */

int ShmemMyPe() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  return states->bootStates->rank;
}

int ShmemNPes() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  return states->bootStates->worldSize;
}

int ShmemNumQpPerPe() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  return states->rdmaStates->commContext->GetNumQpPerPe();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                      Barrier APIs                                             */
/* ---------------------------------------------------------------------------------------------- */

void ShmemBarrierAll() {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  MORI_SHMEM_TRACE("PE {} entering barrier", states->bootStates->rank);
  states->bootStates->bootNet->Barrier();
  MORI_SHMEM_TRACE("PE {} exiting barrier", states->bootStates->rank);
}

void ShmemBarrierOnStream(hipStream_t stream) {
  ShmemStates* states = ShmemStatesSingleton::GetInstance();
  states->CheckStatusValid();

  MORI_SHMEM_TRACE("PE {} launching device barrier on stream", states->bootStates->rank);

  if (states->moduleStates.barrierFunc != nullptr) {
    hipError_t err = hipModuleLaunchKernel(states->moduleStates.barrierFunc, 1, 1, 1, 1, 1, 1, 0,
                                           stream, nullptr, nullptr);
    assert(err == hipSuccess && "ShmemBarrierOnStream launch failed");
  } else if (s_staticBarrierLauncher != nullptr) {
    s_staticBarrierLauncher(stream);
  } else {
    MORI_SHMEM_ERROR(
        "ShmemBarrierOnStream: no barrier kernel available. "
        "Load JIT shmem module (Python) or include shmem.hpp (C++ hipcc).");
    assert(false);
  }
}

}  // namespace shmem
}  // namespace mori
