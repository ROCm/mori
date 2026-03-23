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
#include <vector>

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
static hipModule_t s_shmemModule = nullptr;
static GpuStates* s_deviceGpuStatesAddr = nullptr;
static hipFunction_t s_barrierFunc = nullptr;

GpuStates s_hostGpuStatesCopy{};

using GpuStatesAddrProvider = void* (*)();
static std::vector<GpuStatesAddrProvider> s_gpuStatesAddrProviders;

using BarrierLauncher = void (*)(hipStream_t);
static BarrierLauncher s_staticBarrierLauncher = nullptr;

void RegisterGpuStatesAddrProvider(GpuStatesAddrProvider provider) {
  s_gpuStatesAddrProviders.push_back(provider);
}

void RegisterBarrierLauncher(BarrierLauncher launcher) { s_staticBarrierLauncher = launcher; }

int LoadShmemModule(const char* hsaco_path) {
  if (s_shmemModule != nullptr) return 0;
  hipError_t err = hipModuleLoad(&s_shmemModule, hsaco_path);
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("Failed to load shmem module from {}: {}", hsaco_path, hipGetErrorString(err));
    return -1;
  }
  err = hipModuleGetGlobal(reinterpret_cast<hipDeviceptr_t*>(&s_deviceGpuStatesAddr), nullptr,
                           s_shmemModule, "_ZN4mori5shmem15globalGpuStatesE");
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("globalGpuStates symbol not found in shmem module: {}",
                     hipGetErrorString(err));
    return -1;
  }
  err = hipModuleGetFunction(&s_barrierFunc, s_shmemModule, "mori_shmem_barrier_all_block");
  if (err != hipSuccess) {
    MORI_SHMEM_ERROR("mori_shmem_barrier_all_block not found in shmem module: {}",
                     hipGetErrorString(err));
    return -1;
  }
  MORI_SHMEM_TRACE("Loaded shmem JIT module: globalGpuStates={:p}, barrier={:p}",
                   (void*)s_deviceGpuStatesAddr, (void*)s_barrierFunc);
  return 0;
}

void CopyGpuStatesToDevice(const GpuStates* gpuStates) {
  s_hostGpuStatesCopy = *gpuStates;

  if (s_deviceGpuStatesAddr != nullptr) {
    MORI_SHMEM_TRACE("Copying GpuStates to JIT module globalGpuStates ({:p})",
                     (void*)s_deviceGpuStatesAddr);
    HIP_RUNTIME_CHECK(
        hipMemcpy(s_deviceGpuStatesAddr, gpuStates, sizeof(GpuStates), hipMemcpyHostToDevice));
  }

  for (auto& provider : s_gpuStatesAddrProviders) {
    void* staticAddr = provider();
    if (staticAddr != nullptr) {
      MORI_SHMEM_TRACE("Copying GpuStates to static globalGpuStates ({:p})", staticAddr);
      HIP_RUNTIME_CHECK(hipMemcpy(staticAddr, gpuStates, sizeof(GpuStates), hipMemcpyHostToDevice));
    }
  }

  MORI_SHMEM_TRACE("Successfully copied GpuStates to device (rank={}, worldSize={})",
                   gpuStates->rank, gpuStates->worldSize);
}

void FinalizeRuntime() {
  if (s_shmemModule) {
    hipModuleUnload(s_shmemModule);
    s_shmemModule = nullptr;
    s_deviceGpuStatesAddr = nullptr;
    s_barrierFunc = nullptr;
  }
  s_hostGpuStatesCopy = {};
  s_gpuStatesAddrProviders.clear();
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

  MORI_SHMEM_TRACE("Module globalGpuStates address: {:p} (shmem module address: {:p})",
                   (void*)moduleGlobalGpuStatesAddr, (void*)s_deviceGpuStatesAddr);

  HIP_RUNTIME_CHECK(hipMemcpy(moduleGlobalGpuStatesAddr, &s_hostGpuStatesCopy, sizeof(GpuStates),
                              hipMemcpyHostToDevice));

  MORI_SHMEM_TRACE("Successfully initialized globalGpuStates in module (rank={}, worldSize={})",
                   s_hostGpuStatesCopy.rank, s_hostGpuStatesCopy.worldSize);

  return 0;
}

int CopyGpuStatesToSymbol(void* deviceSymbolAddr) {
  if (deviceSymbolAddr == nullptr) return -1;
  HIP_RUNTIME_CHECK(
      hipMemcpy(deviceSymbolAddr, &s_hostGpuStatesCopy, sizeof(GpuStates), hipMemcpyHostToDevice));
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

  if (s_barrierFunc != nullptr) {
    hipError_t err =
        hipModuleLaunchKernel(s_barrierFunc, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr);
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
