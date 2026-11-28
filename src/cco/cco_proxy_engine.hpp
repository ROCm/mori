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
//
//
// cco_proxy_engine.hpp — host side of ccoProxy (see include/mori/cco/cco_scale_out.hpp, section 6).
//
// Plain C++ interface for cco_init.cpp (a CXX TU); the engine is a HIP TU because
// the KIWI headers are HIP. Builds without it link cco_proxy_engine_stub.cpp.
#pragma once

#include <cstddef>
#include <vector>

#include "mori/cco/cco.hpp"

namespace mori {
namespace application {
class Context;
class BootstrapNetwork;
}  // namespace application

namespace cco {

// Resolved and validated by ccoDevCommCreate.
struct ccoProxyEngineConfig {
  int numSpscQueues{0};  // KIWI SPSC device-to-host queues
  int numMpscQueues{0};  // KIWI MPSC device-to-host queues
  int numThreads{1};     // host proxy threads, 1 .. queue count
  // RC QPs per (thread, peer); a thread creates min(qpsPerPeer, its queue count).
  int qpsPerPeer{1};
  size_t queueCapacity{0};     // ring lines per queue; the engine checks it is a power of two
  int signalCount{0};          // user signal slots (gdaSignalCount); slot signals must be below it
  std::vector<bool> peerMask;  // [worldSize] peers that get a QP per proxy thread
};

class ccoProxyEngine;

// False in builds without the engine (BUILD_CCO_PROXY=OFF or no gfx942/gfx950 target).
bool ccoProxyEngineBuiltIn();

// Collective over bootNet; every rank passes the same queue, thread and QP counts
// and signalCount (ccoDevCommCreate checks). Call with the comm's HIP device
// current. Fills `out` except connType and engineId, or returns nullptr on every
// rank if any rank's setup or connect fails (a bootstrap failure is local).
// Test-only hook: MORI_CCO_PROXY_TEST_FAIL_CONNECT=<rank> fails that rank's connect.
ccoProxyEngine* ccoProxyEngineCreate(application::Context* ctx,
                                     application::BootstrapNetwork* bootNet,
                                     const ccoProxyEngineConfig& config, ccoProxyContext* out);

// 1 and *err filled once the engine has failed (sticky), else 0. Lock-free and
// HIP-free: safe from any thread while kernels run.
int ccoProxyEngineGetError(const ccoProxyEngine* engine, ccoDevCommError* err);

// Local: drains and stops the threads (a failed engine drops queued ops and does
// not wait for in-flight WRs), then frees QPs, queues and device memory. No kernel
// may still use the engine. Null is a no-op. Never throws.
void ccoProxyEngineDestroy(ccoProxyEngine* engine) noexcept;

}  // namespace cco
}  // namespace mori
