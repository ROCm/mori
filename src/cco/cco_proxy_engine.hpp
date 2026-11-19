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
// cco_proxy_engine.hpp — host side of ccoProxy (see include/mori/cco/cco_proxy.hpp).
//
// Plain C++ interface so the host control plane (cco_init.cpp, compiled as CXX
// where ccoComm is visible) can drive it; the implementation is a HIP TU
// because the KIWI headers are HIP.
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

struct ccoProxyEngineConfig {
  int numSpscQueues{0};        // KIWI SPSC device-to-host queues
  int numMpscQueues{0};        // KIWI MPSC device-to-host queues
  int numThreads{0};           // host proxy threads; 0 = 4. Capped at the queue count.
  size_t queueCapacity{0};     // ring lines (power of two); 0 = default
  int qpsPerPeer{1};           // RC QPs per (thread, peer); capped at a thread's queue count
  int signalCount{0};          // user signal slots (gdaSignalCount); slot signals must be below it
  std::vector<bool> peerMask;  // [worldSize] peers that get a QP per proxy thread
};

class ccoProxyEngine;

// Collective over `bootNet`; every rank must pass the same queue, thread and QP
// counts and signalCount (ccoDevCommCreate checks that first). Call with the comm's HIP device
// current (the queues and flush counters are allocated on it). Builds the
// queues, the host-posted RC QPs (on ctx's RDMA device and PD, so cco window
// MRs are valid on them) and starts the proxy threads. Fills `out` on success;
// returns nullptr on failure. A failure in any rank's local setup (before the
// QP exchange) or in connecting and starting makes every rank fail.
// MORI_CCO_PROXY_TEST_FAIL_CONNECT=<rank> makes that rank fail to connect (test
// hook).
ccoProxyEngine* ccoProxyEngineCreate(application::Context* ctx,
                                     application::BootstrapNetwork* bootNet,
                                     const ccoProxyEngineConfig& config, ccoProxyContext* out);

// 1 and *err filled once the engine has failed (sticky), else 0. Lock-free and
// HIP-free: safe from any thread while kernels run.
int ccoProxyEngineGetError(const ccoProxyEngine* engine, ccoDevCommError* err);

// Collective: drains and stops the proxy threads, barriers so no peer still has
// WRs in flight to our QPs, then releases QPs, queues and device copies. No
// kernel using the engine may still be running. A failed engine stops without
// waiting for its outstanding WRs. Never throws: if the barrier fails (a peer
// died), it reports that on stderr and tears down locally.
void ccoProxyEngineDestroy(ccoProxyEngine* engine) noexcept;

}  // namespace cco
}  // namespace mori
