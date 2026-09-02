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
#pragma once

#if !defined(__HIPCC__) && !defined(__CUDACC__)

#include <infiniband/verbs.h>
#include <pthread.h>

#include <atomic>
#include <vector>

#include "mori/core/transport/rdma/proxy/proxy_types.hpp"

namespace mori {
namespace core {

struct InlineBuf {
  uint64_t data[4];
};

struct ProxyQpHandle {
  ibv_qp* qp{nullptr};
  ibv_cq* cq{nullptr};
  uint32_t lkey_override{0};
  uint32_t rkey_override{0};
  void* recv_buf{nullptr};
  uint32_t recv_lkey{0};
  uint32_t recv_count{0};
  bool use_native_atomics{false};
};

class ProxyThread {
 public:
  ProxyThread() = default;
  ~ProxyThread();

  void Init(ProxyRing* ring, std::vector<ProxyQpHandle> qps, int gpuId = 0, uintptr_t heapBase = 0,
            uintptr_t heapEnd = 0);
  void Start();
  void Shutdown();

 private:
  static void* ThreadFunc(void* arg);
  void MainLoop();
  void DrainCq(ProxyQpHandle& qph);
  bool BuildWr(volatile ProxyCmd* cmd, ProxyQpHandle& qph, ibv_send_wr& wr, ibv_sge& sge,
               uint32_t slot_id, InlineBuf& ibuf);

  ProxyRing* ring_{nullptr};
  std::vector<ProxyQpHandle> qps_;
  pthread_t thread_{};
  std::atomic<bool> running_{false};
  uint32_t next_slot_{0};
  uint64_t ops_posted_{0};
  uint64_t ops_completed_{0};
  int gpu_id_{0};
  uintptr_t heap_base_{0};
  uintptr_t heap_end_{0};
};

}  // namespace core
}  // namespace mori

#endif  // !defined(__HIPCC__) && !defined(__CUDACC__)
