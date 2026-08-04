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

struct ProxyQpHandle {
  ibv_qp* qp{nullptr};
  ibv_cq* cq{nullptr};
  uint32_t lkey_override{0};
  uint32_t rkey_override{0};
  void* recv_buf{nullptr};
  uint32_t recv_lkey{0};
  uint32_t recv_count{0};
};

class ProxyThread {
 public:
  ProxyThread() = default;
  ~ProxyThread();

  void Init(ProxyRing* ring, std::vector<ProxyQpHandle> qps, int gpuId = 0);
  void Start();
  void Shutdown();

 private:
  static void* ThreadFunc(void* arg);
  void MainLoop();
  void DrainCq(ProxyQpHandle& qph);

  ProxyRing* ring_{nullptr};
  std::vector<ProxyQpHandle> qps_;
  pthread_t thread_{};
  std::atomic<bool> running_{false};
  uint32_t next_slot_{0};
  uint64_t ops_posted_{0};
  uint64_t ops_completed_{0};
  int gpu_id_{0};
};

}  // namespace core
}  // namespace mori

#endif  // !defined(__HIPCC__) && !defined(__CUDACC__)
