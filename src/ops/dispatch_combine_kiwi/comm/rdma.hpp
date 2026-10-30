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
#pragma once

#include <infiniband/verbs.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mori/application/bootstrap/base_bootstrap.hpp"
#include "mori/application/transport/rdma/rdma.hpp"

// KIWI EP proxy transport on mori's RDMA layer: mori selects the NIC, registers
// memory and connects the RC QPs; this file keeps only the proxy's post, poll
// and send-credit path.
namespace mori::kiwi::ep::rdma {

// What a peer needs to target a registered buffer.
struct rmr_t {
  uint64_t base = 0;
  uint32_t rkey = 0;
  uint32_t reserved = 0;
};

struct mr_t {
  application::RdmaMemoryRegion region;

  bool empty() const { return region.addr == 0; }
  uint32_t lkey() const { return region.lkey; }
  rmr_t rmr() const { return rmr_t{region.addr, region.rkey, 0}; }
};

enum class post_status_t {
  posted,
  // The QP send queue is full; drain completions and retry.
  retry,
};

// One NIC port and protection domain, shared by all devices of an EP context.
class net_context_t {
 public:
  // An empty `device_name` selects MORI_KIWI_DEVICE_NAME, then the NIC that
  // TopoSystem::MatchGpuAndNic pairs with the current HIP device.
  explicit net_context_t(const std::string& device_name);
  ~net_context_t();

  net_context_t(const net_context_t&) = delete;
  net_context_t& operator=(const net_context_t&) = delete;

  mr_t register_memory(void* buffer, size_t size);
  void deregister_memory(mr_t& mr);

  application::RdmaDeviceContext& device_context() const { return *device_context_; }
  uint32_t port() const { return port_; }

 private:
  std::unique_ptr<application::RdmaContext> rdma_;
  std::unique_ptr<application::RdmaDeviceContext> device_context_;
  uint32_t port_ = 0;
};

// One connected RC endpoint per rank, including a loopback endpoint for this
// rank. Each endpoint has its own CQ. A device is driven by a single proxy
// thread and performs no internal locking. Construction is collective over
// `bootstrap`.
class device_t {
 public:
  device_t(net_context_t& net, application::BootstrapNetwork& bootstrap, size_t max_sends,
           size_t max_inline);
  ~device_t();

  device_t(const device_t&) = delete;
  device_t& operator=(const device_t&) = delete;

  // Signaled RDMA WRITE of `size` bytes from a registered local buffer to
  // `rmr.base + remote_disp` on `rank`. One completion is generated.
  post_status_t post_put(int rank, const void* local, size_t size, uint32_t lkey,
                         uint64_t remote_disp, const rmr_t& rmr);

  // Signaled inline RDMA WRITE. The source buffer may be reused as soon as
  // this returns `posted`; one completion is still generated.
  post_status_t post_puts(int rank, const void* local, size_t size, uint64_t remote_disp,
                          const rmr_t& rmr);

  // Polls up to `max` completions from the CQs of ranks with outstanding
  // writes and returns send-queue credit for each.
  size_t poll_cq(ibv_wc* wcs, size_t max);

 private:
  post_status_t post(int rank, ibv_send_wr& wr);

  net_context_t& net_;
  std::vector<application::RdmaEndpoint> endpoints_;
  std::vector<size_t> outstanding_;
  size_t max_sends_ = 0;
  size_t max_inline_ = 0;
  size_t poll_cursor_ = 0;
};

}  // namespace mori::kiwi::ep::rdma
