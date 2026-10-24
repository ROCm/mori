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
#include <string>
#include <vector>

#include "mori/kiwi/rdma/bootstrap.hpp"

// Raw ibverbs transport for KIWI EP proxy threads: one NIC context, per-proxy
// devices that each own a CQ and one RC QP per rank, memory registration, and
// signaled RDMA WRITEs.
namespace mori::kiwi::rdma {

// Remote memory region: what a peer needs to target a registered buffer.
struct rmr_t {
  uint64_t base = 0;
  uint32_t rkey = 0;
  uint32_t reserved = 0;
};

// Local memory region.
struct mr_t {
  ibv_mr* mr = nullptr;
  int dmabuf_fd = -1;
  // Virtual address of the registered buffer. ibv_mr::addr holds the DMA-BUF
  // offset for DMA-BUF registrations, so it cannot serve as the remote base.
  uint64_t base = 0;

  bool empty() const { return mr == nullptr; }
  uint32_t lkey() const { return mr->lkey; }
  rmr_t rmr() const { return rmr_t{base, mr->rkey, 0}; }
};

enum class post_status_t {
  posted,
  // The QP send queue is full; drain completions and retry.
  retry,
};

// One opened NIC port and protection domain, shared by all devices of a
// context.
class net_context_t {
 public:
  // An empty `device_name` selects MORI_KIWI_DEVICE_NAME, then the nearest
  // active NIC for the current HIP device, then the fastest active port.
  explicit net_context_t(const std::string& device_name);
  ~net_context_t();

  net_context_t(const net_context_t&) = delete;
  net_context_t& operator=(const net_context_t&) = delete;

  // GPU memory is registered through DMA-BUF when available; otherwise, or when
  // MORI_KIWI_DISABLE_DMABUF=1, this falls back to ibv_reg_mr.
  mr_t register_memory(void* buffer, size_t size);
  void deregister_memory(mr_t& mr);

  ibv_context* context() const { return context_; }
  ibv_pd* pd() const { return pd_; }
  uint8_t port() const { return port_; }
  const ibv_port_attr& port_attr() const { return port_attr_; }
  const ibv_device_attr& device_attr() const { return device_attr_; }
  int gid_index() const { return gid_index_; }
  const ibv_gid& gid() const { return gid_; }
  const std::string& device_name() const { return device_name_; }

 private:
  ibv_device** device_list_ = nullptr;
  ibv_context* context_ = nullptr;
  ibv_pd* pd_ = nullptr;
  uint8_t port_ = 0;
  ibv_port_attr port_attr_{};
  ibv_device_attr device_attr_{};
  int gid_index_ = -1;
  ibv_gid gid_{};
  std::string device_name_;
  bool use_dmabuf_ = true;
};

// A completion queue plus one connected RC QP per rank, including a loopback
// QP for this rank. A device is driven by a single proxy thread; it performs
// no internal locking. Construction is collective over `bootstrap`.
class device_t {
 public:
  device_t(net_context_t& net, bootstrap_t& bootstrap, size_t max_sends, size_t max_inline);
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

  // Polls up to `max` completions and returns send-queue credit for each.
  size_t poll_cq(ibv_wc* wcs, size_t max);

  size_t max_inline() const { return max_inline_; }

 private:
  post_status_t post(int rank, ibv_send_wr& wr);
  int rank_of_qp(uint32_t qp_num) const;

  net_context_t& net_;
  ibv_cq* cq_ = nullptr;
  std::vector<ibv_qp*> qps_;
  std::vector<size_t> outstanding_;
  std::vector<std::pair<uint32_t, int>> qp_num_to_rank_;
  size_t max_sends_ = 0;
  size_t max_inline_ = 0;
};

}  // namespace mori::kiwi::rdma
