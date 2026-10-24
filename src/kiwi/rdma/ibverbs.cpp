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
#include "mori/kiwi/rdma/ibverbs.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include "mori/kiwi/topology/topology.hpp"

namespace mori::kiwi::rdma {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("mori::kiwi::rdma: " + what);
}

[[noreturn]] void fail_errno(const std::string& what) {
  fail(what + ": " + std::strerror(errno));
}

bool env_flag(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  return std::atoi(value);
}

// Lane width and per-lane speed tables follow the IB specification.
int port_width(uint8_t width) {
  switch (width) {
    case 1:
      return 1;
    case 2:
      return 4;
    case 4:
      return 8;
    case 8:
      return 12;
    case 16:
      return 2;
    default:
      return 0;
  }
}

double port_lane_gbps(uint8_t speed) {
  switch (speed) {
    case 1:
      return 2.5;
    case 2:
      return 5;
    case 4:
    case 8:
      return 10;
    case 16:
      return 14;
    case 32:
      return 25;
    case 64:
      return 50;
    case 128:
      return 100;
    default:
      return 0;
  }
}

bool port_usable(const ibv_port_attr& attr) {
  if (attr.state != IBV_PORT_ACTIVE) return false;
  if (attr.link_layer == IBV_LINK_LAYER_INFINIBAND && attr.lid == 0) return false;
  return port_width(attr.active_width) > 0 && port_lane_gbps(attr.active_speed) > 0;
}

std::string resolve_device_name(const std::string& requested) {
  if (!requested.empty()) return requested;
  const char* env = std::getenv("MORI_KIWI_DEVICE_NAME");
  if (env != nullptr && env[0] != '\0') return env;
  return mori::kiwi::p2p::topology::get_hca_name_for_current_gpu();
}

// Picks the requested device, or the fastest active port when no device is
// requested.
void select_device_port(ibv_device** list, int count, const std::string& requested,
                        ibv_device** device_out, uint8_t* port_out) {
  double best_gbps = 0;
  for (int i = 0; i < count; ++i) {
    const char* name = ibv_get_device_name(list[i]);
    if (!requested.empty() && requested != name) continue;
    ibv_context* ctx = ibv_open_device(list[i]);
    if (ctx == nullptr) continue;
    ibv_device_attr dev_attr{};
    if (ibv_query_device(ctx, &dev_attr) == 0) {
      for (uint8_t port = 1; port <= dev_attr.phys_port_cnt; ++port) {
        ibv_port_attr attr{};
        if (ibv_query_port(ctx, port, &attr) != 0 || !port_usable(attr)) continue;
        const double gbps = port_width(attr.active_width) * port_lane_gbps(attr.active_speed);
        if (gbps > best_gbps) {
          best_gbps = gbps;
          *device_out = list[i];
          *port_out = port;
        }
      }
    }
    ibv_close_device(ctx);
  }
  if (best_gbps <= 0) {
    fail(requested.empty() ? "no active RDMA device port found"
                           : "requested RDMA device '" + requested + "' has no active port");
  }
}

enum class roce_version_t { v2, v1, unknown };

roce_version_t gid_roce_version(const char* device_name, uint8_t port, int index) {
  std::ifstream in("/sys/class/infiniband/" + std::string(device_name) + "/ports/" +
                   std::to_string(port) + "/gid_attrs/types/" + std::to_string(index));
  std::string type;
  std::getline(in, type);
  if (type.rfind("RoCE v2", 0) == 0) return roce_version_t::v2;
  if (type.rfind("IB/RoCE v1", 0) == 0) return roce_version_t::v1;
  return roce_version_t::unknown;
}

bool gid_operational(ibv_context* ctx, ibv_pd* pd, uint8_t port, int index) {
  ibv_gid gid{};
  if (ibv_query_gid(ctx, port, index, &gid) != 0) return false;
  if (gid.global.interface_id == 0) return false;
  ibv_ah_attr ah_attr{};
  ah_attr.port_num = port;
  ah_attr.is_global = 1;
  ah_attr.grh.dgid = gid;
  ah_attr.grh.sgid_index = static_cast<uint8_t>(index);
  ah_attr.grh.hop_limit = 255;
  ah_attr.grh.flow_label = 1;
  ah_attr.dlid = 0xC000;
  ibv_ah* ah = ibv_create_ah(pd, &ah_attr);
  if (ah == nullptr) return false;
  ibv_destroy_ah(ah);
  return true;
}

// Prefers an operational RoCE v2 GID, then RoCE v1, then any operational GID.
int select_roce_gid(const char* device_name, ibv_context* ctx, ibv_pd* pd, uint8_t port,
                    int gid_table_len) {
  int best_index = -1;
  int best_rank = 3;
  for (int i = 0; i < gid_table_len; ++i) {
    const int rank = static_cast<int>(gid_roce_version(device_name, port, i));
    if (rank >= best_rank || !gid_operational(ctx, pd, port, i)) continue;
    best_index = i;
    best_rank = rank;
    if (rank == 0) break;
  }
  return best_index >= 0 ? best_index : 0;
}

int export_dmabuf(void* ptr, size_t size, uint64_t* offset) {
  int fd = -1;
  if (hsa_amd_portable_export_dmabuf(ptr, size, &fd, offset) != HSA_STATUS_SUCCESS) return -1;
  return fd;
}

struct qp_endpoint_t {
  int32_t source_rank;
  int32_t target_rank;
  uint32_t qp_num;
  uint16_t lid;
  uint16_t reserved;
  uint8_t gid[16];
};

}  // namespace

net_context_t::net_context_t(const std::string& requested_device_name) {
  int count = 0;
  device_list_ = ibv_get_device_list(&count);
  if (device_list_ == nullptr || count <= 0) fail("no RDMA devices found");

  const std::string requested = resolve_device_name(requested_device_name);
  ibv_device* device = nullptr;
  select_device_port(device_list_, count, requested, &device, &port_);
  device_name_ = ibv_get_device_name(device);

  context_ = ibv_open_device(device);
  if (context_ == nullptr) fail_errno("ibv_open_device(" + device_name_ + ")");
  pd_ = ibv_alloc_pd(context_);
  if (pd_ == nullptr) fail_errno("ibv_alloc_pd");
  if (ibv_query_device(context_, &device_attr_) != 0) fail_errno("ibv_query_device");
  if (ibv_query_port(context_, port_, &port_attr_) != 0) fail_errno("ibv_query_port");

  gid_index_ = env_int("MORI_KIWI_GID_INDEX", -1);
  if (gid_index_ < 0 && port_attr_.link_layer == IBV_LINK_LAYER_ETHERNET) {
    gid_index_ =
        select_roce_gid(device_name_.c_str(), context_, pd_, port_, port_attr_.gid_tbl_len);
  }
  if (gid_index_ >= 0 && ibv_query_gid(context_, port_, gid_index_, &gid_) != 0) {
    fail_errno("ibv_query_gid(" + std::to_string(gid_index_) + ")");
  }
  use_dmabuf_ = !env_flag("MORI_KIWI_DISABLE_DMABUF");
}

net_context_t::~net_context_t() {
  if (pd_ != nullptr) ibv_dealloc_pd(pd_);
  if (context_ != nullptr) ibv_close_device(context_);
  if (device_list_ != nullptr) ibv_free_device_list(device_list_);
}

mr_t net_context_t::register_memory(void* buffer, size_t size) {
  constexpr int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  mr_t out;
  out.base = reinterpret_cast<uint64_t>(buffer);
  if (use_dmabuf_) {
    const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t begin = reinterpret_cast<uintptr_t>(buffer) & ~(page - 1);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(buffer) + size + page - 1) & ~(page - 1);
    uint64_t offset = 0;
    const int fd = export_dmabuf(reinterpret_cast<void*>(begin), end - begin, &offset);
    if (fd >= 0) {
      out.mr = ibv_reg_dmabuf_mr(pd_, offset, end - begin, begin, fd, access);
      if (out.mr != nullptr) {
        out.dmabuf_fd = fd;
        return out;
      }
      close(fd);
    }
    // Stop trying DMA-BUF after the first failure.
    use_dmabuf_ = false;
  }
  out.mr = ibv_reg_mr(pd_, buffer, size, access);
  if (out.mr == nullptr) fail_errno("ibv_reg_mr(" + std::to_string(size) + " bytes)");
  return out;
}

void net_context_t::deregister_memory(mr_t& mr) {
  if (mr.mr != nullptr) ibv_dereg_mr(mr.mr);
  if (mr.dmabuf_fd >= 0) close(mr.dmabuf_fd);
  mr = mr_t{};
}

device_t::device_t(net_context_t& net, bootstrap_t& bootstrap, size_t max_sends,
                   size_t max_inline)
    : net_(net) {
  const int nranks = bootstrap.size();
  const int me = bootstrap.rank();
  const ibv_device_attr& dev_attr = net.device_attr();
  max_sends_ = std::min(max_sends, static_cast<size_t>(dev_attr.max_qp_wr));
  const size_t cqes = std::min(max_sends_ * static_cast<size_t>(nranks),
                               static_cast<size_t>(dev_attr.max_cqe));

  cq_ = ibv_create_cq(net.context(), static_cast<int>(cqes), nullptr, nullptr, 0);
  if (cq_ == nullptr) fail_errno("ibv_create_cq");

  qps_.assign(static_cast<size_t>(nranks), nullptr);
  outstanding_.assign(static_cast<size_t>(nranks), 0);
  max_inline_ = max_inline;
  std::vector<qp_endpoint_t> local(static_cast<size_t>(nranks));
  for (int rank = 0; rank < nranks; ++rank) {
    ibv_qp_init_attr init{};
    init.send_cq = cq_;
    init.recv_cq = cq_;
    init.cap.max_send_wr = static_cast<uint32_t>(max_sends_);
    init.cap.max_recv_wr = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    init.cap.max_inline_data = static_cast<uint32_t>(max_inline);
    init.qp_type = IBV_QPT_RC;
    init.sq_sig_all = 0;
    ibv_qp* qp = ibv_create_qp(net.pd(), &init);
    if (qp == nullptr) fail_errno("ibv_create_qp for rank " + std::to_string(rank));
    qps_[static_cast<size_t>(rank)] = qp;
    max_inline_ = std::min<size_t>(max_inline_, init.cap.max_inline_data);
    qp_num_to_rank_.emplace_back(qp->qp_num, rank);

    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = net.port();
    attr.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
      fail_errno("ibv_modify_qp(INIT)");
    }

    qp_endpoint_t& ep = local[static_cast<size_t>(rank)];
    ep.source_rank = me;
    ep.target_rank = rank;
    ep.qp_num = qp->qp_num;
    ep.lid = net.port_attr().lid;
    ep.reserved = 0;
    std::memcpy(ep.gid, net.gid().raw, sizeof(ep.gid));
  }
  std::sort(qp_num_to_rank_.begin(), qp_num_to_rank_.end());
  if (max_inline_ < max_inline) {
    fail("QP inline capacity " + std::to_string(max_inline_) + " is below the requested " +
         std::to_string(max_inline));
  }

  // Each rank publishes one endpoint per target; rank r reads row s, column r.
  std::vector<qp_endpoint_t> all(static_cast<size_t>(nranks) * nranks);
  bootstrap.allgather(local.data(), all.data(), local.size() * sizeof(qp_endpoint_t));

  const uint8_t sl = static_cast<uint8_t>(env_int("MORI_KIWI_SL", 0));
  const uint8_t traffic_class = static_cast<uint8_t>(env_int("MORI_KIWI_TRAFFIC_CLASS", 0));
  for (int rank = 0; rank < nranks; ++rank) {
    const qp_endpoint_t& remote =
        all[static_cast<size_t>(rank) * nranks + static_cast<size_t>(me)];
    if (remote.source_rank != rank || remote.target_rank != me) {
      fail("QP endpoint exchange returned mismatched ranks");
    }
    ibv_qp* qp = qps_[static_cast<size_t>(rank)];

    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = net.port_attr().active_mtu;
    attr.dest_qp_num = remote.qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = remote.lid;
    attr.ah_attr.sl = sl;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = net.port();
    ibv_gid remote_gid{};
    std::memcpy(remote_gid.raw, remote.gid, sizeof(remote_gid.raw));
    if (net.gid_index() >= 0 && remote_gid.global.interface_id != 0) {
      attr.ah_attr.is_global = 1;
      attr.ah_attr.grh.dgid = remote_gid;
      attr.ah_attr.grh.sgid_index = static_cast<uint8_t>(net.gid_index());
      attr.ah_attr.grh.hop_limit = 64;
      attr.ah_attr.grh.traffic_class = traffic_class;
    }
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
      fail_errno("ibv_modify_qp(RTR) for rank " + std::to_string(rank));
    }

    attr = ibv_qp_attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 18;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
      fail_errno("ibv_modify_qp(RTS) for rank " + std::to_string(rank));
    }
  }
}

device_t::~device_t() {
  for (ibv_qp* qp : qps_) {
    if (qp != nullptr) ibv_destroy_qp(qp);
  }
  if (cq_ != nullptr) ibv_destroy_cq(cq_);
}

post_status_t device_t::post(int rank, ibv_send_wr& wr) {
  size_t& outstanding = outstanding_[static_cast<size_t>(rank)];
  if (outstanding >= max_sends_) return post_status_t::retry;
  ibv_send_wr* bad = nullptr;
  const int ret = ibv_post_send(qps_[static_cast<size_t>(rank)], &wr, &bad);
  if (ret == ENOMEM) return post_status_t::retry;
  if (ret != 0) {
    errno = ret;
    fail_errno("ibv_post_send to rank " + std::to_string(rank));
  }
  ++outstanding;
  return post_status_t::posted;
}

post_status_t device_t::post_put(int rank, const void* local, size_t size, uint32_t lkey,
                                 uint64_t remote_disp, const rmr_t& rmr) {
  ibv_sge sge{};
  ibv_send_wr wr{};
  if (size > 0) {
    sge.addr = reinterpret_cast<uint64_t>(local);
    sge.length = static_cast<uint32_t>(size);
    sge.lkey = lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
  }
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = rmr.base + remote_disp;
  wr.wr.rdma.rkey = rmr.rkey;
  return post(rank, wr);
}

post_status_t device_t::post_puts(int rank, const void* local, size_t size, uint64_t remote_disp,
                                  const rmr_t& rmr) {
  if (size > max_inline_) {
    fail("inline put of " + std::to_string(size) + " bytes exceeds " +
         std::to_string(max_inline_));
  }
  ibv_sge sge{};
  ibv_send_wr wr{};
  if (size > 0) {
    sge.addr = reinterpret_cast<uint64_t>(local);
    sge.length = static_cast<uint32_t>(size);
    wr.sg_list = &sge;
    wr.num_sge = 1;
  }
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  wr.wr.rdma.remote_addr = rmr.base + remote_disp;
  wr.wr.rdma.rkey = rmr.rkey;
  return post(rank, wr);
}

int device_t::rank_of_qp(uint32_t qp_num) const {
  auto it = std::lower_bound(qp_num_to_rank_.begin(), qp_num_to_rank_.end(),
                             std::make_pair(qp_num, -1));
  if (it == qp_num_to_rank_.end() || it->first != qp_num) return -1;
  return it->second;
}

size_t device_t::poll_cq(ibv_wc* wcs, size_t max) {
  const int n = ibv_poll_cq(cq_, static_cast<int>(max), wcs);
  if (n < 0) fail("ibv_poll_cq failed");
  for (int i = 0; i < n; ++i) {
    const int rank = rank_of_qp(wcs[i].qp_num);
    if (rank >= 0 && outstanding_[static_cast<size_t>(rank)] > 0) {
      --outstanding_[static_cast<size_t>(rank)];
    }
  }
  return static_cast<size_t>(n);
}

}  // namespace mori::kiwi::rdma
