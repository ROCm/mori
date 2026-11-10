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
#include "comm/rdma.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "mori/application/topology/system.hpp"

namespace mori::kiwi::ep::rdma {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("mori::kiwi::ep::rdma: " + what);
}

}  // namespace

net_context_t::net_context_t(const std::string& device_name)
    : rdma_(std::make_unique<application::RdmaContext>(application::RdmaBackendType::IBVerbs)) {
  std::string name = device_name;
  if (const char* env = std::getenv("MORI_KIWI_DEVICE_NAME"); name.empty() && env != nullptr) {
    name = env;
  }
  if (name.empty()) {
    int hip_device = 0;
    if (hipGetDevice(&hip_device) != hipSuccess) fail("hipGetDevice failed");
    name = application::TopoSystem().MatchGpuAndNic(hip_device);
  }
  for (const auto& [device, port] :
       application::GetActiveDevicePortList(rdma_->GetRdmaDeviceList())) {
    if (device->Name() != name) continue;
    port_ = port;
    device_context_.reset(device->CreateRdmaDeviceContext());
    break;
  }
  if (!device_context_) fail("RDMA device '" + name + "' has no active port");
}

net_context_t::~net_context_t() = default;

mr_t net_context_t::register_memory(void* buffer, size_t size) {
  return mr_t{device_context_->RegisterRdmaMemoryRegionAuto(buffer, size)};
}

void net_context_t::deregister_memory(mr_t& mr) {
  if (!mr.empty())
    device_context_->DeregisterRdmaMemoryRegion(reinterpret_cast<void*>(mr.region.addr));
  mr = mr_t{};
}

device_t::device_t(net_context_t& net, application::BootstrapNetwork& bootstrap, size_t max_sends,
                   size_t max_inline)
    : net_(net), max_sends_(max_sends), max_inline_(max_inline) {
  const int nranks = bootstrap.GetWorldSize();
  application::RdmaDeviceContext& ctx = net.device_context();

  application::RdmaEndpointConfig config;
  config.portId = net.port();
  if (const char* gid_index = std::getenv("MORI_IB_GID_INDEX")) {
    config.gidIdx = std::atoi(gid_index);
  }
  config.maxMsgsNum = static_cast<uint32_t>(max_sends);
  config.maxRecvWr = 1;
  config.maxCqeNum = static_cast<uint32_t>(max_sends);
  config.maxInlineData = static_cast<uint32_t>(max_inline);
  config.atomicIbufSlots = 0;

  endpoints_.reserve(static_cast<size_t>(nranks));
  std::vector<application::RdmaEndpointHandle> local(static_cast<size_t>(nranks));
  for (int rank = 0; rank < nranks; ++rank) {
    endpoints_.push_back(ctx.CreateRdmaEndpoint(config));
    local[static_cast<size_t>(rank)] = endpoints_.back().handle;
  }
  std::vector<application::RdmaEndpointHandle> remote(static_cast<size_t>(nranks));
  bootstrap.AllToAll(local.data(), remote.data(), sizeof(application::RdmaEndpointHandle));
  for (int rank = 0; rank < nranks; ++rank) {
    ctx.ConnectEndpoint(local[static_cast<size_t>(rank)], remote[static_cast<size_t>(rank)]);
  }
  outstanding_.assign(static_cast<size_t>(nranks), 0);
}

device_t::~device_t() {
  for (const application::RdmaEndpoint& ep : endpoints_) {
    (void)net_.device_context().DestroyRdmaEndpointNoThrow(ep);
  }
}

post_status_t device_t::post(int rank, ibv_send_wr& wr) {
  size_t& outstanding = outstanding_[static_cast<size_t>(rank)];
  if (outstanding >= max_sends_) return post_status_t::retry;
  ibv_send_wr* bad = nullptr;
  const int ret = ibv_post_send(endpoints_[static_cast<size_t>(rank)].ibvHandle.qp, &wr, &bad);
  if (ret == ENOMEM) return post_status_t::retry;
  if (ret != 0) fail("ibv_post_send to rank " + std::to_string(rank) + " failed: " + strerror(ret));
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
    fail("inline put of " + std::to_string(size) + " bytes exceeds " + std::to_string(max_inline_));
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

size_t device_t::poll_cq(ibv_wc* wcs, size_t max) {
  const size_t nranks = endpoints_.size();
  size_t polled = 0;
  for (size_t i = 0; i < nranks && polled < max; ++i) {
    const size_t rank = (poll_cursor_ + i) % nranks;
    if (outstanding_[rank] == 0) continue;
    const int n =
        ibv_poll_cq(endpoints_[rank].ibvHandle.cq, static_cast<int>(max - polled), wcs + polled);
    if (n < 0) fail("ibv_poll_cq for rank " + std::to_string(rank) + " failed");
    outstanding_[rank] -= std::min(outstanding_[rank], static_cast<size_t>(n));
    polled += static_cast<size_t>(n);
  }
  poll_cursor_ = nranks == 0 ? 0 : (poll_cursor_ + 1) % nranks;
  return polled;
}

}  // namespace mori::kiwi::ep::rdma
