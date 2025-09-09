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

#ifdef ENABLE_BNXT
extern "C" {
#include <infiniband/bnxt_re_dv.h>
#include <infiniband/bnxt_re_hsi.h>
}  // ENABLE_BNXT
#else
extern "C" {
#include "mori/core/transport/rdma/providers/bnxt/bnxt_re_dv.h"
#include "mori/core/transport/rdma/providers/bnxt/bnxt_re_hsi.h"
}
#endif

#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

#ifdef ENABLE_BNXT
/* ---------------------------------------------------------------------------------------------- */
/*                                        Device Attributes                                       */
/* ---------------------------------------------------------------------------------------------- */
static size_t GetBnxtCqeSize() { return BNXT_RE_CQE_SIZE; }

// static size_t GetBnxtSqWqeSize() {
//   return (192 + sizeof(mlx5_wqe_data_seg) + MLX5_SEND_WQE_BB - 1) / MLX5_SEND_WQE_BB *
//          MLX5_SEND_WQE_BB;
// }

/* ---------------------------------------------------------------------------------------------- */
/*                                 Device Data Structure Container                                */
/* ---------------------------------------------------------------------------------------------- */
// TODO: refactor BnxtCqContainer so its structure is similar to BnxtQpContainer
class BnxtCqContainer {
 public:
  BnxtCqContainer(ibv_context* context, const RdmaEndpointConfig& config);
  ~BnxtCqContainer();

 public:
  RdmaEndpointConfig config;
  uint32_t cqeNum;

 public:
  uint32_t cqn{0};
  void* cqUmemAddr{nullptr};
  void* cqDbrUmemAddr{nullptr};
  void* cqUmem{nullptr};
  void* cqDbrUmem{nullptr};
  void* cqUar{nullptr};
  void* cqUarPtr{nullptr};
  ibv_cq* cq{nullptr};
};

class BnxtQpContainer {
 public:
  BnxtQpContainer(ibv_context* context, const RdmaEndpointConfig& config, ibv_cq* cq, ibv_pd* pd);
  ~BnxtQpContainer();

  void ModifyRst2Init();
  void ModifyInit2Rtr(const RdmaEndpointHandle& remote_handle, const ibv_port_attr& portAttr,
                      const ibv_device_attr_ex& deviceAttr);
  void ModifyRtr2Rts(const RdmaEndpointHandle& local_handle,
                     const RdmaEndpointHandle& remote_handle);

  void* GetSqAddress();
  void* GetMsntblAddress();
  void* GetRqAddress();

 private:
  void DestroyQueuePair();

 public:
  ibv_context* context;

 public:
  RdmaEndpointConfig config;
  struct bnxt_re_dv_qp_mem_info qpMemInfo;

 public:
  size_t qpn{0};
  void* sqUmemAddr{nullptr};
  void* rqUmemAddr{nullptr};
  void* msntblUmemAddr{nullptr};
  void* qpDbrUmemAddr{nullptr};
  void* sqUmem{nullptr};
  void* rqUmem{nullptr};
  void* qpDbrUmem{nullptr};
  void* qpUar{nullptr};
  void* qpUarPtr{nullptr};
  ibv_qp* qp{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                        BnxtDeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
class BnxtDeviceContext : public RdmaDeviceContext {
 public:
  BnxtDeviceContext(RdmaDevice* rdma_device, ibv_pd* inPd);
  ~BnxtDeviceContext();

  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) override;
  virtual void ConnectEndpoint(const RdmaEndpointHandle& local,
                               const RdmaEndpointHandle& remote) override;

 private:
  uint32_t pdn;

  std::unordered_map<uint32_t, BnxtCqContainer*> cqPool;
  std::unordered_map<uint32_t, BnxtQpContainer*> qpPool;
};

class BnxtDevice : public RdmaDevice {
 public:
  BnxtDevice(ibv_device* device);
  ~BnxtDevice();

  RdmaDeviceContext* CreateRdmaDeviceContext() override;
};
#endif  // ENABLE_BNXT
}  // namespace application
}  // namespace mori

namespace std {
#ifdef ENABLE_BNXT
static std::ostream& operator<<(std::ostream& s, const bnxt_re_dv_qp_mem_info& m) {
  std::stringstream ss;
  ss << "qp_handle: 0x" << std::hex << m.qp_handle << std::dec << "  sq_va: 0x" << std::hex
     << m.sq_va << std::dec << "  sq_len: " << m.sq_len << "  sq_slots: " << m.sq_slots
     << "  sq_wqe_sz: " << m.sq_wqe_sz << "  sq_psn_sz: " << m.sq_psn_sz
     << "  sq_npsn: " << m.sq_npsn << "  rq_va: 0x" << std::hex << m.rq_va << std::dec
     << "  rq_len: " << m.rq_len << "  rq_slots: " << m.rq_slots << "  rq_wqe_sz: " << m.rq_wqe_sz
     << "  comp_mask: 0x" << std::hex << m.comp_mask << std::dec;
  s << ss.str();
  return s;
}
#endif  // ENABLE_BNXT
}  // namespace std
