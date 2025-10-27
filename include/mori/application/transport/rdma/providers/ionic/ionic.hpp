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

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "mori/core/transport/rdma/providers/ionic/ionic_dv.h"
#include "mori/core/transport/rdma/providers/ionic/ionic_fw.h"
#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace application {

// BNXT UDP sport configuration constants
static constexpr uint32_t IONIC_UDP_SPORT_ARRAY_SIZE = 4;

/* ---------------------------------------------------------------------------------------------- */
/*                                        Device Attributes                                       */
/* ---------------------------------------------------------------------------------------------- */
static size_t GetIonicCqeSize() { return sizeof(struct ionic_v1_cqe); }

/* ---------------------------------------------------------------------------------------------- */
/*                                 Device Data Structure Container                                */
/* ---------------------------------------------------------------------------------------------- */
// TODO: refactor IonicCqContainer so its structure is similar to IonicQpContainer
class IonicCqContainer {
 public:
  IonicCqContainer(ibv_context* context, const RdmaEndpointConfig& config, ibv_pd* pd);
  ~IonicCqContainer();

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

class IonicDeviceContext;  // Forward declaration

typedef struct device_agent {
  hsa_agent_t agent;
  hsa_amd_memory_pool_t pool;
} device_agent_t;

class IonicQpContainer {
 public:
  IonicQpContainer(ibv_context* context, const RdmaEndpointConfig& config, 
                   ibv_cq* cq, ibv_pd* pd, IonicDeviceContext* device_context);
  ~IonicQpContainer();

  void ModifyRst2Init();
  void ModifyInit2Rtr(const RdmaEndpointHandle& remote_handle, const ibv_port_attr& portAttr,
                      uint32_t qpn = 0);
  void ModifyRtr2Rts(const RdmaEndpointHandle& local_handle,
                     const RdmaEndpointHandle& remote_handle);

  void* GetSqAddress();
  void* GetMsntblAddress();
  void* GetRqAddress();
  
  IonicDeviceContext* GetDeviceContext() { return device_context; }

 private:
  void DestroyQueuePair();

 public:
  ibv_context* context;
  IonicDeviceContext* device_context;

 public:
  RdmaEndpointConfig config;
  struct bnxt_re_dv_qp_mem_info qpMemInfo;

 public:
  size_t qpn{0};
  uint16_t wqeNum{0};
  char dev_name[24];
  ionic_dv_ctx dvctx;
  ibv_qp* qp{nullptr};
  
  void* gpu_db_page{nullptr};
  uint64_t *db_page_u64{nullptr};
  uint64_t *gpu_db_page_u64{nullptr};
  uint64_t *gpu_db_ptr{nullptr};
  uint64_t *gpu_db_cq{nullptr};
  uint64_t *gpu_db_sq{nullptr};
  uint64_t *gpu_db_rq{nullptr};
  
  ionic_dv_cq dvcq;
  uint64_t *cq_dbreg{nullptr};
  uint64_t cq_dbval{0};
  uint64_t cq_mask{0};
  struct ionic_v1_cqe *ionic_cq_buf{nullptr};
  
  ionic_dv_qp dvqp;
  uint64_t *sq_dbreg{nullptr};
  uint64_t sq_dbval{0};
  uint64_t sq_mask{0};
  struct ionic_v1_wqe *ionic_sq_buf{nullptr};

  uint64_t *rq_dbreg{nullptr};
  uint64_t rq_dbval{0};
  uint64_t rq_mask{0};
  struct ionic_v1_wqe *ionic_rq_buf{nullptr};
  
  // Atomic internal buffer fields
  void* atomicIbufAddr{nullptr};
  size_t atomicIbufSize{0};
  ibv_mr* atomicIbufMr{nullptr};
};

/* ---------------------------------------------------------------------------------------------- */
/*                                        BnxtDeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
class IonicDeviceContext : public RdmaDeviceContext {
 public:
  IonicDeviceContext(RdmaDevice* rdma_device, ibv_pd* inPd);
  ~IonicDeviceContext();

  virtual RdmaEndpoint CreateRdmaEndpoint(const RdmaEndpointConfig&) override;
  virtual void ConnectEndpoint(const RdmaEndpointHandle& local,
                               const RdmaEndpointHandle& remote, uint32_t qpn = 0) override;

 private:
  uint32_t pdn;
  struct ibv_pd *pd_uxdma;

  std::unordered_map<uint32_t, IonicCqContainer*> cqPool;
  std::unordered_map<uint32_t, IonicQpContainer*> qpPool;
};

class IonicDevice : public RdmaDevice {
 public:
  IonicDevice(ibv_device* device);
  ~IonicDevice();

  static void pd_release(ibv_pd* pd, void* pd_context, void* ptr, uint64_t resource_type);
  static void* pd_alloc_device_uncached(ibv_pd* pd, void* pd_context, size_t size, 
                                        size_t alignment, uint64_t resource_type);
  void create_parent_domain(struct ibv_pd *pd_orig);
  RdmaDeviceContext* CreateRdmaDeviceContext() override;
};
}  // namespace application
}  // namespace mori
