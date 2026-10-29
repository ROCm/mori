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

#include "mori/application/transport/rdma/providers/ionic/ionic.hpp"

#include <hip/hip_runtime_api.h>
#include <infiniband/verbs.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>

#include "mori/application/transport/rdma/providers/ionic/ionic_dv.h"
#include "mori/application/utils/check.hpp"
#include "mori/application/utils/math.hpp"
#include "mori/core/transport/rdma/providers/ionic/ionic_fw.h"
#include "mori/utils/env_utils.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {
/* ---------------------------------------------------------------------------------------------- */
/*                                        Device Attributes                                       */
/* ---------------------------------------------------------------------------------------------- */

namespace {

using FwVersion = std::tuple<int, int, int, int>;
constexpr FwVersion kCcqeMinFwVersion{1, 117, 5, 58};

// Parse "1.117.5-a-58" or "1.117.5-a58" into (1,117,5,58).
std::optional<FwVersion> ParseIonicFwVersion(const char* fw_ver) {
  int major, minor, patch, build;
  char tag;
  if (sscanf(fw_ver, "%d.%d.%d-%c-%d", &major, &minor, &patch, &tag, &build) == 5 ||
      sscanf(fw_ver, "%d.%d.%d-%c%d", &major, &minor, &patch, &tag, &build) == 5) {
    return FwVersion{major, minor, patch, build};
  }
  return std::nullopt;
}

std::optional<FwVersion> ReadIonicFwVersion(const char* dev_name) {
  char path[256];
  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/fw_ver", dev_name);

  FILE* f = fopen(path, "r");
  if (!f) return std::nullopt;

  char buf[64] = {};
  fgets(buf, sizeof(buf), f);
  fclose(f);

  // Strip trailing newline.
  buf[strcspn(buf, "\n")] = '\0';
  return ParseIonicFwVersion(buf);
}

bool IsCcqeSupported(ibv_context* context) {
  const char* disable_ccqe = std::getenv("MORI_DISABLE_IONIC_CCQE");
  if (disable_ccqe && std::strcmp(disable_ccqe, "1") == 0) return false;
  if (IonicDvApi::Instance().create_cq_ex == nullptr) return false;

  /* Minimum firmware version verified by MORI to support CCQE is 1.117.5-a-58. */
  auto ver = ReadIonicFwVersion(context->device->name);
  MORI_APP_TRACE("dev: {} fw_ver {}.{}.{}-a-{}", context->device->name,
                 ver ? std::get<0>(*ver) : -1, ver ? std::get<1>(*ver) : -1,
                 ver ? std::get<2>(*ver) : -1, ver ? std::get<3>(*ver) : -1);
  return ver.has_value() && *ver >= kCcqeMinFwVersion;
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                          IonicCqContainer                            */
/* ---------------------------------------------------------------------------------------------- */
IonicCqContainer::IonicCqContainer(ibv_context* context, const RdmaEndpointConfig& config,
                                   ibv_pd* pd)
    : config(config) {
  int status;
  struct ibv_cq_init_attr_ex cq_attr;
  struct ibv_cq_ex* cq_ex;

  cqeNum = config.maxCqeNum;

  const bool ccqe_enabled = IsCcqeSupported(context);

  memset(&cq_attr, 0, sizeof(struct ibv_cq_init_attr_ex));
  cq_attr.cq_context = nullptr;
  cq_attr.channel = nullptr;
  cq_attr.comp_vector = 0;
  cq_attr.flags = 0;
  cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_PD;
  cq_attr.parent_domain = pd;

  if (ccqe_enabled) {
    MORI_APP_TRACE("cqe mode: ccqe mode");
    struct ionic_cq_init_attr_ex ionic_cq_attr;
    memset(&ionic_cq_attr, 0, sizeof(struct ionic_cq_init_attr_ex));
    ionic_cq_attr.comp_mask = IONIC_CQ_INIT_ATTR_MASK_FLAGS;
    ionic_cq_attr.flags = IONIC_CQ_INIT_ATTR_CCQE;
    cq_attr.cqe = 1;
    cq_ex = IonicDvApi::Instance().create_cq_ex(context, &cq_attr, &ionic_cq_attr);
  } else {
    MORI_APP_TRACE("cqe mode: normal mode");
    cq_attr.cqe = cqeNum * 2;  // from rocshmem, send&recv?
    cq_ex = ibv_create_cq_ex(context, &cq_attr);
  }

  if (cq_ex == nullptr) {
    int savedErrno = errno;
    MORI_APP_ERROR("IONIC CQ creation failed: errno {} ({}), ccqe:{}, cqe:{}", savedErrno,
                   strerror(savedErrno), ccqe_enabled, cq_attr.cqe);
    std::abort();
  }
  cq = ibv_cq_ex_to_cq(cq_ex);
  assert(cq);

  MORI_APP_TRACE("IONIC CQ created: cqn={}, cqeNum={}", cqn, cqeNum * 2);
}

IonicCqContainer::~IonicCqContainer() {
  int err;

  err = ibv_destroy_cq(cq);
  assert(err == 0);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         IonicQpContainer                                       */
/* ---------------------------------------------------------------------------------------------- */

std::vector<device_agent_t> gpu_agents;
std::vector<device_agent_t> cpu_agents;

hsa_status_t rocm_hsa_amd_memory_pool_callback(hsa_amd_memory_pool_t memory_pool, void* data) {
  hsa_amd_memory_pool_global_flag_t pool_flag{};

  hsa_status_t status{
      hsa_amd_memory_pool_get_info(memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &pool_flag)};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to get pool info: 0x%x", status);
    return status;
  }

  if (pool_flag == (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
    *static_cast<hsa_amd_memory_pool_t*>(data) = memory_pool;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t rocm_hsa_agent_callback(hsa_agent_t agent, [[maybe_unused]] void* data) {
  hsa_device_type_t device_type{};

  hsa_status_t status{hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type)};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to get device type: 0x%x", status);
    return status;
  }

  if (device_type == HSA_DEVICE_TYPE_GPU) {
    gpu_agents.emplace_back();
    gpu_agents.back().agent = agent;
    status = hsa_amd_agent_iterate_memory_pools(agent, rocm_hsa_amd_memory_pool_callback,
                                                &(gpu_agents.back().pool));
  }

  if (device_type == HSA_DEVICE_TYPE_CPU) {
    cpu_agents.emplace_back();
    cpu_agents.back().agent = agent;
    status = hsa_amd_agent_iterate_memory_pools(agent, rocm_hsa_amd_memory_pool_callback,
                                                &(cpu_agents.back().pool));
  }

  return status;
}

int rocm_init() {
  hsa_status_t status{hsa_init()};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to open HSA connection: 0x%x", status);
    return 1;
  }

  status = hsa_iterate_agents(rocm_hsa_agent_callback, nullptr);

  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    printf("Failure to iterate HSA agents: 0x%x", status);
    return 1;
  }

  return 0;
}

void rocm_memory_lock_to_fine_grain(void* ptr, size_t size, void** gpu_ptr, int gpu_id) {
  hsa_status_t status{hsa_amd_memory_lock_to_pool(ptr, size, &(gpu_agents[gpu_id].agent), 1,
                                                  cpu_agents[0].pool, 0, gpu_ptr)};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failed to lock memory pool (%p): 0x%x\n", ptr, status);
    exit(-1);
  }
}

IonicQpContainer::IonicQpContainer(ibv_context* context, const RdmaEndpointConfig& config,
                                   ibv_cq* cq, struct ibv_pd* pd_uxdma,
                                   IonicDeviceContext* device_context)
    : context(context), config(config), device_context(device_context) {
  struct ibv_qp_init_attr_ex attr;
  int hip_dev_id{-1};
  int err;
  MORI_APP_TRACE("IonicQpContainer, cq:0x{:x}, pd:0x{:x}, context:0x{:x}, config.maxMsgsNum:{}",
                 reinterpret_cast<uintptr_t>(cq), reinterpret_cast<uintptr_t>(pd_uxdma),
                 reinterpret_cast<uintptr_t>(context), config.maxMsgsNum);
  wqeNum = config.maxMsgsNum;
  uint32_t recvWrNum = config.maxRecvWr != 0 ? config.maxRecvWr : wqeNum;
  memset(&attr, 0, sizeof(struct ibv_qp_init_attr_ex));
  attr.cap.max_send_wr = wqeNum;
  attr.cap.max_recv_wr = recvWrNum;
  attr.cap.max_send_sge = 1;
  attr.cap.max_inline_data = MAX_INLINE_SIZE;
  attr.sq_sig_all = 0;
  attr.qp_type = IBV_QPT_RC;
  attr.comp_mask = IBV_QP_INIT_ATTR_PD;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
  attr.pd = pd_uxdma;
  attr.send_cq = cq;
  attr.recv_cq = cq;
  qp = ibv_create_qp_ex(context, &attr);
  if (qp == nullptr) {
    int savedErrno = errno;
    MORI_APP_ERROR("IONIC QP creation failed: errno {} ({}), max_send_wr:{}, max_recv_wr:{}",
                   savedErrno, strerror(savedErrno), attr.cap.max_send_wr, attr.cap.max_recv_wr);
    std::abort();
  }

  HIP_RUNTIME_CHECK(hipGetDevice(&hip_dev_id));
  IonicDvApi::Instance().get_ctx(&dvctx, context);
  rocm_init();
  rocm_memory_lock_to_fine_grain(dvctx.db_page, 0x1000, &gpu_db_page, hip_dev_id);

  db_page_u64 = reinterpret_cast<uint64_t*>(dvctx.db_page);
  gpu_db_page_u64 = reinterpret_cast<uint64_t*>(gpu_db_page);

  gpu_db_ptr = &gpu_db_page_u64[dvctx.db_ptr - db_page_u64];

  // gpu_db_page = gpu_db_page;
  gpu_db_cq = &gpu_db_ptr[dvctx.cq_qtype];
  gpu_db_sq = &gpu_db_ptr[dvctx.sq_qtype];
  gpu_db_rq = &gpu_db_ptr[dvctx.rq_qtype];

  uint8_t udma_idx = IonicDvApi::Instance().qp_get_udma_idx(qp);
  IonicDvApi::Instance().get_cq(&dvcq, cq, udma_idx);

  cq_dbreg = gpu_db_cq;
  cq_dbval = dvcq.q.db_val;
  cq_mask = dvcq.q.mask;
  ionic_cq_buf = reinterpret_cast<ionic_v1_cqe*>(dvcq.q.ptr);
  MORI_APP_TRACE("cq ptr:0x{:x}, cq size:{}, cq mask:0x{:x}",
                 reinterpret_cast<uintptr_t>(dvcq.q.ptr), dvcq.q.size, dvcq.q.mask);

  ionic_dv_qp dvqp;
  IonicDvApi::Instance().get_qp(&dvqp, qp);

  sq_dbreg = gpu_db_sq;
  sq_dbval = dvqp.sq.db_val;
  sq_mask = dvqp.sq.mask;
  ionic_sq_buf = reinterpret_cast<ionic_v1_wqe*>(dvqp.sq.ptr);
  MORI_APP_TRACE("sq ptr:0x{:x}, sq size:{}, sq mask:0x{:x}",
                 reinterpret_cast<uintptr_t>(dvqp.sq.ptr), dvqp.sq.size, dvqp.sq.mask);
  rq_dbreg = gpu_db_rq;
  rq_dbval = dvqp.rq.db_val;
  rq_mask = dvqp.rq.mask;
  ionic_rq_buf = reinterpret_cast<ionic_v1_wqe*>(dvqp.rq.ptr);
  MORI_APP_TRACE("rq ptr:0x{:x}, rq size:{}, rq mask:0x{:x}",
                 reinterpret_cast<uintptr_t>(dvqp.rq.ptr), dvqp.rq.size, dvqp.rq.mask);
  strncpy(dev_name, qp->context->device->name, sizeof(dev_name));
  dev_name[sizeof(dev_name) - 1] = 0;

  qpn = qp->qp_num;

  MORI_APP_TRACE("IONIC QP created: qpn={}, sqWqeNum={}, cqAddr=0x{:x}, sqAddr=0x{:x}", qpn,
                 config.maxMsgsNum, reinterpret_cast<uintptr_t>(ionic_cq_buf),
                 reinterpret_cast<uintptr_t>(ionic_sq_buf));

  // Allocate and register atomic internal buffer (ibuf)
  atomicIbufSize = (RoundUpPowOfTwo(config.atomicIbufSlots) + 1) * ATOMIC_IBUF_SLOT_SIZE;
  if (config.onGpu) {
    HIP_RUNTIME_CHECK(
        hipExtMallocWithFlags(&atomicIbufAddr, atomicIbufSize, hipDeviceMallocUncached));
    HIP_RUNTIME_CHECK(hipMemset(atomicIbufAddr, 0, atomicIbufSize));
  } else {
    err = posix_memalign(&atomicIbufAddr, config.alignment, atomicIbufSize);
    memset(atomicIbufAddr, 0, atomicIbufSize);
    assert(!err);
  }

  // Register atomic ibuf as independent memory region.
  //
  // A device-resident ibuf is GPU memory just like the CQ/SQ/RQ rings, so a bare
  // VA only resolves if a peer memory client is registered with the RDMA stack.
  // That is not guaranteed: amdgpu's PeerDirect client is optional and resolves
  // ib_register_peer_memory_client via symbol_get at load time, so on a host
  // without it ibv_reg_mr returns NULL here while the dma-buf path works fine.
  // Try both, ordered the same way payload MRs are in RegisterRdmaMemoryRegionAuto.
  int atomicIbufAccessFlag =
      MaybeAddRelaxedOrderingFlag(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                  IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

  // errno of each failed attempt; dmabufErrno stays 0 when the dmabuf export itself
  // failed, since the HSA/HIP export paths do not set errno.
  int dmabufErrno = 0;
  int plainErrno = 0;
  auto tryIbufDmabuf = [&]() -> ibv_mr* {
    if (!config.onGpu) return nullptr;
    uint64_t dmabufOffset = 0;
    int dmabufFd = TryExportDmabufFd(atomicIbufAddr, atomicIbufSize, &dmabufOffset);
    if (dmabufFd < 0) return nullptr;
    ibv_mr* mr = ibv_reg_dmabuf_mr(pd_uxdma, dmabufOffset, atomicIbufSize,
                                   reinterpret_cast<uint64_t>(atomicIbufAddr), dmabufFd,
                                   atomicIbufAccessFlag);
    if (mr == nullptr) dmabufErrno = errno;
    close(dmabufFd);
    return mr;
  };
  auto tryIbufPlain = [&]() -> ibv_mr* {
    ibv_mr* mr = ibv_reg_mr(pd_uxdma, atomicIbufAddr, atomicIbufSize, atomicIbufAccessFlag);
    if (mr == nullptr) plainErrno = errno;
    return mr;
  };

  bool preferDmabuf = env::IsEnvVarEnabled("MORI_ENABLE_DMABUF_REG");
  for (int attempt = 0; attempt < 2 && (atomicIbufMr == nullptr); ++attempt) {
    bool useDmabuf = (attempt == 0) ? preferDmabuf : !preferDmabuf;
    atomicIbufMr = useDmabuf ? tryIbufDmabuf() : tryIbufPlain();
    if (atomicIbufMr) {
      MORI_APP_TRACE("IONIC atomic ibuf registered via {}, addr:{}, size:{}",
                     useDmabuf ? "dmabuf" : "ibv_reg_mr", atomicIbufAddr, atomicIbufSize);
    }
  }

  if (atomicIbufMr == nullptr) {
    std::string dmabufStatus = !config.onGpu      ? "skipped (host memory)"
                               : dmabufErrno != 0 ? strerror(dmabufErrno)
                                                  : "dmabuf export failed";
    MORI_APP_ERROR(
        "IONIC atomic ibuf registration failed (addr:{}, size:{}, onGpu:{}): ibv_reg_mr: "
        "errno {} ({}); dmabuf: {}. Device memory needs either a dma-buf capable "
        "libionic/driver or a registered peer memory client.",
        atomicIbufAddr, atomicIbufSize, config.onGpu, plainErrno, strerror(plainErrno),
        dmabufStatus);
    std::abort();
  }

  MORI_APP_TRACE(
      "IONIC Atomic ibuf allocated: addr=0x{:x}, slots={}, size={}, lkey=0x{:x}, rkey=0x{:x}",
      reinterpret_cast<uintptr_t>(atomicIbufAddr), RoundUpPowOfTwo(config.atomicIbufSlots),
      atomicIbufSize, atomicIbufMr->lkey, atomicIbufMr->rkey);
}

IonicQpContainer::~IonicQpContainer() {
  int err;

  // Clean up atomic internal buffer
  if (atomicIbufMr) {
    ibv_dereg_mr(atomicIbufMr);
    atomicIbufMr = nullptr;
  }

  if (atomicIbufAddr) {
    if (config.onGpu) {
      HIP_RUNTIME_CHECK(hipFree(atomicIbufAddr));
    } else {
      free(atomicIbufAddr);
    }
    atomicIbufAddr = nullptr;
  }

  err = ibv_destroy_qp(qp);
  assert(err == 0);
}

void* IonicQpContainer::GetSqAddress() { return reinterpret_cast<char*>(ionic_sq_buf); }

void* IonicQpContainer::GetRqAddress() { return reinterpret_cast<char*>(ionic_rq_buf); }

void IonicQpContainer::ModifyRst2Init() {
  int err;
  struct ibv_qp_attr attr;
  int attr_mask;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));

  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = config.portId;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_ATOMIC;

  attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  err = ibv_modify_qp(qp, &attr, attr_mask);
  if (err != 0) {
    MORI_APP_ERROR("ionic ModifyRst2Init failed: err:{} ({}), qpn:{}, portId:{}", err,
                   strerror(err), qp ? qp->qp_num : 0, config.portId);
  }
  assert(err == 0);
}

void IonicQpContainer::ModifyInit2Rtr(const RdmaEndpointHandle& local_handle,
                                      const RdmaEndpointHandle& remote_handle,
                                      const ibv_port_attr& portAttr, uint32_t qpn) {
  struct ibv_qp_attr attr;
  int attr_mask;
  int err;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = portAttr.active_mtu;
  attr.min_rnr_timer = 12;
  attr.max_dest_rd_atomic = 15;
  attr.rq_psn = remote_handle.psn;
  attr.dest_qp_num = remote_handle.qpn;

  // ah_atter
  memcpy(&attr.ah_attr.grh.dgid, remote_handle.eth.gid, 16);
  attr.ah_attr.grh.sgid_index = local_handle.eth.gidIdx;
  attr.ah_attr.port_num = config.portId;
  attr.ah_attr.is_global = 1;
  attr.ah_attr.grh.hop_limit = 1;
  attr.ah_attr.sl = ReadRdmaServiceLevelEnv().value_or(0);
  std::optional<uint8_t> tc = ReadRdmaTrafficClassEnv();
  if (tc.has_value()) {
    attr.ah_attr.grh.traffic_class = tc.value();
  }
  MORI_APP_INFO("ionic attr.ah_attr.sl:{} attr.ah_attr.grh.traffic_class:{}", attr.ah_attr.sl,
                attr.ah_attr.grh.traffic_class);

  attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_RQ_PSN | IBV_QP_DEST_QPN | IBV_QP_AV |
              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  MORI_APP_TRACE(
      "ModifyInit2Rtr, remote_handle.psn:{}, remote_handle.qpn:{}, config.gidIdx:{}, "
      "config.portId:{}\n",
      remote_handle.psn, remote_handle.qpn, config.gidIdx, config.portId);
#if 0
  for (int i = 0; i< 16; i++) {
    printf("%02x", remote_handle.eth.gid[i]);
  }
  printf("\n");
#endif
  err = ibv_modify_qp(qp, &attr, attr_mask);
  if (err != 0) {
    char dgidStr[48] = {0};
    for (int i = 0; i < 16; i++) {
      snprintf(dgidStr + i * 2, sizeof(dgidStr) - i * 2, "%02x", remote_handle.eth.gid[i]);
    }
    MORI_APP_ERROR(
        "ionic ModifyInit2Rtr failed: err:{} ({}), local_qpn:{}, dest_qpn:{}, rq_psn:{}, "
        "sgid_index:{}, portId:{}, path_mtu:{}, max_dest_rd_atomic:{}, sl:{}, tc:{}, dgid:{}",
        err, strerror(err), qp ? qp->qp_num : 0, attr.dest_qp_num, attr.rq_psn,
        attr.ah_attr.grh.sgid_index, config.portId, static_cast<int>(attr.path_mtu),
        attr.max_dest_rd_atomic, attr.ah_attr.sl, attr.ah_attr.grh.traffic_class, dgidStr);
  }
  assert(err == 0);
}

void IonicQpContainer::ModifyRtr2Rts(const RdmaEndpointHandle& local_handle,
                                     const RdmaEndpointHandle& remote_handle) {
  struct ibv_qp_attr attr;
  int attr_mask;
  int err;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 15;
  attr.sq_psn = remote_handle.psn;

  attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_TIMEOUT |
              IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY;

  err = ibv_modify_qp(qp, &attr, attr_mask);
  if (err != 0) {
    MORI_APP_ERROR("ionic ModifyRtr2Rts failed: err:{} ({}), local_qpn:{}, sq_psn:{}", err,
                   strerror(err), qp ? qp->qp_num : 0, attr.sq_psn);
  }
  assert(!err);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        IonicDeviceContext                                      */
/* ---------------------------------------------------------------------------------------------- */
void IonicDeviceContext::pd_release(struct ibv_pd* pd, void* pd_context, void* ptr,
                                    uint64_t resource_type) {
  HIP_RUNTIME_CHECK(hipFree(ptr));
}

void* IonicDeviceContext::pd_alloc_device_uncached(struct ibv_pd* pd, void* pd_context, size_t size,
                                                   size_t alignment, uint64_t resource_type) {
  void* dev_ptr{nullptr};
  HIP_RUNTIME_CHECK(
      hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size, hipDeviceMallocUncached));
  memset(dev_ptr, 0, size);
  return dev_ptr;
}

namespace {

constexpr size_t kDmabufRingAlign = 4096;
constexpr size_t kDmabufRingChunkBytes = size_t{16} << 20;

// Payload MR registration falls back to ibv_reg_mr when dma-buf fails; rings have no such
// fallback once pd_set_dmabuf_alloc is set. Export one ring-sized allocation first and keep
// bare-VA rings on a host where that fails.
bool ProbeIonicDmabufExport() {
  int device = -1;
  if (hipGetDevice(&device) != hipSuccess) {
    (void)hipGetLastError();
    MORI_APP_WARN(
        "MORI_ENABLE_DMABUF_REG=1 but there is no current HIP device for the dma-buf ring probe; "
        "CQ/SQ/RQ rings stay bare-VA");
    return false;
  }
  void* devPtr = nullptr;
  hipError_t err = hipExtMallocWithFlags(reinterpret_cast<void**>(&devPtr), kDmabufRingAlign,
                                         hipDeviceMallocUncached);
  if (err != hipSuccess) {
    (void)hipGetLastError();
    MORI_APP_WARN(
        "MORI_ENABLE_DMABUF_REG=1 but the dma-buf ring probe allocation failed ({}); CQ/SQ/RQ "
        "rings stay bare-VA",
        hipGetErrorString(err));
    return false;
  }
  uint64_t offset = 0;
  int fd = TryExportDmabufFd(devPtr, kDmabufRingAlign, &offset);
  if (fd >= 0) close(fd);
  if (hipFree(devPtr) != hipSuccess) (void)hipGetLastError();
  if (fd < 0) {
    MORI_APP_WARN(
        "MORI_ENABLE_DMABUF_REG=1 but dma-buf export of a ring-sized probe failed; CQ/SQ/RQ "
        "rings stay bare-VA (payload MR registration is unchanged)");
    return false;
  }
  return true;
}

}  // namespace

// Rings are slices of a few exported chunks. A ring's fd is the key the provider hands back to
// pd_free_dmabuf, so it stays open for the life of the ring; one export per ring would hold about
// three fds per QP.
int IonicDeviceContext::pd_alloc_dmabuf(struct ibv_pd* pd, void* pd_context, size_t size,
                                        uint64_t resource_type,
                                        struct ionic_dmabuf_alloc_result* result) {
  auto* self = static_cast<IonicDeviceContext*>(pd_context);
  if ((self == nullptr) || (result == nullptr) || (size == 0)) return EINVAL;

  int device = -1;
  if (hipGetDevice(&device) != hipSuccess) {
    (void)hipGetLastError();
    MORI_APP_WARN("pd_alloc_dmabuf: hipGetDevice failed; returning ECANCELED for a bare-VA ring");
    return ECANCELED;
  }
  size_t sliceBytes = (size + kDmabufRingAlign - 1) & ~(kDmabufRingAlign - 1);

  std::lock_guard<std::mutex> lock(self->dmabufRingsMutex);
  DmabufChunk* chunk = nullptr;
  for (auto& entry : self->dmabufChunks) {
    DmabufChunk& c = entry.second;
    if (c.device == device && c.used + sliceBytes <= c.size) {
      chunk = &c;
      break;
    }
  }

  if (chunk == nullptr) {
    size_t chunkBytes = std::max(kDmabufRingChunkBytes, sliceBytes);
    void* base = nullptr;
    hipError_t err =
        hipExtMallocWithFlags(reinterpret_cast<void**>(&base), chunkBytes, hipDeviceMallocUncached);
    if (err != hipSuccess) {
      (void)hipGetLastError();
      MORI_APP_WARN(
          "pd_alloc_dmabuf: hipExtMallocWithFlags({}) failed ({}), resource_type:{:#x}; returning "
          "ECANCELED for a bare-VA ring",
          chunkBytes, hipGetErrorString(err), resource_type);
      return ECANCELED;
    }
    uint64_t exportOffset = 0;
    int fd = TryExportDmabufFd(base, chunkBytes, &exportOffset);
    // The kernel rejects ring offsets that are not page aligned.
    if ((fd >= 0) && (exportOffset % kDmabufRingAlign != 0)) {
      close(fd);
      fd = -1;
    }
    if (fd < 0) {
      MORI_APP_WARN(
          "pd_alloc_dmabuf: dma-buf export of a {}-byte ring chunk failed or is not page aligned, "
          "resource_type:{:#x}; returning ECANCELED for a bare-VA ring",
          chunkBytes, resource_type);
      if (hipFree(base) != hipSuccess) (void)hipGetLastError();
      return ECANCELED;
    }
    DmabufChunk fresh;
    fresh.base = base;
    fresh.size = chunkBytes;
    fresh.exportOffset = exportOffset;
    fresh.device = device;
    fresh.fd = fd;
    chunk = &self->dmabufChunks.emplace(fd, std::move(fresh)).first->second;
    MORI_APP_TRACE("pd_alloc_dmabuf: new ring chunk, addr:{}, size:{}, fd:{}, device:{}, chunks:{}",
                   base, chunkBytes, fd, device, self->dmabufChunks.size());
  }

  size_t sliceOffset = chunk->used;
  chunk->used += sliceBytes;
  void* ptr = static_cast<char*>(chunk->base) + sliceOffset;
  memset(ptr, 0, size);
  uint64_t dmabufOffset = chunk->exportOffset + sliceOffset;
  chunk->live.insert(dmabufOffset);

  result->fd = chunk->fd;
  result->offset = dmabufOffset;
  // Hand back the VA too so the provider can still drive ibv_post_send/recv and ibv_poll_cq.
  result->ptr = ptr;

  MORI_APP_TRACE("pd_alloc_dmabuf, addr:{}, size:{}, fd:{}, offset:{}, resource_type:{:#x}", ptr,
                 size, chunk->fd, dmabufOffset, resource_type);
  return 0;
}

void IonicDeviceContext::pd_free_dmabuf(struct ibv_pd* pd, void* pd_context, int fd,
                                        uint64_t offset, uint64_t resource_type) {
  auto* self = static_cast<IonicDeviceContext*>(pd_context);
  if (self == nullptr) return;

  void* base{nullptr};
  {
    std::lock_guard<std::mutex> lock(self->dmabufRingsMutex);
    auto it = self->dmabufChunks.find(fd);
    if ((it == self->dmabufChunks.end()) || (it->second.live.erase(offset) == 0)) {
      MORI_APP_WARN("pd_free_dmabuf: unknown ring fd:{}, offset:{}, resource_type:{:#x}", fd,
                    offset, resource_type);
      return;
    }
    if (!it->second.live.empty()) {
      MORI_APP_TRACE("pd_free_dmabuf, fd:{}, offset:{}, resource_type:{:#x}, rings left:{}", fd,
                     offset, resource_type, it->second.live.size());
      return;
    }
    base = it->second.base;
    self->dmabufChunks.erase(it);
  }

  // Closed only with the chunk's last ring: the provider borrows the fd, and the same number is
  // shared by every ring in the chunk.
  close(fd);
  HIP_RUNTIME_CHECK(hipFree(base));
  MORI_APP_TRACE("pd_free_dmabuf: released ring chunk, addr:{}, fd:{}, resource_type:{:#x}", base,
                 fd, resource_type);
}

void IonicDeviceContext::create_parent_domain(ibv_context* context, struct ibv_pd* pd_orig) {
  struct ibv_parent_domain_init_attr pattr;

  memset(&pattr, 0, sizeof(struct ibv_parent_domain_init_attr));
  pattr.pd = pd_orig;
  pattr.td = nullptr, pattr.comp_mask = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS;
  pattr.free = IonicDeviceContext::pd_release;
  pattr.pd_context = nullptr;
  pattr.alloc = IonicDeviceContext::pd_alloc_device_uncached;

  // Same switch as payload MRs: MORI_ENABLE_DMABUF_REG=1 enables dma-buf CQ/SQ/RQ
  // rings. Needs libionic >= the release that exports ionic_dv_pd_set_dmabuf_alloc;
  // older libs keep the bare-VA alloc above.
  bool wantDmabufRings = env::IsEnvVarEnabled("MORI_ENABLE_DMABUF_REG");
  bool haveDmabufAlloc = IonicDvApi::Instance().pd_set_dmabuf_alloc != nullptr;
  bool useDmabufRings = wantDmabufRings && haveDmabufAlloc && ProbeIonicDmabufExport();
  if (wantDmabufRings && !haveDmabufAlloc) {
    MORI_APP_WARN(
        "libionic has no ionic_dv_pd_set_dmabuf_alloc; CQ/SQ/RQ rings fall back to bare-VA "
        "allocation");
  }
#if 0
  pd_parent = ibv_alloc_parent_domain(defaultContext, &pattr);
  assert(pd_parent);

  IonicDvApi::Instance().pd_set_sqcmb(pd_parent, false, false, false);
  IonicDvApi::Instance().pd_set_rqcmb(pd_parent, false, false, false);
#endif
  for (int i = 0; i < 2; i++) {
    pd_uxdma[i] = ibv_alloc_parent_domain(context, &pattr);
    assert(pd_uxdma[i]);
    // printf("create_parent_domain, pd_uxdma:%p\n", pd_uxdma[i]);
    if (useDmabufRings) {
      int err = IonicDvApi::Instance().pd_set_dmabuf_alloc(
          pd_uxdma[i], IonicDeviceContext::pd_alloc_dmabuf, IonicDeviceContext::pd_free_dmabuf,
          this);
      if (err) {
        MORI_APP_WARN(
            "ionic_dv_pd_set_dmabuf_alloc failed on pd_uxdma[{}] (err:{}); falling back to bare-VA "
            "rings",
            i, err);
      } else {
        dmabufRingsEnabled = true;
        MORI_APP_TRACE("dmabuf descriptor rings enabled on pd_uxdma[{}]", i);
      }
    }
    IonicDvApi::Instance().pd_set_sqcmb(pd_uxdma[i], false, false, false);
    IonicDvApi::Instance().pd_set_rqcmb(pd_uxdma[i], false, false, false);
    IonicDvApi::Instance().pd_set_udma_mask(pd_uxdma[i], 1u << i);
  }
}

IonicDeviceContext::IonicDeviceContext(RdmaDevice* rdma_device, ibv_context* context, ibv_pd* in_pd)
    : RdmaDeviceContext(rdma_device, in_pd) {
  create_parent_domain(context, in_pd);
}

IonicDeviceContext::~IonicDeviceContext() {
  for (auto& it : qpPool) {
    delete it.second;
  }
  qpPool.clear();

  for (auto& it : cqPool) {
    delete it.second;
  }
  cqPool.clear();

  // Parent domains can only be released once no CQ/QP references them. Only done with
  // dmabuf rings, whose fds must be returned; the default path keeps them until the
  // device is closed, as it always has.
  for (int i = 0; i < 2 && dmabufRingsEnabled; i++) {
    if (pd_uxdma[i] == nullptr) continue;
    int err = ibv_dealloc_pd(pd_uxdma[i]);
    if (err != 0) {
      MORI_APP_WARN("ibv_dealloc_pd failed on pd_uxdma[{}]: err:{} ({})", i, err, strerror(err));
    }
    pd_uxdma[i] = nullptr;
  }
}

RdmaEndpoint IonicDeviceContext::CreateRdmaEndpoint(const RdmaEndpointConfig& config) {
  ibv_context* context = GetIbvContext();
  int ret;

  assert(!config.withCompChannel && !config.enableSrq && "not implemented");

  struct ibv_pd* pd = pd_uxdma[qp_counter & 1];
  qp_counter++;
  IonicCqContainer* cq = new IonicCqContainer(context, config, pd);
  // printf("CreateRdmaEndpoint, context:%p, cq->cq:%p, pd_uxdma:%p\n", context, cq->cq, pd);
  IonicQpContainer* qp = new IonicQpContainer(context, config, cq->cq, pd, this);

  RdmaEndpoint endpoint;
  endpoint.handle.psn = 0;
  endpoint.handle.portId = config.portId;
  endpoint.handle.qpn = qp->qpn;

  const ibv_port_attr* gidPortAttr = GetRdmaDevice()->GetPortAttr(config.portId);
  assert(gidPortAttr);
  GidSelectionResult gidSelection =
      AutoSelectGidIndex(context, config.portId, gidPortAttr, config.gidIdx);
  assert(gidSelection.gidIdx >= 0 && gidSelection.valid);
  memcpy(endpoint.handle.eth.gid, gidSelection.gid.raw, sizeof(endpoint.handle.eth.gid));
  endpoint.handle.eth.gidIdx = gidSelection.gidIdx;
#if 0
  for (int i = 0; i< 16; i++) {
    printf("%02x", endpoint.handle.eth.gid[i]);
  }
  printf("\n");
#endif
  endpoint.vendorId = RdmaDeviceVendorId::Pensando;
  endpoint.wqHandle.sqAddr = qp->GetSqAddress();
  endpoint.wqHandle.rqAddr = qp->GetRqAddress();
  endpoint.wqHandle.dbrAddr = qp->gpu_db_sq;
  endpoint.wqHandle.sqWqeNum = qp->sq_mask + 1;  // qp->wqeNum;
  endpoint.wqHandle.rqWqeNum = qp->rq_mask + 1;  // qp->wqeNum;
  endpoint.wqHandle.rqdbrAddr = qp->gpu_db_rq;

  endpoint.wqHandle.color = true;
  endpoint.wqHandle.sq_dbval = qp->sq_dbval;
  endpoint.wqHandle.rq_dbval = qp->rq_dbval;

  endpoint.cqHandle.cqAddr = qp->ionic_cq_buf;
  endpoint.cqHandle.consIdx = 0;
  endpoint.cqHandle.cqeNum = qp->cq_mask + 1;
  endpoint.cqHandle.cqeSize = GetIonicCqeSize();
  endpoint.cqHandle.dbrAddr = qp->gpu_db_cq;
  endpoint.cqHandle.dbrRecAddr = qp->gpu_db_cq;
  endpoint.cqHandle.cq_dbval = qp->cq_dbval;

  // Set atomic internal buffer information
  endpoint.atomicIbuf.addr = reinterpret_cast<uintptr_t>(qp->atomicIbufAddr);
  endpoint.atomicIbuf.lkey = qp->atomicIbufMr->lkey;
  endpoint.atomicIbuf.rkey = qp->atomicIbufMr->rkey;
  endpoint.atomicIbuf.nslots = RoundUpPowOfTwo(config.atomicIbufSlots);
  // One CQ per endpoint, keyed by its QP's qpn: ionic leaves cqn unset (0 for every CQ).
  // Tracked (and so destroyed at teardown) only with dmabuf rings, to release the ring fd.
  if (dmabufRingsEnabled) cqPool.insert({qp->qpn, cq});
  qpPool.insert({qp->qpn, qp});

  MORI_APP_TRACE(
      "Ionic endpoint created: qpn={}, cqn={}, portId={}, gidIdx={}, atomicIbuf addr=0x{:x}, "
      "nslots={}",
      qp->qpn, cq->cqn, config.portId, config.gidIdx, endpoint.atomicIbuf.addr,
      endpoint.atomicIbuf.nslots);

  return endpoint;
}

void IonicDeviceContext::ConnectEndpoint(const RdmaEndpointHandle& local,
                                         const RdmaEndpointHandle& remote, uint32_t qpn) {
  uint32_t local_qpn = local.qpn;

  assert(qpPool.find(local_qpn) != qpPool.end());
  IonicQpContainer* qp = qpPool.at(local_qpn);

  MORI_APP_TRACE("Ionic connecting endpoint: local_qpn={}, remote_qpn={}, qpId={}", local_qpn,
                 remote.qpn, qpn);

  RdmaDevice* rdmaDevice = GetRdmaDevice();
  const ibv_device_attr_ex* deviceAttr = rdmaDevice->GetDeviceAttr();
  const ibv_port_attr& portAttr = *(rdmaDevice->GetPortAttrMap()->find(local.portId)->second);
  qp->ModifyRst2Init();
  // qpn unused for now, other vendor for udp multi-sport
  qp->ModifyInit2Rtr(local, remote, portAttr, qpn);
  qp->ModifyRtr2Rts(local, remote);

  MORI_APP_TRACE("Ionic endpoint connected successfully: local_qpn={}, remote_qpn={}", local_qpn,
                 remote.qpn);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           IonicDevice                                          */
/* ---------------------------------------------------------------------------------------------- */
IonicDevice::IonicDevice(ibv_device* in_device) : RdmaDevice(in_device) {}
IonicDevice::~IonicDevice() {}

RdmaDeviceContext* IonicDevice::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(defaultContext);
  assert(pd);
  // printf("IonicDevice::CreateRdmaDeviceContext, defaultContext:%p, pd:%p\n", defaultContext, pd);
  return new IonicDeviceContext(this, defaultContext, pd);
}
}  // namespace application
}  // namespace mori
