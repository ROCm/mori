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
#include "mori/application/transport/rdma/providers/mlx5/mlx5.hpp"

#include <hip/hip_runtime_api.h>
#include <infiniband/verbs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "mori/application/transport/rdma/providers/mlx5/mlx5_ifc.hpp"
#include "mori/application/transport/rdma/providers/mlx5/mlx5_prm.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/application/utils/math.hpp"
#include "mori/utils/env_utils.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

namespace {

// Multiple MLX5 QPs can share the same devx UAR reg_addr. hipHostRegister is
// once-per-host-range per process; ref-count so we only unregister after the
// last QP using that address is torn down. hipErrorAlreadyMapped is treated as
// reuse (e.g. shared UAR page, or a prior registration still visible to HIP).
struct Mlx5UarHostRegInfo {
  int refs{0};
  bool registered{false};  // true only if this process called hipHostRegister successfully
};

std::unordered_map<void*, Mlx5UarHostRegInfo> g_mlx5_uar_host_regs;

void Mlx5RegisterUarHost(void* reg_addr, size_t size) {
  auto& info = g_mlx5_uar_host_regs[reg_addr];
  if (info.refs == 0) {
    uint32_t flag = hipHostRegisterPortable | hipHostRegisterMapped;
    hipError_t err = hipHostRegister(reg_addr, size, flag);
    if (err == hipSuccess) {
      info.registered = true;
    } else if (err != hipErrorAlreadyMapped) {
      fprintf(stderr, "[%s:%d] hip failed with %s \n", __FILE__, __LINE__, hipGetErrorString(err));
      exit(-1);
    }
  }
  ++info.refs;
}

void Mlx5UnregisterUarHost(void* reg_addr) {
  auto it = g_mlx5_uar_host_regs.find(reg_addr);
  if (it == g_mlx5_uar_host_regs.end()) return;
  auto& info = it->second;
  assert(info.refs > 0);
  if (--info.refs == 0) {
    if (info.registered) {
      hipError_t err = hipHostUnregister(reg_addr);
      if (err != hipSuccess && err != hipErrorHostMemoryNotRegistered) {
        fprintf(stderr, "[%s:%d] hip failed with %s \n", __FILE__, __LINE__,
                hipGetErrorString(err));
        exit(-1);
      }
    }
    g_mlx5_uar_host_regs.erase(it);
  }
}

// GPU control structures (CQ/WQ rings, doorbell records, atomic ibuf) use one registration
// mechanism per process: ROCr packs them into shared BOs, and amdgpu can't pin a BO in VRAM
// (peermem) and GTT (dmabuf without PCIe P2P to the NIC) at once.
enum class Mlx5GpuRegMode { kPeerMem, kDmabuf };

// ROCr gives allocations of whole 2 MiB units their own BO, at dmabuf offset 0.
constexpr size_t kMlx5OwnBoSize = 2 * 1024 * 1024;
constexpr size_t kMlx5DbrSize = 64;  // doorbell record, right after its ring

// Returns nullptr with errno set. The umem must sit at dmabuf offset 0: rdma-core passes the
// offset to ibv_dontfork_range() as an address, which fails once ibv_fork_init() has run.
mlx5dv_devx_umem* Mlx5RegisterDmabufUmem(ibv_context* context, void* addr, size_t size,
                                         uint32_t accessFlag, uint64_t* offset) {
  *offset = 0;
  int dmabufFd =
      Mlx5DvApi::Instance().devx_umem_reg_ex ? TryExportDmabufFd(addr, size, offset) : -1;
  if (dmabufFd < 0) {
    errno = EOPNOTSUPP;
    return nullptr;
  }
  if (*offset != 0) {
    close(dmabufFd);
    errno = EINVAL;
    return nullptr;
  }
  MoriMlx5DevxUmemIn in{};
  in.addr = reinterpret_cast<void*>(*offset);
  in.size = size;
  in.access = accessFlag;
  in.pgsz_bitmap = ~((1ULL << MLX5_ADAPTER_PAGE_SHIFT) - 1);
  in.comp_mask = MORI_MLX5DV_UMEM_MASK_DMABUF;
  in.dmabuf_fd = dmabufFd;
  mlx5dv_devx_umem* umem = Mlx5DvApi::Instance().devx_umem_reg_ex(context, &in);
  int err = errno;
  close(dmabufFd);
  errno = err;
  return umem;
}

ibv_mr* Mlx5RegisterDmabufMr(ibv_pd* pd, void* addr, size_t size, int accessFlag) {
  uint64_t offset = 0;
  int dmabufFd = TryExportDmabufFd(addr, size, &offset);
  if (dmabufFd < 0) {
    errno = EOPNOTSUPP;
    return nullptr;
  }
  ibv_mr* mr =
      ibv_reg_dmabuf_mr(pd, offset, size, reinterpret_cast<uint64_t>(addr), dmabufFd, accessFlag);
  int err = errno;
  close(dmabufFd);
  errno = err;
  return mr;
}

// MORI_MLX5_DMABUF=force/1 or off/0 pins the mode; otherwise probe once, preferring peermem
// since dmabuf may pin rings in GTT and needs a BO per ring. MORI_ENABLE_DMABUF_REG governs the
// payload MR path instead.
Mlx5GpuRegMode ProbeMlx5GpuRegMode(ibv_context* context) {
  std::optional<std::string> forced = mori::env::GetString("MORI_MLX5_DMABUF");
  if (forced && (*forced == "force" || *forced == "1")) return Mlx5GpuRegMode::kDmabuf;
  if (forced && (*forced == "off" || *forced == "0")) return Mlx5GpuRegMode::kPeerMem;

  constexpr size_t kProbeSize = 4096;
  void* scratch = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&scratch, kMlx5OwnBoSize, hipDeviceMallocUncached));

  auto tryRegister = [](mlx5dv_devx_umem* umem) {
    if (umem == nullptr) return errno;
    Mlx5DvApi::Instance().devx_umem_dereg(umem);
    return 0;
  };
  int peermemErr = tryRegister(
      Mlx5DvApi::Instance().devx_umem_reg(context, scratch, kProbeSize, IBV_ACCESS_LOCAL_WRITE));
  int dmabufErr = 0;
  if (peermemErr != 0) {
    uint64_t offset = 0;
    dmabufErr = tryRegister(
        Mlx5RegisterDmabufUmem(context, scratch, kProbeSize, IBV_ACCESS_LOCAL_WRITE, &offset));
  }
  HIP_RUNTIME_CHECK(hipFree(scratch));

  if (peermemErr == 0) return Mlx5GpuRegMode::kPeerMem;
  if (dmabufErr == 0) return Mlx5GpuRegMode::kDmabuf;
  MORI_APP_ERROR(
      "MLX5: neither peermem (errno={} ({})) nor dmabuf (errno={} ({})) can register GPU "
      "memory with this NIC; set MORI_MLX5_DMABUF to force one and see its error",
      peermemErr, strerror(peermemErr), dmabufErr, strerror(dmabufErr));
  std::abort();
}

Mlx5GpuRegMode GetMlx5GpuRegMode(ibv_context* context) {
  static std::once_flag once;
  static Mlx5GpuRegMode mode = Mlx5GpuRegMode::kPeerMem;
  std::call_once(once, [context] {
    mode = ProbeMlx5GpuRegMode(context);
    MORI_APP_INFO("MLX5 GPU control structures register via {}",
                  mode == Mlx5GpuRegMode::kPeerMem ? "peermem" : "dmabuf");
  });
  return mode;
}

void Mlx5FillControlBuf(bool onGpu, void* ptr, int value, size_t size) {
  if (onGpu) {
    HIP_RUNTIME_CHECK(hipMemset(ptr, value, size));
  } else {
    memset(ptr, value, size);
  }
}

void* Mlx5AllocControlBuf(bool onGpu, size_t size, size_t alignment) {
  void* ptr = nullptr;
  if (onGpu) {
    HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&ptr, size, hipDeviceMallocUncached));
  } else {
    [[maybe_unused]] int status = posix_memalign(&ptr, alignment, size);
    assert(status == 0);
  }
  Mlx5FillControlBuf(onGpu, ptr, 0, size);
  return ptr;
}

// dmabuf-mode umems get a BO of their own (see Mlx5RegisterDmabufUmem).
void* Mlx5AllocControlUmemBuf(ibv_context* context, bool onGpu, size_t size, size_t alignment) {
  if (onGpu && GetMlx5GpuRegMode(context) == Mlx5GpuRegMode::kDmabuf)
    size = (size + kMlx5OwnBoSize - 1) / kMlx5OwnBoSize * kMlx5OwnBoSize;
  return Mlx5AllocControlBuf(onGpu, size, alignment);
}

void Mlx5FreeControlBuf(bool onGpu, void* ptr) {
  if (ptr == nullptr) return;
  if (onGpu) {
    HIP_RUNTIME_CHECK(hipFree(ptr));
  } else {
    free(ptr);
  }
}

[[noreturn]] void Mlx5AbortRegistration(const char* kind, const char* what, const char* how,
                                        void* addr, size_t size, int err,
                                        uint64_t dmabufOffset = 0) {
  MORI_APP_ERROR("MLX5 {} [{}] {} registration failed (addr=0x{:x}, size={}, errno={} ({}))", kind,
                 what, how, reinterpret_cast<uintptr_t>(addr), size, err, strerror(err));
  if (dmabufOffset != 0) {
    MORI_APP_ERROR(
        "MLX5 {} [{}] sits at dmabuf offset 0x{:x}; a dmabuf umem must start its own BO, since "
        "rdma-core fails it at a nonzero offset once ibv_fork_init() has run",
        kind, what, dmabufOffset);
  }
  std::abort();
}

// Host memory and peermem register by VA. No fallback by design: failures abort.
mlx5dv_devx_umem* Mlx5RegisterControlUmem(ibv_context* context, bool onGpu, void* addr, size_t size,
                                          const char* what) {
  const bool dmabuf = onGpu && GetMlx5GpuRegMode(context) == Mlx5GpuRegMode::kDmabuf;
  const char* how = dmabuf ? "dmabuf" : onGpu ? "peermem" : "host";
  uint64_t offset = 0;
  mlx5dv_devx_umem* umem =
      dmabuf ? Mlx5RegisterDmabufUmem(context, addr, size, IBV_ACCESS_LOCAL_WRITE, &offset)
             : Mlx5DvApi::Instance().devx_umem_reg(context, addr, size, IBV_ACCESS_LOCAL_WRITE);
  if (umem == nullptr) Mlx5AbortRegistration("control umem", what, how, addr, size, errno, offset);
  MORI_APP_TRACE("MLX5 control umem [{}] registered via {}: addr=0x{:x}, size={}", what, how,
                 reinterpret_cast<uintptr_t>(addr), size);
  return umem;
}

ibv_mr* Mlx5RegisterControlMr(ibv_context* context, ibv_pd* pd, bool onGpu, void* addr, size_t size,
                              int accessFlag, const char* what) {
  const bool dmabuf = onGpu && GetMlx5GpuRegMode(context) == Mlx5GpuRegMode::kDmabuf;
  const char* how = dmabuf ? "dmabuf" : onGpu ? "peermem" : "host";
  ibv_mr* mr = dmabuf ? Mlx5RegisterDmabufMr(pd, addr, size, accessFlag)
                      : ibv_reg_mr(pd, addr, size, accessFlag);
  if (mr == nullptr) Mlx5AbortRegistration("MR", what, how, addr, size, errno);
  MORI_APP_TRACE("MLX5 MR [{}] registered via {}: addr=0x{:x}, size={}", what, how,
                 reinterpret_cast<uintptr_t>(addr), size);
  return mr;
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                        Device Attributes                                       */
/* ---------------------------------------------------------------------------------------------- */
HcaCapability QueryHcaCap(ibv_context* context) {
  int status;
  uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
      0,
  };
  uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
      0,
  };

  DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
  DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod, HCA_CAP_OPMOD_GET_CUR);

  status = Mlx5DvApi::Instance().devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in),
                                                  cmd_cap_out, sizeof(cmd_cap_out));
  assert(!status);

  HcaCapability hca_cap;

  hca_cap.portType = DEVX_GET(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap.port_type);

  uint32_t logBfRegSize =
      DEVX_GET(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap.log_bf_reg_size);
  hca_cap.dbrRegSize = 1LLU << logBfRegSize;

  MORI_APP_TRACE("MLX5 HCA capabilities: portType={}, dbrRegSize={}", hca_cap.portType,
                 hca_cap.dbrRegSize);

  return hca_cap;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Mlx5CqContainer */
/* ---------------------------------------------------------------------------------------------- */
Mlx5CqContainer::Mlx5CqContainer(ibv_context* context, const RdmaEndpointConfig& config)
    : config(config) {
  int status;
  uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_cq_in)] = {
      0,
  };
  uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_cq_out)] = {
      0,
  };

  // Allocate user memory for CQ
  // TODO: accept memory allocated by user?
  cqeNum = config.maxCqeNum;
  int cqSize = RoundUpPowOfTwo(GetMlx5CqeSize() * cqeNum);
  // TODO: adjust cqe_num after aligning?
  cqSize = (cqSize + config.alignment - 1) / config.alignment * config.alignment;

  const size_t cqUmemSize = cqSize + kMlx5DbrSize;
  cqUmemAddr = Mlx5AllocControlUmemBuf(context, config.onGpu, cqUmemSize, config.alignment);
  cqDbrUmemAddr = static_cast<char*>(cqUmemAddr) + cqSize;
  // Init CQ buffer to 0xff so wqe_counter reads 0xffff ("nothing completed")
  // until the NIC writes a real completion (zero-init would look like WQE 0 done).
  Mlx5FillControlBuf(config.onGpu, cqUmemAddr, 0xff, cqSize);
  cqUmem = Mlx5RegisterControlUmem(context, config.onGpu, cqUmemAddr, cqUmemSize, "cq");

  // Allocate user access region
  uar = Mlx5DvApi::Instance().devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC);
  assert(uar->page_id != 0);

  // Initialize CQ
  DEVX_SET(create_cq_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_CQ);
  DEVX_SET(create_cq_in, cmd_in, cq_umem_valid, 0x1);
  DEVX_SET(create_cq_in, cmd_in, cq_umem_id, cqUmem->umem_id);

  void* cq_context = DEVX_ADDR_OF(create_cq_in, cmd_in, cq_context);
  DEVX_SET(cqc, cq_context, dbr_umem_valid, 0x1);
  DEVX_SET(cqc, cq_context, dbr_umem_id, cqUmem->umem_id);
  DEVX_SET64(cqc, cq_context, dbr_addr, cqSize);  // Byte offset into dbr_umem_id
  // Collapsed CQ: cc=1 collapses all completions into CQE slot 0, oi=1 ignores
  // overrun (no CQ consumer doorbell); progress is tracked via CQE[0].wqe_counter.
  // cqe_sz=0 selects 64B CQEs.
  DEVX_SET(cqc, cq_context, cqe_sz, 0x0);
  DEVX_SET(cqc, cq_context, cc, 0x1);
  DEVX_SET(cqc, cq_context, oi, 0x1);
  DEVX_SET(cqc, cq_context, log_cq_size, LogCeil2(cqeNum));
  DEVX_SET(cqc, cq_context, uar_page, uar->page_id);

  uint32_t eqn;
  status = Mlx5DvApi::Instance().devx_query_eqn(context, 0, &eqn);
  assert(!status);
  DEVX_SET(cqc, cq_context, c_eqn, eqn);

  cq = Mlx5DvApi::Instance().devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out,
                                             sizeof(cmd_out));
  assert(cq);

  cqn = DEVX_GET(create_cq_out, cmd_out, cqn);

  MORI_APP_TRACE("MLX5 CQ created: cqn={}, cqeNum={}, cqSize={}, cqUmemAddr=0x{:x}, uar_page_id={}",
                 cqn, cqeNum, cqSize, reinterpret_cast<uintptr_t>(cqUmemAddr), uar->page_id);
}

Mlx5CqContainer::~Mlx5CqContainer() {
  // Destroy the firmware CQ before releasing the UMEM/UAR it references, then free
  // the CQ/DBR backing memory (previously leaked on every endpoint teardown).
  if (cq) {
    Mlx5DvApi::Instance().devx_obj_destroy(cq);
    cq = nullptr;
  }
  if (cqUmem) Mlx5DvApi::Instance().devx_umem_dereg(cqUmem);
  if (uar) Mlx5DvApi::Instance().devx_free_uar(uar);
  Mlx5FreeControlBuf(config.onGpu, cqUmemAddr);
  cqUmemAddr = nullptr;
  cqDbrUmemAddr = nullptr;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Mlx5QpContainer                                        */
/* ---------------------------------------------------------------------------------------------- */
Mlx5QpContainer::Mlx5QpContainer(ibv_context* context, const RdmaEndpointConfig& config,
                                 uint32_t cqn, uint32_t pdn, Mlx5DeviceContext* device_context)
    : context(context), config(config), device_context(device_context) {
  ComputeQueueAttrs(config);
  CreateQueuePair(cqn, pdn);
}

Mlx5QpContainer::~Mlx5QpContainer() { DestroyQueuePair(); }

void Mlx5QpContainer::ComputeQueueAttrs(const RdmaEndpointConfig& config) {
  // Receive queue attributes
  rqAttrs.wqeSize = GetMlx5RqWqeSize();
  uint32_t rqMaxWr = config.maxRecvWr != 0 ? config.maxRecvWr : config.maxMsgsNum;
  uint32_t maxMsgsNum = RoundUpPowOfTwo(config.maxMsgsNum);
  uint32_t rqMaxWrRounded = RoundUpPowOfTwo(rqMaxWr);
  rqAttrs.wqSize = std::max(rqAttrs.wqeSize * rqMaxWrRounded, uint32_t(MLX5_SEND_WQE_BB));
  rqAttrs.wqeNum = ceil(rqAttrs.wqSize / rqAttrs.wqeSize);
  rqAttrs.wqeShift = log2(rqAttrs.wqeSize - 1) + 1;
  rqAttrs.offset = 0;

  // Send queue attributes
  sqAttrs.offset = rqAttrs.wqSize;
  sqAttrs.wqeSize = GetMlx5SqWqeSize();
  sqAttrs.wqSize = RoundUpPowOfTwo(sqAttrs.wqeSize * maxMsgsNum);
  sqAttrs.wqeNum = ceil(sqAttrs.wqSize / MLX5_SEND_WQE_BB);
  sqAttrs.wqeShift = MLX5_SEND_WQE_SHIFT;

  // Queue pair attributes
  qpTotalSize = RoundUpPowOfTwo(rqAttrs.wqSize + sqAttrs.wqSize);
  qpTotalSize = (qpTotalSize + config.alignment - 1) / config.alignment * config.alignment;

  MORI_APP_TRACE(
      "MLX5 Queue attributes computed - RQ: wqeSize={}, wqSize={}, wqeNum={}, offset={} | SQ: "
      "wqeSize={}, wqSize={}, wqeNum={}, offset={} | Total: {}",
      rqAttrs.wqeSize, rqAttrs.wqSize, rqAttrs.wqeNum, rqAttrs.offset, sqAttrs.wqeSize,
      sqAttrs.wqSize, sqAttrs.wqeNum, sqAttrs.offset, qpTotalSize);
}

void Mlx5QpContainer::CreateQueuePair(uint32_t cqn, uint32_t pdn) {
  uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_qp_in)] = {
      0,
  };
  uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_qp_out)] = {
      0,
  };

  // QP umem: RQ, SQ, then the doorbell record
  const size_t qpDbrOffset =
      (rqAttrs.wqSize + sqAttrs.wqSize + kMlx5DbrSize - 1) / kMlx5DbrSize * kMlx5DbrSize;
  const size_t qpUmemSize = std::max(qpTotalSize, qpDbrOffset + kMlx5DbrSize);
  qpUmemAddr = Mlx5AllocControlUmemBuf(context, config.onGpu, qpUmemSize, config.alignment);
  qpDbrUmemAddr = static_cast<char*>(qpUmemAddr) + qpDbrOffset;
  qpUmem = Mlx5RegisterControlUmem(context, config.onGpu, qpUmemAddr, qpUmemSize, "wq");

  // Allocate and register atomic internal buffer (ibuf) as an independent memory region
  atomicIbufSize = (RoundUpPowOfTwo(config.atomicIbufSlots) + 1) * ATOMIC_IBUF_SLOT_SIZE;
  atomicIbufAddr = Mlx5AllocControlBuf(config.onGpu, atomicIbufSize, config.alignment);
  int atomicIbufAccessFlag =
      MaybeAddRelaxedOrderingFlag(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                  IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
  atomicIbufMr =
      Mlx5RegisterControlMr(context, device_context->GetIbvPd(), config.onGpu, atomicIbufAddr,
                            atomicIbufSize, atomicIbufAccessFlag, "atomic_ibuf");

  MORI_APP_TRACE(
      "MLX5 Atomic ibuf allocated: addr=0x{:x}, slots={}, size={}, lkey=0x{:x}, rkey=0x{:x}",
      reinterpret_cast<uintptr_t>(atomicIbufAddr), RoundUpPowOfTwo(config.atomicIbufSlots),
      atomicIbufSize, atomicIbufMr->lkey, atomicIbufMr->rkey);

  // Allocate user access region
  qpUar = Mlx5DvApi::Instance().devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC);
  assert(qpUar);
  assert(qpUar->page_id != 0);

  if (config.onGpu) {
    Mlx5RegisterUarHost(qpUar->reg_addr, QueryHcaCap(context).dbrRegSize);
    HIP_RUNTIME_CHECK(hipHostGetDevicePointer(&qpUarPtr, qpUar->reg_addr, 0));
  } else {
    qpUarPtr = qpUar->reg_addr;
  }

  // TODO: check for correctness
  uint32_t logRqSize = int(log2(rqAttrs.wqeNum - 1)) + 1;
  uint32_t logRqStride = rqAttrs.wqeShift - 4;
  uint32_t logSqSize = int(log2(sqAttrs.wqeNum - 1)) + 1;

  // Initialize QP
  DEVX_SET(create_qp_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_QP);
  DEVX_SET(create_qp_in, cmd_in, wq_umem_id, qpUmem->umem_id);
  DEVX_SET64(create_qp_in, cmd_in, wq_umem_offset, 0);
  DEVX_SET(create_qp_in, cmd_in, wq_umem_valid, 0x1);

  void* qp_context = DEVX_ADDR_OF(create_qp_in, cmd_in, qpc);
  DEVX_SET(qpc, qp_context, st, MLX5_QPC_ST_RC);
  DEVX_SET(qpc, qp_context, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
  DEVX_SET(qpc, qp_context, pd, pdn);
  DEVX_SET(qpc, qp_context, uar_page, qpUar->page_id);  // BF register
  DEVX_SET(qpc, qp_context, cqn_snd, cqn);
  DEVX_SET(qpc, qp_context, cqn_rcv, cqn);
  DEVX_SET(qpc, qp_context, log_sq_size, logSqSize);
  DEVX_SET(qpc, qp_context, log_rq_size, logRqSize);
  DEVX_SET(qpc, qp_context, log_rq_stride, logRqStride);
  DEVX_SET(qpc, qp_context, ts_format, 0x1);
  DEVX_SET(qpc, qp_context, cs_req, 0);
  DEVX_SET(qpc, qp_context, cs_res, 0);
  DEVX_SET(qpc, qp_context, dbr_umem_valid, 0x1);      // Enable dbr_umem_id
  DEVX_SET64(qpc, qp_context, dbr_addr, qpDbrOffset);  // Byte offset into dbr_umem_id
  DEVX_SET(qpc, qp_context, dbr_umem_id, qpUmem->umem_id);
  DEVX_SET(qpc, qp_context, page_offset, 0);

  qp = Mlx5DvApi::Instance().devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out,
                                             sizeof(cmd_out));
  assert(qp);

  qpn = DEVX_GET(create_qp_out, cmd_out, qpn);

  MORI_APP_TRACE(
      "MLX5 QP created: qpn={}, qpTotalSize={}, sqWqeNum={}, rqWqeNum={}, sqAddr=0x{:x}, "
      "rqAddr=0x{:x}",
      qpn, qpTotalSize, sqAttrs.wqeNum, rqAttrs.wqeNum, reinterpret_cast<uintptr_t>(GetSqAddress()),
      reinterpret_cast<uintptr_t>(GetRqAddress()));
}

void Mlx5QpContainer::DestroyQueuePair() {
  // Destroy the firmware QP first, so the NIC stops referencing the SQ/DBR UMEMs,
  // UAR and atomic MR before we release them (avoids leak + NIC DMA into freed mem).
  if (qp) {
    Mlx5DvApi::Instance().devx_obj_destroy(qp);
    qp = nullptr;
  }

  if (atomicIbufMr) {
    ibv_dereg_mr(atomicIbufMr);
    atomicIbufMr = nullptr;
  }
  Mlx5FreeControlBuf(config.onGpu, atomicIbufAddr);
  atomicIbufAddr = nullptr;

  if (qpUmem) Mlx5DvApi::Instance().devx_umem_dereg(qpUmem);
  Mlx5FreeControlBuf(config.onGpu, qpUmemAddr);
  if (qpUar) {
    if (config.onGpu) {
      Mlx5UnregisterUarHost(qpUar->reg_addr);
    }
    Mlx5DvApi::Instance().devx_free_uar(qpUar);
  }
}

void* Mlx5QpContainer::GetSqAddress() { return static_cast<char*>(qpUmemAddr) + sqAttrs.offset; }

void* Mlx5QpContainer::GetRqAddress() { return static_cast<char*>(qpUmemAddr) + rqAttrs.offset; }

void Mlx5QpContainer::ModifyRst2Init() {
  uint8_t rst2init_cmd_in[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {
      0,
  };
  uint8_t rst2init_cmd_out[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {
      0,
  };

  DEVX_SET(rst2init_qp_in, rst2init_cmd_in, opcode, MLX5_CMD_OP_RST2INIT_QP);
  DEVX_SET(rst2init_qp_in, rst2init_cmd_in, qpn, qpn);

  void* qpc = DEVX_ADDR_OF(rst2init_qp_in, rst2init_cmd_in, qpc);
  DEVX_SET(qpc, qpc, rwe, 1); /* remote write access */
  DEVX_SET(qpc, qpc, rre, 1); /* remote read access */
  DEVX_SET(qpc, qpc, rae, 1);
  DEVX_SET(qpc, qpc, atomic_mode, 0x3);
  DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, config.portId);

  DEVX_SET(qpc, qpc, pm_state, 0x3);
  DEVX_SET(qpc, qpc, counter_set_id, 0x0);

  int status = Mlx5DvApi::Instance().devx_obj_modify(qp, rst2init_cmd_in, sizeof(rst2init_cmd_in),
                                                     rst2init_cmd_out, sizeof(rst2init_cmd_out));
  assert(!status);
}

void Mlx5QpContainer::ModifyInit2Rtr(const RdmaEndpointHandle& local_handle,
                                     const RdmaEndpointHandle& remote_handle,
                                     const ibv_port_attr& portAttr, uint32_t qpId) {
  uint8_t init2rtr_cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {
      0,
  };
  uint8_t init2rtr_cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {
      0,
  };

  DEVX_SET(init2rtr_qp_in, init2rtr_cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
  DEVX_SET(init2rtr_qp_in, init2rtr_cmd_in, qpn, qpn);

  void* qpc = DEVX_ADDR_OF(init2rtr_qp_in, init2rtr_cmd_in, qpc);
  DEVX_SET(qpc, qpc, mtu, portAttr.active_mtu);
  DEVX_SET(qpc, qpc, log_msg_max, 30);
  DEVX_SET(qpc, qpc, remote_qpn, remote_handle.qpn);
  DEVX_SET(qpc, qpc, next_rcv_psn, remote_handle.psn);
  DEVX_SET(qpc, qpc, min_rnr_nak, 12);
  // log_rra_max: clamp to floor(log2(max_qp_rd_atom)) instead of a hardcoded 20,
  // which exceeds the HCA cap and can corrupt atomic (flag) delivery.
  {
    const ibv_device_attr_ex* devAttr = device_context->GetRdmaDevice()->GetDeviceAttr();
    uint32_t rraCap =
        (devAttr && devAttr->orig_attr.max_qp_rd_atom > 0) ? devAttr->orig_attr.max_qp_rd_atom : 1;
    DEVX_SET(qpc, qpc, log_rra_max, static_cast<uint32_t>(log2(static_cast<double>(rraCap))));
  }

  qpc = DEVX_ADDR_OF(init2rtr_qp_in, init2rtr_cmd_in, qpc);
  DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, config.portId);

  // HcaCapability hca_cap = QueryHcaCap(context);
  if (portAttr.link_layer == IBV_LINK_LAYER_ETHERNET) {
    memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip), remote_handle.eth.gid,
           sizeof(remote_handle.eth.gid));

    memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32), remote_handle.eth.mac,
           sizeof(remote_handle.eth.mac));
    DEVX_SET(qpc, qpc, primary_address_path.hop_limit, 64);
    DEVX_SET(qpc, qpc, primary_address_path.src_addr_index, local_handle.eth.gidIdx);
    // UDP sport: default to a single fixed RoCEv2 sport (== 0xC000 on RoCE).
    // MORI_MLX5_ENABLE_UDP_SPORT=1 rotates per-qpId (GetUdpSport) for ECMP spread.
    static const bool enableUdpSport = []() {
      const char* e = std::getenv("MORI_MLX5_ENABLE_UDP_SPORT");
      return e != nullptr && std::atoi(e) != 0;
    }();
    uint16_t selected_udp_sport =
        enableUdpSport ? static_cast<uint16_t>(device_context->GetUdpSport(qpId) | 0xC000)
                       : static_cast<uint16_t>(portAttr.lid | 0xC000);
    DEVX_SET(qpc, qpc, primary_address_path.udp_sport, selected_udp_sport);
    // RoCE QoS: DEVX QPs ignore MORI_RDMA_TC/SL unless dscp/eth_prio are set here
    // (traffic_class = DSCP << 2 | ECN, so DSCP = TC >> 2).
    std::optional<uint8_t> roceTc = ReadRdmaTrafficClassEnv();
    std::optional<uint8_t> roceSl = ReadRdmaServiceLevelEnv();
    if (roceTc.has_value()) {
      DEVX_SET(qpc, qpc, primary_address_path.dscp, roceTc.value() >> 2);
    }
    if (roceSl.has_value()) {
      DEVX_SET(qpc, qpc, primary_address_path.eth_prio, roceSl.value() & 0x7);
    }
    MORI_APP_TRACE("MLX5 QP {} using UDP sport {} (qpId={}, index={})", qpn, selected_udp_sport,
                   qpId, qpId % RDMA_UDP_SPORT_ARRAY_SIZE);
  } else if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    DEVX_SET(qpc, qpc, primary_address_path.rlid, remote_handle.ib.lid);
  } else {
    assert(false);
  }

  int status = Mlx5DvApi::Instance().devx_obj_modify(qp, init2rtr_cmd_in, sizeof(init2rtr_cmd_in),
                                                     init2rtr_cmd_out, sizeof(init2rtr_cmd_out));
  assert(!status);
}

void Mlx5QpContainer::ModifyRtr2Rts(const RdmaEndpointHandle& local_handle) {
  uint8_t rtr2rts_cmd_in[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {
      0,
  };
  uint8_t rtr2rts_cmd_out[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {
      0,
  };

  DEVX_SET(rtr2rts_qp_in, rtr2rts_cmd_in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
  DEVX_SET(rtr2rts_qp_in, rtr2rts_cmd_in, qpn, qpn);

  void* qpc = DEVX_ADDR_OF(rtr2rts_qp_in, rtr2rts_cmd_in, qpc);
  // log_sra_max: clamp to floor(log2(max_qp_rd_atom)) (same rationale as log_rra_max).
  {
    const ibv_device_attr_ex* devAttr = device_context->GetRdmaDevice()->GetDeviceAttr();
    uint32_t sraCap =
        (devAttr && devAttr->orig_attr.max_qp_rd_atom > 0) ? devAttr->orig_attr.max_qp_rd_atom : 1;
    DEVX_SET(qpc, qpc, log_sra_max, static_cast<uint32_t>(log2(static_cast<double>(sraCap))));
  }
  DEVX_SET(qpc, qpc, next_send_psn, local_handle.psn);
  DEVX_SET(qpc, qpc, retry_count, 7);
  DEVX_SET(qpc, qpc, rnr_retry, 7);
  DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, 20);
  DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, config.portId);

  int status = Mlx5DvApi::Instance().devx_obj_modify(qp, rtr2rts_cmd_in, sizeof(rtr2rts_cmd_in),
                                                     rtr2rts_cmd_out, sizeof(rtr2rts_cmd_out));
  assert(!status);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        Mlx5DeviceContext                                       */
/* ---------------------------------------------------------------------------------------------- */
Mlx5DeviceContext::Mlx5DeviceContext(RdmaDevice* rdma_device, ibv_pd* in_pd)
    : RdmaDeviceContext(rdma_device, in_pd) {
  mlx5dv_obj dv_obj{};
  mlx5dv_pd dvpd{};
  dv_obj.pd.in = pd;
  dv_obj.pd.out = &dvpd;
  int status = Mlx5DvApi::Instance().init_obj(&dv_obj, MLX5DV_OBJ_PD);
  assert(!status);
  pdn = dvpd.pdn;
}

Mlx5DeviceContext::~Mlx5DeviceContext() {}

RdmaEndpoint Mlx5DeviceContext::CreateRdmaEndpoint(const RdmaEndpointConfig& config) {
  assert(!config.withCompChannel && !config.enableSrq && "not implemented");
  ibv_context* context = GetIbvContext();

  Mlx5CqContainer* cq = new Mlx5CqContainer(context, config);
  Mlx5QpContainer* qp = new Mlx5QpContainer(context, config, cq->cqn, pdn, this);
  const ibv_device_attr_ex* deviceAttr = GetRdmaDevice()->GetDeviceAttr();

  RdmaEndpoint endpoint;
  endpoint.handle.psn = 0;
  endpoint.handle.portId = config.portId;
  endpoint.handle.maxSge = config.maxMsgSge;

  const ibv_port_attr* portAttr = GetRdmaDevice()->GetPortAttr(config.portId);
  assert(portAttr);
  HcaCapability hca_cap = QueryHcaCap(context);

  endpoint.handle.qpn = qp->qpn;
  if (hca_cap.IsEthernet()) {
    GidSelectionResult gidSelection =
        AutoSelectGidIndex(context, config.portId, portAttr, config.gidIdx);
    assert(gidSelection.gidIdx >= 0 && gidSelection.valid);
    int gidIdx = gidSelection.gidIdx;

    uint32_t out[DEVX_ST_SZ_DW(query_roce_address_out)] = {};
    uint32_t in[DEVX_ST_SZ_DW(query_roce_address_in)] = {};

    DEVX_SET(query_roce_address_in, in, opcode, MLX5_CMD_OP_QUERY_ROCE_ADDRESS);
    DEVX_SET(query_roce_address_in, in, roce_address_index, gidIdx);
    DEVX_SET(query_roce_address_in, in, vhca_port_num, config.portId);

    int status = Mlx5DvApi::Instance().devx_general_cmd(context, in, sizeof(in), out, sizeof(out));
    assert(!status);

    memcpy(endpoint.handle.eth.gid,
           DEVX_ADDR_OF(query_roce_address_out, out, roce_address.source_l3_address),
           sizeof(endpoint.handle.eth.gid));

    memcpy(endpoint.handle.eth.mac,
           DEVX_ADDR_OF(query_roce_address_out, out, roce_address.source_mac_47_32),
           sizeof(endpoint.handle.eth.mac));
    endpoint.handle.eth.gidIdx = gidIdx;
  } else if (hca_cap.IsInfiniBand()) {
    auto mapPtr = GetRdmaDevice()->GetPortAttrMap();
    auto it = mapPtr->find(config.portId);
    if (it != mapPtr->end() && it->second) {
      ibv_port_attr* port_attr = it->second.get();
      endpoint.handle.ib.lid = port_attr->lid;
    } else {
      assert(false && "Port attribute not found for given port ID");
    }
  } else {
    assert(false);
  }

  endpoint.vendorId = RdmaDeviceVendorId::Mellanox;

  endpoint.wqHandle.sqAddr = qp->GetSqAddress();
  endpoint.wqHandle.rqAddr = qp->GetRqAddress();
  endpoint.wqHandle.dbrRecAddr = qp->qpDbrUmemAddr;
  endpoint.wqHandle.dbrAddr = qp->qpUarPtr;
  endpoint.wqHandle.sqWqeNum = qp->sqAttrs.wqeNum;
  endpoint.wqHandle.rqWqeNum = qp->rqAttrs.wqeNum;
  // RQ doorbell record lives in the same QP DBR page as the SQ (mlx5 uses the
  // uint32 at index MORI_MLX5_RCV_DBR=0; SQ uses index 1).
  endpoint.wqHandle.rqdbrAddr = qp->qpDbrUmemAddr;

  endpoint.cqHandle.cqAddr = cq->cqUmemAddr;
  endpoint.cqHandle.consIdx = 0;
  endpoint.cqHandle.cqeNum = cq->cqeNum;
  endpoint.cqHandle.cqeSize = GetMlx5CqeSize();
  endpoint.cqHandle.dbrRecAddr = cq->cqDbrUmemAddr;

  // Set atomic internal buffer information
  endpoint.atomicIbuf.addr = reinterpret_cast<uintptr_t>(qp->atomicIbufAddr);
  endpoint.atomicIbuf.lkey = qp->atomicIbufMr->lkey;
  endpoint.atomicIbuf.rkey = qp->atomicIbufMr->rkey;
  endpoint.atomicIbuf.nslots = RoundUpPowOfTwo(config.atomicIbufSlots);

  cqPool.insert({cq->cqn, std::move(std::unique_ptr<Mlx5CqContainer>(cq))});
  qpPool.insert({qp->qpn, std::move(std::unique_ptr<Mlx5QpContainer>(qp))});

  MORI_APP_TRACE(
      "MLX5 endpoint created: qpn={}, cqn={}, portId={}, gidIdx={}, atomicIbuf addr=0x{:x}, "
      "nslots={}",
      qp->qpn, cq->cqn, config.portId, endpoint.handle.eth.gidIdx, endpoint.atomicIbuf.addr,
      endpoint.atomicIbuf.nslots);

  return endpoint;
}

void Mlx5DeviceContext::ConnectEndpoint(const RdmaEndpointHandle& local,
                                        const RdmaEndpointHandle& remote, uint32_t qpId) {
  uint32_t local_qpn = local.qpn;

  assert(qpPool.find(local_qpn) != qpPool.end());
  Mlx5QpContainer* qp = qpPool.at(local_qpn).get();

  MORI_APP_TRACE("MLX5 connecting endpoint: local_qpn={}, remote_qpn={}, qpId={}", local_qpn,
                 remote.qpn, qpId);

  RdmaDevice* rdmaDevice = GetRdmaDevice();
  const ibv_device_attr_ex* deviceAttr = rdmaDevice->GetDeviceAttr();
  const ibv_port_attr& portAttr = *(rdmaDevice->GetPortAttrMap()->find(local.portId)->second);
  qp->ModifyRst2Init();
  qp->ModifyInit2Rtr(local, remote, portAttr, qpId);
  qp->ModifyRtr2Rts(local);

  MORI_APP_TRACE("MLX5 endpoint connected successfully: local_qpn={}, remote_qpn={}", local_qpn,
                 remote.qpn);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                           Mlx5Device                                           */
/* ---------------------------------------------------------------------------------------------- */
Mlx5Device::Mlx5Device(ibv_device* in_device) : RdmaDevice(in_device) {}
Mlx5Device::~Mlx5Device() {}

RdmaDeviceContext* Mlx5Device::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(defaultContext);
  return new Mlx5DeviceContext(this, pd);
}

}  // namespace application
}  // namespace mori
