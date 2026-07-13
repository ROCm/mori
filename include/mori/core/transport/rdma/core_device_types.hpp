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
//
// Device-safe subset of core RDMA types.
// Only contains POD structs and enums — no ibverbs, no C++ STL.
// Safe to include from both host and device (HIP/CUDA) compilation units.
#pragma once

#include <endian.h>  // BYTE_ORDER / LITTLE_ENDIAN / BIG_ENDIAN
#include <limits.h>
#include <stdint.h>

#include "mori/hip_compat.hpp"  // __device__ / __host__ (no-op under non-hipcc host parse)

namespace mori {
namespace core {

enum ProviderType {
  Unknown = 0,
  // Mellanox direct verbs
  MLX5 = 1,
  // Broadcom direct verbs
  BNXT = 2,
  // Pensando direct verbs
  PSD = 3,
  // Ib verbs
  IBVERBS = 4,
};

// Device-safe mirror of ibverbs' enum ibv_wc_status (same order/values). Lets the
// device-side CQE error decoders report a completion status without pulling the
// host <infiniband/verbs.h> into device translation units. Value parity with
// ibv_wc_status is guarded by static_asserts in a host TU
// (src/application/transport/rdma/rdma.cpp).
enum WcStatus {
  WC_SUCCESS = 0,
  WC_LOC_LEN_ERR,
  WC_LOC_QP_OP_ERR,
  WC_LOC_EEC_OP_ERR,
  WC_LOC_PROT_ERR,
  WC_WR_FLUSH_ERR,
  WC_MW_BIND_ERR,
  WC_BAD_RESP_ERR,
  WC_LOC_ACCESS_ERR,
  WC_REM_INV_REQ_ERR,
  WC_REM_ACCESS_ERR,
  WC_REM_OP_ERR,
  WC_RETRY_EXC_ERR,
  WC_RNR_RETRY_EXC_ERR,
  WC_LOC_RDD_VIOL_ERR,
  WC_REM_INV_RD_REQ_ERR,
  WC_REM_ABORT_ERR,
  WC_INV_EECN_ERR,
  WC_INV_EEC_STATE_ERR,
  WC_FATAL_ERR,
  WC_RESP_TIMEOUT_ERR,
  WC_GENERAL_ERR,
  WC_TM_ERR,
  WC_TM_RNDV_INCOMPLETE,
};

typedef enum {
  AMO_ACK = 1,
  AMO_INC,
  AMO_SET,
  AMO_ADD,
  AMO_AND,
  AMO_OR,
  AMO_XOR,
  AMO_SIGNAL,
  SIGNAL_SET,
  SIGNAL_ADD,
  AMO_SIGNAL_SET = SIGNAL_SET,
  AMO_SIGNAL_ADD = SIGNAL_ADD,
  AMO_END_OF_NONFETCH,
  AMO_FETCH,
  AMO_FETCH_INC,
  AMO_FETCH_ADD,
  AMO_FETCH_AND,
  AMO_FETCH_OR,
  AMO_FETCH_XOR,
  AMO_SWAP,
  AMO_COMPARE_SWAP,
  AMO_OP_SENTINEL = INT_MAX,
} atomicType;

struct WorkQueueHandle {
  uint32_t postIdx{0};     // numbers of wqe that post to work queue
  uint32_t dbTouchIdx{0};  // numbers of wqe that touched doorbell
  uint32_t doneIdx{0};     // numbers of wqe that have been consumed by nic
  uint32_t readyIdx{0};
  union {
    struct {
      uint32_t msntblSlotIdx;
      uint32_t psnIdx;  // for bnxt msn psn index calculate
    };
    uint64_t msnPack{0};
  };
  void* sqAddr{nullptr};
  void* msntblAddr{nullptr};  // for bnxt
  void* rqAddr{nullptr};
  void* dbrRecAddr{nullptr};
  void* dbrAddr{nullptr};
  void* rqdbrAddr{nullptr};
  uint32_t mtuSize{4096};
  uint32_t sqWqeNum{0};
  uint32_t msntblNum{0};
  uint32_t rqWqeNum{0};
  // Persistent RQ producer index for WRITE_WITH_IMM recv WQEs (Phase-6). The SQ
  // side has postIdx; the RQ had no counter, so a per-launch recv pre-post would
  // reuse slot 0 and drive the RQ doorbell record backwards on the 2nd launch.
  // rqPostIdx survives across kernel launches so ShmemPostRecvImm advances the RQ
  // monotonically. Default 0, untouched unless the WRITE_WITH_IMM path is engaged.
  uint32_t rqPostIdx{0};
  uint32_t postSendLock{0};
  bool color;
  uint64_t sq_dbval{0};
  uint64_t rq_dbval{0};
};

struct CompletionQueueHandle {
  void* cqAddr{nullptr};
  void* dbrRecAddr{nullptr};
  void* dbrAddr{nullptr};
  uint32_t consIdx{0};      // numbers of cqe that have been completed
  uint32_t needConsIdx{0};  // numbers of cqe that should be consumed
  uint32_t activeIdx{0};    // numbers of cqe that under processing but not completed
  uint32_t cq_consumer{0};
  uint32_t cq_dbpos{0};
  uint32_t cqeNum{0};
  uint32_t cqeSize{0};
  uint32_t pollCqLock{0};
  uint64_t cq_dbval{0};
};

struct IbufHandle {
  uintptr_t addr{0};
  uint32_t lkey{0};
  uint32_t rkey{0};
  uint32_t nslots{0};
  uint32_t head{0};
  uint32_t tail{0};
};

enum class RdmaDeviceVendorId : uint32_t {
  Unknown = 0,
  Mellanox = 0x02c9,
  Broadcom = 0x14E4,
  Pensando = 0x1dd8,
};

// Device-side view of an RDMA endpoint: a pure device POD over core's WQ/CQ/Ibuf
// handles. Filled on the host from application::RdmaEndpoint, hipMemcpy'd to the
// device, consumed by the RDMA backends (shmem, cco) — which depend down on core.
struct RdmaEndpointDevice {
  RdmaDeviceVendorId vendorId{RdmaDeviceVendorId::Unknown};
  uint32_t qpn{0};  // QP number — extracted from application::RdmaEndpoint::handle.qpn
  WorkQueueHandle wqHandle;
  CompletionQueueHandle cqHandle;
  // WRITE_WITH_IMM receiver (hierarchical AllGather cross-node ring accuracy fix): the
  // recv-side completion queue the responder posts RDMA_WRITE_WITH_IMM CQEs into. Mirrors
  // cqHandle when no dedicated recv CQ is configured (send+recv share one CQ), so device
  // receivers can always read recvCqHandle uniformly. Populated in init.cpp from
  // application::RdmaEndpoint::recvCqHandle.
  CompletionQueueHandle recvCqHandle;
  IbufHandle atomicIbuf;

  __device__ __host__ ProviderType GetProviderType() const {
    switch (vendorId) {
      case RdmaDeviceVendorId::Mellanox:
        return ProviderType::MLX5;
      case RdmaDeviceVendorId::Broadcom:
        return ProviderType::BNXT;
      case RdmaDeviceVendorId::Pensando:
        return ProviderType::PSD;
      default:
        return ProviderType::Unknown;
    }
  }
};

/* ---------------------------------------------------------------------------------------------- */
/*                                        Utility Functions                                       */
/* ---------------------------------------------------------------------------------------------- */
#define BSWAP64(x)                                                                 \
  ((((x) & 0xff00000000000000ull) >> 56) | (((x) & 0x00ff000000000000ull) >> 40) | \
   (((x) & 0x0000ff0000000000ull) >> 24) | (((x) & 0x000000ff00000000ull) >> 8) |  \
   (((x) & 0x00000000ff000000ull) << 8) | (((x) & 0x0000000000ff0000ull) << 24) |  \
   (((x) & 0x000000000000ff00ull) << 40) | (((x) & 0x00000000000000ffull) << 56))

#define BSWAP32(x)                                                                      \
  ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >> 8) | (((x) & 0x0000ff00) << 8) | \
   (((x) & 0x000000ff) << 24))

#define BSWAP16(x) ((((x) & 0xff00) >> 8) | (((x) & 0x00ff) << 8))

#define HTOBE64(x) BSWAP64(x)
#define HTOBE32(x) BSWAP32(x)
#define HTOBE16(x) BSWAP16(x)

#if BYTE_ORDER == LITTLE_ENDIAN
#define BE16TOH(x) BSWAP16(x)
#define BE32TOH(x) BSWAP32(x)
#define BE64TOH(x) BSWAP64(x)
#define LE16TOH(x) (x)
#define LE32TOH(x) (x)
#define LE64TOH(x) (x)
#elif BYTE_ORDER == BIG_ENDIAN
#define BE16TOH(x) (x)
#define BE32TOH(x) (x)
#define BE64TOH(x) (x)
#define LE16TOH(x) BSWAP16(x)
#define LE32TOH(x) BSWAP32(x)
#define LE64TOH(x) BSWAP64(x)
#endif

}  // namespace core
}  // namespace mori
