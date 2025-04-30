#pragma once

#include <stdint.h>

namespace mori {
namespace core {
namespace transport {
namespace ibgda {

enum ProviderType {
  MLX5 = 1,
  BNXTRE = 2,
};

/* ---------------------------------------------------------------------------------------------- */
/*                                        Utility Functions                                       */
/* ---------------------------------------------------------------------------------------------- */
#define BSWAP64(x)                                                             \
  ((((x)&0xff00000000000000ull) >> 56) | (((x)&0x00ff000000000000ull) >> 40) | \
   (((x)&0x0000ff0000000000ull) >> 24) | (((x)&0x000000ff00000000ull) >> 8) |  \
   (((x)&0x00000000ff000000ull) << 8) | (((x)&0x0000000000ff0000ull) << 24) |  \
   (((x)&0x000000000000ff00ull) << 40) | (((x)&0x00000000000000ffull) << 56))

#define BSWAP32(x)                                                                \
  ((((x)&0xff000000) >> 24) | (((x)&0x00ff0000) >> 8) | (((x)&0x0000ff00) << 8) | \
   (((x)&0x000000ff) << 24))

#define BSWAP16(x) ((((x)&0xff00) >> 8) | (((x)&0x00ff) << 8))

#define HTOBE64(x) BSWAP64(x)
#define HTOBE32(x) BSWAP32(x)
#define HTOBE16(x) BSWAP16(x)

#if BYTE_ORDER == LITTLE_ENDIAN
#define BE32TOH(x) BSWAP32(x)
#define LE32TOH(x) (x)
#define BE64TOH(x) BSWAP64(x)
#define LE64TOH(x) (x)
#elif BYTE_ORDER == BIG_ENDIAN
#define BE32TOH(x) (x)
#define LE32TOH(x) BSWAP32(x)
#define BE64TOH(x) (x)
#define LE64TOH(x) BSWAP64(x)
#endif

struct QueuePairHandle {
  uint32_t qpn;
  uint32_t post_idx;
  void* next_wqe_addr;
  void* dbr_rec_addr;
  void* dbr_addr;
};

struct MemoryRegion {
  uintptr_t addr;
  uint32_t lkey;
  uint32_t rkey;
  size_t length;
};

struct IbgdaWriteReq {
  QueuePairHandle qp_handle;
  MemoryRegion local_mr;
  MemoryRegion remote_mr;
  size_t bytes_count;
};

struct CompletionQueueHandle {
  void* cq_addr{nullptr};
  // TODO: consumer_idx should be tracked globally
  uint32_t consumer_idx{0};
  uint32_t cqe_num{0};
  uint32_t cqe_size{0};
};

}  // namespace ibgda
}  // namespace transport
}  // namespace core
}  // namespace mori