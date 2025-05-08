#pragma once

#include <stdint.h>
// TODO: remove application dependencies
#include "mori/application/transport/rdma/rdma.hpp"

namespace mori {
namespace core {

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

struct CompletionQueueHandle {
  void* cqAddr{nullptr};
  void* dbrRecAddr{nullptr};
  // TODO: consIdx should be tracked globally
  uint32_t consIdx{0};
  uint32_t cqeNum{0};
  uint32_t cqeSize{0};
};

}  // namespace core
}  // namespace mori