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

#include <limits.h>
#include <stdint.h>

namespace mori {
namespace core {

enum ProviderType {
  Unknown = 0,
  // Mellanox direct verbs
  MLX5 = 1,
  // Broadcom direct verbs
  BNXT = 2,
  // Ib verbs
  IBVERBS = 3,
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
