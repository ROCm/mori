// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include <cstdint>

namespace mori {
namespace core {

enum ProxyCmdOp : uint32_t {
  PROXY_NOP = 0,
  PROXY_RDMA_WRITE = 1,
  PROXY_RDMA_WRITE_INLINE = 2,
  PROXY_ATOMIC_FETCH_ADD = 3,   // standalone atomic (barrier) → SEND_WITH_IMM
  PROXY_ATOMIC_CMP_SWAP = 4,
  PROXY_SIGNAL_WRITE = 5,      // signal paired with data → RDMA_WRITE (same PCIe path)
};

enum ProxyInlineTag : uint32_t {
  PROXY_INLINE_NONE = 0,
  PROXY_INLINE_SCALAR_WRITE = 1,
};

enum ProxyCmdStatus : uint32_t {
  PROXY_FREE = 0,
  PROXY_PENDING = 1,
  PROXY_COMPLETED = 3,
  PROXY_ERROR = 4,
};

static constexpr uint32_t PROXY_MAX_INLINE_DATA = 48;

struct alignas(128) ProxyCmd {
  uint32_t op;
  uint32_t qp_idx;
  uint64_t src_addr;
  uint64_t dst_addr;
  uint32_t length;
  uint32_t lkey;
  uint32_t rkey;
  uint32_t flags;
  uint64_t atomic_arg;
  uint64_t atomic_swap;
  volatile uint32_t status;
  uint32_t pad0;
  volatile uint64_t result;
  uint8_t inline_data[PROXY_MAX_INLINE_DATA];
  uint32_t inline_tag;
  uint32_t inline_len;
};

static_assert(sizeof(ProxyCmd) == 128, "ProxyCmd must be 128 bytes");

static constexpr uint32_t PROXY_RING_SIZE = 65536;
static constexpr uint32_t PROXY_RING_MASK = PROXY_RING_SIZE - 1;
static constexpr int PROXY_MAX_NICS = 8;

struct ProxyRing {
  volatile uint32_t gpu_head;
  uint32_t pad1[15];
  volatile uint32_t shutdown;
  uint32_t pad2[15];
  ProxyCmd cmds[PROXY_RING_SIZE];
};

}  // namespace core
}  // namespace mori
