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
  PROXY_ATOMIC_FETCH_ADD = 3,
  PROXY_ATOMIC_CMP_SWAP = 4,
};

enum ProxyCmdStatus : uint32_t {
  PROXY_FREE = 0,
  PROXY_PENDING = 1,
  PROXY_COMPLETED = 3,
  PROXY_ERROR = 4,
};

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
  uint8_t pad1[128 - 72];
};

static constexpr uint32_t PROXY_RING_SIZE = 1024;
static constexpr uint32_t PROXY_RING_MASK = PROXY_RING_SIZE - 1;

struct ProxyRing {
  volatile uint32_t gpu_head;
  uint32_t pad1[15];
  volatile uint32_t shutdown;
  uint32_t pad2[15];
  ProxyCmd cmds[PROXY_RING_SIZE];
};

}  // namespace core
}  // namespace mori
