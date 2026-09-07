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
// Level 1: Host-only unit test for proxy types
// Tests: struct sizes, alignment, enum values, ring layout
// No GPU, no RDMA — pure compile+run on any machine
//
// Build: g++ -std=c++17 -I<mori>/include -o test_proxy_types test_proxy_types.cpp &&
// ./test_proxy_types

#include <cassert>
#include <cstdio>
#include <cstring>

#include "mori/core/transport/rdma/proxy/proxy_types.hpp"

using namespace mori::core;

void test_enum_values() {
  assert(PROXY_NOP == 0);
  assert(PROXY_RDMA_WRITE == 1);
  assert(PROXY_RDMA_WRITE_INLINE == 2);
  assert(PROXY_ATOMIC_FETCH_ADD == 3);
  assert(PROXY_ATOMIC_CMP_SWAP == 4);

  assert(PROXY_FREE == 0);
  assert(PROXY_PENDING == 1);
  assert(PROXY_COMPLETED == 3);
  assert(PROXY_ERROR == 4);
  printf("  enum_values: PASS\n");
}

void test_proxy_cmd_layout() {
  assert(sizeof(ProxyCmd) == 128);
  assert(alignof(ProxyCmd) == 128);

  ProxyCmd cmd{};
  assert(cmd.op == 0);
  assert(cmd.status == PROXY_FREE);
  assert(cmd.result == 0);

  cmd.op = PROXY_RDMA_WRITE;
  cmd.qp_idx = 3;
  cmd.src_addr = 0xDEAD0000;
  cmd.dst_addr = 0xBEEF0000;
  cmd.length = 4096;
  cmd.lkey = 100;
  cmd.rkey = 200;
  cmd.flags = 1;
  cmd.status = PROXY_PENDING;

  assert(cmd.op == PROXY_RDMA_WRITE);
  assert(cmd.qp_idx == 3);
  assert(cmd.length == 4096);
  assert(cmd.status == PROXY_PENDING);
  printf("  proxy_cmd_layout: PASS\n");
}

void test_ring_constants() {
  assert(PROXY_RING_SIZE == 1024);
  assert(PROXY_RING_MASK == 1023);
  assert((PROXY_RING_SIZE & PROXY_RING_MASK) == 0);
  printf("  ring_constants: PASS\n");
}

void test_ring_layout() {
  // Verify gpu_head and shutdown are on different cache lines
  ProxyRing ring{};
  uintptr_t head_off = (uintptr_t)&ring.gpu_head - (uintptr_t)&ring;
  uintptr_t shut_off = (uintptr_t)&ring.shutdown - (uintptr_t)&ring;
  assert(head_off == 0);
  assert(shut_off == 64);  // gpu_head(4) + pad1[15](60) = 64

  assert(ring.gpu_head == 0);
  assert(ring.shutdown == 0);

  // All cmds should be zero-initialized
  for (uint32_t i = 0; i < PROXY_RING_SIZE; i++) {
    assert(ring.cmds[i].status == PROXY_FREE);
    assert(ring.cmds[i].op == PROXY_NOP);
  }
  printf("  ring_layout: PASS\n");
}

void test_ring_slot_independence() {
  ProxyRing ring{};

  ring.cmds[0].status = PROXY_PENDING;
  ring.cmds[0].op = PROXY_RDMA_WRITE;
  ring.cmds[0].length = 100;

  ring.cmds[1].status = PROXY_COMPLETED;
  ring.cmds[1].op = PROXY_ATOMIC_FETCH_ADD;
  ring.cmds[1].length = 8;

  // Slots are independent (128-byte aligned, no false sharing)
  assert(ring.cmds[0].status == PROXY_PENDING);
  assert(ring.cmds[0].length == 100);
  assert(ring.cmds[1].status == PROXY_COMPLETED);
  assert(ring.cmds[1].length == 8);

  // Wrap-around indexing
  uint32_t seq = PROXY_RING_SIZE + 5;
  uint32_t slot = seq & PROXY_RING_MASK;
  assert(slot == 5);
  printf("  ring_slot_independence: PASS\n");
}

void test_ring_size() {
  printf("  ProxyCmd size: %zu bytes\n", sizeof(ProxyCmd));
  printf("  ProxyRing size: %zu bytes (%.1f KB)\n", sizeof(ProxyRing), sizeof(ProxyRing) / 1024.0);
  printf("  ring_size: PASS\n");
}

int main() {
  printf("=== Level 1: proxy_types unit test ===\n");
  test_enum_values();
  test_proxy_cmd_layout();
  test_ring_constants();
  test_ring_layout();
  test_ring_slot_independence();
  test_ring_size();
  printf("=== ALL PASS ===\n");
  return 0;
}
