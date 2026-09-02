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
// Level 3: GPU + CPU proxy test
// Tests: GPU kernel writes commands to proxy ring, CPU thread posts via ibv_post_send
// Uses loopback RDMA + HIP GPU kernel
// Requires: RDMA device + GPU
//
// Build: hipcc -std=c++17 -O2 -I . -o test_proxy_gpu
//          test_proxy_gpu.cpp proxy_thread.cpp -libverbs -lpthread --offload-arch=gfx950
// Run:   ./test_proxy_gpu -d ionic_0 -g 1

#include <arpa/inet.h>
#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mori/core/transport/rdma/proxy/proxy_device_primitives.hpp"
#include "mori/core/transport/rdma/proxy/proxy_thread.hpp"
#include "mori/core/transport/rdma/proxy/proxy_types.hpp"

#define HIP_CHECK(x)                                            \
  do {                                                          \
    hipError_t e = (x);                                         \
    if (e != hipSuccess) {                                      \
      fprintf(stderr, "HIP %d %s:%d\n", e, __FILE__, __LINE__); \
      exit(1);                                                  \
    }                                                           \
  } while (0)

using namespace mori::core;

// GPU kernel: submit N RDMA writes via proxy ring
__global__ void gpu_proxy_write_kernel(volatile ProxyRing* ring, uint32_t qp_idx, uint64_t src,
                                       uint32_t lkey, uint64_t dst, uint32_t rkey,
                                       uint32_t xfer_size, int num_ops, volatile int* result) {
  if (threadIdx.x || blockIdx.x) return;

  uint32_t first_seq = ring->gpu_head;
  for (int i = 0; i < num_ops; i++) {
    ProxyPostWrite(ring, qp_idx, src, lkey, dst, rkey, xfer_size);
    if (i > 0 && i % 100 == 0) {
      printf("GPU: submitted %d/%d\n", i, num_ops);
    }
  }
  printf("GPU: all %d submitted, first_seq=%u, now quieting...\n", num_ops, first_seq);

  ProxyQuiet(ring, first_seq, num_ops);
  printf("GPU: quiet done\n");
  *result = num_ops;
}

// GPU kernel: submit inline writes
__global__ void gpu_proxy_write_inline_kernel(volatile ProxyRing* ring, uint32_t qp_idx,
                                              uint64_t src, uint32_t lkey, uint64_t dst,
                                              uint32_t rkey, uint32_t xfer_size, int num_ops,
                                              volatile int* result) {
  if (threadIdx.x || blockIdx.x) return;

  uint32_t first_seq = ring->gpu_head;
  for (int i = 0; i < num_ops; i++) {
    ProxyPostWriteInline(ring, qp_idx, src, lkey, dst, rkey, xfer_size);
  }
  ProxyQuiet(ring, first_seq, num_ops);
  *result = num_ops;
}

// GPU kernel: multi-warp test (simulates EP where multiple warps post concurrently)
__global__ void gpu_proxy_multi_warp_kernel(volatile ProxyRing* ring, uint32_t qp_idx, uint64_t src,
                                            uint32_t lkey, uint64_t dst, uint32_t rkey,
                                            uint32_t xfer_size, volatile int* completed_count) {
  int warp_id = threadIdx.x / 64;
  int lane_id = threadIdx.x % 64;
  if (lane_id != 0) return;  // only lane 0 per warp

  // Each warp submits 10 writes
  for (int i = 0; i < 10; i++) {
    ProxyPostWrite(ring, qp_idx, src, lkey, dst, rkey, xfer_size);
  }

  __syncthreads();

  // Warp 0 does quiet + reports
  // 4 warps × 10 ops = 40 total, first_seq was captured before submissions
  if (warp_id == 0) {
    // Wait for all 40 ops (simple: scan last 40 slots from current head)
    uint32_t head_now = ring->gpu_head;
    // All warps submitted before syncthreads, so head_now = first + 40
    ProxyQuietRange(ring, head_now - 40, head_now);
    *completed_count = 1;
  }
}

struct TestCtx {
  ibv_context* ctx;
  ibv_pd* pd;
  ibv_cq* cq;
  ibv_qp* qp;
  ibv_mr* mr;
  void* gpu_buf;
  size_t buf_size;
};

static TestCtx setup_loopback(const char* dev_name, int gid_idx) {
  TestCtx t{};
  int nd;
  ibv_device** dl = ibv_get_device_list(&nd);
  ibv_device* d = nullptr;
  for (int i = 0; i < nd; i++)
    if (!strcmp(dl[i]->name, dev_name)) d = dl[i];
  assert(d);
  t.ctx = ibv_open_device(d);
  t.pd = ibv_alloc_pd(t.ctx);
  t.cq = ibv_create_cq(t.ctx, 256, nullptr, nullptr, 0);
  ibv_qp_init_attr qa{};
  qa.send_cq = t.cq;
  qa.recv_cq = t.cq;
  qa.qp_type = IBV_QPT_RC;
  qa.cap = {128, 128, 1, 1, 0};
  t.qp = ibv_create_qp(t.pd, &qa);
  t.buf_size = 64 * 1024;
  HIP_CHECK(hipMalloc(&t.gpu_buf, t.buf_size));
  HIP_CHECK(hipMemset(t.gpu_buf, 0xAB, t.buf_size));
  t.mr = ibv_reg_mr(t.pd, t.gpu_buf, t.buf_size,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  assert(t.mr);
  ibv_gid gid;
  ibv_query_gid(t.ctx, 1, gid_idx, &gid);
  {
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_INIT;
    a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
    ibv_modify_qp(t.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  }
  {
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = IBV_MTU_4096;
    a.dest_qp_num = t.qp->qp_num;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 12;
    memcpy(&a.ah_attr.grh.dgid, &gid, 16);
    a.ah_attr.grh.sgid_index = gid_idx;
    a.ah_attr.grh.hop_limit = 1;
    a.ah_attr.is_global = 1;
    a.ah_attr.port_num = 1;
    ibv_modify_qp(t.qp, &a,
                  IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_AV |
                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  }
  {
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.max_rd_atomic = 1;
    ibv_modify_qp(t.qp, &a,
                  IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                      IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
  }
  ibv_free_device_list(dl);
  return t;
}

int main(int argc, char** argv) {
  const char* dev = "ionic_0";
  int gid = 1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-d"))
      dev = argv[++i];
    else if (!strcmp(argv[i], "-g"))
      gid = atoi(argv[++i]);
  }

  setbuf(stdout, NULL);
  printf("=== Level 3: GPU proxy test (dev=%s gid=%d) ===\n", dev, gid);
  HIP_CHECK(hipSetDevice(0));

  TestCtx t = setup_loopback(dev, gid);
  printf("  QP loopback (qpn=%u), GPU buf=%p, MR lkey=%u\n", t.qp->qp_num, t.gpu_buf, t.mr->lkey);

  // Allocate proxy ring (host-pinned, coherent)
  ProxyRing* ring;
  HIP_CHECK(hipHostMalloc(&ring, sizeof(ProxyRing), hipHostMallocMapped | hipHostMallocCoherent));
  memset(ring, 0, sizeof(ProxyRing));

  int* result;
  HIP_CHECK(hipHostMalloc(&result, sizeof(int), hipHostMallocMapped | hipHostMallocCoherent));

  // Start proxy thread
  ProxyThread proxy;
  std::vector<ProxyQpHandle> qps = {{t.qp, t.cq}};
  proxy.Init(ring, qps);
  proxy.Start();

  // ── Test 1: GPU single-thread, 10 writes ──
  printf("\n  Test 1: GPU single-thread, 10 RDMA writes...\n");
  *result = 0;
  hipLaunchKernelGGL(gpu_proxy_write_kernel, dim3(1), dim3(1), 0, 0, (volatile ProxyRing*)ring, 0,
                     (uint64_t)t.gpu_buf, t.mr->lkey, (uint64_t)t.gpu_buf + 4096, t.mr->rkey, 4096u,
                     10, result);
  HIP_CHECK(hipDeviceSynchronize());
  printf("  Test 1: completed=%d %s\n", *result, *result == 10 ? "PASS" : "FAIL");
  assert(*result == 10);

  // Shutdown and restart proxy for clean state
  proxy.Shutdown();
  memset(ring, 0, sizeof(ProxyRing));
  proxy.Init(ring, qps);
  proxy.Start();

  // ── Test 2: GPU single-thread, 500 writes (tests ring wrap) ──
  printf("\n  Test 2: GPU single-thread, 500 RDMA writes (ring wrap)...\n");
  *result = 0;
  hipLaunchKernelGGL(gpu_proxy_write_kernel, dim3(1), dim3(1), 0, 0, (volatile ProxyRing*)ring, 0,
                     (uint64_t)t.gpu_buf, t.mr->lkey, (uint64_t)t.gpu_buf + 4096, t.mr->rkey, 256u,
                     500, result);
  HIP_CHECK(hipDeviceSynchronize());
  printf("  Test 2: completed=%d %s\n", *result, *result == 500 ? "PASS" : "FAIL");
  assert(*result == 500);

  proxy.Shutdown();
  memset(ring, 0, sizeof(ProxyRing));
  proxy.Init(ring, qps);
  proxy.Start();

  // ── Test 3: GPU multi-warp (4 warps × 10 writes = 40 ops) ──
  printf("\n  Test 3: GPU multi-warp (4 warps × 10 writes)...\n");
  *result = 0;
  hipLaunchKernelGGL(gpu_proxy_multi_warp_kernel, dim3(1), dim3(256), 0, 0,
                     (volatile ProxyRing*)ring, 0, (uint64_t)t.gpu_buf, t.mr->lkey,
                     (uint64_t)t.gpu_buf + 4096, t.mr->rkey, 256u, result);
  HIP_CHECK(hipDeviceSynchronize());
  printf("  Test 3: completed=%d %s\n", *result, *result == 1 ? "PASS" : "FAIL");
  assert(*result == 1);

  proxy.Shutdown();
  memset(ring, 0, sizeof(ProxyRing));
  proxy.Init(ring, qps);
  proxy.Start();

  // ── Test 4: Throughput benchmark ──
  printf("\n  Test 4: GPU→proxy throughput benchmark...\n");
  int num_ops = 5000;
  *result = 0;
  auto t0 = std::chrono::high_resolution_clock::now();
  hipLaunchKernelGGL(gpu_proxy_write_kernel, dim3(1), dim3(1), 0, 0, (volatile ProxyRing*)ring, 0,
                     (uint64_t)t.gpu_buf, t.mr->lkey, (uint64_t)t.gpu_buf + 4096, t.mr->rkey, 4096u,
                     num_ops, result);
  HIP_CHECK(hipDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  printf("  Test 4: %d ops in %.1f ms = %.0f ops/s, %.2f GB/s, %.1f us/op %s\n", *result, us / 1e3,
         num_ops / (us / 1e6), (double)num_ops * 4096 / (us / 1e6) / 1e9, us / num_ops,
         *result == num_ops ? "PASS" : "FAIL");
  assert(*result == num_ops);

  // Cleanup
  proxy.Shutdown();
  ibv_destroy_qp(t.qp);
  ibv_destroy_cq(t.cq);
  ibv_dereg_mr(t.mr);
  hipFree(t.gpu_buf);
  hipHostFree(ring);
  hipHostFree(result);
  ibv_dealloc_pd(t.pd);
  ibv_close_device(t.ctx);

  printf("\n=== ALL PASS ===\n");
  return 0;
}
