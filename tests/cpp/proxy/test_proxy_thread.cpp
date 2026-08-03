// Level 2: CPU-only test for proxy thread
// Tests: proxy thread picks up commands and posts via ibv_post_send
// Uses loopback RDMA (same node, self-connected QP)
// Requires: RDMA device available (ionic or mlx5)
//
// Build: g++ -std=c++17 -O2 -I<mori>/include -I<mori> -o test_proxy_thread \
//          test_proxy_thread.cpp proxy_thread.cpp -libverbs -lpthread
// Run:   ./test_proxy_thread -d <rdma_device> -g <gid_index>

#include "mori/core/transport/rdma/proxy/proxy_types.hpp"
#include "mori/core/transport/rdma/proxy/proxy_thread.hpp"

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace mori::core;

struct TestCtx {
  ibv_context* ctx;
  ibv_pd* pd;
  ibv_cq* cq;
  ibv_qp* qp;
  ibv_mr* mr;
  void* buf;
  size_t buf_size;
};

static TestCtx setup_loopback(const char* dev_name, int gid_idx) {
  TestCtx t{};
  int nd;
  ibv_device** dl = ibv_get_device_list(&nd);
  ibv_device* d = nullptr;
  for (int i = 0; i < nd; i++) {
    if (!strcmp(dl[i]->name, dev_name)) d = dl[i];
  }
  if (!d) { fprintf(stderr, "Device %s not found\n", dev_name); exit(1); }

  t.ctx = ibv_open_device(d);
  t.pd = ibv_alloc_pd(t.ctx);
  t.cq = ibv_create_cq(t.ctx, 256, nullptr, nullptr, 0);

  ibv_qp_init_attr qa{};
  qa.send_cq = t.cq; qa.recv_cq = t.cq; qa.qp_type = IBV_QPT_RC;
  qa.cap = {128, 128, 1, 1, 0};
  t.qp = ibv_create_qp(t.pd, &qa);

  t.buf_size = 64 * 1024;
  t.buf = calloc(1, t.buf_size);
  t.mr = ibv_reg_mr(t.pd, t.buf, t.buf_size,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

  // Self-connect QP (loopback)
  ibv_gid gid;
  ibv_query_gid(t.ctx, 1, gid_idx, &gid);

  { ibv_qp_attr a{}; a.qp_state = IBV_QPS_INIT; a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
    ibv_modify_qp(t.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS); }

  { ibv_qp_attr a{}; a.qp_state = IBV_QPS_RTR; a.path_mtu = IBV_MTU_4096;
    a.dest_qp_num = t.qp->qp_num; a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
    memcpy(&a.ah_attr.grh.dgid, &gid, 16); a.ah_attr.grh.sgid_index = gid_idx;
    a.ah_attr.grh.hop_limit = 1; a.ah_attr.is_global = 1; a.ah_attr.port_num = 1;
    ibv_modify_qp(t.qp, &a, IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER); }

  { ibv_qp_attr a{}; a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7;
    a.rnr_retry = 7; a.max_rd_atomic = 1;
    ibv_modify_qp(t.qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC); }

  ibv_free_device_list(dl);
  return t;
}

static void cleanup(TestCtx& t) {
  ibv_destroy_qp(t.qp);
  ibv_destroy_cq(t.cq);
  ibv_dereg_mr(t.mr);
  free(t.buf);
  ibv_dealloc_pd(t.pd);
  ibv_close_device(t.ctx);
}

void test_single_write(TestCtx& t) {
  ProxyRing ring{};
  memset(&ring, 0, sizeof(ring));

  ProxyThread proxy;
  std::vector<ProxyQpHandle> qps = {{t.qp, t.cq}};
  proxy.Init(&ring, qps);
  proxy.Start();

  // Write pattern to src region
  memset((char*)t.buf, 0xAA, 4096);
  memset((char*)t.buf + 4096, 0x00, 4096);

  // Submit a write command: copy 4096 bytes from offset 0 to offset 4096
  ring.cmds[0].op = PROXY_RDMA_WRITE;
  ring.cmds[0].qp_idx = 0;
  ring.cmds[0].src_addr = (uint64_t)t.buf;
  ring.cmds[0].dst_addr = (uint64_t)t.buf + 4096;
  ring.cmds[0].length = 4096;
  ring.cmds[0].lkey = t.mr->lkey;
  ring.cmds[0].rkey = t.mr->rkey;
  ring.cmds[0].flags = 1;
  ring.cmds[0].status = PROXY_PENDING;
  ring.gpu_head = 1;

  // Wait for completion
  int spins = 0;
  while (ring.cmds[0].status == PROXY_PENDING && spins < 1000000) {
    usleep(10);
    spins++;
  }

  proxy.Shutdown();

  assert(ring.cmds[0].status == PROXY_COMPLETED);

  // Verify data was written
  int match = memcmp(t.buf, (char*)t.buf + 4096, 4096);
  assert(match == 0);

  printf("  single_write: PASS\n");
}

void test_multiple_writes(TestCtx& t) {
  ProxyRing ring{};
  memset(&ring, 0, sizeof(ring));

  ProxyThread proxy;
  std::vector<ProxyQpHandle> qps = {{t.qp, t.cq}};
  proxy.Init(&ring, qps);
  proxy.Start();

  int num_ops = 100;
  // Fill src buffer with sequential pattern
  for (int i = 0; i < 4096; i++) {
    ((uint8_t*)t.buf)[i] = i & 0xFF;
  }
  memset((char*)t.buf + 4096, 0, 4096);

  // Submit 100 writes of 32 bytes each at different offsets
  for (int i = 0; i < num_ops; i++) {
    uint32_t slot = i & PROXY_RING_MASK;
    ring.cmds[slot].op = PROXY_RDMA_WRITE;
    ring.cmds[slot].qp_idx = 0;
    ring.cmds[slot].src_addr = (uint64_t)t.buf + (i % 128) * 32;
    ring.cmds[slot].dst_addr = (uint64_t)t.buf + 4096 + (i % 128) * 32;
    ring.cmds[slot].length = 32;
    ring.cmds[slot].lkey = t.mr->lkey;
    ring.cmds[slot].rkey = t.mr->rkey;
    ring.cmds[slot].flags = 1;
    ring.cmds[slot].status = PROXY_PENDING;
    ring.gpu_head = i + 1;
  }

  // Wait for all completions
  int spins = 0;
  while (spins < 5000000) {
    bool all_done = true;
    for (int i = 0; i < num_ops; i++) {
      uint32_t slot = i & PROXY_RING_MASK;
      if (ring.cmds[slot].status != PROXY_COMPLETED &&
          ring.cmds[slot].status != PROXY_FREE) {
        all_done = false;
        break;
      }
    }
    if (all_done) break;
    usleep(10);
    spins++;
  }

  proxy.Shutdown();

  int completed = 0;
  for (int i = 0; i < num_ops; i++) {
    uint32_t slot = i & PROXY_RING_MASK;
    if (ring.cmds[slot].status == PROXY_COMPLETED) completed++;
  }
  assert(completed == num_ops);

  printf("  multiple_writes (%d ops): PASS\n", num_ops);
}

void test_throughput(TestCtx& t) {
  ProxyRing ring{};
  memset(&ring, 0, sizeof(ring));

  ProxyThread proxy;
  std::vector<ProxyQpHandle> qps = {{t.qp, t.cq}};
  proxy.Init(&ring, qps);
  proxy.Start();

  int num_ops = 1000;
  uint32_t xfer_size = 4096;

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_ops; i++) {
    uint32_t slot = i & PROXY_RING_MASK;

    // Wait for slot to be free
    while (ring.cmds[slot].status == PROXY_PENDING) {
      usleep(0);
    }

    ring.cmds[slot].op = PROXY_RDMA_WRITE;
    ring.cmds[slot].qp_idx = 0;
    ring.cmds[slot].src_addr = (uint64_t)t.buf;
    ring.cmds[slot].dst_addr = (uint64_t)t.buf + 4096;
    ring.cmds[slot].length = xfer_size;
    ring.cmds[slot].lkey = t.mr->lkey;
    ring.cmds[slot].rkey = t.mr->rkey;
    ring.cmds[slot].flags = 1;
    ring.cmds[slot].status = PROXY_PENDING;
    ring.gpu_head = i + 1;
  }

  // Wait for last slot
  uint32_t last_slot = (num_ops - 1) & PROXY_RING_MASK;
  while (ring.cmds[last_slot].status == PROXY_PENDING) {
    usleep(1);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

  proxy.Shutdown();

  double ops_per_sec = num_ops / (us / 1e6);
  double bw = (double)num_ops * xfer_size / (us / 1e6) / 1e9;
  printf("  throughput: %d ops in %.1f ms = %.0f ops/s, %.2f GB/s, %.1f us/op PASS\n",
         num_ops, us / 1e3, ops_per_sec, bw, us / num_ops);
}

int main(int argc, char** argv) {
  const char* dev = "ionic_0";
  int gid = 1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-d")) dev = argv[++i];
    else if (!strcmp(argv[i], "-g")) gid = atoi(argv[++i]);
  }

  printf("=== Level 2: proxy_thread test (dev=%s gid=%d) ===\n", dev, gid);
  TestCtx t = setup_loopback(dev, gid);
  printf("  QP loopback connected (qpn=%u)\n", t.qp->qp_num);

  test_single_write(t);
  test_multiple_writes(t);
  test_throughput(t);

  cleanup(t);
  printf("=== ALL PASS ===\n");
  return 0;
}
