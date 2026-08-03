/*
 * test_cross_nic_dma.cpp — Can ionic_N DMA GPU M's VRAM? (M != N's affinity GPU)
 *
 * Tests: allocate buffer on GPU 0, register MR on ionic_3's PD,
 * do a loopback RDMA write through ionic_3.
 *
 * Build: hipcc -std=c++17 -O2 -Wno-unused-result -o test_cross_nic_dma \
 *          test_cross_nic_dma.cpp -libverbs -I/opt/rocm/include --offload-arch=gfx950
 *
 * Run:  ./test_cross_nic_dma -g 0 -d ionic_3 --gid 1
 *       (allocate on GPU 0, RDMA through ionic_3)
 */

#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP %d at %s:%d\n",e,__FILE__,__LINE__);exit(1);}} while(0)

int main(int argc, char** argv) {
  int gpu = 0;
  const char* dev = "ionic_3";
  int gid_idx = 1;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-g")) gpu = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-d")) dev = argv[++i];
    else if (!strcmp(argv[i], "--gid")) gid_idx = atoi(argv[++i]);
  }

  setbuf(stdout, NULL);
  printf("=== Cross-NIC DMA Test ===\n");
  printf("  GPU: %d, NIC: %s, GID index: %d\n\n", gpu, dev, gid_idx);

  // Set GPU
  HIP_CHECK(hipSetDevice(gpu));

  // Find NIC
  int nd;
  ibv_device** dl = ibv_get_device_list(&nd);
  ibv_device* d = nullptr;
  for (int i = 0; i < nd; i++) if (!strcmp(dl[i]->name, dev)) d = dl[i];
  if (!d) { fprintf(stderr, "Device %s not found\n", dev); return 1; }

  ibv_context* ctx = ibv_open_device(d);
  ibv_pd* pd = ibv_alloc_pd(ctx);
  printf("  NIC %s opened, PD allocated\n", dev);

  // Allocate GPU buffer on GPU `gpu`
  size_t buf_size = 64 * 1024;
  void* gpu_buf;
  HIP_CHECK(hipMalloc(&gpu_buf, buf_size));
  HIP_CHECK(hipMemset(gpu_buf, 0xAB, buf_size));
  printf("  GPU %d buffer: %p (%zu bytes)\n", gpu, gpu_buf, buf_size);

  // Register MR on this NIC's PD for GPU buffer
  printf("  Registering MR on %s PD for GPU %d buffer...\n", dev, gpu);
  ibv_mr* mr = ibv_reg_mr(pd, gpu_buf, buf_size,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  if (!mr) {
    printf("  ibv_reg_mr FAILED: %s\n", strerror(errno));
    printf("  *** CROSS-NIC DMA NOT SUPPORTED ***\n");
    hipFree(gpu_buf);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    return 1;
  }
  printf("  MR registered: lkey=%u rkey=%u\n", mr->lkey, mr->rkey);

  // Create loopback QP
  ibv_cq* cq = ibv_create_cq(ctx, 64, nullptr, nullptr, 0);
  ibv_qp_init_attr qa{};
  qa.send_cq = cq; qa.recv_cq = cq; qa.qp_type = IBV_QPT_RC;
  qa.cap = {32, 32, 1, 1, 0};
  ibv_qp* qp = ibv_create_qp(pd, &qa);
  assert(qp);

  ibv_gid gid;
  ibv_query_gid(ctx, 1, gid_idx, &gid);

  { ibv_qp_attr a{}; a.qp_state = IBV_QPS_INIT; a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
    ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS); }
  { ibv_qp_attr a{}; a.qp_state = IBV_QPS_RTR; a.path_mtu = IBV_MTU_4096;
    a.dest_qp_num = qp->qp_num; a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
    memcpy(&a.ah_attr.grh.dgid, &gid, 16); a.ah_attr.grh.sgid_index = gid_idx;
    a.ah_attr.grh.hop_limit = 1; a.ah_attr.is_global = 1; a.ah_attr.port_num = 1;
    ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER); }
  { ibv_qp_attr a{}; a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7;
    a.rnr_retry = 7; a.max_rd_atomic = 1;
    ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC); }
  printf("  QP loopback connected (qpn=%u)\n", qp->qp_num);

  // RDMA write: first 4KB → second 4KB
  printf("\n  Posting RDMA write (4KB, src=offset 0, dst=offset 4096)...\n");
  ibv_sge sge{};
  sge.addr = (uint64_t)gpu_buf;
  sge.length = 4096;
  sge.lkey = mr->lkey;

  ibv_send_wr wr{};
  wr.wr_id = 1;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uint64_t)gpu_buf + 4096;
  wr.wr.rdma.rkey = mr->rkey;

  ibv_send_wr* bad = nullptr;
  int ret = ibv_post_send(qp, &wr, &bad);
  if (ret) {
    printf("  ibv_post_send FAILED: %s\n", strerror(ret));
  } else {
    ibv_wc wc{};
    int polls = 0;
    bool ok = false;
    while (polls < 100000) {
      if (ibv_poll_cq(cq, 1, &wc) > 0) { ok = true; break; }
      usleep(10);
      polls++;
    }
    if (ok && wc.status == IBV_WC_SUCCESS) {
      printf("  RDMA write: PASS (polls=%d)\n", polls);
      printf("\n  *** CROSS-NIC DMA WORKS: %s can DMA GPU %d's VRAM ***\n", dev, gpu);
    } else {
      printf("  RDMA write: FAIL (status=%d %s polls=%d)\n",
             ok ? (int)wc.status : -1, ok ? ibv_wc_status_str(wc.status) : "timeout", polls);
      printf("\n  *** CROSS-NIC DMA FAILED ***\n");
    }
  }

  // Cleanup
  ibv_destroy_qp(qp);
  ibv_destroy_cq(cq);
  ibv_dereg_mr(mr);
  hipFree(gpu_buf);
  ibv_dealloc_pd(pd);
  ibv_close_device(ctx);
  ibv_free_device_list(dl);

  return 0;
}
