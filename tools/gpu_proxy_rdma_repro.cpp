/*
 * gpu_proxy_rdma_repro.cpp — GPU-initiated RDMA via CPU proxy thread
 *
 * Proof-of-concept for ionic AINIC where GPU IBGDA WQE posting doesn't work.
 * Instead: GPU writes descriptors to a shared ring, CPU thread calls ibv_post_send.
 *
 * Build:  hipcc -std=c++17 -O2 -o gpu_proxy_rdma_repro \
 *           gpu_proxy_rdma_repro.cpp -libverbs -lpthread -I/opt/rocm/include \
 *           --offload-arch=gfx950
 *
 * Run:  Node 0: ./gpu_proxy_rdma_repro -d ionic_0 -g 1 -s
 *       Node 1: ./gpu_proxy_rdma_repro -d ionic_0 -g 1 -c <node0_ip>
 */

#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <unistd.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP error %d at %s:%d\n",e,__FILE__,__LINE__);exit(1);}} while(0)

// ── Shared command ring between GPU and CPU proxy ──────────────────────────

#define RING_SIZE 256
#define RING_MASK (RING_SIZE - 1)

struct ProxyCmd {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t length;
    uint32_t lkey;
    uint32_t rkey;
    uint32_t flags;       // 1 = signaled
    volatile uint32_t status;  // 0=free, 1=pending, 2=posted, 3=completed, 4=error
    uint32_t pad[3];
};

struct ProxyRing {
    volatile uint32_t gpu_head;    // GPU writes (next slot to fill)
    uint32_t pad1[15];
    volatile uint32_t cpu_tail;    // CPU writes (last slot processed)
    uint32_t pad2[15];
    volatile uint32_t gpu_done_count;  // CPU increments on completion
    uint32_t pad3[15];
    volatile uint32_t shutdown;    // set to 1 to stop proxy thread
    uint32_t pad4[15];
    ProxyCmd cmds[RING_SIZE];
};

// ── GPU kernel: post N RDMA writes via proxy ring ──────────────────────────

__global__ void gpu_rdma_via_proxy(
    volatile ProxyRing* ring,
    uint64_t local_buf, uint32_t lkey,
    uint64_t remote_buf, uint32_t rkey,
    uint32_t xfer_size,
    int num_ops,
    volatile int* result)  // [0]=ops_submitted, [1]=ops_completed
{
    if (threadIdx.x || blockIdx.x) return;

    int submitted = 0;
    int completed = 0;

    uint32_t base = ring->gpu_head;
    for (int i = 0; i < num_ops; i++) {
        uint32_t seq = base + i;
        uint32_t slot = seq & RING_MASK;
        int spins = 0;
        while (__hip_atomic_load((uint32_t*)&ring->cmds[slot].status, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 0 &&
               __hip_atomic_load((uint32_t*)&ring->cmds[slot].status, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 3) {
            if (++spins > 200000000) {
                result[0] = submitted;
                result[1] = -1;  // timeout waiting for free slot
                return;
            }
            if (spins % 100000 == 0) __builtin_amdgcn_s_sleep(1);
        }

        if (i > 0 && i % 100 == 0) {
            printf("GPU: submitted %d, completed %d, slot %u status %u\n",
                   submitted, completed, slot,
                   __hip_atomic_load((uint32_t*)&ring->cmds[slot].status, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        }

        // Fill the command
        ring->cmds[slot].src_addr = local_buf + (i % 16) * xfer_size;
        ring->cmds[slot].dst_addr = remote_buf + (i % 16) * xfer_size;
        ring->cmds[slot].length = xfer_size;
        ring->cmds[slot].lkey = lkey;
        ring->cmds[slot].rkey = rkey;
        ring->cmds[slot].flags = 1;  // signaled

        // Fence before status write to ensure cmd fields are visible
        __threadfence_system();

        // Mark as pending — CPU proxy will pick it up
        ring->cmds[slot].status = 1;

        // Advance head
        __threadfence_system();
        ring->gpu_head = seq + 1;

        submitted++;
    }

    // Wait for all completions — poll slot status directly
    int spins = 0;
    while (completed < num_ops) {
        // Check if any submitted slots have completed
        for (int c = completed; c < submitted; c++) {
            uint32_t cslot = (base + c) & RING_MASK;
            uint32_t st = __hip_atomic_load(
                (uint32_t*)&ring->cmds[cslot].status, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            if (st == 3) {
                completed = c + 1;
            } else {
                break;  // completions must be in order
            }
        }
        if (++spins > 500000000) {
            result[0] = submitted;
            result[1] = completed;
            return;
        }
        if (spins % 100000 == 0) __builtin_amdgcn_s_sleep(1);
    }

    result[0] = submitted;
    result[1] = completed;
}

// ── CPU proxy thread ───────────────────────────────────────────────────────

struct ProxyCtx {
    ProxyRing* ring;
    ibv_qp* qp;
    ibv_cq* cq;
    uint32_t next_slot;
    uint64_t ops_posted;
    uint64_t ops_completed;
    uint64_t cq_polls;
};

void* proxy_thread_func(void* arg) {
    ProxyCtx* ctx = (ProxyCtx*)arg;
    ProxyRing* ring = ctx->ring;
    uint32_t next = 0;

    while (!ring->shutdown) {
        // Check for new commands from GPU
        uint32_t head = ring->gpu_head;
        while (next < head) {
            uint32_t slot = next & RING_MASK;
            volatile ProxyCmd* cmd = &ring->cmds[slot];

            // Wait for GPU to finish writing the command
            while (cmd->status != 1) {
                if (ring->shutdown) goto done;
                usleep(0);
            }

            // Build ibv_post_send
            ibv_sge sge{};
            sge.addr = cmd->src_addr;
            sge.length = cmd->length;
            sge.lkey = cmd->lkey;

            ibv_send_wr wr{};
            wr.wr_id = next;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.send_flags = (cmd->flags & 1) ? IBV_SEND_SIGNALED : 0;
            wr.wr.rdma.remote_addr = cmd->dst_addr;
            wr.wr.rdma.rkey = cmd->rkey;

            ibv_send_wr* bad = nullptr;
            int ret = ibv_post_send(ctx->qp, &wr, &bad);
            if (ret) {
                if (ret == ENOMEM) {
                    // SQ full — drain CQ until space frees up
                    ibv_wc dwc[32];
                    int dn;
                    int drained = 0;
                    while (drained < 16) { // drain at least some before retry
                        dn = ibv_poll_cq(ctx->cq, 32, dwc);
                        if (dn <= 0) { usleep(0); continue; }
                        for (int di = 0; di < dn; di++) {
                            uint32_t ds = dwc[di].wr_id & RING_MASK;
                            ring->cmds[ds].status = (dwc[di].status == IBV_WC_SUCCESS) ? 3 : 4;
                            ctx->ops_completed++;
                        }
                        __atomic_store_n((uint32_t*)&ring->gpu_done_count, ctx->ops_completed, __ATOMIC_RELEASE);
                        drained += dn;
                    }
                    // Retry post
                    ret = ibv_post_send(ctx->qp, &wr, &bad);
                }
                if (ret) {
                    fprintf(stderr, "proxy: ibv_post_send failed: %s (ret=%d)\n", strerror(ret), ret);
                    cmd->status = 4;  // error
            }} else {
                cmd->status = 2;  // posted
                ctx->ops_posted++;
            }

            next++;
        }

        // Poll CQ for completions
        ibv_wc wc[16];
        int n = ibv_poll_cq(ctx->cq, 16, wc);
        ctx->cq_polls++;
        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                fprintf(stderr, "proxy: CQE error: wr_id=%lu status=%d (%s)\n",
                        wc[i].wr_id, wc[i].status, ibv_wc_status_str(wc[i].status));
                uint32_t slot = wc[i].wr_id & RING_MASK;
                ring->cmds[slot].status = 4;
            } else {
                uint32_t slot = wc[i].wr_id & RING_MASK;
                ring->cmds[slot].status = 3;  // completed
            }
            ctx->ops_completed++;
            // Update done counter for GPU
            __atomic_store_n((uint32_t*)&ring->gpu_done_count, ctx->ops_completed, __ATOMIC_RELEASE);
        }

        // Spin — don't sleep, latency matters
    }

done:
    return nullptr;
}

// ── TCP exchange ───────────────────────────────────────────────────────────

struct QPX { uint32_t qpn, psn; ibv_gid gid; uint32_t rkey; uint64_t addr; };

static void xchg(QPX* m, QPX* p, bool srv, const char* h, int port) {
    int fd;
    if (srv) {
        int l = socket(AF_INET, SOCK_STREAM, 0);
        int on = 1; setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
        bind(l, (sockaddr*)&a, sizeof(a)); listen(l, 1);
        fd = accept(l, 0, 0); close(l);
    } else {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
        inet_pton(AF_INET, h, &a.sin_addr);
        while (connect(fd, (sockaddr*)&a, sizeof(a)) < 0) usleep(100000);
    }
    write(fd, m, sizeof(*m)); read(fd, p, sizeof(*p)); close(fd);
}

// ── main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* dev = "ionic_0";
    int gid = 1;
    bool srv = false;
    const char* peer = nullptr;
    int port = 19877;
    int num_ops = 1000;
    uint32_t xfer_size = 4096;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-d")) dev = argv[++i];
        else if (!strcmp(argv[i], "-g")) gid = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s")) srv = true;
        else if (!strcmp(argv[i], "-c")) peer = argv[++i];
        else if (!strcmp(argv[i], "-p")) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")) num_ops = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-S")) xfer_size = atoi(argv[++i]);
    }
    if (!srv && !peer) {
        fprintf(stderr, "Usage: %s -d <dev> -g <gid> [-s|-c <ip>] [-n ops] [-S size]\n", argv[0]);
        return 1;
    }

    setbuf(stdout, NULL);
    printf("============================================================\n");
    printf("  GPU Proxy RDMA Reproducer\n");
    printf("  Dev:%s GID:%d Role:%s Ops:%d Size:%u\n",
           dev, gid, srv ? "server" : "client", num_ops, xfer_size);
    printf("============================================================\n\n");

    HIP_CHECK(hipSetDevice(0));

    // ── Setup RDMA ─────────────────────────────────────────────────
    int nd;
    ibv_device** dl = ibv_get_device_list(&nd);
    ibv_device* d = nullptr;
    for (int i = 0; i < nd; i++) if (!strcmp(dl[i]->name, dev)) d = dl[i];
    assert(d);

    ibv_context* ctx = ibv_open_device(d);
    ibv_pd* pd = ibv_alloc_pd(ctx);
    ibv_cq* cq = ibv_create_cq(ctx, 256, nullptr, nullptr, 0);
    assert(cq);

    ibv_qp_init_attr qa{};
    qa.send_cq = cq; qa.recv_cq = cq; qa.qp_type = IBV_QPT_RC;
    qa.cap = {128, 128, 1, 1, 0};
    ibv_qp* qp = ibv_create_qp(pd, &qa);
    assert(qp);

    // GPU data buffer (16 × xfer_size)
    size_t buf_size = 16 * xfer_size;
    void* gpu_buf;
    HIP_CHECK(hipMalloc(&gpu_buf, buf_size));
    HIP_CHECK(hipMemset(gpu_buf, 0xAB, buf_size));
    ibv_mr* mr = ibv_reg_mr(pd, gpu_buf, buf_size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    assert(mr);
    printf("MR: addr=%p size=%zu lkey=%u rkey=%u\n", gpu_buf, buf_size, mr->lkey, mr->rkey);

    // Connect QP
    ibv_gid mg;
    ibv_query_gid(ctx, 1, gid, &mg);
    QPX lx{qp->qp_num, 0, mg, mr->rkey, (uint64_t)gpu_buf}, rx{};
    xchg(&lx, &rx, srv, peer, port);

    {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_INIT; a.port_num = 1;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    }
    {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_RTR; a.path_mtu = IBV_MTU_4096;
        a.dest_qp_num = rx.qpn; a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
        memcpy(&a.ah_attr.grh.dgid, &rx.gid, 16);
        a.ah_attr.grh.sgid_index = gid; a.ah_attr.grh.hop_limit = 1;
        a.ah_attr.is_global = 1; a.ah_attr.port_num = 1;
        ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    }
    {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7;
        a.rnr_retry = 7; a.max_rd_atomic = 1;
        ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
            IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
    }
    printf("QP connected (qpn=%u -> remote qpn=%u)\n", qp->qp_num, rx.qpn);

    // Sync both sides
    { QPX d{}; xchg(&d, &d, srv, peer, port + 1); }

    // ── Allocate proxy ring (host-pinned, GPU+CPU visible) ─────────
    ProxyRing* ring;
    HIP_CHECK(hipHostMalloc(&ring, sizeof(ProxyRing),
        hipHostMallocMapped | hipHostMallocCoherent));
    memset((void*)ring, 0, sizeof(ProxyRing));
    printf("Proxy ring: %p (%zu bytes, %d slots)\n", ring, sizeof(ProxyRing), RING_SIZE);

    // GPU result buffer
    int* result;
    HIP_CHECK(hipHostMalloc(&result, 8, hipHostMallocMapped | hipHostMallocCoherent));
    result[0] = 0; result[1] = 0;

    // ── Start proxy thread ─────────────────────────────────────────
    ProxyCtx pctx{};
    pctx.ring = ring;
    pctx.qp = qp;
    pctx.cq = cq;

    pthread_t proxy_tid;
    pthread_create(&proxy_tid, nullptr, proxy_thread_func, &pctx);
    printf("Proxy thread started\n\n");

    // ── Benchmark (no warmup) ─────────────────────────────────────
    printf("── Benchmark: %d ops, %u bytes each ──\n", num_ops, xfer_size);
    auto t0 = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(gpu_rdma_via_proxy, dim3(1), dim3(1), 0, 0,
        (volatile ProxyRing*)ring,
        (uint64_t)gpu_buf, mr->lkey,
        rx.addr, rx.rkey,
        xfer_size, num_ops, result);
    HIP_CHECK(hipDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double elapsed_s = elapsed_us / 1e6;

    printf("Result: submitted=%d completed=%d\n", result[0], result[1]);
    printf("Proxy stats: posted=%lu completed=%lu cq_polls=%lu\n",
           pctx.ops_posted, pctx.ops_completed, pctx.cq_polls);

    if (result[1] == num_ops) {
        double ops_per_sec = num_ops / elapsed_s;
        double bw_gbps = (double)num_ops * xfer_size / elapsed_s / 1e9;
        double lat_us = elapsed_us / num_ops;
        printf("\n  PASS\n");
        printf("  Time: %.2f ms\n", elapsed_us / 1e3);
        printf("  Ops/s: %.0f\n", ops_per_sec);
        printf("  Bandwidth: %.2f GB/s\n", bw_gbps);
        printf("  Avg latency: %.1f us/op\n", lat_us);
    } else {
        printf("\n  FAIL (completed %d / %d)\n", result[1], num_ops);
    }

    // Sweep removed for simplicity — add back once basic benchmark works

    // ── Cleanup ────────────────────────────────────────────────────
    ring->shutdown = 1;
    pthread_join(proxy_tid, nullptr);

    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(mr);
    hipFree(gpu_buf);
    hipHostFree(ring);
    hipHostFree(result);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dl);

    printf("\nDone.\n");
    return 0;
}
