// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#include "mori/core/transport/rdma/proxy/proxy_thread.hpp"

#include <arpa/inet.h>
#include <hip/hip_runtime_api.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace mori {
namespace core {

ProxyThread::~ProxyThread() { Shutdown(); }

void ProxyThread::Init(ProxyRing* ring, std::vector<ProxyQpHandle> qps, int gpuId) {
  ring_ = ring;
  qps_ = std::move(qps);
  next_slot_ = 0;
  ops_posted_ = 0;
  ops_completed_ = 0;
  gpu_id_ = gpuId;
}

void ProxyThread::Start() {
  if (running_.load()) return;
  running_.store(true);
  pthread_create(&thread_, nullptr, ThreadFunc, this);
}

void ProxyThread::Shutdown() {
  if (!running_.load()) return;
  if (ring_) ring_->shutdown = 1;
  running_.store(false);
  pthread_join(thread_, nullptr);
}

void* ProxyThread::ThreadFunc(void* arg) {
  auto* self = static_cast<ProxyThread*>(arg);
  self->MainLoop();
  return nullptr;
}

void ProxyThread::DrainCq(ProxyQpHandle& qph) {
  if (!qph.cq) return;
  ibv_wc wc[32];
  int n;
  while ((n = ibv_poll_cq(qph.cq, 32, wc)) > 0) {
    for (int i = 0; i < n; i++) {
      // Recv CQE: incoming SEND_WITH_IMM carrying atomic emulation payload
      if (wc[i].opcode == IBV_WC_RECV || wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        if (wc[i].status == IBV_WC_SUCCESS && wc[i].byte_len >= 16) {
          // Read [dst_addr, add_value] from recv buffer
          uint32_t recv_idx = static_cast<uint32_t>(wc[i].wr_id);
          if (recv_idx < qph.recv_count && qph.recv_buf) {
            struct { uint64_t addr; uint64_t val; } payload;
            memcpy(&payload, reinterpret_cast<char*>(qph.recv_buf) + recv_idx * 64, 16);
            // Atomic add on GPU VRAM from CPU. With hipDeviceMallocUncached,
            // GPU reads bypass L2 cache so they see CPU writes directly.
            //
            // IMPORTANT: The prior RDMA_WRITE (data) and this SEND_WITH_IMM
            // (signal) were sent with FENCE on the same QP, but take different
            // paths: data goes NIC→GPU VRAM via DMA, signal goes NIC→CPU recv
            // buffer→here. We must ensure the data DMA completed before we
            // increment the signal counter that tells the GPU "data is ready".
            // Read-back from the data destination address forces PCIe ordering.
            // Fence: ensure prior RDMA_WRITE data has landed in GPU VRAM
            // before incrementing the signal counter that tells the GPU
            // "data is ready". Read from the data region base address to
            // force PCIe posted write ordering, then do the signal atomic.
            // The data region starts at the base of symmetric memory (page-aligned),
            // so read from a page-aligned address near the signal to force ordering.
            uintptr_t page_addr = payload.addr & ~0xFFFULL;
            volatile uint64_t fence_read = *reinterpret_cast<volatile uint64_t*>(page_addr);
            (void)fence_read;
            asm volatile("mfence" ::: "memory");

            volatile uint64_t* target = reinterpret_cast<volatile uint64_t*>(payload.addr);
            __atomic_fetch_add(target, payload.val, __ATOMIC_SEQ_CST);
            asm volatile("clflush (%0)" :: "r"(target) : "memory");
            asm volatile("sfence" ::: "memory");
            recv_atomics_++;
            // Re-post recv WR
            ibv_sge rsge{};
            rsge.addr = reinterpret_cast<uintptr_t>(qph.recv_buf) + recv_idx * 64;
            rsge.length = 64;
            rsge.lkey = qph.recv_lkey;
            ibv_recv_wr rwr{}, *rbad = nullptr;
            rwr.wr_id = recv_idx;
            rwr.sg_list = &rsge;
            rwr.num_sge = 1;
            ibv_post_recv(qph.qp, &rwr, &rbad);
          }
        } else if (wc[i].status != IBV_WC_SUCCESS) {
          fprintf(stderr, "proxy: RECV CQE error status=%d (%s) ibvQP=%u\n",
                  wc[i].status, ibv_wc_status_str(wc[i].status),
                  qph.qp ? qph.qp->qp_num : 0);
        }
        continue;
      }
      // Send CQE: our outgoing op completed
      uint32_t slot = static_cast<uint32_t>(wc[i].wr_id) & PROXY_RING_MASK;
      if (wc[i].status == IBV_WC_SUCCESS) {
        ring_->cmds[slot].status = PROXY_COMPLETED;
      } else {
        fprintf(stderr, "proxy: CQE error slot=%u status=%d (%s) wr_id=%lu ibvQP=%u\n",
                slot, wc[i].status, ibv_wc_status_str(wc[i].status), wc[i].wr_id,
                qph.qp ? qph.qp->qp_num : 0);
        ring_->cmds[slot].status = PROXY_ERROR;
      }
      ops_completed_++;
    }
  }
}

void ProxyThread::MainLoop() {
  // Set HIP device context for correct GPU VRAM BAR mapping.
  // Only called once at thread start — no further HIP calls in the loop.
  hipSetDevice(gpu_id_);
  while (!ring_->shutdown) {
    bool did_work = false;

    // Try to post ONE pending command
    uint32_t head = ring_->gpu_head;
    if (next_slot_ < head) {
      uint32_t slot = next_slot_ & PROXY_RING_MASK;
      volatile ProxyCmd* cmd = &ring_->cmds[slot];

      if (cmd->status == PROXY_PENDING) {
        uint32_t qi = cmd->qp_idx;
        if (qi >= qps_.size()) {
          fprintf(stderr, "proxy: qp_idx=%u out of range (%zu)\n", qi, qps_.size());
          cmd->status = PROXY_ERROR;
          next_slot_++;
          continue;
        }
        ProxyQpHandle& qph = qps_[qi];
        if (qph.qp == nullptr) {
          // Non-RDMA peer slot — shouldn't happen in normal flow
          fprintf(stderr, "proxy: null QP at idx=%u\n", qi);
          cmd->status = PROXY_ERROR;
          next_slot_++;
          continue;
        }

        ibv_sge sge{};
        sge.addr = cmd->src_addr;
        sge.length = cmd->length;
        bool isAtomic = (cmd->op == PROXY_ATOMIC_FETCH_ADD || cmd->op == PROXY_ATOMIC_CMP_SWAP);
        sge.lkey = (isAtomic) ? cmd->lkey
                   : (qph.lkey_override != 0) ? qph.lkey_override : cmd->lkey;

        if (ops_posted_ < 3) {
          fprintf(stderr, "[MoRI-PROXY] post #%lu: qp_idx=%u op=%u len=%u\n",
                  ops_posted_, qi, cmd->op, cmd->length);
        }

        ibv_send_wr wr{};
        wr.wr_id = next_slot_;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.send_flags = IBV_SEND_SIGNALED;

        switch (cmd->op) {
          case PROXY_RDMA_WRITE:
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.wr.rdma.remote_addr = cmd->dst_addr;
            wr.wr.rdma.rkey = (qph.rkey_override != 0) ? qph.rkey_override : cmd->rkey;
            break;
          case PROXY_RDMA_WRITE_INLINE:
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.send_flags |= IBV_SEND_INLINE;
            wr.wr.rdma.remote_addr = cmd->dst_addr;
            wr.wr.rdma.rkey = (qph.rkey_override != 0) ? qph.rkey_override : cmd->rkey;
            break;
          case PROXY_ATOMIC_FETCH_ADD:
          case PROXY_ATOMIC_CMP_SWAP: {
            // Pensando AINIC: atomics return CQE OK but don't modify remote memory.
            // Emulate via SEND_WITH_IMM: send [dst_addr, add_value] inline.
            // Receiver proxy thread does CPU atomic add on the GPU address.
            struct { uint64_t addr; uint64_t val; } payload;
            payload.addr = cmd->dst_addr;
            payload.val = cmd->atomic_arg;
            memcpy(reinterpret_cast<void*>(sge.addr), &payload, 16);
            sge.length = 16;
            wr.opcode = IBV_WR_SEND_WITH_IMM;
            wr.imm_data = htonl(0xA70C); // magic marker
            wr.send_flags |= IBV_SEND_FENCE | IBV_SEND_INLINE;
            break;
          }
          default:
            cmd->status = PROXY_ERROR;
            next_slot_++;
            continue;
        }

        ibv_send_wr* bad = nullptr;
        int ret = ibv_post_send(qph.qp, &wr, &bad);

        if (ret == ENOMEM) {
          // SQ full — drain CQ until we can post
          for (int attempt = 0; attempt < 1000; attempt++) {
            DrainCq(qph);
            ret = ibv_post_send(qph.qp, &wr, &bad);
            if (ret != ENOMEM) break;
            usleep(0);
          }
        }

        if (ret) {
          fprintf(stderr, "proxy: ibv_post_send failed: %s (ret=%d) op=%u\n",
                  strerror(ret), ret, cmd->op);
          cmd->status = PROXY_ERROR;
        } else {
          ops_posted_++;
        }
        next_slot_++;
        did_work = true;
      }
    }

    // Only drain CQ after we've posted at least one command
    if (ops_posted_ > 0) {
      for (auto& qph : qps_) {
        if (qph.qp) DrainCq(qph);
      }
    }

    if (!did_work) idle_count_++;
    else idle_count_ = 0;
  }
}

}  // namespace core
}  // namespace mori
