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
      // Recv CQE: incoming SEND_WITH_IMM for barrier atomic emulation
      if (wc[i].opcode == IBV_WC_RECV || wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        if (wc[i].status == IBV_WC_SUCCESS && wc[i].byte_len >= 16) {
          uint32_t recv_idx = static_cast<uint32_t>(wc[i].wr_id);
          if (recv_idx < qph.recv_count && qph.recv_buf) {
            struct { uint64_t addr; uint64_t val; } payload;
            memcpy(&payload, reinterpret_cast<char*>(qph.recv_buf) + recv_idx * 64, 16);
            // Barrier atomics have no data-ordering requirement, so CPU
            // atomic on GPU VRAM is safe (no concurrent NIC DMA to race with).
            volatile uint64_t* target = reinterpret_cast<volatile uint64_t*>(payload.addr);
            __atomic_fetch_add(target, payload.val, __ATOMIC_SEQ_CST);
            asm volatile("clflush (%0)" :: "r"(target) : "memory");
            asm volatile("sfence" ::: "memory");
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
  hipSetDevice(gpu_id_);
  while (!ring_->shutdown) {
    bool did_work = false;

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
          fprintf(stderr, "proxy: null QP at idx=%u\n", qi);
          cmd->status = PROXY_ERROR;
          next_slot_++;
          continue;
        }

        ibv_sge sge{};
        sge.addr = cmd->src_addr;
        sge.length = cmd->length;
        sge.lkey = (qph.lkey_override != 0) ? qph.lkey_override : cmd->lkey;

        if (ops_posted_ < 3 || (ops_posted_ % 100 == 0)) {
          fprintf(stderr, "[MoRI-PROXY] post #%lu: qp_idx=%u op=%u len=%u head=%u next=%u\n",
                  ops_posted_, qi, cmd->op, cmd->length, head, next_slot_);
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
          case PROXY_SIGNAL_WRITE: {
            // Signal paired with data: RDMA_WRITE so both go through same
            // NIC → PCIe → GPU VRAM path. RC QP guarantees responder-side
            // ordering — data write completes before signal write at the
            // remote GPU without needing IBV_SEND_FENCE.
            uint64_t val = cmd->atomic_arg;
            memcpy(reinterpret_cast<void*>(sge.addr), &val, 8);
            sge.length = 8;
            sge.lkey = cmd->lkey;
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.send_flags |= IBV_SEND_INLINE;
            wr.wr.rdma.remote_addr = cmd->dst_addr;
            wr.wr.rdma.rkey = (qph.rkey_override != 0) ? qph.rkey_override : cmd->rkey;
            break;
          }
          case PROXY_ATOMIC_FETCH_ADD:
          case PROXY_ATOMIC_CMP_SWAP: {
            // Standalone atomic (barrier): SEND_WITH_IMM so receiver proxy
            // does CPU __atomic_fetch_add. No data-ordering requirement.
            struct { uint64_t addr; uint64_t val; } payload;
            payload.addr = cmd->dst_addr;
            payload.val = cmd->atomic_arg;
            memcpy(reinterpret_cast<void*>(sge.addr), &payload, 16);
            sge.length = 16;
            wr.opcode = IBV_WR_SEND_WITH_IMM;
            wr.imm_data = htonl(0xA70C);
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

    if (ops_posted_ > 0) {
      for (auto& qph : qps_) {
        if (qph.qp) DrainCq(qph);
      }
    }

    if (!did_work && ops_posted_ > 0 && ops_completed_ < ops_posted_) {
      static thread_local uint64_t idle = 0;
      static thread_local int dump_count = 0;
      if (++idle == 50000000 && dump_count < 3) {
        fprintf(stderr, "[MoRI-PROXY] STALL: posted=%lu completed=%lu head=%u next=%u gpu=%d\n",
                ops_posted_, ops_completed_, ring_->gpu_head, next_slot_, gpu_id_);
        int pending = 0;
        for (uint32_t s = 0; s < PROXY_RING_SIZE && pending < 5; s++) {
          uint32_t st = ring_->cmds[s].status;
          if (st != PROXY_FREE && st != PROXY_COMPLETED) {
            fprintf(stderr, "[MoRI-PROXY] PENDING slot=%u status=%u op=%u qp_idx=%u len=%u\n",
                    s, st, ring_->cmds[s].op, ring_->cmds[s].qp_idx, ring_->cmds[s].length);
            pending++;
          }
        }
        if (pending == 0) fprintf(stderr, "[MoRI-PROXY] No pending slots — GPU waiting for data\n");
        dump_count++;
        idle = 0;
      }
    }
  }
}

}  // namespace core
}  // namespace mori
