// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#include "mori/core/transport/rdma/proxy/proxy_thread.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace mori {
namespace core {

ProxyThread::~ProxyThread() { Shutdown(); }

void ProxyThread::Init(ProxyRing* ring, std::vector<ProxyQpHandle> qps) {
  ring_ = ring;
  qps_ = std::move(qps);
  next_slot_ = 0;
  ops_posted_ = 0;
  ops_completed_ = 0;
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
            // Pensando AINIC: real RDMA atomics silently fail (CQE OK, no
            // memory modification). No contention on signal entries — each
            // receiver GPU gets signals from exactly one sender GPU — so a
            // plain RDMA_WRITE of the value works.
            //
            // Both the data RDMA_WRITE and this signal RDMA_WRITE go through
            // the same NIC → PCIe → GPU VRAM path. FENCE on the same QP
            // guarantees the data write completes before the signal write.
            // This eliminates the CPU-vs-NIC PCIe ordering issue that caused
            // the 2-token data corruption with SEND_WITH_IMM.
            uint64_t val = cmd->atomic_arg;
            memcpy(reinterpret_cast<void*>(sge.addr), &val, 8);
            sge.length = 8;
            sge.lkey = cmd->lkey;  // ibuf's own lkey (not perNic override)
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.send_flags |= IBV_SEND_FENCE;
            wr.wr.rdma.remote_addr = cmd->dst_addr;
            wr.wr.rdma.rkey = (qph.rkey_override != 0) ? qph.rkey_override : cmd->rkey;
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
  }
}

}  // namespace core
}  // namespace mori
