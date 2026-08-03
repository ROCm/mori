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
  ibv_wc wc[32];
  int n;
  while ((n = ibv_poll_cq(qph.cq, 32, wc)) > 0) {
    for (int i = 0; i < n; i++) {
      uint32_t slot = static_cast<uint32_t>(wc[i].wr_id) & PROXY_RING_MASK;
      if (wc[i].status == IBV_WC_SUCCESS) {
        if (wc[i].opcode == IBV_WC_FETCH_ADD || wc[i].opcode == IBV_WC_COMP_SWAP) {
          // For fetch atomics, the result is already in the ibuf.
          // The GPU reads it from ibuf_addr after seeing COMPLETED.
        }
        ring_->cmds[slot].status = PROXY_COMPLETED;
      } else {
        fprintf(stderr, "proxy: CQE error slot=%u status=%d (%s) wr_id=%lu\n",
                slot, wc[i].status, ibv_wc_status_str(wc[i].status), wc[i].wr_id);
        ring_->cmds[slot].status = PROXY_ERROR;
      }
      ops_completed_++;
    }
  }
}

void ProxyThread::MainLoop() {
  while (!ring_->shutdown) {
    bool did_work = false;

    // Try to post ONE pending command
    uint32_t head = ring_->gpu_head;
    if (next_slot_ < head) {
      uint32_t slot = next_slot_ & PROXY_RING_MASK;
      volatile ProxyCmd* cmd = &ring_->cmds[slot];

      if (cmd->status == PROXY_PENDING) {
        uint32_t qi = cmd->qp_idx;
        if (qi >= qps_.size()) qi = 0;
        ProxyQpHandle& qph = qps_[qi];

        ibv_sge sge{};
        sge.addr = cmd->src_addr;
        sge.length = cmd->length;
        sge.lkey = (qph.lkey_override != 0) ? qph.lkey_override : cmd->lkey;

        ibv_send_wr wr{};
        wr.wr_id = next_slot_;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.send_flags = IBV_SEND_SIGNALED;

        switch (cmd->op) {
          case PROXY_RDMA_WRITE:
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.wr.rdma.remote_addr = cmd->dst_addr;
            wr.wr.rdma.rkey = cmd->rkey;
            break;
          case PROXY_RDMA_WRITE_INLINE:
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.send_flags |= IBV_SEND_INLINE;
            wr.wr.rdma.remote_addr = cmd->dst_addr;
            wr.wr.rdma.rkey = cmd->rkey;
            break;
          case PROXY_ATOMIC_FETCH_ADD:
            wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
            wr.wr.atomic.remote_addr = cmd->dst_addr;
            wr.wr.atomic.rkey = cmd->rkey;
            wr.wr.atomic.compare_add = cmd->atomic_arg;
            break;
          case PROXY_ATOMIC_CMP_SWAP:
            wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
            wr.wr.atomic.remote_addr = cmd->dst_addr;
            wr.wr.atomic.rkey = cmd->rkey;
            wr.wr.atomic.compare_add = cmd->atomic_arg;
            wr.wr.atomic.swap = cmd->atomic_swap;
            break;
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

    // ALWAYS drain CQ — this is critical for freeing SQ slots and completing GPU waits
    for (auto& qph : qps_) {
      DrainCq(qph);
    }
  }
}

}  // namespace core
}  // namespace mori
