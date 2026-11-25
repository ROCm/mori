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
//
//
// cco_proxy_engine.hip.cpp — host proxy engine behind ccoProxy.
//
// Each proxy thread is the only consumer of its share of the KIWI queues and owns
// min(qpsPerPeer, its queue count) RC QPs (own CQ each) per peer. A queue posts on
// one QP slot, so its ops to a peer stay in order. Threads drain queues with
// Context::progress(), which runs one ProxyWorker::On* callback per descriptor.
// The QPs share the window MRs' PD, so kernel-resolved lkeys/rkeys are valid as-is.
//
// The k-th WR on a QP has wr_id k; RC completes in order, so a CQE for k retires
// every WR <= k. Every WR is signaled.
//
// A FLUSH posts its queue's pending WRs and retires in FIFO order (flushDone[q] += 1)
// once that queue's last WR to every peer has completed (GPU side: ccoProxy::flushPush
// in cco_scale_out.hpp).
//
// The first error fails the engine for good (Raise); workers then drain and drop
// (EnterFailed).

// cco headers first: core/utils.hpp (pulled in by the application headers)
// defines a `warpSize` macro that would break cco.hpp's impl::warpSize().
#include "mori/cco/cco_scale_out.hpp"
// clang-format off
#include "cco_proxy_engine.hpp"
#include "hip/hip_runtime.h"
#include "mori/cco/proxy/invoke/invoke.hpp"
#include "mori/application/bootstrap/base_bootstrap.hpp"
#include "mori/application/context/context.hpp"
#include "mori/application/transport/rdma/providers/ibverbs/ibverbs.hpp"
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/cpu_affinity.hpp"
#include "mori/utils/env_utils.hpp"
#include "mori/utils/mori_log.hpp"
// clang-format on

#include <immintrin.h>
#include <infiniband/verbs.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mori {
namespace cco {

namespace {

constexpr uint32_t kSqDepth = 512;  // WRs per QP
constexpr int kPopBatch = 64;       // callbacks run per queue per pass
constexpr int kCqPollBatch = 32;
constexpr size_t kScratchStride = 64;  // per-thread atomic fetch-result slot
constexpr uint32_t kInlineBytes = 8;   // putValue and SignalSet write inline

// First error of an engine, shared by its workers.
struct ProxyErrorState {
  std::atomic<int> state{0};  // 0 none, 1 claiming, 2 published
  ccoDevCommError info{};     // written once, before state = 2
  std::atomic<uint64_t> suppressed{0};
};

ccoDevCommError MakeError(int code, int peer, int queue, int op) {
  ccoDevCommError e{};
  e.code = code;
  e.rank = -1;
  e.peer = peer;
  e.queue = queue;
  e.op = op;
  e.wcStatus = -1;
  return e;
}

const char* OpName(const ccoDevCommError& e) {
  if (e.code == CCO_PROXY_ERR_BAD_REQUEST) {
    switch (e.op) {
      case ccoProxyOpPut:
        return "put";
      case ccoProxyOpPutSignal:
        return "put+signal";
      case ccoProxyOpSignal:
        return "signal";
      case ccoProxyOpFlush:
        return "flush";
      case ccoProxyOpPutValue:
        return "put value";
      case ccoProxyOpPutValueSignal:
        return "put value+signal";
      default:
        return nullptr;
    }
  }
  switch (e.op) {
    case IBV_WR_RDMA_WRITE:
      return "RDMA_WRITE";
    case IBV_WR_ATOMIC_FETCH_AND_ADD:
      return "ATOMIC_FETCH_AND_ADD";
    default:
      return nullptr;
  }
}

// One stderr line: what failed, the known fields, then `tail`.
std::string DescribeError(const ccoDevCommError& e, const char* tail) {
  std::string fields;
  char buf[160];
  auto add = [&](const char* part) {
    fields += fields.empty() ? "" : ", ";
    fields += part;
  };
  if (e.peer >= 0) {
    std::snprintf(buf, sizeof(buf), "peer %d", e.peer);
    add(buf);
  }
  if (e.queue >= 0) {
    std::snprintf(buf, sizeof(buf), "queue %d", e.queue);
    add(buf);
  }
  if (const char* name = OpName(e)) {
    std::snprintf(buf, sizeof(buf), "op %s", name);
    add(buf);
  } else if (e.op >= 0) {
    std::snprintf(buf, sizeof(buf), "op %d", e.op);
    add(buf);
  }
  if (e.raddr != 0 || e.rkey != 0 || e.len != 0) {
    std::snprintf(buf, sizeof(buf), "raddr 0x%llx rkey 0x%x len %u",
                  static_cast<unsigned long long>(e.raddr), e.rkey, e.len);
    add(buf);
  }
  if (e.code == CCO_PROXY_ERR_COMPLETION) {
    std::snprintf(buf, sizeof(buf), "wc %s vendor_err 0x%x wr_id %llu",
                  ibv_wc_status_str(static_cast<ibv_wc_status>(e.wcStatus)), e.vendorErr,
                  static_cast<unsigned long long>(e.wrId));
    add(buf);
  }
  if (e.sysErrno != 0) {
    std::snprintf(buf, sizeof(buf), "errno %d", e.sysErrno);
    add(buf);
  }
  std::string line = "[mori cco] ERROR: ccoProxy rank " + std::to_string(e.rank) + ": " + e.message;
  if (!fields.empty()) line += " (" + fields + ")";
  return line + "; " + tail;
}

// CPUs for the busy-polling proxy threads: MORI_CCO_PROXY_CPUS (thread t on entry
// t % size), else the GPU-local CPUs in the affinity mask (or the whole mask), one
// physical core per thread from the highest, skipping CPU 0's core, then wrapping
// onto SMT siblings.
std::vector<int> ChooseProxyCpus(int numThreads, std::string* source) {
  std::vector<int> order;
  if (const char* env = std::getenv("MORI_CCO_PROXY_CPUS"); env != nullptr && env[0] != '\0') {
    order = ::mori::ParseCpuList(env).value_or(std::vector<int>{});
    if (order.empty()) {
      std::fprintf(stderr,
                   "[mori cco] WARNING: ccoProxy: MORI_CCO_PROXY_CPUS=\"%s\" names no CPUs; "
                   "using the default choice\n",
                   env);
    } else {
      *source = "MORI_CCO_PROXY_CPUS";
    }
  }
  if (order.empty()) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    std::vector<int> allowed;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
      for (int c = 0; c < CPU_SETSIZE; c++) {
        if (CPU_ISSET(c, &mask)) allowed.push_back(c);
      }
    }
    std::vector<int> candidates;
    int dev = -1, numaNode = -1;
    if (hipGetDevice(&dev) == hipSuccess) {
      if (auto list = application::detail::GpuLocalCpuList(dev, numaNode)) {
        for (int c : ::mori::ParseCpuList(*list).value_or(std::vector<int>{})) {
          if (std::find(allowed.begin(), allowed.end(), c) != allowed.end()) {
            candidates.push_back(c);
          }
        }
      }
    }
    *source = candidates.empty() ? "affinity mask" : "GPU-local cores in the affinity mask";
    if (candidates.empty()) candidates = allowed;

    std::vector<std::vector<int>> cores = application::detail::GroupBySiblings(candidates);
    if (cores.size() > 1) {
      cores.erase(std::remove_if(cores.begin(), cores.end(),
                                 [](const std::vector<int>& core) {
                                   return std::find(core.begin(), core.end(), 0) != core.end();
                                 }),
                  cores.end());
    }
    std::reverse(cores.begin(), cores.end());
    for (size_t sibling = 0;; sibling++) {
      bool any = false;
      for (const auto& core : cores) {
        if (sibling < core.size()) {
          order.push_back(core[sibling]);
          any = true;
        }
      }
      if (!any) break;
    }
  }
  std::vector<int> cpus(numThreads, -1);
  if (!order.empty()) {
    for (int t = 0; t < numThreads; t++) cpus[t] = order[t % order.size()];
  }
  return cpus;
}

void PinCallingThread(int worker, int cpu) {
  if (cpu < 0) return;
  cpu_set_t target;
  CPU_ZERO(&target);
  CPU_SET(cpu, &target);
  if (sched_setaffinity(0, sizeof(target), &target) != 0) {
    std::fprintf(stderr,
                 "[mori cco] WARNING: ccoProxy worker %d: cannot pin to CPU %d: %s; running "
                 "unpinned\n",
                 worker, cpu, strerror(errno));
  }
}

void HipOrThrow(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
  }
}

// Host side of one KIWI invoke endpoint (one queue), independent of its
// producer type.
class HostQueue {
 public:
  virtual ~HostQueue() = default;
  virtual bool Ready() = 0;              // hint: the next message has started to arrive
  virtual size_t Progress(int max) = 0;  // runs up to `max` callbacks; returns how many
};

template <typename Q>
class HostQueueImpl final : public HostQueue {
 public:
  HostQueueImpl(mori::cco::proxy::invoke::Context<Q>& ctx, size_t endpoint)
      : ctx_(ctx), endpoint_(endpoint), poll_(ctx.raw_queue(endpoint)->host_poll_state()) {}
  bool Ready() override { return poll_.ready(); }
  size_t Progress(int max) override {
    size_t n = 0;
    try {
      n = ctx_.progress(endpoint_, max);
    } catch (...) {
      poll_.refresh();  // the failing message was popped; Ready() must not stay stale
      throw;
    }
    if (n > 0) poll_.refresh();
    return n;
  }

 private:
  mori::cco::proxy::invoke::Context<Q>& ctx_;
  size_t endpoint_;
  typename Q::HostPollState poll_;
};

struct PendingFlush {
  std::vector<std::pair<int, uint64_t>> waits;  // (index in qps, WR seq) that must complete
};

struct OwnedQueue {
  std::unique_ptr<HostQueue> queue;
  int index{0};                           // in flushDone (SPSC pool, then MPSC pool)
  int qpSlot{0};                          // which of its thread's QPs per peer it posts on
  volatile uint64_t* flushDone{nullptr};  // fine-grained device memory, host-written
  uint64_t flushed{0};                    // FLUSHes retired (*flushDone without the failed bit)
  std::vector<uint64_t> lastSeq;          // [worldSize] seq of the last WR on its QP to the peer
  std::deque<PendingFlush> pending;
  bool broken{false};  // corrupted ring: never read again
};

// What a WR targeted, kept for error reports.
struct WrShadow {
  uint64_t raddr;
  uint32_t rkey;
  uint32_t len;
  int queue;
  int opcode;  // ibv_wr_opcode
};

struct ProxyQp {
  application::RdmaEndpoint ep;
  bool active{false};
  int peer{-1};
  uint64_t postedSeq{0};     // WRs posted so far (wr_id of the last one)
  uint64_t completedSeq{0};  // every WR <= this has completed
  // WRs built during the current pass, posted as one chain at its end. Their
  // seqs are assigned when built: postedSeq + index + 1.
  std::vector<ibv_send_wr> wrs;
  std::vector<ibv_sge> sges;
  // Inline payload of wrs[i] (IBV_SEND_INLINE); copied by ibv_post_send, so it
  // only lives until the chain is posted.
  std::vector<uint64_t> inl;
  // [kSqDepth] by seq % kSqDepth; Reserve keeps every tracked seq in one window.
  std::vector<WrShadow> shadow;
};

class ProxyWorker {
 public:
  ProxyWorker(int id, int worldSize, int qpsPerPeer, int signalCount, int rank,
              ProxyErrorState* err)
      : qps(static_cast<size_t>(worldSize) * qpsPerPeer),
        worldSize_(worldSize),
        qpsPerPeer_(qpsPerPeer),
        signalCount_(signalCount),
        id_(id),
        rank_(rank),
        err_(err) {}

  // QP `slot` of this thread to `peer`.
  ProxyQp& Qp(int peer, int slot) { return qps[static_cast<size_t>(peer) * qpsPerPeer_ + slot]; }

  // Binds the next owned queue to a QP slot, round robin.
  void AddQueue(OwnedQueue* oq) {
    oq->qpSlot = static_cast<int>(queues.size() % qpsPerPeer_);
    queues.push_back(oq);
  }

  void Start(int cpu, const std::atomic<bool>* stop) {
    stop_ = stop;
    for (int peer = 0; peer < worldSize_; peer++) {
      if (!Qp(peer, 0).active) continue;
      activePeers_.push_back(peer);
      for (int slot = 0; slot < qpsPerPeer_; slot++) {
        ProxyQp& qp = Qp(peer, slot);
        activeQps_.push_back(static_cast<int>(&qp - qps.data()));
        qp.peer = peer;
        qp.wrs.reserve(kSqDepth);
        qp.sges.reserve(kSqDepth);
        qp.inl.reserve(kSqDepth);
        qp.shadow.resize(kSqDepth);
      }
    }
    thread_ = std::thread([this, cpu] {
      PinCallingThread(id_, cpu);
      current_ = this;
      try {
        Run();
      } catch (const std::exception& e) {
        Raise(MakeError(CCO_PROXY_ERR_INTERNAL, -1, -1, -1),
              std::string("proxy thread failed: ") + e.what());
      } catch (...) {
        Raise(MakeError(CCO_PROXY_ERR_INTERNAL, -1, -1, -1), "proxy thread failed");
      }
      // A kernel still flushing (e.g. during destroy) traps instead of spinning.
      Poison();
    });
  }

  void Join() {
    if (thread_.joinable()) thread_.join();
  }

  // Registers the ccoProxyOp callbacks on a pool's invoke context, in op order
  // so the callback ids are the ops.
  template <typename Q>
  static void RegisterCallbacks(mori::cco::proxy::invoke::Context<Q>& ctx) {
    Register(ctx, ccoProxyOpPut, &ProxyWorker::OnPut);
    Register(ctx, ccoProxyOpPutSignal, &ProxyWorker::OnPutSignal);
    Register(ctx, ccoProxyOpSignal, &ProxyWorker::OnSignal);
    Register(ctx, ccoProxyOpFlush, &ProxyWorker::OnFlush);
    Register(ctx, ccoProxyOpPutValue, &ProxyWorker::OnPutValue);
    Register(ctx, ccoProxyOpPutValueSignal, &ProxyWorker::OnPutValueSignal);
    static_assert(ccoProxyOpCount == 6, "register every ccoProxyOp");
  }

  std::vector<ProxyQp> qps;  // [worldSize * qpsPerPeer] by (peer, slot); active for masked peers
  std::vector<OwnedQueue*> queues;  // queues this thread consumes (AddQueue)
  uint64_t scratchAddr{0};          // 8-byte host slot for atomic fetch results (discarded)
  uint32_t scratchLkey{0};

 private:
  // A callback runs on the proxy thread that called progress(), for the queue
  // in its cur_.
  template <typename Q, typename Desc>
  static void Register(mori::cco::proxy::invoke::Context<Q>& ctx, ccoProxyOp op,
                       void (ProxyWorker::*fn)(const Desc&)) {
    const uint8_t id =
        ctx.register_callback(std::function<void(Desc)>([fn](Desc d) { (current_->*fn)(d); }));
    if (id != op) throw std::runtime_error("ccoProxy callback id mismatch");
  }

  template <typename Q>
  static void Register(mori::cco::proxy::invoke::Context<Q>& ctx, ccoProxyOp op,
                       void (ProxyWorker::*fn)()) {
    const uint8_t id = ctx.register_callback(std::function<void()>([fn] { (current_->*fn)(); }));
    if (id != op) throw std::runtime_error("ccoProxy callback id mismatch");
  }

  void Run() {
    for (;;) {
      if (!failed_ && ErrorPublished()) EnterFailed();
      size_t popped = 0;
      for (OwnedQueue* oq : queues) {
        if (oq->broken || !oq->queue->Ready()) continue;
        cur_ = oq;
        popped += Progress(*oq);
      }
      bool inFlight = false;
      bool flushPending = false;
      if (!failed_) {
        PostAndPollAll(&inFlight);
        for (OwnedQueue* oq : queues) flushPending = flushPending || !oq->pending.empty();
      }
      if (popped == 0) {
        // Stop only once drained: nothing waiting and, unless failed, every WR
        // completed and every FLUSH answered (a failed QP may never complete).
        if (stop_->load(std::memory_order_acquire) && !AnyReady() &&
            (failed_ || (!inFlight && !flushPending))) {
          break;
        }
        _mm_pause();
      }
    }
  }

  bool AnyReady() {
    for (OwnedQueue* oq : queues) {
      if (!oq->broken && oq->queue->Ready()) return true;
    }
    return false;
  }

  // Runs up to kPopBatch of oq's callbacks; nonzero if it made progress.
  size_t Progress(OwnedQueue& oq) {
    try {
      return oq.queue->Progress(kPopBatch);
    } catch (const mori::cco::proxy::queue::queue_error& e) {
      oq.broken = true;
      Raise(MakeError(CCO_PROXY_ERR_INTERNAL, -1, oq.index, -1),
            std::string("corrupted queue: ") + e.what());
    } catch (const std::bad_alloc&) {
      Raise(MakeError(CCO_PROXY_ERR_INTERNAL, -1, oq.index, -1), "out of host memory");
    } catch (const std::exception& e) {
      // KIWI invoke: unknown callback id or argument size; the message is popped.
      Raise(MakeError(CCO_PROXY_ERR_BAD_REQUEST, -1, oq.index, -1),
            std::string("bad descriptor: ") + e.what());
    }
    return 1;
  }

  // Records the engine's first error (later ones only count as suppressed),
  // prints it, and switches this worker to drain-and-drop.
  void Raise(ccoDevCommError e, const std::string& what) {
    e.rank = rank_;
    std::snprintf(e.message, sizeof(e.message), "%s", what.c_str());
    int expected = 0;
    if (err_->state.compare_exchange_strong(expected, 1, std::memory_order_relaxed)) {
      err_->info = e;
      err_->state.store(2, std::memory_order_release);
      const bool destroying = stop_ != nullptr && stop_->load(std::memory_order_acquire);
      std::fprintf(
          stderr, "%s\n",
          DescribeError(e, destroying
                               ? "while destroying the DevComm: a peer that destroyed first "
                                 "drops the work requests still in flight to it"
                               : "DevComm failed: flush() / waitSignal() now trap and abort the "
                                 "process, their timeout overloads return 2; peers are not told. "
                                 "To recover, every rank destroys and recreates the DevComm")
              .c_str());
    } else {
      err_->suppressed.fetch_add(1, std::memory_order_relaxed);
      while (!ErrorPublished()) _mm_pause();  // poison only once GetError can report it
    }
    EnterFailed();
  }

  // True once ccoProxyEngineGetError can report the engine's first error.
  bool ErrorPublished() const { return err_->state.load(std::memory_order_acquire) == 2; }

  // Drain-and-drop from now on: keep popping so producers never wedge, post
  // nothing, and answer every FLUSH, pending or later, with failure.
  void EnterFailed() {
    if (failed_) return;
    for (OwnedQueue* oq : queues) RetireFlushes(*oq);  // satisfied FLUSHes still succeed
    failed_ = true;
    for (int i : activeQps_) {
      qps[i].wrs.clear();
      qps[i].sges.clear();
      qps[i].inl.clear();
    }
    for (OwnedQueue* oq : queues) oq->pending.clear();
    Poison();
  }

  // Sets ccoProxyFailedBit in every owned queue's flushDone.
  void Poison() { PublishFlushDone(queues, ccoProxyFailedBit); }

  // Release-stores flushed | bits into each queue's flushDone (this worker is its
  // only writer), then one sfence for the batch.
  template <typename Queues>
  static void PublishFlushDone(const Queues& qs, uint64_t bits) {
    for (OwnedQueue* oq : qs) {
      __atomic_store_n(oq->flushDone, oq->flushed | bits, __ATOMIC_RELEASE);
    }
    _mm_sfence();  // drain the write-combining buffer toward the device
  }

  // The current queue's QP to `peer`. Null, after raising BAD_REQUEST, if the
  // peer has no proxy QP.
  ProxyQp* PeerQp(uint32_t peer, ccoProxyOp op) {
    if (peer < static_cast<uint32_t>(worldSize_)) {
      ProxyQp& qp = Qp(static_cast<int>(peer), cur_->qpSlot);
      if (qp.active) return &qp;
    }
    const int p =
        peer > static_cast<uint32_t>(std::numeric_limits<int>::max()) ? -1 : static_cast<int>(peer);
    Raise(MakeError(CCO_PROXY_ERR_BAD_REQUEST, p, cur_->index, op),
          "descriptor for peer " + std::to_string(peer) +
              " without a proxy QP (check proxyConnectionType)");
    return nullptr;
  }

  // A signal targets an 8-byte aligned word and, for a slot, one below
  // gdaSignalCount: past it lie the peer's consumed counts, counters and barrier
  // state, which the NIC would write without complaint. Raises BAD_REQUEST otherwise.
  bool SignalValid(const ccoProxySignalDesc& d, ccoProxyOp op) {
    const bool aligned = (d.raddr & 7) == 0;
    const uint64_t slot = d.raddr / sizeof(uint64_t);
    if (aligned && (d.slot == 0 || slot < static_cast<uint64_t>(signalCount_))) return true;
    ccoDevCommError e =
        MakeError(CCO_PROXY_ERR_BAD_REQUEST, static_cast<int>(d.peer), cur_->index, op);
    e.raddr = d.raddr;
    e.rkey = d.rkey;
    e.len = sizeof(uint64_t);
    Raise(e, aligned ? "signal slot " + std::to_string(slot) + " out of range (gdaSignalCount " +
                           std::to_string(signalCount_) + ")"
                     : std::string("signal address not 8-byte aligned"));
    return false;
  }

  void OnPut(const ccoProxyPutDesc& d) {
    if (failed_) return;
    ProxyQp* qp = PeerQp(d.peer, ccoProxyOpPut);
    if (qp == nullptr || !Reserve(*qp, 1)) return;
    cur_->lastSeq[d.peer] = AddWrite(*qp, d);
  }

  void OnPutSignal(const ccoProxyPutSignalDesc& d) {
    if (failed_) return;
    ProxyQp* qp = PeerQp(d.put.peer, ccoProxyOpPutSignal);
    if (qp == nullptr || !SignalValid(d.signal, ccoProxyOpPutSignal) || !Reserve(*qp, 2)) {
      return;
    }
    AddWrite(*qp, d.put);
    cur_->lastSeq[d.put.peer] = AddSignal(*qp, d.signal);
  }

  void OnSignal(const ccoProxySignalDesc& d) {
    if (failed_) return;
    ProxyQp* qp = PeerQp(d.peer, ccoProxyOpSignal);
    if (qp == nullptr || !SignalValid(d, ccoProxyOpSignal) || !Reserve(*qp, 1)) return;
    cur_->lastSeq[d.peer] = AddSignal(*qp, d);
  }

  void OnPutValue(const ccoProxyPutValueDesc& d) {
    if (failed_) return;
    ProxyQp* qp = PeerQp(d.peer, ccoProxyOpPutValue);
    if (qp == nullptr || !Reserve(*qp, 1)) return;
    cur_->lastSeq[d.peer] = AddInline(*qp, d.raddr, d.rkey, d.value, d.bytes);
  }

  void OnPutValueSignal(const ccoProxyPutValueSignalDesc& d) {
    if (failed_) return;
    ProxyQp* qp = PeerQp(d.put.peer, ccoProxyOpPutValueSignal);
    if (qp == nullptr || !SignalValid(d.signal, ccoProxyOpPutValueSignal) || !Reserve(*qp, 2)) {
      return;
    }
    AddInline(*qp, d.put.raddr, d.put.rkey, d.put.value, d.put.bytes);
    cur_->lastSeq[d.put.peer] = AddSignal(*qp, d.signal);
  }

  void OnFlush() {
    if (failed_) return;  // the poisoned flushDone answers it
    OwnedQueue& oq = *cur_;
    PendingFlush f;
    for (int p : activePeers_) {
      ProxyQp& qp = Qp(p, oq.qpSlot);
      PostChain(qp);
      if (failed_) return;
      if (oq.lastSeq[p] > qp.completedSeq) {
        f.waits.emplace_back(static_cast<int>(&qp - qps.data()), oq.lastSeq[p]);
      }
    }
    oq.pending.push_back(std::move(f));
    RetireFlushes(oq);
  }

  // Retire satisfied flushes in FIFO order and publish the new count. Never
  // once failed: that would clear the failed bit.
  void RetireFlushes(OwnedQueue& oq) {
    if (failed_) return;
    uint64_t retired = 0;
    while (!oq.pending.empty()) {
      const PendingFlush& f = oq.pending.front();
      const bool satisfied = std::all_of(f.waits.begin(), f.waits.end(), [&](const auto& w) {
        return qps[w.first].completedSeq >= w.second;
      });
      if (!satisfied) break;
      retired++;
      oq.pending.pop_front();
    }
    if (retired != 0) {
      oq.flushed += retired;
      PublishFlushDone(std::initializer_list<OwnedQueue*>{&oq}, 0);
    }
  }

  // Make room for n more WRs in qp's SQ (posted + pending chain). While it is
  // full, posts every QP's chain and polls every CQ, so other QPs' work and
  // flushes keep moving. False if the DevComm failed meanwhile.
  bool Reserve(ProxyQp& qp, size_t n) {
    while (qp.postedSeq + qp.wrs.size() + n - qp.completedSeq > kSqDepth) {
      if (ErrorPublished()) EnterFailed();
      if (failed_) return false;
      PostAndPollAll();
      if (failed_) return false;
    }
    return true;
  }

  // Posts every QP's chain, polls the CQs with WRs outstanding and, if any
  // completed, retires flushes. Sets *inFlight if a polled QP still has WRs
  // outstanding. Once failed, chains are empty and nothing is polled.
  void PostAndPollAll(bool* inFlight = nullptr) {
    bool completed = false;
    for (int i : activeQps_) {
      ProxyQp& qp = qps[i];
      PostChain(qp);
      if (!failed_ && qp.completedSeq < qp.postedSeq) {
        completed = PollCq(qp) || completed;
        if (inFlight != nullptr) *inFlight = *inFlight || qp.completedSeq < qp.postedSeq;
      }
    }
    if (completed) {
      for (OwnedQueue* oq : queues) RetireFlushes(*oq);
    }
  }

  // Appends a WR to qp's pending chain and records its target; returns its
  // sequence number. `inl` is the payload of an IBV_SEND_INLINE WR.
  uint64_t Append(ProxyQp& qp, const ibv_send_wr& wr, const ibv_sge& sge, uint64_t raddr,
                  uint32_t rkey, uint32_t len, uint64_t inl = 0) {
    qp.wrs.push_back(wr);
    qp.sges.push_back(sge);
    qp.inl.push_back(inl);
    const uint64_t seq = qp.postedSeq + qp.wrs.size();
    qp.shadow[seq % kSqDepth] =
        WrShadow{raddr, rkey, len, cur_->index, static_cast<int>(wr.opcode)};
    return seq;
  }

  // Error for WR `seq` of qp, with its target if the seq is still tracked.
  static ccoDevCommError WrError(int code, const ProxyQp& qp, uint64_t seq) {
    ccoDevCommError e = MakeError(code, qp.peer, -1, -1);
    e.wrId = seq;
    if (seq > qp.completedSeq && seq <= qp.postedSeq + qp.wrs.size()) {
      const WrShadow& w = qp.shadow[seq % kSqDepth];
      e.queue = w.queue;
      e.op = w.opcode;
      e.raddr = w.raddr;
      e.rkey = w.rkey;
      e.len = w.len;
    }
    return e;
  }

  uint64_t AddWrite(ProxyQp& qp, const ccoProxyPutDesc& d) {
    ibv_send_wr wr{};
    ibv_sge sge{};
    sge.addr = d.laddr;  // MR address (window offset + iovaBias)
    sge.length = d.bytes;
    sge.lkey = d.lkey;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.num_sge = d.bytes ? 1 : 0;
    wr.wr.rdma.remote_addr = d.raddr;
    wr.wr.rdma.rkey = d.rkey;
    return Append(qp, wr, sge, d.raddr, d.rkey, d.bytes);
  }

  // RDMA write of the low `bytes` bytes of `value`, inline (no local MR).
  uint64_t AddInline(ProxyQp& qp, uint64_t raddr, uint32_t rkey, uint64_t value, uint32_t bytes) {
    ibv_send_wr wr{};
    ibv_sge sge{};
    sge.length = bytes;  // addr: the chain's inline slot, set in PostChain
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_INLINE;
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey = rkey;
    return Append(qp, wr, sge, raddr, rkey, bytes, value);
  }

  // A signal: an 8-byte inline write (set) or an atomic fetch-add.
  uint64_t AddSignal(ProxyQp& qp, const ccoProxySignalDesc& d) {
    if (d.set) return AddInline(qp, d.raddr, d.rkey, d.value, sizeof(uint64_t));
    return AddAtomic(qp, d);
  }

  uint64_t AddAtomic(ProxyQp& qp, const ccoProxySignalDesc& d) {
    ibv_send_wr wr{};
    ibv_sge sge{};
    sge.addr = scratchAddr;
    sge.length = sizeof(uint64_t);
    sge.lkey = scratchLkey;
    wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
    wr.num_sge = 1;
    wr.wr.atomic.remote_addr = d.raddr;
    wr.wr.atomic.rkey = d.rkey;
    wr.wr.atomic.compare_add = d.value;
    return Append(qp, wr, sge, d.raddr, d.rkey, sizeof(uint64_t));
  }

  // Posts the pending chain with one ibv_post_send.
  void PostChain(ProxyQp& qp) {
    const size_t n = qp.wrs.size();
    if (n == 0) return;
    for (size_t i = 0; i < n; i++) {
      ibv_send_wr& wr = qp.wrs[i];
      wr.wr_id = qp.postedSeq + i + 1;
      if (wr.send_flags & IBV_SEND_INLINE) {
        qp.sges[i].addr = reinterpret_cast<uintptr_t>(&qp.inl[i]);
      }
      wr.sg_list = wr.num_sge ? &qp.sges[i] : nullptr;
      wr.next = (i + 1 < n) ? &qp.wrs[i + 1] : nullptr;
      wr.send_flags |= IBV_SEND_SIGNALED;
    }
    ibv_send_wr* bad = nullptr;
    const int ret = ibv_post_send(qp.ep.ibvHandle.qp, qp.wrs.data(), &bad);
    if (ret != 0) {
      // The WRs before `bad` were posted.
      if (bad >= qp.wrs.data() && bad < qp.wrs.data() + n) qp.postedSeq += bad - qp.wrs.data();
      ccoDevCommError e = WrError(CCO_PROXY_ERR_POST, qp, qp.postedSeq + 1);
      e.sysErrno = ret;
      Raise(e, std::string("ibv_post_send failed: ") + strerror(ret));  // drops the chain
      return;
    }
    qp.postedSeq += n;
    qp.wrs.clear();
    qp.sges.clear();
    qp.inl.clear();
  }

  // Returns true if any CQE was polled. An error CQE raises COMPLETION.
  bool PollCq(ProxyQp& qp) {
    ibv_wc wc[kCqPollBatch];
    const int n = ibv_poll_cq(qp.ep.ibvHandle.cq, kCqPollBatch, wc);
    if (n < 0) {
      ccoDevCommError e = MakeError(CCO_PROXY_ERR_INTERNAL, qp.peer, -1, -1);
      e.sysErrno = -n;
      Raise(e, "ibv_poll_cq failed");
      return false;
    }
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        ccoDevCommError e = WrError(CCO_PROXY_ERR_COMPLETION, qp, wc[i].wr_id);
        e.wcStatus = wc[i].status;
        e.vendorErr = wc[i].vendor_err;
        Raise(e, "work request failed");
        return true;
      }
      if (wc[i].wr_id > qp.completedSeq) qp.completedSeq = wc[i].wr_id;
    }
    return n > 0;
  }

  static inline thread_local ProxyWorker* current_ = nullptr;  // set on each proxy thread

  int worldSize_;
  int qpsPerPeer_;
  int signalCount_;  // user signal slots (gdaSignalCount), the same on every rank
  int id_;
  int rank_;
  ProxyErrorState* err_;                    // the engine's
  const std::atomic<bool>* stop_{nullptr};  // the engine's; set while destroying
  bool failed_{false};                      // drain-and-drop
  OwnedQueue* cur_{nullptr};                // queue whose callbacks are running
  std::vector<int> activePeers_;            // peers with QPs
  std::vector<int> activeQps_;              // indices in qps of every active QP
  std::thread thread_;
};

// Per-rank setup result, Allgather'd at each agreement point.
struct ProxyCreateInfo {
  int32_t ok;  // this rank's step succeeded
  int32_t pad;
  uint64_t maxMsgBytes;  // the port's max_msg_sz
};

// Allgathers every rank's result. If *error is still empty, sets it to the first
// rank that failed, so all ranks fail or go on together.
std::vector<ProxyCreateInfo> AgreeOk(application::BootstrapNetwork* bootNet, ProxyCreateInfo mine,
                                     const char* what, std::string* error) {
  const int worldSize = bootNet->GetWorldSize();
  std::vector<ProxyCreateInfo> all(worldSize);
  bootNet->Allgather(&mine, all.data(), sizeof(ProxyCreateInfo));
  for (int r = 0; r < worldSize && error->empty(); r++) {
    if (!all[r].ok) *error = "rank " + std::to_string(r) + " " + what;
  }
  return all;
}

}  // namespace

class ccoProxyEngine {
 public:
  ~ccoProxyEngine() {
    StopAndJoin();
    if (ibv) {
      for (auto& w : workers) {
        for (auto& qp : w->qps) {
          if (qp.active) ibv->DestroyRdmaEndpointNoThrow(qp.ep);
        }
      }
    }
    workers.clear();
    queues.clear();
    spsc.reset();  // frees the KIWI queues
    mpsc.reset();
    ibv.reset();  // deregisters the scratch MR; leaves the borrowed PD alone
    free(scratch);
    if (flushDone) (void)hipFree(flushDone);
    if (flushTicket) (void)hipFree(flushTicket);
  }

  void StopAndJoin() {
    stop.store(true, std::memory_order_release);
    for (auto& w : workers) w->Join();
  }

  std::unique_ptr<application::IBVerbsDeviceContext> ibv;
  std::unique_ptr<mori::cco::proxy::invoke::Context<mori::cco::proxy::queue::SPSCD2HQueue>>
      spsc;  // SPSC pool
  std::unique_ptr<mori::cco::proxy::invoke::Context<mori::cco::proxy::queue::MPSCD2HQueue>>
      mpsc;                                         // MPSC pool
  std::vector<std::unique_ptr<OwnedQueue>> queues;  // SPSC pool, then MPSC pool
  std::vector<std::unique_ptr<ProxyWorker>> workers;
  ProxyErrorState err;
  int rank{0};
  void* scratch{nullptr};
  uint64_t* flushDone{nullptr};    // [numSpsc + numMpsc]
  uint64_t* flushTicket{nullptr};  // [numSpsc + numMpsc]
  std::atomic<bool> stop{false};
};

ccoProxyEngine* ccoProxyEngineCreate(application::Context* ctx,
                                     application::BootstrapNetwork* bootNet,
                                     const ccoProxyEngineConfig& config, ccoProxyContext* out) {
  auto engine = std::make_unique<ccoProxyEngine>();
  const int worldSize = bootNet->GetWorldSize();
  const int rank = bootNet->GetLocalRank();
  engine->rank = rank;
  // Logs and stops any started threads (they have posted nothing); the engine
  // destructor releases the rest.
  auto fail = [&](const std::string& why) -> ccoProxyEngine* {
    MORI_SHMEM_ERROR("ccoProxyEngineCreate: rank={} failed: {}", rank, why);
    engine->StopAndJoin();
    return nullptr;
  };
  const int numSpsc = config.numSpscQueues;
  const int numMpsc = config.numMpscQueues;
  // Summed in 64 bits and validated below before any use as an int.
  const int64_t numQueues64 = int64_t{numSpsc} + int64_t{numMpsc};
  const int numQueues =
      static_cast<int>(std::clamp<int64_t>(numQueues64, 0, std::numeric_limits<int>::max()));
  // Never more threads than queues: a thread without a queue would only spin.
  const int numThreads =
      std::min(config.numThreads > 0 ? config.numThreads : kDefaultProxyThreads, numQueues);
  // QPs per peer of thread t: threadQps[t] = min(config.qpsPerPeer, t's queue
  // count), so no QP lacks a queue. Handle (peer, t, slot) sits at
  // local[peer * qpsPerRank + qpBase[t] + slot]; every rank computes the same
  // layout from the agreed counts.
  std::vector<int> threadQps;
  std::vector<int> qpBase;
  int qpsPerRank = 0;  // QPs to one peer, summed over the threads

  // Local setup, up to the QP handle exchange.
  ProxyCreateInfo mine{};
  std::string error;
  std::vector<application::RdmaEndpointHandle> local;
  try {
    application::RdmaDeviceContext* devCtx = ctx->GetRdmaDeviceContext();
    if (devCtx == nullptr) throw std::runtime_error("no RDMA device context");
    if (numSpsc < 0 || numMpsc < 0 || numQueues64 <= 0 ||
        numQueues64 > std::numeric_limits<int>::max()) {
      throw std::runtime_error("bad queue counts: " + std::to_string(numSpsc) + " SPSC + " +
                               std::to_string(numMpsc) + " MPSC");
    }
    if (static_cast<int>(config.peerMask.size()) != worldSize) {
      throw std::runtime_error("peerMask size != worldSize");
    }
    if (config.qpsPerPeer < 1) {
      throw std::runtime_error("bad qpsPerPeer " + std::to_string(config.qpsPerPeer));
    }
    for (int t = 0; t < numThreads; t++) {
      const int owned = numQueues / numThreads + (t < numQueues % numThreads ? 1 : 0);
      threadQps.push_back(std::min(config.qpsPerPeer, owned));
      qpBase.push_back(qpsPerRank);
      qpsPerRank += threadQps.back();
    }
    local.resize(static_cast<size_t>(worldSize) * qpsPerRank);
    const size_t capacity =
        config.queueCapacity ? config.queueCapacity : kDefaultProxyQueueCapacity;
    if ((capacity & (capacity - 1)) != 0 ||
        capacity < mori::cco::proxy::queue::message_lines(sizeof(ccoProxyPutSignalDesc))) {
      throw std::runtime_error("queue capacity must be a power of two >= the largest descriptor");
    }
    int dev = -1, largeBar = 0;
    HipOrThrow(hipGetDevice(&dev), "hipGetDevice");
    HipOrThrow(hipDeviceGetAttribute(&largeBar, hipDeviceAttributeIsLargeBar, dev),
               "hipDeviceGetAttribute(IsLargeBar)");
    if (!largeBar) {
      throw std::runtime_error("GPU " + std::to_string(dev) +
                               " has no host-mapped VRAM (large BAR), which the proxy threads "
                               "need to write the flush counters");
    }

    // Host-posted ibverbs QPs on the PD that owns the window MRs (borrowed).
    // They never receive, so skip the receive buffers mori's shmem proxy uses.
    engine->ibv = std::make_unique<application::IBVerbsDeviceContext>(
        devCtx->GetRdmaDevice(), devCtx->GetIbvPd(), /*ownsPd=*/false);
    engine->ibv->SetProxyRecvBuffersEnabled(false);
    application::RdmaEndpointConfig epConfig = ctx->GetRdmaEndpointConfig();  // port, GID index
    epConfig.maxMsgsNum = kSqDepth;
    epConfig.maxCqeNum = kSqDepth;
    epConfig.maxRecvWr = 1;
    epConfig.maxInlineData = kInlineBytes;
    epConfig.onGpu = false;
    epConfig.withCompChannel = false;
    epConfig.enableSrq = false;
    epConfig.atomicIbufSlots = 0;
    const ibv_port_attr* portAttr = devCtx->GetRdmaDevice()->GetPortAttr(epConfig.portId);
    if (portAttr == nullptr || portAttr->max_msg_sz == 0) {
      throw std::runtime_error("port " + std::to_string(epConfig.portId) + " max_msg_sz unknown");
    }
    mine.maxMsgBytes = portAttr->max_msg_sz;

    if (posix_memalign(&engine->scratch, 4096, kScratchStride * numThreads) != 0) {
      throw std::runtime_error("scratch allocation failed");
    }
    std::memset(engine->scratch, 0, kScratchStride * numThreads);
    application::RdmaMemoryRegion scratchMr =
        engine->ibv->RegisterRdmaMemoryRegion(engine->scratch, kScratchStride * numThreads);

    for (int t = 0; t < numThreads; t++) {
      auto w = std::make_unique<ProxyWorker>(t, worldSize, threadQps[t], config.signalCount, rank,
                                             &engine->err);
      w->scratchAddr = reinterpret_cast<uint64_t>(engine->scratch) + t * kScratchStride;
      w->scratchLkey = scratchMr.lkey;
      engine->workers.push_back(std::move(w));
    }

    HipOrThrow(hipExtMallocWithFlags(reinterpret_cast<void**>(&engine->flushDone),
                                     sizeof(uint64_t) * numQueues, hipDeviceMallocFinegrained),
               "hipExtMallocWithFlags");
    HipOrThrow(hipMemset(engine->flushDone, 0, sizeof(uint64_t) * numQueues), "hipMemset");
    HipOrThrow(
        hipMalloc(reinterpret_cast<void**>(&engine->flushTicket), sizeof(uint64_t) * numQueues),
        "hipMalloc");
    HipOrThrow(hipMemset(engine->flushTicket, 0, sizeof(uint64_t) * numQueues), "hipMemset");
    HipOrThrow(hipDeviceSynchronize(), "hipDeviceSynchronize");

    // One invoke context per pool. Queues: SPSC pool [0, numSpsc), MPSC pool
    // [numSpsc, numQueues); queue g is consumed by thread g % numThreads.
    if (numSpsc > 0) {
      engine->spsc = std::make_unique<
          mori::cco::proxy::invoke::Context<mori::cco::proxy::queue::SPSCD2HQueue>>(numSpsc,
                                                                                    capacity);
      ProxyWorker::RegisterCallbacks(*engine->spsc);
    }
    if (numMpsc > 0) {
      engine->mpsc = std::make_unique<
          mori::cco::proxy::invoke::Context<mori::cco::proxy::queue::MPSCD2HQueue>>(numMpsc,
                                                                                    capacity);
      ProxyWorker::RegisterCallbacks(*engine->mpsc);
    }
    for (int g = 0; g < numQueues; g++) {
      auto oq = std::make_unique<OwnedQueue>();
      oq->index = g;
      oq->flushDone = engine->flushDone + g;
      if (g < numSpsc) {
        oq->queue = std::make_unique<HostQueueImpl<mori::cco::proxy::queue::SPSCD2HQueue>>(
            *engine->spsc, g);
      } else {
        oq->queue = std::make_unique<HostQueueImpl<mori::cco::proxy::queue::MPSCD2HQueue>>(
            *engine->mpsc, g - numSpsc);
      }
      oq->lastSeq.assign(worldSize, 0);
      engine->workers[g % numThreads]->AddQueue(oq.get());
      engine->queues.push_back(std::move(oq));
    }

    // threadQps[t] RC QPs per (proxy thread t, masked peer).
    for (int peer = 0; peer < worldSize; peer++) {
      if (peer == rank || !config.peerMask[peer]) continue;
      for (int t = 0; t < numThreads; t++) {
        for (int slot = 0; slot < threadQps[t]; slot++) {
          ProxyQp& qp = engine->workers[t]->Qp(peer, slot);
          qp.ep = engine->ibv->CreateRdmaEndpoint(epConfig);
          qp.active = true;
          local[static_cast<size_t>(peer) * qpsPerRank + qpBase[t] + slot] = qp.ep.handle;
        }
      }
    }
  } catch (const std::exception& e) {
    error = e.what();
  }

  mine.ok = error.empty() ? 1 : 0;
  const std::vector<ProxyCreateInfo> all =
      AgreeOk(bootNet, mine, "failed to set up its proxy engine", &error);
  if (!error.empty()) return fail(error);
  // One RDMA write must fit every port.
  uint64_t maxMsgBytes = ccoProxyMaxPutBytes;
  for (const ProxyCreateInfo& info : all) maxMsgBytes = std::min(maxMsgBytes, info.maxMsgBytes);

  // Exchange QP handles the way Context does. A broken bootstrap leaves nothing
  // to agree on.
  std::vector<application::RdmaEndpointHandle> remote(local.size());
  try {
    bootNet->AllToAll(local.data(), remote.data(),
                      sizeof(application::RdmaEndpointHandle) * qpsPerRank);
  } catch (const std::exception& e) {
    return fail(e.what());
  }

  // Connect and start, then agree: a rank going on with QPs whose peer failed
  // would hang in later collectives or fail its first put.
  try {
    if (env::GetInt("MORI_CCO_PROXY_TEST_FAIL_CONNECT") == rank) {
      throw std::runtime_error("MORI_CCO_PROXY_TEST_FAIL_CONNECT");  // test hook
    }
    for (int peer = 0; peer < worldSize; peer++) {
      if (peer == rank || !config.peerMask[peer]) continue;
      for (int i = 0; i < qpsPerRank; i++) {
        const size_t h = static_cast<size_t>(peer) * qpsPerRank + i;
        engine->ibv->ConnectEndpoint(local[h], remote[h], i);
      }
    }

    std::string cpuSource;
    const std::vector<int> cpus = ChooseProxyCpus(numThreads, &cpuSource);
    std::string cpuList;
    for (int t = 0; t < numThreads; t++) {
      cpuList += (t ? "," : "") + std::to_string(cpus[t]);
      engine->workers[t]->Start(cpus[t], &engine->stop);
    }
    MORI_SHMEM_INFO("ccoProxy: rank={} {} proxy threads on CPUs [{}] (from {})", rank, numThreads,
                    cpuList, cpuSource);
    std::vector<int> sortedCpus = cpus;
    std::sort(sortedCpus.begin(), sortedCpus.end());
    if (!sortedCpus.empty() && sortedCpus.front() >= 0 &&
        std::adjacent_find(sortedCpus.begin(), sortedCpus.end()) != sortedCpus.end()) {
      // Busy-polling threads sharing a CPU cut the message rate ~100x; stderr so
      // it shows at the default log level.
      std::fprintf(stderr,
                   "[mori cco] WARNING: ccoProxy rank %d: %d proxy threads share CPUs [%s] (from "
                   "%s); widen the affinity mask (e.g. mpirun --bind-to none), set "
                   "MORI_CCO_PROXY_CPUS per rank (it applies to the whole process tree, so one "
                   "value for every rank of a node puts their threads on the same CPUs), or use "
                   "fewer threads\n",
                   rank, numThreads, cpuList.c_str(), cpuSource.c_str());
    }
  } catch (const std::exception& e) {
    error = e.what();
  }
  ProxyCreateInfo connected{};
  connected.ok = error.empty() ? 1 : 0;
  AgreeOk(bootNet, connected, "failed to connect or start", &error);
  if (!error.empty()) return fail(error);

  const int numPeers =
      static_cast<int>(std::count(config.peerMask.begin(), config.peerMask.end(), true)) -
      (config.peerMask[rank] ? 1 : 0);
  MORI_SHMEM_INFO(
      "ccoProxyEngineCreate: rank={} spscQueues={} mpscQueues={} threads={} qpsPerPeer={} "
      "(requested {}) peers={} QPs={} sqDepth={} maxPutBytes={}",
      rank, numSpsc, numMpsc, numThreads, threadQps[0], config.qpsPerPeer, numPeers,
      qpsPerRank * numPeers, kSqDepth, maxMsgBytes);
  out->numSpscQueues = numSpsc;
  out->numMpscQueues = numMpsc;
  out->spscContext = engine->spsc ? engine->spsc->device_context() : nullptr;
  out->mpscContext = engine->mpsc ? engine->mpsc->device_context() : nullptr;
  out->flushDone = engine->flushDone;
  out->flushTicket = engine->flushTicket;
  out->maxPutBytes = static_cast<uint32_t>(maxMsgBytes);  // <= ccoProxyMaxPutBytes
  out->engine = engine.get();
  return engine.release();
}

int ccoProxyEngineGetError(const ccoProxyEngine* engine, ccoDevCommError* err) {
  if (engine == nullptr || engine->err.state.load(std::memory_order_acquire) != 2) return 0;
  *err = engine->err.info;
  err->suppressed = engine->err.suppressed.load(std::memory_order_relaxed);
  return 1;
}

void ccoProxyEngineDestroy(ccoProxyEngine* engine) noexcept {
  if (engine == nullptr) return;
  engine->StopAndJoin();
  ccoDevCommError err;
  if (ccoProxyEngineGetError(engine, &err)) {
    std::fprintf(stderr,
                 "[mori cco] ccoProxy rank %d: destroying a failed DevComm (error code %d, %llu "
                 "later errors suppressed)\n",
                 engine->rank, err.code, static_cast<unsigned long long>(err.suppressed));
  }
  delete engine;
}

}  // namespace cco
}  // namespace mori
