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
// A fixed set of proxy threads. Each owns a subset of the KIWI device-to-host
// queues (SPSC and MPSC; it is their only consumer) and one host-posted RC QP,
// with its own CQ, per peer. Each queue pool is one KIWI invoke context: a
// thread drains its queues with Context::progress(), which runs one registered
// callback (ProxyWorker::On*) per descriptor on that thread. The QPs live on
// the PD of the comm's RDMA device context, which owns the cco window MRs, so
// the lkeys/rkeys kernels resolve from ccoWindowDevice are valid on them as-is.
//
// Per-QP work-request sequence numbers: the k-th WR posted on a QP has wr_id k.
// RC completes WRs in order, so a CQE for WR k retires every WR <= k
// (completedSeq). Every WR is signaled (v1). SQ credit is postedSeq -
// completedSeq.
//
// Flush: each queue tracks, per peer, the sequence of the last WR it produced.
// A popped FLUSH first posts every pending WR, then becomes a pending flush
// that waits for this queue's last WR on every peer that still has work
// outstanding. Pending flushes retire in FIFO order (a later FLUSH never waits
// for less than an earlier one), and each retired FLUSH adds 1 to
// flushDone[q]. The GPU side turns that count into "my FLUSH is satisfied"
// (see cco_proxy.hpp).

// cco headers first: core/utils.hpp (pulled in by the application headers)
// defines a `warpSize` macro that would break cco.hpp's impl::warpSize().
#include "mori/cco/cco_proxy.hpp"
// clang-format off
#include "cco_proxy_engine.hpp"
#include "hip/hip_runtime.h"
#include "mori/cco/proxy/invoke/invoke.hpp"
#include "mori/application/bootstrap/base_bootstrap.hpp"
#include "mori/application/context/context.hpp"
#include "mori/application/transport/rdma/providers/ibverbs/ibverbs.hpp"
#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/cpu_affinity.hpp"
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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mori {
namespace cco {

namespace {

constexpr size_t kDefaultQueueCapacity = 4096;  // lines (256 KiB ring per queue)
constexpr int kDefaultMaxThreads = 4;
constexpr uint32_t kSqDepth = 512;  // WRs per QP
constexpr int kPopBatch = 64;       // callbacks run per queue per pass
constexpr int kCqPollBatch = 32;
constexpr size_t kScratchStride = 64;  // per-thread atomic fetch-result slot

[[noreturn]] void ProxyFatal(int worker, const std::string& what) {
  MORI_SHMEM_ERROR("ccoProxy worker {}: {}", worker, what);
  std::abort();
}

// Proxy thread CPU placement. Each proxy thread busy-polls, so it gets a core of
// its own:
//  1. MORI_CCO_PROXY_CPUS (a Linux cpulist, e.g. "8-11" or "8,9,10,11") wins:
//     thread t runs on entry t % size. It applies to every rank in the process
//     tree, so with several ranks per node set it per rank.
//  2. Otherwise the candidates are the CPUs in the calling thread's affinity
//     mask that are local to the current GPU (ccoCommCreate already narrowed
//     the mask to this GPU's share of its NUMA node's cores), or the whole
//     mask if none of them is GPU-local.
//  3. The candidates are grouped into physical cores. The core holding CPU 0
//     (kernel housekeeping: IRQs, softirqs, kworkers) is dropped unless it is
//     the only one. Cores are taken from the END of the list, one thread per
//     core (its first sibling), so proxy threads stay off the low CPUs the
//     rank's own threads and the system favour, and off each other's SMT
//     siblings. With more threads than cores they wrap onto the remaining
//     siblings, then around again.
std::vector<int> ChooseProxyCpus(int numThreads, std::string* source) {
  std::vector<int> order;
  if (const char* env = std::getenv("MORI_CCO_PROXY_CPUS"); env != nullptr && env[0] != '\0') {
    order = application::detail::ParseCpuList(env);
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
        for (int c : application::detail::ParseCpuList(*list)) {
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
    const size_t n = ctx_.progress(endpoint_, max);
    if (n > 0) poll_.refresh();
    return n;
  }

 private:
  mori::cco::proxy::invoke::Context<Q>& ctx_;
  size_t endpoint_;
  typename Q::HostPollState poll_;
};

struct PendingFlush {
  std::vector<std::pair<int, uint64_t>> waits;  // (peer, WR seq) that must complete
};

struct OwnedQueue {
  std::unique_ptr<HostQueue> queue;
  volatile uint64_t* flushDone{nullptr};  // fine-grained device memory, host-written
  uint64_t flushed{0};                    // FLUSHes retired (the value of *flushDone)
  std::vector<uint64_t> lastSeq;          // [worldSize] seq of the last WR this queue produced
  std::deque<PendingFlush> pending;
};

struct ProxyQp {
  application::RdmaEndpoint ep;
  bool active{false};
  uint64_t postedSeq{0};     // WRs posted so far (wr_id of the last one)
  uint64_t completedSeq{0};  // every WR <= this has completed
  // WRs built during the current pass, posted as one chain at its end. Their
  // seqs are assigned when built: postedSeq + index + 1.
  std::vector<ibv_send_wr> wrs;
  std::vector<ibv_sge> sges;
};

class ProxyWorker {
 public:
  ProxyWorker(int id, int worldSize) : qps(worldSize), id_(id) {}

  void Start(int cpu, const std::atomic<bool>* stop) {
    for (int peer = 0; peer < static_cast<int>(qps.size()); peer++) {
      if (qps[peer].active) {
        activePeers_.push_back(peer);
        qps[peer].wrs.reserve(kSqDepth);
        qps[peer].sges.reserve(kSqDepth);
      }
    }
    thread_ = std::thread([this, cpu, stop] {
      PinCallingThread(id_, cpu);
      current_ = this;
      try {
        Run(stop);
      } catch (const std::exception& e) {  // e.g. KIWI invoke: unknown callback, bad size
        ProxyFatal(id_, e.what());
      }
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
    const uint8_t flushId =
        ctx.register_callback(std::function<void()>([] { current_->OnFlush(); }));
    if (flushId != ccoProxyOpFlush) throw std::runtime_error("ccoProxy callback id mismatch");
  }

  std::vector<ProxyQp> qps;         // [worldSize]; active for masked peers
  std::vector<OwnedQueue*> queues;  // queues this thread consumes
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

  void Run(const std::atomic<bool>* stop) {
    for (;;) {
      size_t popped = 0;
      for (OwnedQueue* oq : queues) {
        if (!oq->queue->Ready()) continue;
        cur_ = oq;
        popped += oq->queue->Progress(kPopBatch);
      }
      bool inFlight = false;
      bool completed = false;
      for (int peer : activePeers_) {
        ProxyQp& qp = qps[peer];
        PostChain(qp);
        if (qp.completedSeq < qp.postedSeq) {
          completed = PollCq(qp) || completed;
          inFlight = inFlight || qp.completedSeq < qp.postedSeq;
        }
      }
      bool flushPending = false;
      for (OwnedQueue* oq : queues) {
        if (completed) RetireFlushes(*oq);
        flushPending = flushPending || !oq->pending.empty();
      }
      if (popped == 0) {
        // Stop only once drained: nothing waiting, every WR completed, every
        // FLUSH answered.
        if (!inFlight && !flushPending && stop->load(std::memory_order_acquire) && !AnyReady()) {
          break;
        }
        _mm_pause();
      }
    }
  }

  bool AnyReady() {
    for (OwnedQueue* oq : queues) {
      if (oq->queue->Ready()) return true;
    }
    return false;
  }

  ProxyQp& PeerQp(uint32_t peer) {
    if (peer >= qps.size() || !qps[peer].active) {
      ProxyFatal(id_, "descriptor for peer " + std::to_string(peer) +
                          " without a proxy QP (check gdaConnectionType)");
    }
    return qps[peer];
  }

  void OnPut(const ccoProxyPutDesc& d) {
    ProxyQp& qp = PeerQp(d.peer);
    Reserve(qp, 1);
    cur_->lastSeq[d.peer] = AddWrite(qp, d);
  }

  void OnPutSignal(const ccoProxyPutSignalDesc& d) {
    ProxyQp& qp = PeerQp(d.put.peer);
    Reserve(qp, 2);  // both in one chain: data, then the atomic, on the same QP
    AddWrite(qp, d.put);
    cur_->lastSeq[d.put.peer] = AddAtomic(qp, d.signal);
  }

  void OnSignal(const ccoProxySignalDesc& d) {
    ProxyQp& qp = PeerQp(d.peer);
    Reserve(qp, 1);
    cur_->lastSeq[d.peer] = AddAtomic(qp, d);
  }

  void OnFlush() {
    OwnedQueue& oq = *cur_;
    // Everything before the FLUSH is popped; post it, then wait for this
    // queue's last WR on every peer that still has one outstanding.
    PendingFlush f;
    for (int p : activePeers_) {
      PostChain(qps[p]);
      if (oq.lastSeq[p] > qps[p].completedSeq) f.waits.emplace_back(p, oq.lastSeq[p]);
    }
    oq.pending.push_back(std::move(f));
    RetireFlushes(oq);
  }

  // Retire satisfied flushes in FIFO order and publish the new count.
  void RetireFlushes(OwnedQueue& oq) {
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
      __atomic_store_n(oq.flushDone, oq.flushed, __ATOMIC_RELEASE);
      _mm_sfence();  // drain the write-combining buffer toward the device
    }
  }

  // Make room for n more WRs in qp's SQ (posted + pending chain).
  void Reserve(ProxyQp& qp, size_t n) {
    while (qp.postedSeq + qp.wrs.size() + n - qp.completedSeq > kSqDepth) {
      PostChain(qp);
      if (PollCq(qp)) {
        for (OwnedQueue* oq : queues) RetireFlushes(*oq);
      }
    }
  }

  // Append a WR to qp's pending chain; returns its sequence number.
  uint64_t Append(ProxyQp& qp, const ibv_send_wr& wr, const ibv_sge& sge) {
    qp.wrs.push_back(wr);
    qp.sges.push_back(sge);
    return qp.postedSeq + qp.wrs.size();
  }

  uint64_t AddWrite(ProxyQp& qp, const ccoProxyPutDesc& d) {
    ibv_send_wr wr{};
    ibv_sge sge{};
    sge.addr = d.laddr;  // iova=0 window offset
    sge.length = d.bytes;
    sge.lkey = d.lkey;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.num_sge = d.bytes ? 1 : 0;
    wr.wr.rdma.remote_addr = d.raddr;
    wr.wr.rdma.rkey = d.rkey;
    return Append(qp, wr, sge);
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
    return Append(qp, wr, sge);
  }

  // Post the pending chain with one ibv_post_send. Every WR is signaled (v1).
  void PostChain(ProxyQp& qp) {
    const size_t n = qp.wrs.size();
    if (n == 0) return;
    for (size_t i = 0; i < n; i++) {
      ibv_send_wr& wr = qp.wrs[i];
      wr.wr_id = qp.postedSeq + i + 1;
      wr.sg_list = wr.num_sge ? &qp.sges[i] : nullptr;
      wr.next = (i + 1 < n) ? &qp.wrs[i + 1] : nullptr;
      wr.send_flags = IBV_SEND_SIGNALED;
    }
    ibv_send_wr* bad = nullptr;
    const int ret = ibv_post_send(qp.ep.ibvHandle.qp, qp.wrs.data(), &bad);
    if (ret != 0) {
      ProxyFatal(id_, "ibv_post_send failed: " + std::string(strerror(ret)));
    }
    qp.postedSeq += n;
    qp.wrs.clear();
    qp.sges.clear();
  }

  // Returns true if completedSeq advanced.
  bool PollCq(ProxyQp& qp) {
    ibv_wc wc[kCqPollBatch];
    const int n = ibv_poll_cq(qp.ep.ibvHandle.cq, kCqPollBatch, wc);
    if (n < 0) ProxyFatal(id_, "ibv_poll_cq failed");
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        ProxyFatal(id_, std::string("WR failed: ") + ibv_wc_status_str(wc[i].status) + " (wr_id " +
                            std::to_string(wc[i].wr_id) + ")");
      }
      if (wc[i].wr_id > qp.completedSeq) qp.completedSeq = wc[i].wr_id;
    }
    return n > 0;
  }

  static inline thread_local ProxyWorker* current_ = nullptr;  // set on each proxy thread

  int id_;
  OwnedQueue* cur_{nullptr};  // queue whose callbacks are running
  std::vector<int> activePeers_;
  std::thread thread_;
};

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

  application::BootstrapNetwork* bootNet{nullptr};
  std::unique_ptr<application::IBVerbsDeviceContext> ibv;
  std::unique_ptr<mori::cco::proxy::invoke::Context<mori::cco::proxy::queue::SPSCD2HQueue>>
      spsc;  // SPSC pool
  std::unique_ptr<mori::cco::proxy::invoke::Context<mori::cco::proxy::queue::MPSCD2HQueue>>
      mpsc;                                         // MPSC pool
  std::vector<std::unique_ptr<OwnedQueue>> queues;  // SPSC pool, then MPSC pool
  std::vector<std::unique_ptr<ProxyWorker>> workers;
  void* scratch{nullptr};
  uint64_t* flushDone{nullptr};    // fine-grained device [numSpsc + numMpsc]
  uint64_t* flushTicket{nullptr};  // device [numSpsc + numMpsc]
  std::atomic<bool> stop{false};
};

namespace {

// Per-rank setup result, Allgather'd before the QP exchange.
struct ProxyCreateInfo {
  int32_t ok;  // this rank's local setup succeeded
  int32_t pad;
  uint64_t maxMsgBytes;  // the port's max_msg_sz
};

}  // namespace

ccoProxyEngine* ccoProxyEngineCreate(application::Context* ctx,
                                     application::BootstrapNetwork* bootNet,
                                     const ccoProxyEngineConfig& config, ccoProxyContext* out) {
  auto engine = std::make_unique<ccoProxyEngine>();
  engine->bootNet = bootNet;
  const int worldSize = bootNet->GetWorldSize();
  const int rank = bootNet->GetLocalRank();
  const int numSpsc = config.numSpscQueues;
  const int numMpsc = config.numMpscQueues;
  // Summed in 64 bits and validated below before any use as an int.
  const int64_t numQueues64 = int64_t{numSpsc} + int64_t{numMpsc};
  const int numQueues =
      static_cast<int>(std::clamp<int64_t>(numQueues64, 0, std::numeric_limits<int>::max()));
  // Never more threads than queues: a thread without a queue would only spin.
  const int numThreads =
      std::min(config.numThreads > 0 ? config.numThreads : kDefaultMaxThreads, numQueues);

  // Local setup: everything up to the QP handle exchange. A failure here is
  // reported to every rank by the Allgather below, so all ranks fail together.
  ProxyCreateInfo mine{};
  std::string error;
  std::vector<application::RdmaEndpointHandle> local(static_cast<size_t>(worldSize) *
                                                     std::max(numThreads, 0));
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
    const size_t capacity = config.queueCapacity ? config.queueCapacity : kDefaultQueueCapacity;
    if ((capacity & (capacity - 1)) != 0 ||
        capacity < mori::cco::proxy::queue::message_lines(sizeof(ccoProxyPutSignalDesc))) {
      throw std::runtime_error("queue capacity must be a power of two >= the largest descriptor");
    }
    // The proxy threads store the flush counters into fine-grained VRAM, so the
    // CPU must be able to map device memory (large BAR).
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
    epConfig.maxInlineData = 0;
    epConfig.onGpu = false;
    epConfig.withCompChannel = false;
    epConfig.enableSrq = false;
    epConfig.atomicIbufSlots = 0;  // atomics return into the per-thread scratch slot below
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
      auto w = std::make_unique<ProxyWorker>(t, worldSize);
      w->scratchAddr = reinterpret_cast<uint64_t>(engine->scratch) + t * kScratchStride;
      w->scratchLkey = scratchMr.lkey;
      engine->workers.push_back(std::move(w));
    }

    // flushDone: fine-grained device memory (the GPU polls it locally, the
    // proxy writes it over PCIe). Flush tickets: plain device memory, only
    // touched by GPU atomics.
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
      oq->flushDone = engine->flushDone + g;
      if (g < numSpsc) {
        oq->queue = std::make_unique<HostQueueImpl<mori::cco::proxy::queue::SPSCD2HQueue>>(
            *engine->spsc, g);
      } else {
        oq->queue = std::make_unique<HostQueueImpl<mori::cco::proxy::queue::MPSCD2HQueue>>(
            *engine->mpsc, g - numSpsc);
      }
      oq->lastSeq.assign(worldSize, 0);
      engine->workers[g % numThreads]->queues.push_back(oq.get());
      engine->queues.push_back(std::move(oq));
    }

    // One RC QP per (proxy thread, masked peer).
    for (int peer = 0; peer < worldSize; peer++) {
      if (peer == rank || !config.peerMask[peer]) continue;
      for (int t = 0; t < numThreads; t++) {
        ProxyQp& qp = engine->workers[t]->qps[peer];
        qp.ep = engine->ibv->CreateRdmaEndpoint(epConfig);
        qp.active = true;
        local[peer * numThreads + t] = qp.ep.handle;
      }
    }
  } catch (const std::exception& e) {
    error = e.what();
  }

  // Every rank must have set up before the exchange. All ranks see the same
  // table, so they all fail or all go on. (The caller has already checked that
  // every rank passed the same queue and thread counts, so numThreads, the
  // handles sent per peer, matches.) One RDMA write must fit every port.
  mine.ok = error.empty() ? 1 : 0;
  std::vector<ProxyCreateInfo> all(worldSize);
  bootNet->Allgather(&mine, all.data(), sizeof(ProxyCreateInfo));
  uint64_t maxMsgBytes = ccoProxyMaxPutBytes;
  for (int r = 0; r < worldSize && error.empty(); r++) {
    if (!all[r].ok) {
      error = "rank " + std::to_string(r) + " failed to set up its proxy engine";
    } else {
      maxMsgBytes = std::min(maxMsgBytes, all[r].maxMsgBytes);
    }
  }
  if (!error.empty()) {
    MORI_SHMEM_ERROR("ccoProxyEngineCreate: rank={} failed: {}", rank, error);
    return nullptr;  // engine destructor releases what was built
  }

  try {
    // Exchange QP handles the way Context does, then connect and start.
    std::vector<application::RdmaEndpointHandle> remote(local.size());
    bootNet->AllToAll(local.data(), remote.data(),
                      sizeof(application::RdmaEndpointHandle) * numThreads);
    for (int peer = 0; peer < worldSize; peer++) {
      if (peer == rank || !config.peerMask[peer]) continue;
      for (int t = 0; t < numThreads; t++) {
        engine->ibv->ConnectEndpoint(local[peer * numThreads + t], remote[peer * numThreads + t],
                                     t);
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
                   "MORI_CCO_PROXY_CPUS, or use fewer threads\n",
                   rank, numThreads, cpuList.c_str(), cpuSource.c_str());
    }
  } catch (const std::exception& e) {
    MORI_SHMEM_ERROR("ccoProxyEngineCreate: rank={} failed: {}", rank, e.what());
    return nullptr;  // engine destructor stops threads and releases what was built
  }

  MORI_SHMEM_INFO(
      "ccoProxyEngineCreate: rank={} spscQueues={} mpscQueues={} threads={} sqDepth={} "
      "maxPutBytes={}",
      rank, numSpsc, numMpsc, numThreads, kSqDepth, maxMsgBytes);
  out->numSpscQueues = numSpsc;
  out->numMpscQueues = numMpsc;
  out->spscContext = engine->spsc ? engine->spsc->device_context() : nullptr;
  out->mpscContext = engine->mpsc ? engine->mpsc->device_context() : nullptr;
  out->flushDone = engine->flushDone;
  out->flushTicket = engine->flushTicket;
  out->maxPutBytes = maxMsgBytes;
  out->engine = engine.get();
  return engine.release();
}

void ccoProxyEngineDestroy(ccoProxyEngine* engine) {
  if (engine == nullptr) return;
  engine->StopAndJoin();       // drain: every WR we posted completed, every FLUSH answered
  engine->bootNet->Barrier();  // ... and every peer's WRs to our QPs too
  delete engine;
}

}  // namespace cco
}  // namespace mori
