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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>

#include "hip/hip_runtime_api.h"
#include "mori/cco/cco.hpp"
#include "mori/utils/mori_log.hpp"

#if BUILD_CCO_SDMA
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "mori/application/application.hpp"
#include "mori/application/transport/sdma/anvil.hpp"
#endif

namespace mori {
namespace cco {

#if BUILD_CCO_SDMA

namespace {

// One packet per 32-byte unit, zero-padded (zero dwords are NOPs), so a packet never
// straddles the end of the power-of-two ring.
constexpr uint64_t kUnit = 32;
constexpr uint32_t kUnitDwords = kUnit / sizeof(uint32_t);
constexpr uint32_t kPollInterval = 0x10;
constexpr uint32_t kPollRetryForever = 0xFFF;

using Unit = uint32_t[kUnitDwords];

static_assert(sizeof(SDMA_PKT_POLL_REGMEM) <= kUnit, "POLL must fit one unit");
static_assert(sizeof(SDMA_PKT_COPY_LINEAR) <= kUnit, "COPY must fit one unit");
static_assert(sizeof(SDMA_PKT_ATOMIC) == kUnit, "ATOMIC must be exactly one unit");

void PollUnit(Unit u, const void* gate, uint32_t target, uint32_t func) {
  SDMA_PKT_POLL_REGMEM p{};
  p.HEADER_UNION.op = SDMA_OP_POLL_REGMEM;
  p.HEADER_UNION.mem_poll = 1;
  p.HEADER_UNION.func = func;
  p.ADDR_LO_UNION.addr_31_0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(gate));
  p.ADDR_HI_UNION.addr_63_32 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(gate) >> 32);
  p.VALUE_UNION.value = target;
  p.MASK_UNION.mask = 0xFFFFFFFFu;
  p.DW5_UNION.interval = kPollInterval;
  p.DW5_UNION.retry_count = kPollRetryForever;
  std::memset(u, 0, kUnit);
  std::memcpy(u, &p, sizeof(p));
}

void CopyUnit(Unit u, const void* src, void* dst, uint64_t bytes) {
  SDMA_PKT_COPY_LINEAR p{};
  p.HEADER_UNION.op = SDMA_OP_COPY;
  p.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
  p.COUNT_UNION.count = static_cast<uint32_t>(bytes - 1);
  p.SRC_ADDR_LO_UNION.src_addr_31_0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src));
  p.SRC_ADDR_HI_UNION.src_addr_63_32 =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src) >> 32);
  p.DST_ADDR_LO_UNION.dst_addr_31_0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst));
  p.DST_ADDR_HI_UNION.dst_addr_63_32 =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst) >> 32);
  std::memset(u, 0, kUnit);
  std::memcpy(u, &p, sizeof(p));
}

void AtomicAddUnit(Unit u, void* addr) {
  SDMA_PKT_ATOMIC p{};
  p.HEADER_UNION.op = SDMA_OP_ATOMIC;
  p.HEADER_UNION.operation = SDMA_ATOMIC_ADD64;
  p.ADDR_LO_UNION.addr_31_0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr));
  p.ADDR_HI_UNION.addr_63_32 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr) >> 32);
  p.SRC_DATA_LO_UNION.src_data_31_0 = 1;
  std::memcpy(u, &p, sizeof(p));
}

}  // namespace

struct ccoTriggerQueue {
  int pe{-1};
  std::unique_ptr<anvil::SdmaQueue> queue;
  anvil::SdmaQueueHostHandle handle{};
  uint64_t writeIndex{0};   // host shadow of the monotonic write index
  uint64_t kickedIndex{0};  // last index published to the engine

  uint64_t ReadIndex() const { return __atomic_load_n(handle.rptr, __ATOMIC_ACQUIRE); }

  void WriteUnit(uint64_t at, const Unit u) {
    auto* dst = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(handle.ring) +
                                            (at & (handle.ringBytes - 1)));
#if defined(__x86_64__) || defined(__i386__)
    // Two aligned 16-byte stores per unit.
    _mm_store_si128(reinterpret_cast<__m128i*>(dst),
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(u)));
    _mm_store_si128(reinterpret_cast<__m128i*>(dst + 4),
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(u + 4)));
#else
    for (uint32_t i = 0; i < kUnitDwords; i++) {
      __atomic_store_n(dst + i, u[i], __ATOMIC_RELAXED);
    }
#endif
  }
};

int ccoTriggerQueueCreate(ccoComm* comm, int pe, int engineOffset, uint64_t ringBytes,
                          ccoTriggerQueue_t* queue) {
  if (comm == nullptr || queue == nullptr) return -1;
  *queue = nullptr;
  const int lsa = pe - comm->myNodeStart;
  if (pe == comm->rank || lsa < 0 || lsa >= comm->lsaSize || pe >= comm->worldSize) {
    MORI_SHMEM_ERROR("ccoTriggerQueueCreate: rank {} is not another rank of this LSA team", pe);
    return -1;
  }
  if (!comm->ctx->IsSdmaEnabled() || !comm->ctx->GetPeerCapabilities(pe).canSDMA) {
    MORI_SHMEM_ERROR("ccoTriggerQueueCreate: no SDMA to rank {} (MORI_ENABLE_SDMA unset?)", pe);
    return -1;
  }
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, comm->hipDev) != hipSuccess) return -1;
  const bool gfx942 = std::strncmp(prop.gcnArchName, "gfx942", 6) == 0;
  const bool gfx950 = std::strncmp(prop.gcnArchName, "gfx950", 6) == 0;
  if (!gfx942 && !gfx950) {
    MORI_SHMEM_ERROR("ccoTriggerQueueCreate: supported on gfx942 and gfx950, not {}",
                     prop.gcnArchName);
    return -1;
  }
  // On gfx950 the link's sibling xGMI engine is known to hang.
  if (engineOffset != 0 && !gfx942) {
    MORI_SHMEM_ERROR("ccoTriggerQueueCreate: engine offsets are supported on gfx942 only");
    return -1;
  }
  const int srcNode = comm->ctx->LocalKfdNode();
  const int dstNode = comm->ctx->KfdNodeId(pe);
  if (srcNode < 0 || dstNode < 0) {
    MORI_SHMEM_ERROR("ccoTriggerQueueCreate: KFD node unknown (local {}, rank {}: {})", srcNode, pe,
                     dstNode);
    return -1;
  }
  if (ringBytes == 0) ringBytes = anvil::SDMA_QUEUE_SIZE;
  auto q = anvil::anvil.createPrivateQueue(srcNode, dstNode, engineOffset, ringBytes);
  if (!q) return -1;
  if (q->hostHandle().hdpFlush == nullptr) {
    MORI_SHMEM_ERROR("ccoTriggerQueueCreate: no HDP flush register for this GPU");
    return -1;
  }

  auto t = std::make_unique<ccoTriggerQueue>();
  t->pe = pe;
  t->handle = q->hostHandle();
  t->queue = std::move(q);
  t->writeIndex = __atomic_load_n(t->handle.wptr, __ATOMIC_RELAXED);
  t->kickedIndex = t->writeIndex;
  *queue = t.release();
  return 0;
}

int ccoTriggerQueueDestroy(ccoTriggerQueue_t queue) {
  if (queue == nullptr) return 0;
  const bool pending = queue->ReadIndex() < queue->kickedIndex;
  if (pending) {
    MORI_SHMEM_ERROR(
        "ccoTriggerQueueDestroy: queue to rank {} destroyed with kicked work not consumed "
        "(read {} < {})",
        queue->pe, queue->ReadIndex(), queue->kickedIndex);
  }
  delete queue;
  return pending ? 1 : 0;
}

int ccoTriggerAppendChain(ccoTriggerQueue_t queue, const void* gate, uint32_t target,
                          ccoSdmaCmp cmp, const ccoTriggerCopy* copies, int numCopies,
                          void* arrival, uint64_t* endIndex) {
  if (queue == nullptr || gate == nullptr || arrival == nullptr || numCopies < 0 ||
      (numCopies > 0 && copies == nullptr))
    return -1;
  if ((reinterpret_cast<uintptr_t>(gate) & 3) != 0 ||
      (reinterpret_cast<uintptr_t>(arrival) & 7) != 0) {
    MORI_SHMEM_ERROR("ccoTriggerAppendChain: gate must be 4-byte and arrival 8-byte aligned");
    return -1;
  }
  if (cmp > ccoSdmaCmpGt) return -1;
  const uint64_t ringUnits = queue->handle.ringBytes / kUnit;
  uint64_t units = 2;
  for (int i = 0; i < numCopies && units < ringUnits; i++) {
    if (copies[i].bytes == 0 || copies[i].src == nullptr || copies[i].dst == nullptr) return -1;
    if (copies[i].bytes > ringUnits * CCO_SDMA_MAX_COPY_BYTES) return -1;  // can never fit
    units += (copies[i].bytes + CCO_SDMA_MAX_COPY_BYTES - 1) / CCO_SDMA_MAX_COPY_BYTES;
  }
  if (units >= ringUnits) return -1;  // can never fit
  const uint64_t bytes = units * kUnit;
  // Never fill the ring: the engine reads wptr == rptr (mod ring) as empty.
  const uint64_t end = queue->writeIndex + bytes;
  if (end - queue->ReadIndex() >= queue->handle.ringBytes) return 1;

  uint64_t at = queue->writeIndex;
  Unit u;
  PollUnit(u, gate, target, cmp);
  queue->WriteUnit(at, u);
  at += kUnit;
  for (int i = 0; i < numCopies; i++) {
    const char* src = static_cast<const char*>(copies[i].src);
    char* dst = static_cast<char*>(copies[i].dst);
    for (uint64_t off = 0; off < copies[i].bytes; off += CCO_SDMA_MAX_COPY_BYTES) {
      const uint64_t n = std::min<uint64_t>(copies[i].bytes - off, CCO_SDMA_MAX_COPY_BYTES);
      CopyUnit(u, src + off, dst + off, n);
      queue->WriteUnit(at, u);
      at += kUnit;
    }
  }
  AtomicAddUnit(u, arrival);
  queue->WriteUnit(at, u);
  at += kUnit;
  queue->writeIndex = at;
  if (endIndex != nullptr) *endIndex = at;
  return 0;
}

int ccoTriggerKickAll(const ccoTriggerQueue_t* queues, int numQueues) {
  if (numQueues < 0 || (numQueues > 0 && queues == nullptr)) return -1;
  for (int i = 0; i < numQueues; i++)
    if (queues[i] == nullptr) return -1;
  // Packet stores to the VRAM rings can still sit in the CPU's write-combining
  // buffers or the GPU's HDP when the doorbell arrives: sfence drains the CPU side,
  // and the flush register's write plus read-back drains the HDP (once per GPU).
#if defined(__x86_64__) || defined(__i386__)
  _mm_sfence();
#endif
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  for (int i = 0; i < numQueues; i++) {
    if (queues[i]->writeIndex == queues[i]->kickedIndex) continue;
    volatile uint32_t* hdp = queues[i]->handle.hdpFlush;
    bool seen = false;
    for (int j = 0; j < i && !seen; j++)
      seen = queues[j]->writeIndex != queues[j]->kickedIndex && queues[j]->handle.hdpFlush == hdp;
    if (seen) continue;
    *hdp = 1u;
    (void)*hdp;
  }
  for (int i = 0; i < numQueues; i++) {
    ccoTriggerQueue_t q = queues[i];
    if (q->writeIndex == q->kickedIndex) continue;
    __atomic_store_n(q->handle.wptr, q->writeIndex, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(q->handle.doorbell, q->writeIndex, __ATOMIC_RELAXED);
    q->kickedIndex = q->writeIndex;
  }
  return 0;
}

int ccoTriggerKick(ccoTriggerQueue_t queue) {
  if (queue == nullptr) return -1;
  return ccoTriggerKickAll(&queue, 1);
}

int ccoTriggerQuiesce(ccoTriggerQueue_t queue, uint64_t index, uint64_t timeoutNs) {
  if (queue == nullptr) return -1;
  if (index == UINT64_MAX) index = queue->kickedIndex;
  if (index > queue->kickedIndex) {
    MORI_SHMEM_ERROR("ccoTriggerQuiesce: index {} was never kicked (kicked up to {})", index,
                     queue->kickedIndex);
    return -1;
  }
  constexpr uint64_t kMaxTimeoutNs = 1ull << 62;  // keeps the deadline from overflowing
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::nanoseconds(timeoutNs < kMaxTimeoutNs ? timeoutNs : kMaxTimeoutNs);
  for (uint32_t spin = 0;; spin++) {
    if (queue->ReadIndex() >= index) return 0;
    if ((spin & 255) == 0 && std::chrono::steady_clock::now() >= deadline) return 1;
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#endif
  }
}

int ccoTriggerQueueState(ccoTriggerQueue_t queue, uint64_t* readIndex, uint64_t* writeIndex) {
  if (queue == nullptr) return -1;
  if (readIndex != nullptr) *readIndex = queue->ReadIndex();
  if (writeIndex != nullptr) *writeIndex = queue->writeIndex;
  return 0;
}

int ccoTriggerQueueGetInfo(ccoTriggerQueue_t queue, ccoTriggerQueueInfo* info) {
  if (queue == nullptr || info == nullptr) return -1;
  info->peer = queue->pe;
  info->engineId = queue->handle.engineId;
  info->ringBytes = queue->handle.ringBytes;
  return 0;
}

int ccoTriggerQueueReadRing(ccoTriggerQueue_t queue, uint64_t index, void* out, uint64_t bytes) {
  if (queue == nullptr || out == nullptr || ((index | bytes) & 3) != 0 ||
      bytes > queue->handle.ringBytes)
    return -1;
  auto* dst = static_cast<uint32_t*>(out);
  const auto* ring = reinterpret_cast<const volatile uint32_t*>(queue->handle.ring);
  const uint64_t mask = (queue->handle.ringBytes - 1) / sizeof(uint32_t);
  for (uint64_t i = 0; i < bytes / sizeof(uint32_t); i++) {
    dst[i] = ring[(index / sizeof(uint32_t) + i) & mask];
  }
  return 0;
}

#else  // !BUILD_CCO_SDMA

struct ccoTriggerQueue {};

int ccoTriggerQueueCreate(ccoComm*, int, int, uint64_t, ccoTriggerQueue_t* queue) {
  if (queue != nullptr) *queue = nullptr;
  MORI_SHMEM_ERROR("ccoTriggerQueueCreate: this mori was built BUILD_CCO_SDMA=OFF");
  return -1;
}
int ccoTriggerQueueDestroy(ccoTriggerQueue_t) { return 0; }
int ccoTriggerAppendChain(ccoTriggerQueue_t, const void*, uint32_t, ccoSdmaCmp,
                          const ccoTriggerCopy*, int, void*, uint64_t*) {
  return -1;
}
int ccoTriggerKick(ccoTriggerQueue_t) { return -1; }
int ccoTriggerKickAll(const ccoTriggerQueue_t*, int) { return -1; }
int ccoTriggerQuiesce(ccoTriggerQueue_t, uint64_t, uint64_t) { return -1; }
int ccoTriggerQueueState(ccoTriggerQueue_t, uint64_t*, uint64_t*) { return -1; }
int ccoTriggerQueueGetInfo(ccoTriggerQueue_t, ccoTriggerQueueInfo*) { return -1; }
int ccoTriggerQueueReadRing(ccoTriggerQueue_t, uint64_t, void*, uint64_t) { return -1; }

#endif  // BUILD_CCO_SDMA

}  // namespace cco
}  // namespace mori
