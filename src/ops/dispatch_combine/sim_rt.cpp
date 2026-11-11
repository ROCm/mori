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
// Runtime sim mode. The op runs on a simulated HIP runtime (e.g. Sundial's virtual GPU, libvgpu,
// loaded as libamdhip64) that executes no kernel; for the EP dispatch / combine and local expert
// count kernels it calls the handlers below, at every eager launch and every replay of a captured
// graph. Dispatch draws this rank's routing (MORI_EP_SIM_ROUTING) into the caller's top-k buffer
// and posts it through the runtime's all-to-all, which times the op from the rows each rank sends
// to each other rank. Launches never wait: an op's outputs are written once every rank has posted
// it, and the runtime's read hook blocks for that before the host reads them. Token data is not
// moved. Every handler call that found its op posts exactly one op, also on an error path (an empty
// one), so the ranks' op sequences stay aligned.
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/ops/dispatch_combine/sim.hpp"

namespace mori {
namespace moe {
namespace sim {
namespace rt {
namespace {

// The runtime's interface (version 1), looked up in the process: mori links no simulator.
struct Hctx {  // one handler call
  uint32_t version, size;
  const char* kname;
  const void* kargs;  // the packed kernel arguments
  size_t kn;
  uint32_t grid[3], block[3];
  void* stream;
  int phase;  // 0 eager launch, 2 graph replay
  int kernelTime;
  uint64_t graph;
  uint32_t node;
  void* user;
  int (*ptrOk)(const void* p, size_t n);  // host-accessible
  void (*wrote)(void* p, size_t n);       // reports a host write of device memory
};
struct A2aMsg {  // this rank's part of an all-to-all op
  uint64_t group;
  const uint32_t* rowsTo;  // rows sent to each rank; null with transposeOf
  uint64_t transposeOf;    // > 0: the rows of that earlier op, sent back (combine)
  double rowBytes;
  const void* payload;  // readable by the peers until each releases the op
  size_t payloadLen;
};
using Handler = int (*)(Hctx*);  // 0 handled, 1 not ours, < 0 error
using ReadHook = void (*)(void* user, const void* p, size_t n, int block);
constexpr unsigned kPrefix = 1, kComm = 2;  // handler flags: name prefix; may post

struct Api {
  int (*registerHandler)(const char* name, unsigned flags, Handler fn, void* user);
  uint64_t (*a2aOpen)(const char* key, int members, int me, int node, size_t payloadCap);
  int (*a2aPost)(Hctx* c, const A2aMsg* m, uint64_t* seq);
  const void* (*a2aPayload)(uint64_t group, uint64_t seq, int member, size_t* len, int block);
  int (*a2aReady)(uint64_t group, uint64_t seq);  // every member posted op seq
  void (*a2aRelease)(uint64_t group, uint64_t seq);
  // a member must release op s - nslot before it posts op s
  int (*a2aInfo)(uint64_t group, int* members, int* me, uint64_t* posted, uint32_t* nslot);
  // called before the host reads device memory (block = 1) and at syncs (p = null)
  int (*registerReadHook)(ReadHook hook, void* user);
  // optional
  void (*hostWrote)(void* p, size_t n);
  int (*valueRegister)(const char* name, const void* p, uint32_t count,
                       int type);  // kernel-time keys
  int (*a2aStats)(uint64_t* v, int n);
  const char* (*a2aStatName)(int i);
  bool ok;
};

const Api& Rt() {
  static const Api api = [] {
    Api a{};
    auto sym = [](auto& fn, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(RTLD_DEFAULT, name));
      return fn != nullptr;
    };
    a.ok = sym(a.registerHandler, "vgpu_kl_register_handler") & sym(a.a2aOpen, "vgpu_a2a_open") &
           sym(a.a2aPost, "vgpu_kl_a2a_post") & sym(a.a2aPayload, "vgpu_a2a_payload") &
           sym(a.a2aReady, "vgpu_a2a_ready") & sym(a.a2aRelease, "vgpu_a2a_release") &
           sym(a.a2aInfo, "vgpu_a2a_info") & sym(a.registerReadHook, "vgpu_register_read_hook");
    sym(a.hostWrote, "vgpu_host_wrote");
    sym(a.valueRegister, "vgpu_value_register");
    sym(a.a2aStats, "vgpu_a2a_stats");
    sym(a.a2aStatName, "vgpu_a2a_stat_name");
    return a;
  }();
  return api;
}

uint64_t NowNs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

struct Item {
  char kind;     // 'D' dispatch, 'C' combine, 'L' local expert count
  uint64_t seq;  // D, C: the all-to-all op
  int phase, n, sent;
  bool empty, resetTotal;
  index_t *outIdx, *srcMap, *totalRecv, *destMap;  // D; C: totalRecv to reset
  const index_t *lecIdx, *lecTotal;                // L
  int* lecOut;
  uint64_t t0;
};

// Values of the last dispatch the runtime may key kernel times with (mori.ep[.op<i>].<name>).
struct Values {
  int32_t recvRows, expertsHit, maxExpertRows, pad;
};

struct Op {
  OpSpec spec;
  std::unique_ptr<Routing> routing;  // null: MORI_EP_SIM_ROUTING=input, the caller's top-k as it is
  uint64_t seed{0}, group{0}, calls{0}, lastDispatch{0};
  std::mutex mu;
  std::deque<Item> q;            // in launch order
  std::vector<int32_t> payload;  // {n, rowsTo[world], ids[n * topk]}
  std::vector<int> lec;          // the last dispatch's local expert counts
  const index_t* lastOutIdx{nullptr};
  std::vector<std::pair<uintptr_t, uintptr_t>> outputs;  // what the read hook guards
  Values vals{};
  FILE* log{nullptr};  // MORI_EP_SIM_LOG_DIR: one line per dispatch
  // counters
  uint64_t disp[3]{}, comb[3]{}, cnt[3]{};  // eager, replay, error
  uint64_t items{0}, deferred{0}, maxPending{0}, blockedReads{0}, blockedNs{0}, recvRows{0};
  uint64_t overflow{0}, missing{0}, emptyPosts{0}, lecChecked{0}, lecMismatch{0};

  void AddOutput(const void* p, size_t n) {
    const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
    for (auto& r : outputs)
      if (r.first == lo) return;
    outputs.emplace_back(lo, lo + n);
  }
};

std::mutex gMu;
std::vector<std::unique_ptr<Op>> gOps;
std::unordered_map<uintptr_t, Op*> gByKey;

Op* Find(uint64_t key) {
  std::lock_guard<std::mutex> g(gMu);
  auto it = gByKey.find(key);
  return it == gByKey.end() ? nullptr : it->second;
}

void Wrote(Hctx* c, void* p, size_t n) {
  if (!p || !n) return;
  if (c && c->wrote)
    c->wrote(p, n);
  else if (Rt().hostWrote)
    Rt().hostWrote(p, n);
}

// The destination of each of a token's top-k slots, or -1: a token goes once to each rank (keyed by
// its first expert there). Expert ids < 0: v1 drops them, v2's kernel truncates (-1 / epr = rank
// 0).
void Dests(const OpSpec& s, const int32_t* row, int* d) {
  for (int k = 0; k < s.topk; ++k) {
    const int32_t e = row[k];
    d[k] = e < 0 && s.version == 1 ? -1 : e / s.expertsPerRank;
    if (d[k] < 0 || d[k] >= s.world) d[k] = -1;
    for (int j = 0; j < k && d[k] >= 0; ++j)
      if (d[j] == d[k]) d[k] = -1;
  }
}

void Count(const OpSpec& s, const index_t* idx, int rows, int* count) {
  const int lo = s.rank * s.expertsPerRank;
  std::fill(count, count + s.expertsPerRank, 0);
  for (int i = 0; i < rows * s.topk; ++i) {
    const int e = idx[i] - lo;
    if (e >= 0 && e < s.expertsPerRank) ++count[e];
  }
}

// The receive side of dispatch `it` and this rank's destination map, as the kernel writes them;
// false while a peer has not posted (block: wait for it). Received rows are in source-rank, then
// source-token order (the kernel's arrival order is another valid one).
bool RunDispatch(Op& op, const Item& it, Hctx* c, bool block) {
  const Api& rt = Rt();
  const OpSpec& s = op.spec;
  if (!block && !rt.a2aReady(op.group, it.seq)) return false;
  if (!it.empty) {
    std::vector<int64_t> next(s.world, 0);  // this rank's first slot on each rank
    const int32_t* mine = nullptr;
    int myN = 0, recv = 0;
    // a token comes here if one of its experts is here (v2: ids in (-epr, 0) are rank 0's too)
    const int32_t hi = (s.rank + 1) * s.expertsPerRank,
                  lo = s.version == 2 && s.rank == 0 ? 1 - s.expertsPerRank : hi - s.expertsPerRank;
    for (int j = 0; j < s.world; ++j) {
      size_t len = 0;
      const auto* p = static_cast<const int32_t*>(rt.a2aPayload(op.group, it.seq, j, &len, block));
      if (!p || len < (1 + s.world) * 4u ||
          len < (1 + s.world) * 4u + static_cast<size_t>(std::max(p[0], 0)) * s.topk * 4) {
        if (!block) return false;
        ++op.missing;  // the runtime timed out waiting for it
        continue;
      }
      const int n = p[0];
      const int32_t* ids = p + 1 + s.world;
      if (j < s.rank)
        for (int d = 0; d < s.world; ++d) next[d] += static_cast<uint32_t>(p[1 + d]);
      if (j == s.rank) mine = ids, myN = n;
      for (int t = 0; t < n; ++t) {
        const int32_t* row = ids + static_cast<size_t>(t) * s.topk;
        bool hit = false;
        for (int k = 0; k < s.topk && !hit; ++k) hit = row[k] >= lo && row[k] < hi;
        if (!hit) continue;
        if (recv >= s.maxRecv) {
          ++op.overflow;
          continue;
        }
        std::memcpy(it.outIdx + static_cast<size_t>(recv) * s.topk, row, s.topk * 4);
        it.srcMap[recv++] = static_cast<index_t>(j * s.srcStride + t);
      }
    }
    std::vector<int> d(s.topk);
    for (int t = 0; t < myN; ++t) {
      Dests(s, mine + static_cast<size_t>(t) * s.topk, d.data());
      for (int k = 0; k < s.topk; ++k) {
        const int64_t slot = d[k] < 0 ? s.maxRecv : next[d[k]]++;
        it.destMap[t * s.topk + k] = static_cast<index_t>(
            slot < s.maxRecv ? d[k] * s.destStride + slot : s.world * s.destStride);
      }
    }
    *it.totalRecv = recv;
    Wrote(c, it.outIdx, static_cast<size_t>(recv) * s.topk * 4);
    Wrote(c, it.srcMap, static_cast<size_t>(recv) * 4);
    Wrote(c, it.destMap, static_cast<size_t>(myN) * s.topk * 4);
    Wrote(c, it.totalRecv, 4);
    Count(s, it.outIdx, recv, op.lec.data());
    op.lastOutIdx = it.outIdx;
    op.vals = {recv,
               static_cast<int32_t>(
                   std::count_if(op.lec.begin(), op.lec.end(), [](int x) { return x > 0; })),
               *std::max_element(op.lec.begin(), op.lec.end()), 0};
    op.recvRows += recv;
    if (op.log)
      std::fprintf(op.log, "%llu %d %d %d %llu %d\n", static_cast<unsigned long long>(it.seq),
                   it.phase, it.n, recv, static_cast<unsigned long long>((NowNs() - it.t0) / 1000),
                   it.sent);
  }
  rt.a2aRelease(op.group, it.seq);
  return true;
}

// Runs the queued items in launch order while their inputs are there (block: wait for the peers);
// limit > 0: stops at the first all-to-all op past it.
void Drain(Op& op, Hctx* c, bool block, uint64_t limit = 0) {
  const OpSpec& s = op.spec;
  while (!op.q.empty()) {
    Item& it = op.q.front();
    if (limit && it.kind != 'L' && it.seq > limit) return;
    if (it.kind == 'D') {
      if (!RunDispatch(op, it, c, block)) return;
    } else if (it.kind == 'C') {
      if (it.resetTotal) {  // as the v1 combine kernel does
        *it.totalRecv = 0;
        Wrote(c, it.totalRecv, 4);
      }
      Rt().a2aRelease(op.group, it.seq);
    } else {
      Count(s, it.lecIdx, it.lecTotal ? *it.lecTotal : 0, it.lecOut);
      Wrote(c, it.lecOut, s.expertsPerRank * 4);
      if (it.lecIdx == op.lastOutIdx) {  // a count of another dispatch = items out of order
        ++op.lecChecked;
        op.lecMismatch += !std::equal(op.lec.begin(), op.lec.end(), it.lecOut);
      }
    }
    op.q.pop_front();
    ++op.items;
  }
}

// Posts this rank's next all-to-all op; 0 if the runtime refused it. The runtime reuses the slot of
// op seq - nslot, so this rank first runs its own items up to there.
uint64_t Post(Op& op, Hctx* c, const A2aMsg& m) {
  int members = 0, me = 0;
  uint64_t posted = 0, seq = 0;
  uint32_t nslot = 0;
  Rt().a2aInfo(op.group, &members, &me, &posted, &nslot);
  if (nslot && posted + 1 > nslot) Drain(op, c, true, posted + 1 - nslot);
  return Rt().a2aPost(c, &m, &seq) == 0 ? seq : 0;
}

void Push(Op& op, Hctx* c, Item it) {
  it.t0 = NowNs();
  it.phase = c->phase;
  const bool behind = !op.q.empty();
  op.q.push_back(it);
  op.maxPending = std::max<uint64_t>(op.maxPending, op.q.size());
  const size_t before = op.q.size();
  Drain(op, c, false);
  op.deferred += behind || op.q.size() == before;
}

int PostEmpty(Op& op, Hctx* c, char kind) {
  int32_t* p = op.payload.data();
  std::fill(p, p + 1 + op.spec.world, 0);
  const uint64_t seq = Post(op, c,
                            {op.group, reinterpret_cast<uint32_t*>(p + 1), 0, 0, p,
                             (1 + op.spec.world) * sizeof(int32_t)});
  if (!seq) return -1;
  ++op.emptyPosts;
  if (kind == 'D') op.lastDispatch = seq;
  Item it{kind, seq};
  it.empty = true;
  Push(op, c, it);
  return -1;
}

// Draws (or takes) the routing of this rank's n tokens, writes it into the caller's top-k, posts
// it.
int PostDispatch(Op& op, Hctx* c, index_t* tokIdx, int n, double rowBytes, Item it) {
  const OpSpec& s = op.spec;
  const int k = s.topk;
  const size_t bytes = static_cast<size_t>(std::max(n, 0)) * k * 4;
  if (n < 0 || n > s.maxTok || (n && !c->ptrOk(tokIdx, bytes))) return PostEmpty(op, c, 'D');
  int32_t* p = op.payload.data();
  int32_t* ids = p + 1 + s.world;
  const uint64_t call = op.calls++;
  if (op.routing) {
    const uint64_t rng =
        op.seed ^ (static_cast<uint64_t>(s.rank) << 40) ^ (call * 0x9E3779B97F4A7C15ull);
    op.routing->Draw(call, rng, n, k, ids);
    std::memcpy(tokIdx, ids, bytes);  // the MoE and combine see the drawn routing
    Wrote(c, tokIdx, bytes);
  } else {
    std::memcpy(ids, tokIdx, bytes);
  }
  uint32_t* rowsTo = reinterpret_cast<uint32_t*>(p + 1);
  p[0] = n;
  std::fill(rowsTo, rowsTo + s.world, 0u);
  std::vector<int> d(k);
  for (int t = 0; t < n; ++t) {
    Dests(s, ids + static_cast<size_t>(t) * k, d.data());
    for (int x : d)
      if (x >= 0) ++rowsTo[x], ++it.sent;
  }
  it.seq = Post(op, c, {op.group, rowsTo, 0, rowBytes, p, (1 + s.world) * 4 + bytes});
  if (!it.seq) return -1;
  op.lastDispatch = it.seq;
  it.kind = 'D';
  it.n = n;
  Push(op, c, it);
  return 0;
}

int PostCombine(Op& op, Hctx* c, double rowBytes, Item it) {
  if (!op.lastDispatch) return PostEmpty(op, c, 'C');  // nothing to send back
  it.seq = Post(op, c, {op.group, nullptr, op.lastDispatch, rowBytes, nullptr, 0});
  if (!it.seq) return -1;
  it.kind = 'C';
  Push(op, c, it);
  return 0;
}

void MaybeDumpStats();

// Handler prologue: the op this launch belongs to (locked), or null.
template <class Args>
Op* Begin(Hctx* c, Args* a, uint64_t (*key)(Hctx*, const Args&)) {
  MaybeDumpStats();
  if (!c || c->version != 1 || c->size < sizeof(Hctx) || !c->kargs || c->kn < sizeof(Args))
    return nullptr;
  std::memcpy(a, c->kargs, sizeof(Args));
  return Find(key(c, *a));
}

bool Ok(Hctx* c, const void* p, size_t n) { return c->ptrOk(p, n); }
int Tally(uint64_t* calls, Hctx* c, int r) {
  ++calls[r != 0 ? 2 : c->phase ? 1 : 0];
  return r;
}

double ElemBytes(const std::string& kname) {
  if (kname.find("_fp4") != std::string::npos) return 0.5;
  if (kname.find("_fp8") != std::string::npos) return 1;
  if (kname.find("_f32") != std::string::npos) return 4;
  return 2;
}

// ---- ops v1: the kernel argument is the EpDispatchCombineArgsRaw struct, by value ----
uint64_t V1Key(Hctx*, const EpDispatchCombineArgsRaw& a) {
  return reinterpret_cast<uintptr_t>(a.dispatchGridBarrier);
}

int V1Dispatch(Hctx* c) {
  EpDispatchCombineArgsRaw a;
  Op* op = Begin(c, &a, V1Key);
  if (!op) return 1;
  std::lock_guard<std::mutex> g(op->mu);
  const OpSpec& s = op->spec;
  const size_t k = s.topk, mr = s.maxRecv, mt = s.maxTok;
  Item it{};
  it.outIdx = a.shmemOutIndicesMemObj.cpu
                  ? static_cast<index_t*>(a.shmemOutIndicesMemObj.cpu->localPtr)
                  : nullptr;
  it.srcMap = a.dispTokIdToSrcTokIdMemObj.cpu
                  ? static_cast<index_t*>(a.dispTokIdToSrcTokIdMemObj.cpu->localPtr)
                  : nullptr;
  it.totalRecv = a.totalRecvTokenNum;
  it.destMap = a.dispDestTokIdMap;
  // routing replay is not simulated; unreadable outputs: an empty post keeps the ranks aligned
  if (a.replayMode || !Ok(c, it.outIdx, mr * k * 4) || !Ok(c, it.srcMap, mr * 4) ||
      !Ok(c, it.totalRecv, 4) || !Ok(c, it.destMap, mt * k * 4))
    return Tally(op->disp, c, PostEmpty(*op, c, 'D'));
  op->AddOutput(it.outIdx, mr * k * 4);
  op->AddOutput(it.srcMap, mr * 4);
  op->AddOutput(it.totalRecv, 4);
  op->AddOutput(it.destMap, mt * k * 4);
  const EpDispatchCombineConfig& cfg = a.config;
  const double rowBytes = cfg.hiddenDim * ElemBytes(c->kname) +
                          static_cast<double>(cfg.scaleDim) * cfg.scaleTypeSize +
                          cfg.numExpertPerToken * 8.0 + 4;
  return Tally(op->disp, c, PostDispatch(*op, c, a.tokenIndices, a.curRankNumToken, rowBytes, it));
}

int V1Combine(Hctx* c) {
  EpDispatchCombineArgsRaw a;
  Op* op = Begin(c, &a, V1Key);
  if (!op) return 1;
  std::lock_guard<std::mutex> g(op->mu);
  const std::string n = c->kname ? c->kname : "";
  double eb = ElemBytes(n), extra = 0;
  if (n.find("fp4_blockwise") != std::string::npos) {
    eb = 0.5;
    extra = a.fp8BlockwiseCombineScaleDim * 4.0;
  } else if (n.find("fp8_blockwise") != std::string::npos) {
    eb = 1;
    extra = a.fp8BlockwiseCombineScaleDim * 4.0;
  } else if (n.find("fp8cast") != std::string::npos) {
    eb = 1;
  }
  Item it{};
  it.totalRecv = a.totalRecvTokenNum;
  it.resetTotal = Ok(c, it.totalRecv, 4);
  return Tally(op->comb, c, PostCombine(*op, c, a.config.hiddenDim * eb + extra, it));
}

int LocalExpertCount(Hctx* c, Op* op, const index_t* idx, const index_t* total, int* out) {
  std::lock_guard<std::mutex> g(op->mu);
  const size_t bytes = op->spec.expertsPerRank * 4;
  if (!Ok(c, out, bytes) || !Ok(c, idx, 4)) return Tally(op->cnt, c, -1);
  op->AddOutput(out, bytes);
  Item it{'L'};
  it.lecIdx = idx;
  it.lecTotal = total;
  it.lecOut = out;
  Push(*op, c, it);
  return Tally(op->cnt, c, 0);
}

int V1LocalExpertCount(Hctx* c) {
  LocalExpertCountArgs a;
  Op* op = Begin(
      c, &a, +[](Hctx*, const LocalExpertCountArgs& x) {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x.totalRecvTokenNum));
      });
  return op ? LocalExpertCount(c, op, a.indices, a.totalRecvTokenNum, a.localExpertCount) : 1;
}

// ---- ops v2 (FlyDSL kernels): positional 8-byte pointer arguments ----
struct V2Args {
  uint64_t w[10];
};
uint64_t Word(Hctx* c, size_t off) {
  uint64_t v = 0;
  if (off + 8 <= c->kn) std::memcpy(&v, static_cast<const uint8_t*>(c->kargs) + off, 8);
  return v;
}

// ep_dispatch(arena, inp_tok, inp_idx, inp_wts, tok_map, dest_pe_ctr, disp_bar, total_recv,
// inp_scales, rank, n)
int V2Dispatch(Hctx* c) {
  V2Args a;
  Op* op = Begin(c, &a, +[](Hctx* h, const V2Args&) { return h->kn >= 80 ? Word(h, 48) : 0; });
  if (!op) return 1;
  std::lock_guard<std::mutex> g(op->mu);
  const OpSpec& s = op->spec;
  const size_t k = s.topk, mr = s.maxRecv, mt = s.maxTok;
  Item it{};
  it.outIdx = reinterpret_cast<index_t*>(s.outIdx);
  it.srcMap = reinterpret_cast<index_t*>(s.srcMap);
  it.totalRecv = reinterpret_cast<index_t*>(Word(c, 56));
  it.destMap = reinterpret_cast<index_t*>(Word(c, 32));
  if (!Ok(c, it.outIdx, mr * k * 4) || !Ok(c, it.srcMap, mr * 4) || !Ok(c, it.totalRecv, 4) ||
      !Ok(c, it.destMap, mt * k * 4))
    return Tally(op->disp, c, PostEmpty(*op, c, 'D'));
  op->AddOutput(it.outIdx, mr * k * 4);
  op->AddOutput(it.srcMap, mr * 4);
  op->AddOutput(it.totalRecv, 4);
  op->AddOutput(it.destMap, mt * k * 4);
  int32_t n = 0;
  std::memcpy(&n, static_cast<const uint8_t*>(c->kargs) + 76, 4);
  return Tally(
      op->disp, c,
      PostDispatch(*op, c, reinterpret_cast<index_t*>(Word(c, 16)), n, s.dispRowBytes, it));
}

// ep_combine(arena, tok_map, comb_bar, xdb_flag, total_recv, out, out_wts, rank, n) and
// ep_combine_s: found by any of the op's keys among the pointer words
int V2Combine(Hctx* c) {
  MaybeDumpStats();
  if (!c || c->version != 1 || !c->kargs) return 1;
  Op* op = nullptr;
  for (size_t off = 8; off + 8 <= c->kn && off < 128 && !op; off += 8) op = Find(Word(c, off));
  if (!op || op->spec.version != 2) return 1;
  std::lock_guard<std::mutex> g(op->mu);
  return Tally(op->comb, c, PostCombine(*op, c, op->spec.combRowBytes, Item{}));
}

// local_expert_count_kernel(out_idx, total_recv, count)
int V2LocalExpertCount(Hctx* c) {
  V2Args a;
  Op* op = Begin(c, &a, +[](Hctx* h, const V2Args&) { return h->kn >= 24 ? Word(h, 8) : 0; });
  return op ? LocalExpertCount(c, op, reinterpret_cast<index_t*>(Word(c, 0)),
                               reinterpret_cast<index_t*>(Word(c, 8)),
                               reinterpret_cast<int*>(Word(c, 16)))
            : 1;
}

void OnRead(void*, const void* p, size_t n, int block) {
  std::vector<Op*> ops;
  {
    std::lock_guard<std::mutex> g(gMu);
    for (auto& o : gOps) ops.push_back(o.get());
  }
  const uintptr_t lo = reinterpret_cast<uintptr_t>(p), hi = lo + std::max<size_t>(n, 1);
  for (Op* op : ops) {
    std::lock_guard<std::mutex> g(op->mu);
    if (op->q.empty()) continue;
    bool mine = block && !p;
    for (const auto& r : op->outputs) mine = mine || (block && lo < r.second && r.first < hi);
    const uint64_t t0 = NowNs();
    Drain(*op, nullptr, mine);
    if (mine) ++op->blockedReads, op->blockedNs += NowNs() - t0;
  }
}

std::string Calls(const uint64_t* v) {
  return "{\"eager\":" + std::to_string(v[0]) + ",\"replay\":" + std::to_string(v[1]) +
         ",\"error\":" + std::to_string(v[2]) + "}";
}

void DumpStats() {
  const char* dir = std::getenv("MORI_EP_SIM_STATS_DIR");
  if (!dir || !*dir) return;
  const std::string path = std::string(dir) + "/ep_sim." + std::to_string(getpid()) + ".json";
  if (FILE* f = std::fopen((path + ".tmp").c_str(), "w")) {
    std::fputs(Stats().c_str(), f);
    std::fclose(f);
    std::rename((path + ".tmp").c_str(), path.c_str());
  }
}

// Serving processes are usually killed, not exited: keep the stats file current (every 2 s at
// most).
void MaybeDumpStats() {
  static const bool on =
      std::getenv("MORI_EP_SIM_STATS_DIR") && *std::getenv("MORI_EP_SIM_STATS_DIR");
  static std::atomic<uint64_t> last{0};
  uint64_t t = last.load(), now = NowNs();
  if (on && now - t >= 2000000000ull && last.compare_exchange_strong(t, now)) DumpStats();
}

}  // namespace

bool Available() { return Rt().ok; }

void Register(const OpSpec& s) {
  const Api& rt = Rt();
  if (!rt.ok) throw std::runtime_error("MORI_EP_SIM_MODE=runtime needs a simulated HIP runtime");
  if (s.world < 1 || s.rank < 0 || s.rank >= s.world || s.topk < 1 || s.expertsPerRank < 1 ||
      s.maxTok < 1 || s.maxRecv < 1 || s.group.empty() || s.keys.empty())
    throw std::invalid_argument("EP sim: bad op spec");
  static std::once_flag once;
  std::call_once(once, [&] {
    // v1: IntraNode / IntraNodeLL launch one kernel per phase; the InterNode / AsyncLL sequences
    // are handled on the kernel that carries the transfer, with the intranode receive layout.
    const struct {
      const char* name;
      unsigned flags;
      Handler fn;
    } hs[] = {{"EpDispatchIntraNodeKernel_", kPrefix | kComm, V1Dispatch},
              {"EpDispatchIntraNodeLLKernel_", kPrefix | kComm, V1Dispatch},
              {"EpDispatchInterNodeV1Kernel", kPrefix | kComm, V1Dispatch},
              {"EpDispatchInterNodeKernel_", kPrefix | kComm, V1Dispatch},
              {"EpDispatchLowLatencyAsyncSendTransfer_", kPrefix | kComm, V1Dispatch},
              {"EpCombineIntraNodeKernel_", kPrefix | kComm, V1Combine},
              {"EpCombineInterNodeV1Kernel", kPrefix | kComm, V1Combine},
              {"EpCombineInterNodeKernel_", kPrefix | kComm, V1Combine},
              {"EpCombineLowLatencyAsyncSendTransfer_", kPrefix | kComm, V1Combine},
              {"LocalExpertCountKernel", 0, V1LocalExpertCount},
              {"ep_dispatch", kPrefix | kComm, V2Dispatch},
              {"ep_combine", kPrefix | kComm, V2Combine},
              {"local_expert_count_kernel", kPrefix, V2LocalExpertCount}};
    for (const auto& h : hs)
      if (rt.registerHandler(h.name, h.flags, h.fn, nullptr) != 0)
        throw std::runtime_error(std::string("EP sim: the runtime refused a handler for ") +
                                 h.name);
    rt.registerReadHook(OnRead, nullptr);
    std::atexit(DumpStats);
  });
  auto op = std::make_unique<Op>();
  op->spec = s;
  const char* routing = std::getenv("MORI_EP_SIM_ROUTING");
  if (!routing || std::string(routing) != "input")
    op->routing = std::make_unique<Routing>(s.world * s.expertsPerRank);
  const char* seed = std::getenv("MORI_EP_SIM_SEED");
  op->seed = seed && *seed ? std::strtoull(seed, nullptr, 10) : 0;
  op->lec.assign(s.expertsPerRank, 0);
  const size_t ints = 1 + s.world + static_cast<size_t>(s.maxTok) * s.topk;
  op->payload.resize(ints);
  const std::string key = "mori_ep_sim." + s.group;
  const char* node = std::getenv("MORI_NODE_ID");
  op->group =
      rt.a2aOpen(key.c_str(), s.world, s.rank,
                 node && *node ? std::atoi(node) : s.rank / std::max(1, s.gpuPerNode), ints * 4);
  if (!op->group) throw std::runtime_error("EP sim: the runtime refused all-to-all group " + key);
  op->AddOutput(&op->vals, sizeof(op->vals));
  std::lock_guard<std::mutex> g(gMu);
  if (rt.valueRegister) {
    const char* names[] = {"recv_rows", "experts_hit", "max_expert_rows"};
    const int32_t* vals[] = {&op->vals.recvRows, &op->vals.expertsHit, &op->vals.maxExpertRows};
    for (int i = 0; i < 3; ++i) {
      rt.valueRegister(("mori.ep.op" + std::to_string(gOps.size()) + "." + names[i]).c_str(),
                       vals[i], 1, 0);
      if (gOps.empty())
        rt.valueRegister((std::string("mori.ep.") + names[i]).c_str(), vals[i], 1, 0);
    }
  }
  if (const char* d = std::getenv("MORI_EP_SIM_LOG_DIR"); d && *d) {
    const std::string path =
        std::string(d) + "/ep_sim." + std::to_string(getpid()) + "." + s.group + ".log";
    if ((op->log = std::fopen(path.c_str(), "w"))) {
      std::setvbuf(op->log, nullptr, _IOLBF, 0);  // killed processes keep every line
      std::fprintf(op->log, "# seq phase n_own recv_rows lag_us sent_rows (rank %d)\n", s.rank);
    }
  }
  for (uintptr_t k : s.keys) gByKey[k] = op.get();
  gOps.push_back(std::move(op));
}

std::string Stats() {
  std::ostringstream o;
  o << "{\"pid\":" << getpid() << ",\"ops\":[";
  {
    std::lock_guard<std::mutex> g(gMu);
    for (size_t i = 0; i < gOps.size(); ++i) {
      Op& op = *gOps[i];
      std::lock_guard<std::mutex> h(op.mu);
      o << (i ? "," : "") << "{\"group\":\"" << op.spec.group
        << "\",\"version\":" << op.spec.version << ",\"rank\":" << op.spec.rank
        << ",\"world\":" << op.spec.world << ",\"dispatch\":" << Calls(op.disp)
        << ",\"combine\":" << Calls(op.comb) << ",\"local_expert_count\":" << Calls(op.cnt)
        << ",\"items\":" << op.items << ",\"deferred\":" << op.deferred
        << ",\"pending\":" << op.q.size() << ",\"max_pending\":" << op.maxPending
        << ",\"blocked_reads\":" << op.blockedReads << ",\"blocked_read_ns\":" << op.blockedNs
        << ",\"recv_rows\":" << op.recvRows << ",\"overflow\":" << op.overflow
        << ",\"missing_posts\":" << op.missing << ",\"empty_posts\":" << op.emptyPosts
        << ",\"lec_checked\":" << op.lecChecked << ",\"lec_mismatch\":" << op.lecMismatch << "}";
    }
  }
  o << "]";
  const Api& rt = Rt();
  if (rt.a2aStats && rt.a2aStatName) {  // the runtime's side, e.g. the ops' device time
    std::vector<uint64_t> v(std::max(rt.a2aStats(nullptr, 0), 0));
    rt.a2aStats(v.data(), static_cast<int>(v.size()));
    o << ",\"runtime_a2a\":{";
    for (size_t i = 0; i < v.size(); ++i)
      o << (i ? "," : "") << "\"" << rt.a2aStatName(i) << "\":" << v[i];
    o << "}";
  }
  o << "}";
  return o.str();
}

}  // namespace rt
}  // namespace sim
}  // namespace moe
}  // namespace mori
