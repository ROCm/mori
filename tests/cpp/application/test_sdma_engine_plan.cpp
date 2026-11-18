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

// Host-only tests of anvil's SDMA engine plan. The oracle is AnvilLib::connect()
// as it was before the plan: KFD's mask, the OAM table for a 14-engine mask, and
// the MI308X fold with per-link balancing.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include "mori/application/transport/sdma/sdma_engine_plan.hpp"

namespace {

using anvil::plan::Arch;
using anvil::plan::Bits;
using anvil::plan::Classify;
using anvil::plan::NodeCaps;
using anvil::plan::SourcePlan;

using Channels = std::map<int, std::vector<uint32_t>>;  // dst -> engine of each channel
using MaskFn = std::function<uint32_t(int src, int dst)>;
using OamFn = std::function<std::optional<uint32_t>(int src, int dst)>;

const NodeCaps kMi300x{Arch::Gfx9Xgmi, 2, 14, 8};  // also MI325X, MI355X
const NodeCaps kMi308x{Arch::Gfx9Xgmi, 2, 6, 8};
const NodeCaps kDpx{Arch::Gfx9Xgmi, 2, 6, 8};
const NodeCaps kQpx{Arch::Gfx9Xgmi, 2, 2, 8};
const NodeCaps kCpx{Arch::Gfx9Xgmi, 2, 0, 8};
const NodeCaps kGfx1250{Arch::Gfx125x, 16, 0, 6};

// AnvilLib::mi300xOamMap.
constexpr int kOam[8][8] = {{0, 7, 6, 1, 2, 4, 5, 3}, {7, 0, 1, 5, 4, 2, 3, 6},
                            {5, 1, 0, 6, 7, 3, 2, 4}, {1, 6, 5, 0, 3, 7, 4, 2},
                            {2, 4, 7, 3, 0, 5, 6, 1}, {4, 2, 3, 7, 6, 0, 1, 5},
                            {5, 3, 2, 4, 6, 1, 0, 7}, {3, 6, 4, 2, 1, 5, 7, 0}};

std::optional<uint32_t> OamEngine(int s, int d) { return kOam[s][d] * 2; }
std::optional<uint32_t> NoOam(int, int) { return std::nullopt; }

// recommended_sdma_engine_id_mask on an 8x MI308X host, KFD nodes 2..9.
const std::map<int, std::map<int, uint32_t>> kMi308xMasks = {
    {2, {{3, 0x40}, {4, 0xfc}, {5, 0x20}, {6, 0x80}, {7, 0x8}, {8, 0x10}, {9, 0x4}}},
    {3, {{2, 0x20}, {4, 0x40}, {5, 0xfc}, {6, 0x8}, {7, 0x80}, {8, 0x4}, {9, 0x10}}},
    {4, {{2, 0xfc}, {3, 0x40}, {5, 0x80}, {6, 0x10}, {7, 0x4}, {8, 0x20}, {9, 0x8}}},
    {5, {{2, 0x20}, {3, 0xfc}, {4, 0x80}, {6, 0x4}, {7, 0x10}, {8, 0x8}, {9, 0x40}}},
    {6, {{2, 0x80}, {3, 0x8}, {4, 0x10}, {5, 0x4}, {7, 0x20}, {8, 0xfc}, {9, 0x40}}},
    {7, {{2, 0x8}, {3, 0x80}, {4, 0x4}, {5, 0x10}, {6, 0x40}, {8, 0x20}, {9, 0xfc}}},
    {8, {{2, 0x10}, {3, 0x4}, {4, 0x20}, {5, 0x8}, {6, 0xfc}, {7, 0x40}, {9, 0x80}}},
    {9, {{2, 0x4}, {3, 0x10}, {4, 0x8}, {5, 0x40}, {6, 0x20}, {7, 0xfc}, {8, 0x80}}}};

uint32_t Mi308xMask(int s, int d) { return s == d ? 0 : kMi308xMasks.at(s).at(d); }
uint32_t BareMetalMask(int s, int d) { return s == d ? 0 : 1u << (kOam[s][d] * 2); }

// AnvilLib::connect() over the plan; `create` says whether KFD gives a queue.
struct Source {
  explicit Source(NodeCaps caps) : plan(caps) {}
  SourcePlan plan;
  Channels ch;

  bool Connect(int src, int dst, int n, uint32_t mask, std::optional<uint32_t> oam,
               const std::function<bool(uint32_t)>& create = nullptr) {
    const bool loopback = src == dst;
    if (loopback || !plan.WantsOam(mask)) oam.reset();
    auto& v = ch[dst];
    for (int c = static_cast<int>(v.size()); c < n; ++c) {
      bool ok = false;
      for (uint32_t e : plan.Candidates(dst, loopback, c, n, mask, oam)) {
        if (create && !create(e)) {
          plan.MarkFull(e);
          continue;
        }
        plan.Commit(dst, loopback, e);
        v.push_back(e);
        ok = true;
        break;
      }
      if (!ok) return false;
    }
    return true;
  }
};

// AnvilLib::connect() before the plan. False where it threw, or where KFD would
// have refused a queue (more than `cap` on an engine) and it exit()ed.
struct MainSource {
  bool mi308x = false;
  int cap = 8;
  std::array<int, 32> created{};
  std::array<int, 32> links{}, queues{};
  std::map<int, uint32_t> home;
  Channels ch;

  bool Connect(int src, int dst, int n, uint32_t mask, std::optional<uint32_t> oam) {
    std::vector<uint32_t> engines;
    if (src == dst) {
      engines.push_back(0);
    } else {
      engines = Bits(mask);
      if (engines.size() == 14 && oam && (mask & (1u << *oam))) engines.assign(1, *oam);
      if (engines.empty()) {
        if (!oam) return false;
        engines.push_back(*oam);
      }
    }
    if (mi308x) {
      std::vector<uint32_t> even;
      for (uint32_t e : engines)
        if (std::find(even.begin(), even.end(), e & ~1u) == even.end()) even.push_back(e & ~1u);
      engines.swap(even);
    }
    auto& v = ch[dst];
    if (static_cast<int>(v.size()) >= n) return true;
    const bool balance = mi308x && src != dst;
    const std::array<uint32_t, 3> xgmi = {2, 4, 6};
    if (balance && !home.count(dst)) {
      auto rank = [&](uint32_t e) {
        const bool notRec = std::find(engines.begin(), engines.end(), e) == engines.end();
        return std::make_tuple(links[e], notRec, e);
      };
      uint32_t best = xgmi[0];
      for (uint32_t e : xgmi)
        if (rank(e) < rank(best)) best = e;
      links[best]++;
      home[dst] = best;
    }
    for (int c = static_cast<int>(v.size()); c < n; ++c) {
      uint32_t e = engines[c % engines.size()];
      if (balance) {
        e = home[dst];
        if (queues[e] >= 8) {
          e = *std::min_element(xgmi.begin(), xgmi.end(),
                                [&](uint32_t a, uint32_t b) { return queues[a] < queues[b]; });
          if (queues[e] >= 8) return false;
        }
        queues[e]++;
      }
      if (++created[e] > cap) return false;
      v.push_back(e);
    }
    return true;
  }
};

// Every subset of `nodes` with at least two GPUs.
std::vector<std::vector<int>> Subsets(const std::vector<int>& nodes) {
  std::vector<std::vector<int>> out;
  for (uint32_t m = 0; m < (1u << nodes.size()); ++m) {
    std::vector<int> s;
    for (size_t i = 0; i < nodes.size(); ++i)
      if (m & (1u << i)) s.push_back(nodes[i]);
    if (s.size() >= 2) out.push_back(s);
  }
  return out;
}

std::vector<int> Range(int lo, int hi) {
  std::vector<int> v;
  for (int i = lo; i < hi; ++i) v.push_back(i);
  return v;
}

// Each rank connects to every rank of the job in rank order, itself included,
// once per entry of `counts` (a later, larger count adds channels).
template <typename S>
bool ConnectJob(S& s, int src, const std::vector<int>& job, const std::vector<int>& counts,
                const MaskFn& mask, const OamFn& oam) {
  for (int n : counts)
    for (int dst : job)
      if (!s.Connect(src, dst, n, mask(src, dst), oam(src, dst))) return false;
  return true;
}

// Plan and main agree wherever main succeeded; returns the number of cases compared.
int ExpectSameAsMain(NodeCaps caps, bool mi308x, const std::vector<int>& nodes, const MaskFn& mask,
                     const OamFn& oam, const std::vector<std::vector<int>>& countsList) {
  int compared = 0;
  for (const auto& job : Subsets(nodes)) {
    for (int src : job) {
      for (const auto& counts : countsList) {
        MainSource m;
        m.mi308x = mi308x;
        m.cap = caps.cap;
        if (!ConnectJob(m, src, job, counts, mask, oam)) continue;
        Source s(caps);
        EXPECT_TRUE(ConnectJob(s, src, job, counts, mask, oam));
        EXPECT_EQ(s.ch, m.ch) << "src " << src << ", " << job.size() << " GPUs, " << counts.back()
                              << " channels";
        if (s.ch != m.ch) return compared;
        ++compared;
      }
    }
  }
  return compared;
}

const std::vector<std::vector<int>> kCounts1to8 = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}};

TEST(SdmaEnginePlan, ClassifiesByGfxVersion) {
  EXPECT_EQ(Classify(90402), Arch::Gfx9Xgmi);  // gfx942
  EXPECT_EQ(Classify(90500), Arch::Gfx9Xgmi);  // gfx950
  EXPECT_EQ(Classify(120500), Arch::Gfx125x);  // gfx1250
  EXPECT_EQ(Classify(90010), Arch::Generic);   // gfx90a
  EXPECT_EQ(Classify(120001), Arch::Generic);  // gfx1201
  EXPECT_EQ(SourcePlan(kCpx).Caps().arch, Arch::Generic);
  EXPECT_EQ(SourcePlan(kQpx).Caps().arch, Arch::Gfx9Xgmi);
  EXPECT_EQ(SourcePlan(kDpx).Caps().arch, Arch::Gfx9Xgmi);
}

// One recommended engine per link: unchanged.
TEST(SdmaEnginePlan, BareMetalHiveIsUnchanged) {
  EXPECT_GT(ExpectSameAsMain(kMi300x, false, Range(0, 8), BareMetalMask, OamEngine, kCounts1to8),
            0);
}

// A VF lists all 14 xGMI engines; the OAM table picks one per link: unchanged.
TEST(SdmaEnginePlan, VfWithOamIsUnchanged) {
  auto mask = [](int s, int d) { return s == d ? 0u : 0xfffcu; };
  EXPECT_GT(ExpectSameAsMain(kMi300x, false, Range(0, 8), mask, OamEngine, kCounts1to8), 0);
}

TEST(SdmaEnginePlan, Mi308xIsUnchanged) {
  const std::vector<std::vector<int>> counts = {{1}, {2}, {3}, {4}, {6}, {8}, {2, 3}, {1, 3}};
  EXPECT_GT(ExpectSameAsMain(kMi308x, true, Range(2, 10), Mi308xMask, NoOam, counts), 0);
}

// gfx942/gfx950: peer queues only on even xGMI engines, links balanced, and a
// link's queues on one engine until it is full.
void ExpectGfx9Placement(NodeCaps caps, const std::vector<int>& nodes, const MaskFn& mask,
                         const OamFn& oam) {
  std::vector<uint32_t> pool;
  for (uint32_t e = caps.nGeneral; e < caps.nGeneral + caps.nXgmi; e += 2) pool.push_back(e);
  for (const auto& job : Subsets(nodes)) {
    for (int src : job) {
      for (int n = 1; n <= 8; ++n) {
        Source s(caps);
        // Fails only when the job asks for more queues than the even engines hold.
        const bool ok = ConnectJob(s, src, job, {n}, mask, oam);
        if (!ok) {
          EXPECT_GT(static_cast<int>(job.size() - 1) * n, static_cast<int>(pool.size()) * caps.cap);
          continue;
        }
        std::map<uint32_t, int> homes;
        for (const auto& [dst, engines] : s.ch) {
          if (dst == src) {
            EXPECT_EQ(engines, std::vector<uint32_t>(n, 0));
            continue;
          }
          const uint32_t home = *s.plan.HomeOf(dst);
          homes[home]++;
          for (uint32_t e : engines) {
            EXPECT_EQ(e % 2, 0u) << "odd engine " << e;
            EXPECT_NE(std::find(pool.begin(), pool.end(), e), pool.end());
            if (e != home) {
              EXPECT_EQ(s.plan.QueuesOn(home), caps.cap);
            }
          }
        }
        const int links = static_cast<int>(job.size()) - 1;
        const int fair =
            (links + static_cast<int>(pool.size()) - 1) / static_cast<int>(pool.size());
        for (const auto& [e, k] : homes) EXPECT_LE(k, fair) << "engine " << e;
        for (uint32_t e = 0; e < 32; ++e) EXPECT_LE(s.plan.QueuesOn(e), caps.cap);
      }
    }
  }
}

TEST(SdmaEnginePlan, VfWithoutOamUsesEvenEngines) {
  auto mask = [](int s, int d) { return s == d ? 0u : 0xfffcu; };
  ExpectGfx9Placement(kMi300x, Range(0, 8), mask, NoOam);
}

TEST(SdmaEnginePlan, DpxUsesEvenEngines) {
  auto mask = [](int s, int d) { return s == d ? 0u : 0xfcu; };
  ExpectGfx9Placement(kDpx, Range(0, 8), mask, NoOam);
}

TEST(SdmaEnginePlan, Mi308xUsesEvenEngines) {
  ExpectGfx9Placement(kMi308x, Range(2, 10), Mi308xMask, NoOam);
}

// QPX keeps one even xGMI engine, CPX none: KFD's mask folded to even engines.
TEST(SdmaEnginePlan, PartitionModesUseEvenEngines) {
  for (uint32_t pm : {0x3u, 0xcu, 0x8u}) {
    auto mask = [pm](int s, int d) { return s == d ? 0u : pm; };
    ExpectGfx9Placement(kQpx, Range(0, 4), mask, NoOam);
  }
  // CPX: a general-engine mask folds to engine 0; an xGMI one names no engine there.
  auto general = [](int s, int d) { return s == d ? 0u : 0x3u; };
  int checked = 0;
  for (const auto& job : Subsets(Range(0, 4))) {
    for (int n = 1; n <= 8; ++n) {
      Source s(kCpx);
      if (!ConnectJob(s, job[0], job, {n}, general, NoOam)) {
        EXPECT_GT(static_cast<int>(job.size()) * n, kCpx.cap);
        continue;
      }
      ++checked;
      for (const auto& [dst, engines] : s.ch)
        for (uint32_t e : engines) EXPECT_EQ(e, 0u);
    }
  }
  EXPECT_GT(checked, 0);
  EXPECT_TRUE(SourcePlan(kCpx).HasEngine(false, 0x3, std::nullopt));
  for (uint32_t pm : {0xcu, 0x8u})
    EXPECT_FALSE(SourcePlan(kCpx).HasEngine(false, pm, std::nullopt));
}

// gfx1250: a link's queues spread over an aligned block of engines.
TEST(SdmaEnginePlan, Gfx1250SpreadsLinksOverBlocks) {
  auto mask = [](int, int) { return 0xffffu; };  // peers; loopback spreads over the host link's
  for (int w : {2, 4, 8}) {
    for (int n : {1, 2, 4, 8}) {
      for (int src = 0; src < w; ++src) {
        Source s(kGfx1250);
        ASSERT_TRUE(ConnectJob(s, src, Range(0, w), {n}, mask, NoOam));
        std::set<uint32_t> used;
        bool disjoint = true;
        for (const auto& [dst, engines] : s.ch) {
          EXPECT_EQ(std::set<uint32_t>(engines.begin(), engines.end()).size(), engines.size());
          if (dst == src) continue;
          for (uint32_t e : engines) disjoint &= used.insert(e).second;
        }
        // Peers fit side by side: no two share an engine.
        if ((w - 1) * n <= 16 - n) {
          EXPECT_TRUE(disjoint) << w << " GPUs, " << n << " channels";
        }
        // AsyncLL (n = GPUs per node): peer d at offset o uses queue (d + o) % n, and
        // peers get distinct engines.
        for (int o = 0; n >= w && o < n; ++o) {
          std::set<uint32_t> engines;
          for (int d = 0; d < w; ++d) {
            if (d == src) continue;
            EXPECT_TRUE(engines.insert(s.ch[d][(d + o) % n]).second)
                << w << " GPUs, " << n << " channels, src " << src << ", offset " << o;
          }
        }
        int most = 0;
        for (uint32_t e = 0; e < 32; ++e) most = std::max(most, s.plan.QueuesOn(e));
        EXPECT_LE(most, (w * n + 15) / 16 + 1);
        EXPECT_LE(most, kGfx1250.cap);
      }
    }
  }
}

// gfx1250, a link asking for more queues later: they wrap over its block, so peers
// with their own block, or a block as large as the final count, keep distinct
// AsyncLL engines.
TEST(SdmaEnginePlan, Gfx1250GrowthStaysInBlock) {
  auto mask = [](int, int) { return 0xffffu; };
  const std::vector<std::tuple<int, std::vector<int>, bool>> cases = {
      {4, {1, 4}, true}, {4, {2, 4}, true},  {8, {2, 8}, true},
      {8, {8, 2}, true}, {8, {1, 8}, false}, {8, {4, 8}, false}};
  for (const auto& [w, counts, distinct] : cases) {
    const int n = *std::max_element(counts.begin(), counts.end());
    for (int src = 0; src < w; ++src) {
      Source s(kGfx1250);
      ASSERT_TRUE(ConnectJob(s, src, Range(0, w), counts, mask, NoOam));
      for (uint32_t e = 0; e < 32; ++e) EXPECT_LE(s.plan.QueuesOn(e), kGfx1250.cap);
      for (int o = 0; distinct && o < n; ++o) {
        std::set<uint32_t> engines;
        for (int d = 0; d < w; ++d) {
          if (d == src) continue;
          EXPECT_TRUE(engines.insert(s.ch[d][(d + o) % n]).second)
              << w << " GPUs, " << counts.front() << " then " << n << ", src " << src;
        }
      }
    }
  }
}

// ROCr or another process holds an engine's slots: the queue goes elsewhere.
TEST(SdmaEnginePlan, FullEngineIsSkipped) {
  Source s(kMi308x);
  auto no6 = [](uint32_t e) { return e != 6; };
  for (int dst = 2; dst < 10; ++dst)
    ASSERT_TRUE(s.Connect(2, dst, 2, Mi308xMask(2, dst), std::nullopt, no6));
  for (const auto& [dst, engines] : s.ch) {
    for (uint32_t e : engines) EXPECT_NE(e, 6u);
    if (dst != 2) {
      EXPECT_EQ(std::set<uint32_t>(engines.begin(), engines.end()).size(), 1u);
    }
  }

  Source g(kGfx1250);  // ROCr's blit queues fill engines 0 and 1
  auto from2 = [](uint32_t e) { return e >= 2; };
  ASSERT_TRUE(g.Connect(0, 0, 8, 0xffff, std::nullopt, from2));
  for (uint32_t e : g.ch[0]) EXPECT_GE(e, 2u);
  for (int dst = 1; dst < 8; ++dst) ASSERT_TRUE(g.Connect(0, dst, 8, 0xffff, std::nullopt, from2));
  for (uint32_t e = 0; e < 32; ++e) EXPECT_LE(g.plan.QueuesOn(e), kGfx1250.cap);

  Source lb(kMi300x);  // engine 0 full: loopback moves to even xGMI engines
  auto no0 = [](uint32_t e) { return e != 0; };
  ASSERT_TRUE(lb.Connect(0, 0, 8, 0, std::nullopt, no0));
  for (uint32_t e : lb.ch[0]) EXPECT_TRUE(e >= 2 && e % 2 == 0) << e;

  Source none(kMi300x);
  EXPECT_FALSE(none.Connect(0, 1, 1, 0xfffc, std::nullopt, [](uint32_t) { return false; }));
}

}  // namespace
