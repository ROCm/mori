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
#pragma once

// Which SDMA engine each anvil queue goes on. Pure (no sysfs, no hsakmt) so it can
// be tested on a host without GPUs; AnvilLib::connect() feeds it KFD's topology.
//
// Gfx9Xgmi (gfx942, gfx950; also as SR-IOV VFs):
//   - Peer links use the even xGMI engines only. A copy started by a doorbell that
//     device code rings never runs on an odd engine (MI300X VF, MI355X, MI355X VF,
//     MI308X, where the odd ids are also fused off), so a recommended odd engine
//     stands for its even partner (id & ~1).
//   - Each link gets one home engine: a usable one with the fewest of this GPU's
//     peer links, ties to a recommended engine, then the lowest id. All of a link's
//     queues share it: AsyncLL uses one queue per peer per op, and spreading a
//     link's queues measured 28-53% slower on MI308X. On a bare-metal 8-GPU hive
//     KFD already gives every link its own engine, so nothing changes there.
//   - When KFD lists every xGMI engine for a link (a VF, a hive of another size)
//     on a 2 + 14 engine part, the static OAM table supplies the preference.
//   - Loopback stays on engine 0, moving to an xGMI engine only when 0 is full.
//   - A partition mode without xGMI engines (CPX) uses KFD's mask folded to even ids.
// Gfx125x (gfx1250): all 16 engines are general and every link may use them all.
//   One engine cannot fill a link there, so a link's queues spread over
//   consecutive engines. Links take aligned blocks of bit_ceil(queues) engines, so
//   AsyncLL's queue (destPe + offset) % n gives distinct peers distinct engines.
//   The block is sized by the link's first connect; queues added later wrap over it.
// Generic (any other arch): KFD's mask, channel c on engine c % n, as before.
// In every class a queue that cannot be created marks its engine full and the next
// candidate is tried; queue counts only order the candidates, because KFD's slots
// are shared with ROCr and other processes.

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace anvil {
namespace plan {

enum class Arch { Generic, Gfx9Xgmi, Gfx125x };

// gfx_target_version, as libhsakmt splits it: major = v / 10000 % 100, minor = v / 100 % 100.
inline Arch Classify(uint64_t gfxTargetVersion) {
  const uint64_t major = gfxTargetVersion / 10000 % 100;
  const uint64_t minor = gfxTargetVersion / 100 % 100;
  if (major == 9 && (minor == 4 || minor == 5)) return Arch::Gfx9Xgmi;
  if (major == 12 && minor == 5) return Arch::Gfx125x;
  return Arch::Generic;
}

inline const char* ArchName(Arch a) {
  switch (a) {
    case Arch::Gfx9Xgmi:
      return "gfx9-xgmi";
    case Arch::Gfx125x:
      return "gfx125x";
    default:
      return "generic";
  }
}

constexpr uint32_t kMaxEngines = 32;

// One GPU's SDMA layout, from its KFD node properties.
struct NodeCaps {
  Arch arch = Arch::Generic;
  uint32_t nGeneral = 0;  // num_sdma_engines: engines 0..nGeneral-1
  uint32_t nXgmi = 0;     // num_sdma_xgmi_engines: the next nXgmi ids
  int cap = 0;            // num_sdma_queues_per_engine; 0 = unknown
};

inline std::vector<uint32_t> Bits(uint32_t mask) {
  std::vector<uint32_t> v;
  for (uint32_t b = 0; b < kMaxEngines; ++b)
    if (mask & (1u << b)) v.push_back(b);
  return v;
}

// The engine choices of one source GPU. Not thread-safe; AnvilLib holds its lock.
class SourcePlan {
 public:
  explicit SourcePlan(NodeCaps caps) : caps_(caps) {
    if (caps_.arch == Arch::Gfx9Xgmi) {
      for (uint32_t e = caps_.nGeneral; e < caps_.nGeneral + caps_.nXgmi && e < kMaxEngines; ++e)
        if (!(e & 1u)) pool_.push_back(e);
      // CPX has no xGMI engine: KFD's mask, folded to even ids.
      if (pool_.empty()) {
        caps_.arch = Arch::Generic;
        evenOnly_ = true;
      }
    }
  }

  const NodeCaps& Caps() const { return caps_; }

  // Whether the OAM-table engine matters for a peer link with this KFD mask: the
  // mask is empty, or (on a 2 + 14 part) lists every xGMI engine.
  bool WantsOam(uint32_t mask) const {
    if (mask == 0) return true;
    return caps_.arch == Arch::Gfx9Xgmi && caps_.nGeneral == 2 && caps_.nXgmi == 14 &&
           mask == XgmiMask();
  }

  // Engines to try, in order, for channel `channel` of the link to `dst` (the
  // link's first call fixes its home from `nRequested`). `mask` is KFD's
  // recommendation (loopback: the engines to spread over, 0 for engine 0); `oam`
  // the static OAM-table engine, when WantsOam(mask).
  std::vector<uint32_t> Candidates(int dst, bool loopback, int channel, int nRequested,
                                   uint32_t mask, std::optional<uint32_t> oam) {
    if (loopback) return LoopbackCandidates(channel, mask);
    switch (caps_.arch) {
      case Arch::Gfx9Xgmi:
        return Gfx9Candidates(dst, mask, oam);
      case Arch::Gfx125x:
        if (mask) return Gfx125xCandidates(dst, channel, nRequested, Bits(mask));
        [[fallthrough]];
      default:
        return GenericCandidates(channel, mask, oam);
    }
  }

  // Whether the link has any engine at all, full or not.
  bool HasEngine(bool loopback, uint32_t mask, std::optional<uint32_t> oam) const {
    if (loopback || caps_.arch == Arch::Gfx9Xgmi || (caps_.arch == Arch::Gfx125x && mask))
      return true;
    return !GenericEngines(mask, oam).empty();
  }

  // A queue was created on `e` for the link to `dst`.
  void Commit(int dst, bool loopback, uint32_t e) {
    queues_[e]++;
    if (loopback) return;
    auto it = links_.find(dst);
    if (it == links_.end()) return;
    Link& l = it->second;
    // Gfx9: the home could not take the link's first queue, so the link moves where
    // it went. A gfx1250 link keeps its aligned block.
    if (caps_.arch == Arch::Gfx9Xgmi && l.queues == 0 && e != l.home) {
      linksOn_[l.home]--;
      linksOn_[e]++;
      l.home = e;
    }
    l.queues++;
  }

  // Creating a queue on `e` failed: KFD has no slot left there.
  void MarkFull(uint32_t e) { full_ |= 1u << e; }

  int QueuesOn(uint32_t e) const { return queues_[e]; }
  int LinksOn(uint32_t e) const { return linksOn_[e]; }
  std::optional<uint32_t> HomeOf(int dst) const {
    auto it = links_.find(dst);
    if (it == links_.end()) return std::nullopt;
    return it->second.home;
  }

 private:
  struct Link {
    uint32_t home = 0;
    int queues = 0;
    size_t block = 1;  // Gfx125x: engines in the link's block
  };

  uint32_t XgmiMask() const {
    uint32_t m = 0;
    for (uint32_t e = caps_.nGeneral; e < caps_.nGeneral + caps_.nXgmi && e < kMaxEngines; ++e)
      m |= 1u << e;
    return m;
  }

  bool Usable(uint32_t e) const {
    if (full_ & (1u << e)) return false;
    return caps_.cap <= 0 || queues_[e] < caps_.cap;
  }

  // Queues on `e`; a full engine counts as at the cap, or as far beyond any real
  // load when the cap is unknown.
  int Load(uint32_t e) const {
    if (Usable(e)) return queues_[e];
    return caps_.cap > 0 ? std::max(caps_.cap, queues_[e]) : (1 << 16);
  }

  // `pool` by fewest queues, then id, keeping usable engines only.
  std::vector<uint32_t> ByLoad(std::vector<uint32_t> pool) const {
    std::stable_sort(pool.begin(), pool.end(), [&](uint32_t a, uint32_t b) {
      return std::make_tuple(queues_[a], a) < std::make_tuple(queues_[b], b);
    });
    std::vector<uint32_t> out;
    for (uint32_t e : pool)
      if (Usable(e)) out.push_back(e);
    return out;
  }

  std::vector<uint32_t> Gfx9Candidates(int dst, uint32_t mask, std::optional<uint32_t> oam) {
    auto it = links_.find(dst);
    if (it == links_.end()) {
      // Preferred engines: the recommendation folded onto even ids, or the OAM
      // engine when KFD lists them all.
      uint32_t preferred = 0;
      if (WantsOam(mask)) {
        if (oam && *oam < kMaxEngines) preferred = 1u << *oam;
      } else {
        for (uint32_t e : Bits(mask)) preferred |= 1u << (e & ~1u);
      }
      uint32_t best = pool_[0];
      auto rank = [&](uint32_t e) {
        return std::make_tuple(!Usable(e), linksOn_[e], !(preferred & (1u << e)), e);
      };
      for (uint32_t e : pool_)
        if (rank(e) < rank(best)) best = e;
      linksOn_[best]++;
      it = links_.emplace(dst, Link{best, 0}).first;
    }
    const uint32_t home = it->second.home;
    std::vector<uint32_t> out;
    if (Usable(home)) out.push_back(home);
    for (uint32_t e : ByLoad(pool_))
      if (e != home) out.push_back(e);
    return out;
  }

  std::vector<uint32_t> Gfx125xCandidates(int dst, int channel, int nRequested,
                                          const std::vector<uint32_t>& ring) {
    const size_t n = ring.size();
    auto it = links_.find(dst);
    if (it == links_.end()) {
      size_t block = 1;
      while (block < static_cast<size_t>(std::max(1, nRequested))) block <<= 1;
      block = std::min(block, n);
      // Rank block starts by the load in the block, then links homed there.
      size_t bestStart = 0;
      auto rank = [&](size_t start) {
        int load = 0;
        for (size_t i = 0; i < block; ++i) load += Load(ring[(start + i) % n]);
        return std::make_tuple(load, linksOn_[ring[start]], start);
      };
      for (size_t start = 0; start < n; start += block)
        if (rank(start) < rank(bestStart)) bestStart = start;
      linksOn_[ring[bestStart]]++;
      it = links_.emplace(dst, Link{ring[bestStart], 0, block}).first;
    }
    const size_t homeIdx = std::find(ring.begin(), ring.end(), it->second.home) - ring.begin();
    const size_t first = homeIdx + static_cast<size_t>(channel) % it->second.block;
    std::vector<uint32_t> out;
    for (size_t i = 0; i < n; ++i) {
      const uint32_t e = ring[(first + i) % n];
      if (Usable(e)) out.push_back(e);
    }
    return out;
  }

  std::vector<uint32_t> LoopbackCandidates(int channel, uint32_t mask) {
    std::vector<uint32_t> out;
    if (mask != 0) {  // gfx1250: spread over the given engines, channel c on engine c % n
      const auto ring = Bits(mask);
      for (size_t i = 0; i < ring.size(); ++i) {
        const uint32_t e = ring[(static_cast<size_t>(channel) + i) % ring.size()];
        if (Usable(e)) out.push_back(e);
      }
      return out;
    }
    if (Usable(0)) out.push_back(0);
    // Gfx9: an xGMI engine when engine 0 is full; never engine 1 (odd).
    if (caps_.arch == Arch::Gfx9Xgmi)
      for (uint32_t e : ByLoad(pool_)) out.push_back(e);
    return out;
  }

  // The engines of a Generic link: KFD's mask, or the OAM engine for an empty mask.
  std::vector<uint32_t> GenericEngines(uint32_t mask, std::optional<uint32_t> oam) const {
    std::vector<uint32_t> engines = Bits(mask);
    if (engines.empty() && oam && *oam < kMaxEngines) engines.push_back(*oam);
    if (!evenOnly_) return engines;
    std::vector<uint32_t> even;
    for (uint32_t e : engines)
      if ((e & ~1u) < caps_.nGeneral + caps_.nXgmi &&
          std::find(even.begin(), even.end(), e & ~1u) == even.end())
        even.push_back(e & ~1u);
    return even;
  }

  std::vector<uint32_t> GenericCandidates(int channel, uint32_t mask,
                                          std::optional<uint32_t> oam) const {
    const std::vector<uint32_t> engines = GenericEngines(mask, oam);
    std::vector<uint32_t> out;
    for (size_t i = 0; i < engines.size(); ++i) {
      const uint32_t e = engines[(static_cast<size_t>(channel) + i) % engines.size()];
      if (Usable(e)) out.push_back(e);
    }
    return out;
  }

  NodeCaps caps_;
  std::vector<uint32_t> pool_;  // Gfx9Xgmi: the even xGMI engines
  bool evenOnly_ = false;       // a gfx9 part without xGMI engines
  std::array<int, kMaxEngines> queues_{};
  std::array<int, kMaxEngines> linksOn_{};
  uint32_t full_ = 0;
  std::unordered_map<int, Link> links_;
};

}  // namespace plan
}  // namespace anvil
