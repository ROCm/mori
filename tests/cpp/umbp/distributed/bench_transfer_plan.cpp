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

// Plan-building cost of one layer-wise ranged restore: the buffered Plan()
// against the incremental plan builder.
//
// WHAT IT MODELS
//
// The local half of PoolClient::BatchGetRanges restoring a long cached prefix
// from the DRAM tier into a GPU KV cache, stopped before anything is submitted:
// the copies are not what changed, the work that produces the plan set is.  The
// shape is one KV block per key and one range per attention layer per key, as
// sglang's umbp_direct_linker builds them.  The defaults are moonshotai/Kimi-K3
// at TP8 with an FP8 KV cache, a 999,936-token prefix:
//
//   * 24 of its 93 layers are MLA (linear_attn_config.full_attn_layers).  The
//     other 69 are KDA, whose state is per request rather than per token and is
//     left out: one block's worth is a few hundred fragments next to ~10^5.
//   * An MLA token is kv_lora_rank + qk_rope_head_dim = 576 bytes in FP8, and
//     TP8 replicates the latent cache, so every rank restores all of it.
//   * A hybrid model's attention block is sized so one block covers a KDA state
//     page (~814 KB per layer per rank here), which lands on 1536 tokens; the
//     prefix is then exactly 651 blocks.
//   * The DRAM tier is one buffer and each GPU layer tensor is its own
//     registered region, so the restore spans 24 (src, dst) pairs.
//
// Items are built the way BuildLocalRangeTransfers builds them (one
// registered-region lookup per range, one TransferItem per page fragment), and
// the engine is the composite PoolClient assembles (LocalCopyEngine, then
// HbmCopyEngine).  Planning reads descriptors only, so every address is
// synthetic and nothing is allocated on, or copied to, a GPU.
//
// WHICH ENGINE PLANS IT
//
// The local half of BatchGetRanges reaches three engines.  One that overrides
// CreatePlanBuilder plans as items arrive; one that does not sits behind the
// default buffered builder, which still holds every item until Finish.
//
//   --dst=gpu       DRAM tier -> GPU: HbmCopyEngine (the default).
//   --dst=cpu       DRAM tier -> host memory: LocalCopyEngine.
//   --file-pct=N    N% of keys live in SSD-tier files and are built the way
//                   BuildLocalFileRangeTransfers builds them, one item per
//                   range, for GdsEngine.  100 is an all-SSD restore; anything
//                   lower is a mixed batch, which takes the composite's
//                   partitioning path.  GPU destinations only, as for GDS.
//
// GdsEngine is compiled only where the hipfile headers are.  Without them the
// file engine is a stand-in whose CanHandle and Plan are GdsEngine's, verbatim;
// neither depends on how items reach Plan, which is all a builder changes.
// MoriIoEngine is not reachable from the local half and is not modelled.
//
// TWO FLOWS
//
//   buffered     every fragment of every key into one vector, then Plan().
//   incremental  one key's fragments at a time, fed to a TransferPlanBuilder.
//
// The same source builds against a tree with and without the builder; the
// incremental flow is compiled in only where TransferEngine::CreatePlanBuilder
// exists.  Every iteration prints one CSV row, including a checksum of the plan
// set, so two trees can be checked for building the same plans.
//
//   bench_umbp_transfer_plan --page-kib=32 --desc=rdma --tree=pr758

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "umbp/common/host_registration.h"
#include "umbp/distributed/range_map.h"
#include "umbp/distributed/transfer/composite_transfer_engine.h"
#include "umbp/distributed/transfer/hbm_copy_engine.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp/distributed/transfer/transfer_engine.h"
#ifdef UMBP_ENABLE_GDS
#include "umbp/distributed/transfer/gds_engine.h"
#endif

namespace {

using mori::io::MemoryLocationType;
using mori::umbp::CompositeTransferEngine;
using mori::umbp::ForEachRangePageFragment;
using mori::umbp::HbmCopyEngine;
using mori::umbp::LocalCopyEngine;
using mori::umbp::PageLocation;
using mori::umbp::TransferDirection;
using mori::umbp::TransferEngine;
using mori::umbp::TransferHandle;
using mori::umbp::TransferItem;
using mori::umbp::TransferPlan;
using mori::umbp::TransferPlanSet;
using mori::umbp::TransferRef;
using Clock = std::chrono::steady_clock;

#ifdef UMBP_ENABLE_GDS
using FileEngine = mori::umbp::GdsEngine;
constexpr const char* kFileEngine = "GdsEngine";
#else
// GdsEngine's CanHandle and Plan, verbatim (gds_engine.cpp), for builds without
// the hipfile headers.  Nothing here is submitted.
class FileEngine final : public TransferEngine {
 public:
  const char* Name() const override { return "GdsEngine"; }

  TransferRef RegisterMemory(void*, size_t, MemoryLocationType, int) override {
    return TransferRef{};
  }
  void Deregister(const TransferRef&) override {}

  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override {
    return src.IsFile() && !dst.IsFile() && dst.HasHostPtr() && IsGpu(dst);
  }

  TransferPlanSet Plan(const std::vector<TransferItem>& items) const override {
    TransferPlanSet out;
    out.plans.reserve(items.size());
    for (const auto& item : items) {
      if (item.size == 0) continue;
      if (!CanHandle(item.src, item.dst)) {
        out.rejected_tags.push_back(item.tag);
        continue;
      }
      if (item.size > item.dst.size || item.dst_offset > item.dst.size - item.size) {
        out.rejected_tags.push_back(item.tag);
        continue;
      }
      TransferPlan plan;
      plan.src = item.src;
      plan.dst = item.dst;
      plan.dir = TransferDirection::kLocal;
      plan.src_offsets.push_back(static_cast<size_t>(item.src_offset));
      plan.dst_offsets.push_back(static_cast<size_t>(item.dst_offset));
      plan.sizes.push_back(static_cast<size_t>(item.size));
      plan.tags.push_back(item.tag);
      out.plans.push_back(std::move(plan));
    }
    return out;
  }

  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan>) override { return nullptr; }

 private:
  static bool IsGpu(const TransferRef& r) { return r.loc == MemoryLocationType::GPU; }
};
constexpr const char* kFileEngine = "GdsEngine stand-in (no hipfile headers)";
#endif

template <class Engine, class = void>
struct HasPlanBuilder : std::false_type {};
template <class Engine>
struct HasPlanBuilder<
    Engine, std::void_t<decltype(std::declval<const Engine&>().CreatePlanBuilder(size_t{}))>>
    : std::true_type {};

struct Options {
  size_t tokens = 999936;
  size_t layers = 24;
  size_t token_bytes = 576;
  size_t block_tokens = 1536;
  size_t page_kib = 64;
  // Fill each TransferRef's MemoryDesc the way a pool registered with mori-io
  // does.  Its three std::strings make every TransferItem copy allocate.
  bool descriptors = false;
  // Place pages at random slots instead of in allocation order, as a long-lived
  // pool's bitmap allocator would; fragments then stop coalescing.
  bool scattered = false;
  // Restore into host memory (LocalCopyEngine) instead of a GPU KV cache.
  bool cpu_dst = false;
  // Share of keys served from SSD-tier files (GdsEngine), 0..100.
  size_t file_pct = 0;
  int warmup = 3;
  int iters = 12;
  std::string tree = "?";
};

bool ParseSize(const char* arg, const char* name, size_t* out) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) != 0 || arg[n] != '=') return false;
  *out = std::strtoull(arg + n + 1, nullptr, 10);
  return true;
}

bool ParseString(const char* arg, const char* name, std::string* out) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) != 0 || arg[n] != '=') return false;
  *out = arg + n + 1;
  return true;
}

Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    size_t v = 0;
    std::string s;
    if (ParseSize(argv[i], "--tokens", &o.tokens) || ParseSize(argv[i], "--layers", &o.layers) ||
        ParseSize(argv[i], "--token-bytes", &o.token_bytes) ||
        ParseSize(argv[i], "--block-tokens", &o.block_tokens) ||
        ParseSize(argv[i], "--page-kib", &o.page_kib) ||
        ParseSize(argv[i], "--file-pct", &o.file_pct) || ParseString(argv[i], "--tree", &o.tree)) {
      continue;
    }
    if (ParseSize(argv[i], "--warmup", &v)) {
      o.warmup = static_cast<int>(v);
    } else if (ParseSize(argv[i], "--iters", &v)) {
      o.iters = static_cast<int>(v);
    } else if (ParseString(argv[i], "--desc", &s) && (s == "none" || s == "rdma")) {
      o.descriptors = (s == "rdma");
    } else if (ParseString(argv[i], "--pages", &s) && (s == "sequential" || s == "scattered")) {
      o.scattered = (s == "scattered");
    } else if (ParseString(argv[i], "--dst", &s) && (s == "gpu" || s == "cpu")) {
      o.cpu_dst = (s == "cpu");
    } else {
      std::fprintf(stderr,
                   "usage: %s [--tokens=N] [--layers=N] [--token-bytes=N] [--block-tokens=N]\n"
                   "          [--page-kib=N] [--desc=none|rdma] [--pages=sequential|scattered]\n"
                   "          [--dst=gpu|cpu] [--file-pct=0..100]\n"
                   "          [--warmup=N] [--iters=N] [--tree=LABEL]\n",
                   argv[0]);
      std::exit(2);
    }
  }
  if (o.block_tokens == 0 || o.tokens % o.block_tokens != 0) {
    std::fprintf(stderr, "--tokens must be a positive multiple of --block-tokens\n");
    std::exit(2);
  }
  if (o.file_pct > 100 || (o.file_pct > 0 && o.cpu_dst)) {
    std::fprintf(stderr, "--file-pct must be 0..100, and file keys need --dst=gpu (as GDS does)\n");
    std::exit(2);
  }
  return o;
}

// Synthetic and never dereferenced; 1 GiB aligned like a hugetlb mapping, so
// host registration pieces fall where a real pool's would.
constexpr uintptr_t kDramBase = uintptr_t{0x7e00} << 32;
constexpr uintptr_t kUserBase = uintptr_t{0x7f00} << 32;
constexpr uint64_t kUserAlign = uint64_t{2} << 20;
// SSD tier: objects packed into 1 GiB segment files, one fd per segment.
constexpr uint64_t kSegmentBytes = uint64_t{1} << 30;
constexpr int kFirstSegmentFd = 1000;

// PoolClient::ObjectRange.
struct Range {
  void* user;
  size_t size;
  size_t object_offset;
};

// PoolClient::RegisteredRegion.
struct Region {
  uintptr_t base;
  uint64_t size;
  TransferRef ref;
};

struct World {
  uint64_t page_size = 0;
  uint64_t object_size = 0;
  size_t layer_bytes = 0;  // one layer of one block
  TransferRef dram;        // PageBackend::BufferRef(0)
  std::vector<Region> regions;
  std::vector<std::vector<PageLocation>> pages;  // per key
  // Resolved file_ref per key; invalid for a key the DRAM tier holds.
  std::vector<TransferRef> file_refs;
  // The batch_get_ranges_into_ptr arguments, per key.
  std::vector<std::vector<void*>> dsts;
  std::vector<std::vector<size_t>> sizes;
  std::vector<std::vector<size_t>> offsets;
};

mori::io::MemoryDesc Describe(uintptr_t base, uint64_t size, MemoryLocationType loc, int device,
                              uint64_t id) {
  mori::io::MemoryDesc d;
  d.engineKey = "smci355-ccs-aus-n06-21";  // a node id, past libstdc++'s 15-byte SSO
  d.id = id;
  d.deviceId = device;
  d.deviceBusId = device >= 0 ? "0000:75:00.0" : "";
  d.data = base;
  d.size = size;
  d.loc = loc;
  d.numaNode = 0;
  return d;
}

World MakeWorld(const Options& o) {
  World w;
  w.page_size = uint64_t{o.page_kib} << 10;
  w.layer_bytes = o.block_tokens * o.token_bytes;
  w.object_size = uint64_t{w.layer_bytes} * o.layers;
  const size_t keys = o.tokens / o.block_tokens;
  const size_t pages_per_object =
      static_cast<size_t>((w.object_size + w.page_size - 1) / w.page_size);
  const size_t total_pages = keys * pages_per_object;

  w.dram = TransferRef::HostBytes(reinterpret_cast<void*>(kDramBase), total_pages * w.page_size);
  if (o.descriptors) w.dram.mem = Describe(kDramBase, w.dram.size, MemoryLocationType::CPU, -1, 1);

  // One tensor per layer, sized for the prefix plus the null block 0.
  const MemoryLocationType user_loc = o.cpu_dst ? MemoryLocationType::CPU : MemoryLocationType::GPU;
  const int user_device = o.cpu_dst ? -1 : 0;
  const uint64_t region_bytes = uint64_t{keys + 1} * w.layer_bytes;
  const uint64_t stride = (region_bytes + kUserAlign - 1) / kUserAlign * kUserAlign;
  for (size_t l = 0; l < o.layers; ++l) {
    const uintptr_t base = kUserBase + l * stride;
    Region r{
        base, region_bytes,
        TransferRef::HostBytes(reinterpret_cast<void*>(base), region_bytes, user_loc, user_device)};
    if (o.descriptors) r.ref.mem = Describe(base, region_bytes, user_loc, user_device, 2 + l);
    w.regions.push_back(std::move(r));
  }

  // Spread file keys evenly through the batch: key k is on SSD when it falls in
  // the first file_pct of each run of 100.
  const size_t keys_per_segment = std::max<size_t>(1, kSegmentBytes / w.object_size);
  w.file_refs.resize(keys);
  for (size_t k = 0; k < keys; ++k) {
    if (k % 100 >= o.file_pct) continue;
    const int fd = kFirstSegmentFd + static_cast<int>(k / keys_per_segment);
    w.file_refs[k] = TransferRef::File(fd, (k % keys_per_segment) * w.object_size, w.object_size,
                                       reinterpret_cast<void*>(uintptr_t{0x5000} + fd));
  }

  std::vector<uint32_t> slots(total_pages);
  std::iota(slots.begin(), slots.end(), 0u);
  if (o.scattered) std::shuffle(slots.begin(), slots.end(), std::mt19937(758));

  w.pages.resize(keys);
  w.dsts.resize(keys);
  w.sizes.resize(keys);
  w.offsets.resize(keys);
  for (size_t k = 0; k < keys; ++k) {
    for (size_t p = 0; p < pages_per_object; ++p) {
      w.pages[k].push_back(PageLocation{0, slots[k * pages_per_object + p]});
    }
    // Block k + 1: a fresh engine hands out consecutive block ids after the null block.
    for (size_t l = 0; l < o.layers; ++l) {
      w.dsts[k].push_back(reinterpret_cast<void*>(w.regions[l].base + (k + 1) * w.layer_bytes));
      w.sizes[k].push_back(w.layer_bytes);
      w.offsets[k].push_back(l * w.layer_bytes);
    }
  }
  return w;
}

// PoolClient::FindRegisteredRegionLocked.
const Region* FindRegion(const World& w, const void* p, size_t n) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  auto it = std::upper_bound(w.regions.begin(), w.regions.end(), a,
                             [](uintptr_t v, const Region& r) { return v < r.base; });
  if (it == w.regions.begin()) return nullptr;
  --it;
  if (a + n > it->base + it->size) return nullptr;
  return &*it;
}

// FillReadRanges in pool_client.cpp.
void FillReadRanges(const std::vector<void*>& dsts, const std::vector<size_t>& sizes,
                    const std::vector<size_t>& offsets, std::vector<Range>* out) {
  out->clear();
  out->reserve(dsts.size());
  for (size_t r = 0; r < dsts.size(); ++r) out->push_back({dsts[r], sizes[r], offsets[r]});
}

// PoolClient::BuildLocalRangeTransfers with to_backend = false.
bool BuildLocalRangeTransfers(const World& w, size_t key, const std::vector<Range>& ranges,
                              std::vector<TransferItem>* items) {
  std::vector<TransferRef> user_refs;
  std::vector<uint64_t> user_offsets;
  user_refs.reserve(ranges.size());
  user_offsets.reserve(ranges.size());
  for (const auto& range : ranges) {
    const Region* reg = FindRegion(w, range.user, range.size);
    if (reg == nullptr) return false;
    user_refs.push_back(reg->ref);
    user_offsets.push_back(reinterpret_cast<uintptr_t>(range.user) - reg->base);
  }
  for (size_t r = 0; r < ranges.size(); ++r) {
    const auto& range = ranges[r];
    const TransferRef& user_ref = user_refs[r];
    const uint64_t user_base = user_offsets[r];
    const bool walked = ForEachRangePageFragment(
        w.pages[key], w.page_size, w.object_size, range.object_offset, range.size,
        [&](size_t /*page_index*/, uint64_t tier_offset, size_t fragment, size_t copied) {
          TransferRef buf = w.dram;
          TransferItem item;
          item.size = fragment;
          item.tag = key;
          item.src = std::move(buf);
          item.src_offset = tier_offset;
          item.dst = user_ref;
          item.dst_offset = user_base + copied;
          items->push_back(std::move(item));
          return true;
        });
    if (!walked) return false;
  }
  return true;
}

// PoolClient::UserBufferRef: the registered ref and the offset into it, by value.
std::pair<TransferRef, uint64_t> UserBufferRef(const World& w, void* ptr, size_t size) {
  if (const Region* reg = FindRegion(w, ptr, size)) {
    return {reg->ref, reinterpret_cast<uintptr_t>(ptr) - reg->base};
  }
  return {TransferRef::HostBytes(ptr, size), 0};
}

// PoolClient::BuildLocalFileRangeTransfers: one item per range, no page walk.
bool BuildLocalFileRangeTransfers(const World& w, const TransferRef& file_ref,
                                  const std::vector<Range>& ranges, size_t tag,
                                  std::vector<TransferItem>* items) {
  if (!file_ref.IsFile() || ranges.empty()) return false;
  items->reserve(items->size() + ranges.size());
  for (const Range& r : ranges) {
    if (r.size == 0) continue;
    const auto [dst_ref, dst_base_offset] = UserBufferRef(w, r.user, r.size);
    TransferItem item;
    item.src = file_ref;
    item.src_offset = r.object_offset;
    item.dst = dst_ref;
    item.dst_offset = dst_base_offset;
    item.size = r.size;
    item.tag = tag;
    items->push_back(std::move(item));
  }
  return true;
}

// The per-key dispatch in BatchGetRanges: a file-backed key skips the page walk.
bool BuildKeyItems(const World& w, size_t key, const std::vector<Range>& ranges,
                   std::vector<TransferItem>* items) {
  return w.file_refs[key].IsFile()
             ? BuildLocalFileRangeTransfers(w, w.file_refs[key], ranges, key, items)
             : BuildLocalRangeTransfers(w, key, ranges, items);
}

long MinorFaults() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_minflt;
}

double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Sample {
  double build_ms = 0;
  double plan_ms = 0;
  double total_ms = 0;
  double teardown_ms = 0;
  long faults = 0;
  size_t items = 0;
  size_t plans = 0;
  size_t segments = 0;
  size_t rejected = 0;
  size_t peak_item_bytes = 0;  // the largest TransferItem vector held at once
  uint64_t checksum = 0;
};

void Mix(uint64_t* h, uint64_t v) {
  *h ^= v;
  *h *= 1099511628211ULL;
}

// Order-insensitive across plans: the buffered composite partitions a mixed
// batch through an unordered_map, so its plans come back in hash order, while
// the builder keeps first-appearance order.  Submit does not care which.
void Summarize(const TransferPlanSet& planned, Sample* s) {
  std::vector<uint64_t> per_plan;
  per_plan.reserve(planned.plans.size());
  for (const auto& plan : planned.plans) {
    uint64_t h = 1469598103934665603ULL;
    Mix(&h, reinterpret_cast<uintptr_t>(plan.src.host_ptr));
    Mix(&h, reinterpret_cast<uintptr_t>(plan.dst.host_ptr));
    Mix(&h, static_cast<uint64_t>(plan.src.file_fd));
    Mix(&h, plan.src.file_offset);
    Mix(&h, std::hash<std::string>{}(plan.engine != nullptr ? plan.engine->Name() : ""));
    for (size_t i = 0; i < plan.sizes.size(); ++i) {
      Mix(&h, plan.src_offsets[i]);
      Mix(&h, plan.dst_offsets[i]);
      Mix(&h, plan.sizes[i]);
    }
    for (size_t tag : plan.tags) Mix(&h, tag);
    per_plan.push_back(h);
    s->segments += plan.sizes.size();
  }
  std::sort(per_plan.begin(), per_plan.end());
  std::vector<size_t> rejected = planned.rejected_tags;
  std::sort(rejected.begin(), rejected.end());
  uint64_t h = 1469598103934665603ULL;
  for (uint64_t v : per_plan) Mix(&h, v);
  for (size_t tag : rejected) Mix(&h, ~uint64_t{tag});
  s->plans = planned.plans.size();
  s->rejected = planned.rejected_tags.size();
  s->checksum = h;
}

size_t RangeTotal(const World& w) {
  size_t n = 0;
  for (const auto& per_key : w.sizes) n += per_key.size();
  return n;
}

// BatchGetRanges before the builder: build everything, then plan once.
Sample RunBuffered(const CompositeTransferEngine& engine, const World& w) {
  Sample s;
  const long f0 = MinorFaults();
  const auto t0 = Clock::now();
  std::vector<TransferItem> local_items;
  local_items.reserve(RangeTotal(w));
  std::vector<Range> key_ranges;
  for (size_t i = 0; i < w.dsts.size(); ++i) {
    const size_t before = local_items.size();
    FillReadRanges(w.dsts[i], w.sizes[i], w.offsets[i], &key_ranges);
    if (!BuildKeyItems(w, i, key_ranges, &local_items)) local_items.resize(before);
  }
  const auto t1 = Clock::now();
  TransferPlanSet planned = engine.Plan(local_items);
  const auto t2 = Clock::now();
  s.items = local_items.size();
  s.peak_item_bytes = local_items.capacity() * sizeof(TransferItem);
  Summarize(planned, &s);
  const auto t3 = Clock::now();
  {
    std::vector<TransferItem> drop_items = std::move(local_items);
    TransferPlanSet drop_plans = std::move(planned);
  }
  const auto t4 = Clock::now();
  s.build_ms = Ms(t0, t1);
  s.plan_ms = Ms(t1, t2);
  s.total_ms = Ms(t0, t2);
  s.teardown_ms = Ms(t3, t4);
  s.faults = MinorFaults() - f0;
  return s;
}

// BatchGetRanges with the builder: buffer one key, then hand it to the builder.
template <class Engine>
Sample RunIncremental(const Engine& engine, const World& w) {
  Sample s;
  if constexpr (HasPlanBuilder<Engine>::value) {
    const long f0 = MinorFaults();
    const auto t0 = Clock::now();
    Clock::duration plan{};
    auto plan_builder = engine.CreatePlanBuilder(RangeTotal(w));
    std::vector<TransferItem> key_items;
    std::vector<Range> key_ranges;
    for (size_t i = 0; i < w.dsts.size(); ++i) {
      key_items.clear();
      FillReadRanges(w.dsts[i], w.sizes[i], w.offsets[i], &key_ranges);
      if (!BuildKeyItems(w, i, key_ranges, &key_items)) {
        key_items.clear();
        continue;
      }
      const auto a0 = Clock::now();
      for (const auto& item : key_items) plan_builder->Add(item);
      plan += Clock::now() - a0;
      s.items += key_items.size();
      s.peak_item_bytes = std::max(s.peak_item_bytes, key_items.capacity() * sizeof(TransferItem));
    }
    const auto f_begin = Clock::now();
    TransferPlanSet planned = plan_builder->Finish();
    const auto t2 = Clock::now();
    plan += t2 - f_begin;
    Summarize(planned, &s);
    const auto t3 = Clock::now();
    {
      std::vector<TransferItem> drop_items = std::move(key_items);
      auto drop_builder = std::move(plan_builder);
      TransferPlanSet drop_plans = std::move(planned);
    }
    const auto t4 = Clock::now();
    s.plan_ms = std::chrono::duration<double, std::milli>(plan).count();
    s.total_ms = Ms(t0, t2);
    s.build_ms = s.total_ms - s.plan_ms;
    s.teardown_ms = Ms(t3, t4);
    s.faults = MinorFaults() - f0;
  } else {
    (void)engine;
    (void)w;
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = Parse(argc, argv);
  const World w = MakeWorld(o);

  // PoolClient's order; the file engine is registered only where GDS is usable,
  // so it joins only when some key is file-backed.
  CompositeTransferEngine engine;
  engine.AddEngine(std::make_unique<LocalCopyEngine>());
  engine.AddEngine(std::make_unique<HbmCopyEngine>());
  if (o.file_pct > 0) engine.AddEngine(std::make_unique<FileEngine>());

  const bool incremental = HasPlanBuilder<CompositeTransferEngine>::value;
  std::printf(
      "# tree=%s incremental_api=%s sizeof(TransferItem)=%zu keys=%zu ranges=%zu "
      "object_bytes=%llu page_bytes=%llu host_piece_bytes=%zu dst=%s file_engine=%s\n",
      o.tree.c_str(), incremental ? "yes" : "no", sizeof(TransferItem), w.dsts.size(),
      RangeTotal(w), static_cast<unsigned long long>(w.object_size),
      static_cast<unsigned long long>(w.page_size), mori::umbp::HostRegistrationChunkBytes(),
      o.cpu_dst ? "cpu(LocalCopyEngine)" : "gpu(HbmCopyEngine)",
      o.file_pct > 0 ? kFileEngine : "-");
  std::printf(
      "tree,flow,tokens,block_tokens,page_kib,desc,pages,dst,file_pct,iter,build_ms,plan_ms,"
      "total_ms,teardown_ms,minor_faults,items,plans,segments,rejected,peak_item_mib,checksum\n");

  std::vector<std::string> flows = {"buffered"};
  if (incremental) flows.push_back("incremental");
  for (int it = -o.warmup; it < o.iters; ++it) {
    // Alternate which flow goes first so drift lands on both alike.
    std::reverse(flows.begin(), flows.end());
    for (const auto& flow : flows) {
      const Sample s = flow == "buffered" ? RunBuffered(engine, w) : RunIncremental(engine, w);
      if (it < 0) continue;
      std::printf(
          "%s,%s,%zu,%zu,%zu,%s,%s,%s,%zu,%d,%.3f,%.3f,%.3f,%.3f,%ld,%zu,%zu,%zu,%zu,%.2f,%"
          "016llx\n",
          o.tree.c_str(), flow.c_str(), o.tokens, o.block_tokens, o.page_kib,
          o.descriptors ? "rdma" : "none", o.scattered ? "scattered" : "sequential",
          o.cpu_dst ? "cpu" : "gpu", o.file_pct, it, s.build_ms, s.plan_ms, s.total_ms,
          s.teardown_ms, s.faults, s.items, s.plans, s.segments, s.rejected,
          s.peak_item_bytes / 1048576.0, static_cast<unsigned long long>(s.checksum));
    }
    std::fflush(stdout);
  }
  return 0;
}
