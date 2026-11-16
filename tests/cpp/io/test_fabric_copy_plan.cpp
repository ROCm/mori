// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// CPU-only: also runnable with c++ -std=c++17 -I. this.cpp copy_plan.cpp.
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "src/io/fabric/copy_plan.hpp"

using namespace mori::io::fabric;
#define CHECK(x)                            \
  do {                                      \
    if (!(x)) throw std::runtime_error(#x); \
  } while (false)

void CheckCoverage(const CopyPlan& plan, uint32_t maxBlocks) {
  size_t offset = 0;
  for (const auto& dispatch : plan.dispatches) {
    std::visit(
        [&](const auto& a) {
          uint64_t chunks = 0;
          for (uint32_t i = 0; i < a.numSegs; ++i) {
            const auto& seg = plan.segments.at(offset++);
            CHECK(a.src[i] == seg.src && a.dst[i] == seg.dst && a.size[i] == seg.size);
            CHECK(a.size[i] != 0);
            uint64_t chunk = a.chunkShift == 64 ? UINT64_MAX : uint64_t{1} << a.chunkShift;
            chunks += a.chunkShift == 64 ? 1 : seg.size / chunk + (seg.size % chunk != 0);
            CHECK(chunks <= UINT32_MAX && chunks == a.chunkEnd[i]);
          }
          CHECK(dispatch.grid == std::min<uint64_t>(chunks, maxBlocks));
          CHECK(dispatch.grid > 0);
        },
        dispatch.args);
  }
  CHECK(offset == plan.segments.size());
}

int main() {
  try {
    CHECK(PlanCopy({}, 1536, true).dispatches.empty());
    CHECK(PlanCopy({{1, 2, 0}, {3, 4, 0}}, 1536, true).segments.empty());
    CHECK(PlanCopy({{1, 2, 0}, {32, 64, 17}}, 1536, true).segments.size() == 1);
    for (size_t count : {1, 16, 17, 512, 513, 5000}) {
      std::vector<CopySegment> segs;
      for (size_t i = 0; i < count; ++i) segs.push_back({i * 8192, i * 16384, 4096});
      auto plan = PlanCopy(segs, 1536, true);
      CHECK(plan.dispatches.size() == (count + 511) / 512);
      for (size_t i = 0; i < plan.dispatches.size(); ++i) {
        bool small = std::min<size_t>(count - i * 512, 512) <= 16;
        CHECK(std::holds_alternative<FabricCopyArgs<kSmallK>>(plan.dispatches[i].args) == small);
      }
      CheckCoverage(plan, 1536);
      std::reverse(segs.begin(), segs.end());
      CheckCoverage(PlanCopy(segs, 1536, true), 1536);
    }
    for (uint64_t size : {1ull, 4095ull, 4096ull, 4097ull, 8193ull, 1ull << 33, 1ull << 40}) {
      auto plan = PlanCopy({{0, 0, size}}, 1536, true);
      CheckCoverage(plan, 1536);
    }
    constexpr uint64_t boundary = 1536 * kMinChunk;
    for (uint64_t bytes : {boundary - 1, boundary, boundary + 1}) {
      auto plan = PlanCopy({{0, 0, bytes}}, 1536, true);
      auto a = std::get<FabricCopyArgs<kSmallK>>(plan.dispatches[0].args);
      CHECK(a.chunkShift == (bytes <= boundary ? 12 : 13));
      CHECK(plan.dispatches[0].grid == (bytes <= boundary ? 1536 : 769));
    }
    std::vector<CopySegment> mixed{{0, 0, uint64_t{1} << 30}};
    for (uint64_t i = 0; i < 511; ++i)
      mixed.push_back({(uint64_t{1} << 30) + i * 2 + 1, (uint64_t{1} << 30) + i * 2 + 1, 1});
    auto mixedPlan = PlanCopy(mixed, 1536, true);
    auto mixedArgs = std::get<FabricCopyArgs<kLargeK>>(mixedPlan.dispatches[0].args);
    CHECK(mixedArgs.chunkShift == 20 && mixedArgs.chunkEnd[511] == 1535);
    CheckCoverage(mixedPlan, 1536);
    auto prefixes = PlanCopy({{0, 0, 4095}, {8192, 8192, 4096}, {16384, 16384, 4097}}, 1536, true);
    auto a = std::get<FabricCopyArgs<kSmallK>>(prefixes.dispatches[0].args);
    CHECK(a.chunkEnd[0] == 1 && a.chunkEnd[1] == 2 && a.chunkEnd[2] == 4);
    auto split =
        PlanCopy({{0, 0, (uint64_t{UINT32_MAX} - 1) * 4096}, {1, 1, 1}, {3, 3, 1}, {5, 5, 1}},
                 UINT32_MAX, true);
    CHECK(split.dispatches.size() == 2);
    CheckCoverage(split, UINT32_MAX);
    // Total bytes can also exceed uint64_t even when individual segments do not.
    auto wide = PlanCopy({{0, 0, UINT64_MAX}, {1, 1, UINT64_MAX}}, 1536, true);
    CHECK(wide.dispatches.size() == 2);
    CheckCoverage(wide, 1536);
    auto maximal = PlanCopy({{0, 0, UINT64_MAX}}, 1, true);
    CHECK(std::get<FabricCopyArgs<kSmallK>>(maximal.dispatches[0].args).chunkShift == 64);
    CheckCoverage(maximal, 1);

    auto merge = PlanCopy(
        {{32, 132, 8}, {0, 100, 16}, {16, 116, 16}, {40, 200, 8}, {80, 208, 8}, {7, 110, 0}}, 1536,
        true);
    CHECK(merge.segments.size() == 3 && merge.segments[0].size == 40);
    CheckCoverage(merge, 1536);
    auto read = PlanCopy({{16, 116, 16}, {0, 100, 16}}, 1536, true, true);
    CHECK(read.segments.size() == 1 && read.segments[0].size == 32);
    constexpr uint64_t mib = 1024 * 1024;
    CHECK(PlanCopy({{0, 0, 4 * mib - 1}}, 1536, false).error.empty());
    CHECK(!PlanCopy({{0, 0, 4 * mib}}, 1536, false).error.empty());
    auto fallback = PlanCopy({{0, 0, 3 * mib}, {3 * mib, 3 * mib, 3 * mib}}, 1536, false);
    CHECK(fallback.segments.size() == 1 && !fallback.error.empty());
    CHECK(fallback.dispatches.empty());
    CHECK(fallback.error.find("livelock") != std::string::npos);
    CHECK(PlanCopy({{0, 0, 3 * mib}, {4 * mib, 4 * mib, 3 * mib}}, 1536, false).error.empty());
    bool invalid = false;
    try {
      (void)PlanCopy({}, 0, true);
    } catch (const std::invalid_argument&) {
      invalid = true;
    }
    CHECK(invalid);
    std::cout << "PASS: Fabric copy planner\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
