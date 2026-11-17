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
          using Args = std::decay_t<decltype(a)>;
          CHECK(a.numSegs > 0);
          CHECK((std::is_same_v<Args, FabricCopyArgs<kSingleK>>) == (a.numSegs == 1));
          CHECK((std::is_same_v<Args, FabricCopyArgs<kSmallK>>) ==
                (a.numSegs >= 2 && a.numSegs <= 16));
          CHECK((std::is_same_v<Args, FabricCopyArgs<kLargeK>>) ==
                (a.numSegs >= 17 && a.numSegs <= 512));
          uint64_t chunks = 0;
          for (uint32_t i = 0; i < a.numSegs; ++i) {
            const auto& seg = plan.segments.at(offset++);
            CHECK(a.src[i] == seg.src && a.dst[i] == seg.dst && a.size[i] == seg.size);
            CHECK(a.size[i] != 0);
            if (a.numSegs == 1) continue;  // Flat copies have no chunk prefix.
            uint64_t chunk = a.chunkShift == 64 ? UINT64_MAX : uint64_t{1} << a.chunkShift;
            chunks += a.chunkShift == 64 ? 1 : seg.size / chunk + (seg.size % chunk != 0);
            CHECK(chunks <= UINT32_MAX && chunks == a.chunkEnd[i]);
          }
          if (a.numSegs > 1) CHECK(dispatch.grid == std::min<uint64_t>(chunks, maxBlocks));
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
    for (size_t count : {1, 2, 16, 17, 512, 513, 5000}) {
      std::vector<CopySegment> segs;
      for (size_t i = 0; i < count; ++i) segs.push_back({i * 8192, i * 16384, 4096});
      auto plan = PlanCopy(segs, 1536, true);
      CHECK(plan.dispatches.size() == (count + 511) / 512);
      for (size_t i = 0; i < plan.dispatches.size(); ++i) {
        size_t n = std::min<size_t>(count - i * 512, 512);
        bool small = n >= 2 && n <= 16;
        CHECK(std::holds_alternative<FabricCopyArgs<kSingleK>>(plan.dispatches[i].args) ==
              (n == 1));
        CHECK(std::holds_alternative<FabricCopyArgs<kSmallK>>(plan.dispatches[i].args) == small);
        CHECK(std::holds_alternative<FabricCopyArgs<kLargeK>>(plan.dispatches[i].args) == (n > 16));
      }
      CheckCoverage(plan, 1536);
      std::reverse(segs.begin(), segs.end());
      CheckCoverage(PlanCopy(segs, 1536, true), 1536);
    }
    // Explicit old-kernel grid boundaries, including ceil(size / 16), the cap,
    // >4 GiB, and UINT64_MAX without overflowing a ceil-division addition.
    struct GridCase {
      uint64_t bytes;
      uint32_t grid;
    };
    for (auto test : {GridCase{1, 1},
                      {15, 1},
                      {16, 1},
                      {4095, 1},
                      {4096, 1},
                      {4097, 2},
                      {8192, 2},
                      {8193, 3},
                      {65534ull * 4096 + 1, 65535},
                      {65535ull * 4096, 65535},
                      {65535ull * 4096 + 1, 65535},
                      {(1ull << 32) + 1, 65535},
                      {1ull << 40, 65535},
                      {UINT64_MAX, 65535}}) {
      for (uint32_t maxBlocks : {1u, 1536u, UINT32_MAX}) {
        auto plan = PlanCopy({{17, 19, test.bytes}}, maxBlocks, true);
        CHECK(plan.dispatches.size() == 1 && plan.dispatches[0].grid == test.grid);
        CheckCoverage(plan, maxBlocks);
      }
    }
    constexpr uint64_t boundary = 1536 * kMinChunk;
    for (uint64_t bytes : {boundary - 1, boundary, boundary + 1}) {
      // Keep two noncontiguous segments to exercise adaptive multi-segment chunks.
      auto plan = PlanCopy({{0, 0, bytes - 1}, {bytes + 16, bytes + 16, 1}}, 1536, true);
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
    // Trimming reaches one segment when its rounded chunk count already fills
    // uint32. Still larger segments must also terminate through the flat path.
    constexpr uint64_t huge = (uint64_t{UINT32_MAX} - 1) * 4096 + 1;
    for (uint64_t size : {huge, uint64_t{UINT32_MAX} * 4096 + 1, UINT64_MAX - 1}) {
      auto single = PlanCopy({{0, 0, size}}, UINT32_MAX, true);
      CHECK(single.dispatches.size() == 1 && single.dispatches[0].grid == 65535);
      CheckCoverage(single, UINT32_MAX);
      auto trimmed = PlanCopy({{0, 0, size}, {1, 1, 1}}, UINT32_MAX, true);
      CHECK(!trimmed.dispatches.empty());
      if (size == huge) {
        CHECK(trimmed.dispatches.size() == 2);
        CHECK(std::holds_alternative<FabricCopyArgs<kSingleK>>(trimmed.dispatches[0].args));
      }
      CheckCoverage(trimmed, UINT32_MAX);
    }
    // Total bytes can also exceed uint64_t even when individual segments do not.
    auto wide = PlanCopy({{0, 0, UINT64_MAX}, {1, 1, UINT64_MAX}}, 1536, true);
    CHECK(wide.dispatches.size() == 2);
    CheckCoverage(wide, 1536);
    auto maximal = PlanCopy({{0, 0, UINT64_MAX}}, 1, true);
    CHECK(std::holds_alternative<FabricCopyArgs<kSingleK>>(maximal.dispatches[0].args));
    CheckCoverage(maximal, 1);
    auto maximalChunks = PlanCopy({{0, 0, UINT64_MAX - 1}, {1, 1, 1}}, 1, true);
    CHECK(std::get<FabricCopyArgs<kSmallK>>(maximalChunks.dispatches[0].args).chunkShift == 64);
    CheckCoverage(maximalChunks, 1);

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
