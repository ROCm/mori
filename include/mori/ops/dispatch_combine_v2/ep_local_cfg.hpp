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
// The EP local kernels: post-dispatch passes over this rank's own arena rows
// (local_expert_count, the StdMoE converts). No window, no peer: every pointer
// is local, so they get their own small Cfg and Args instead of EpCfg/EpArgs.
//
// HIP-free and attribute-free, like ep_cfg.hpp.

#pragma once

#include <cstddef>
#include <string>

#include "mori/jit/v2/render.hpp"
#include "mori/ops/dispatch_combine_v2/ep_cfg.hpp"

namespace mori {
namespace ops {
namespace v2 {

struct EpLocalCfg {
  int worldSize = 8;
  int hiddenDim = 7168;
  int maxTokPerRank = 128;
  int numExpertPerRank = 8;
  int numExpertPerToken = 8;
  int maxRecv = 0;  // 0 = worldSize * maxTokPerRank
  EpDType dtype = EpDType::Bf16;
  int blockNum = 64;
  int warpPerBlock = 16;
  int waveSize = 64;
};

template <typename Self, typename Visit>
inline void VisitFields(Self& c, const EpLocalCfg& d, Visit&& v) {
#define MORI_FIELD(x) v(#x, c.x, d.x)
  MORI_FIELD(worldSize);
  MORI_FIELD(hiddenDim);
  MORI_FIELD(maxTokPerRank);
  MORI_FIELD(numExpertPerRank);
  MORI_FIELD(numExpertPerToken);
  MORI_FIELD(maxRecv);
  MORI_FIELD(dtype);
  MORI_FIELD(blockNum);
  MORI_FIELD(warpPerBlock);
  MORI_FIELD(waveSize);
#undef MORI_FIELD
}

MORI_JIT_ASSERT_FIELD_COUNT(EpLocalCfg, 10,
                            "added an EpLocalCfg field -- update VisitFields(EpLocalCfg) too");

inline std::string Render(const EpLocalCfg& c) {
  const EpLocalCfg d{};
  mori::jit::v2::Fields f;
  VisitFields(c, d, [&f](const char* name, const auto& value, const auto& dflt) {
    f.Put(name, value, dflt);
  });
  return mori::jit::v2::BraceInit("EpLocalCfg", f);
}

inline std::string Describe(const EpLocalCfg& c) {
  std::string out;
  VisitFields(c, c, [&out](const char* name, const auto& value, const auto&) {
    using mori::jit::v2::RenderValue;
    out += name;
    out += "=";
    out += RenderValue(value);
    out += "\n";
  });
  return out;
}

// What a caller asks for: the Cfg less waveSize, which the arch decides.
struct EpLocalRequest {
  int worldSize = 8;
  int hiddenDim = 7168;
  int maxTokPerRank = 128;
  int numExpertPerRank = 8;
  int numExpertPerToken = 8;
  int maxRecv = 0;
  EpDType dtype = EpDType::Bf16;
  int blockNum = 64;
  int warpPerBlock = 16;
};

template <typename Self, typename Visit>
inline void VisitFields(Self& r, const EpLocalRequest& d, Visit&& v) {
#define MORI_FIELD(x) v(#x, r.x, d.x)
  MORI_FIELD(worldSize);
  MORI_FIELD(hiddenDim);
  MORI_FIELD(maxTokPerRank);
  MORI_FIELD(numExpertPerRank);
  MORI_FIELD(numExpertPerToken);
  MORI_FIELD(maxRecv);
  MORI_FIELD(dtype);
  MORI_FIELD(blockNum);
  MORI_FIELD(warpPerBlock);
#undef MORI_FIELD
}

MORI_JIT_ASSERT_FIELD_COUNT(EpLocalRequest, 9,
                            "added an EpLocalRequest field -- update VisitFields(EpLocalRequest)");

// Runtime arguments, one struct for every local kernel. No field is named `window`
// or off<Region>: the binding would take those for arena regions.
struct EpLocalArgs {
  const int* outIdx = nullptr;     // [maxRecv * topk] forwarded expert ids
  const int* totalRecv = nullptr;  // [1] rows dispatch delivered
  int* expertCount = nullptr;      // [numExpertPerRank] counts / packed_count
  const void* dispOut = nullptr;   // [maxRecv * hidden] dispatched tokens
  const int* recvToSrc = nullptr;  // [maxRecv] recv row -> source token
  const float* outWts = nullptr;   // [maxRecv * topk] forwarded weights
  void* outTok = nullptr;          // [maxRecv * hidden] combine input
  void* packedX = nullptr;         // [numExpertPerRank * EpLocalBucket * hidden]
  int* packedSrc = nullptr;        // [numExpertPerRank * EpLocalBucket]
  long long* slotMap = nullptr;    // [maxRecv * topk] packed row, -1 = not local
  int rank = 0;
};

#define MORI_EP_LOCAL_ARGS_FIELDS(X) \
  X(outIdx, "p")                     \
  X(totalRecv, "p")                  \
  X(expertCount, "p")                \
  X(dispOut, "p")                    \
  X(recvToSrc, "p")                  \
  X(outWts, "p")                     \
  X(outTok, "p")                     \
  X(packedX, "p")                    \
  X(packedSrc, "p")                  \
  X(slotMap, "p")                    \
  X(rank, "i32")

#define MORI_EP_LOCAL_ARGS_SCHEMA_ENTRY(name, tag) #name ":" tag ","
#define MORI_EP_LOCAL_ARGS_SCHEMA MORI_EP_LOCAL_ARGS_FIELDS(MORI_EP_LOCAL_ARGS_SCHEMA_ENTRY)

namespace detail {

#define MORI_EP_LOCAL_ARGS_OFFSET(name, tag) offsetof(::mori::ops::v2::EpLocalArgs, name),
inline constexpr size_t kEpLocalArgsOffsets[] = {
    MORI_EP_LOCAL_ARGS_FIELDS(MORI_EP_LOCAL_ARGS_OFFSET)};
#undef MORI_EP_LOCAL_ARGS_OFFSET

constexpr size_t kEpLocalArgsFieldCount =
    sizeof(kEpLocalArgsOffsets) / sizeof(kEpLocalArgsOffsets[0]);

constexpr bool EpLocalArgsOffsetsAscend() {
  for (size_t i = 1; i < kEpLocalArgsFieldCount; ++i)
    if (kEpLocalArgsOffsets[i] <= kEpLocalArgsOffsets[i - 1]) return false;
  return true;
}

}  // namespace detail

static_assert(detail::kEpLocalArgsFieldCount == 11,
              "added an EpLocalArgs field -- add it to MORI_EP_LOCAL_ARGS_FIELDS and bump this");
static_assert(detail::EpLocalArgsOffsetsAscend(),
              "MORI_EP_LOCAL_ARGS_FIELDS is not in EpLocalArgs declaration order");

constexpr int EpLocalMaxRecv(const EpLocalCfg& c) {
  return c.maxRecv > 0 ? c.maxRecv : c.worldSize * c.maxTokPerRank;
}

// Rows of one StdMoE expert bucket: every token of every rank, FlyDSL's layout.
constexpr int EpLocalBucket(const EpLocalCfg& c) { return c.worldSize * c.maxTokPerRank; }

constexpr int EpLocalBlockThreads(const EpLocalCfg& c) { return c.warpPerBlock * c.waveSize; }

constexpr int EpLocalMaxExperts = 8192;

constexpr bool EpLocalCfgIsValid(const EpLocalCfg& c) {
  return c.worldSize > 0 && c.hiddenDim > 0 && c.maxTokPerRank > 0 && c.numExpertPerRank > 0 &&
         c.numExpertPerToken > 0 && c.maxRecv >= 0 && c.blockNum > 0 && c.warpPerBlock > 0 &&
         (c.waveSize == 32 || c.waveSize == 64) && EpLocalBlockThreads(c) <= 1024 &&
         // One lane per top-k slot fills the per-warp source pointers.
         c.numExpertPerToken <= c.waveSize &&
         // The count's per-block histogram lives in LDS.
         c.numExpertPerRank <= EpLocalMaxExperts &&
         // Rows move as whole 16 B vectors.
         (c.hiddenDim * EpElemSize(c.dtype)) % 16 == 0;
}

}  // namespace v2
}  // namespace ops
}  // namespace mori
