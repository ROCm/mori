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

// ---------------------------------------------------------------------------
// The internode EP kernels' argument surface.
//
// Same shape as EpArgs (ep_cfg.hpp), for the same reasons: one arena window
// handle, one byte offset per region, the local buffers as plain pointers, and a
// `name:tag` schema published from the field list so the Python binding builds
// its struct from what C++ declares instead of keeping a parallel copy. Nothing
// here crosses as an opaque blob except the device communicator, which is cco's
// struct rather than EP's.
//
// Offsets are runtime fields, not Cfg constants. EpArgs records why, and it
// applies unchanged here: as constants they make every arena layout its own
// binary, and they measured SLOWER on gfx942 (VGPR 9 -> 22) because the compiler
// stops treating the base as uniform and rematerialises the address per lane.
//
// Nothing in this file names anything from v1. That is the point: this header
// and the kernel that includes it used to reach dispatch_combine.hpp for the
// argument struct, the config, the index helpers and index_t, which cost 219
// transitive headers on every JIT compile and carried a struct whose tail is
// #ifdef'd on build macros the JIT toolchain never defines -- so the host library
// and the device module could disagree about its size with nothing to catch it.
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

// mori/core/utils/utils.hpp defines `warpSize` as an object-like macro, while
// cco.hpp declares mori::cco::impl::warpSize(). Whichever lands first wins, so
// including cco from EP code is otherwise include-order dependent. Hide the
// macro across the cco header and restore it after.
#pragma push_macro("warpSize")
#undef warpSize
#include "mori/cco/cco.hpp"
#pragma pop_macro("warpSize")

namespace mori {
namespace ops {
namespace v2 {

// v1 spells these index_t and QuantType. Declared here so this header owns them.
using ep_index_t = int32_t;

enum class EpQuantType {
  None = 0,
  Fp8DirectCast = 1,
  Fp8BlockwiseQuant = 2,
  Fp4BlockwiseQuant = 3,
};

// ---------------------------------------------------------------------------
// A region view: (arena window, byte offset) plus what turns a world rank into
// an LSA one. Built on the fly from the flat args below, never stored -- which
// is why it carries no size and why the three members are the three things an
// access actually needs.
// ---------------------------------------------------------------------------
struct EpInterNodeRegion {
  uint64_t win{0};
  uint64_t off{0};
  // World rank of LSA rank 0 on this node (myWorldRank - myLsaRank).
  //
  // GetAs(pe) takes a WORLD rank because that is what the kernel computes and
  // what v1's peer table was indexed by; ccoGetLsaPeerPtr takes an LSA rank. On
  // one node the two coincide and the difference is invisible. At two nodes
  // world rank 12 is LSA rank 4, and using it unconverted reads eight slots past
  // this rank's window -- silently, because that VA belongs to another region.
  int32_t lsaBase{0};

  __device__ __host__ const EpInterNodeRegion* operator->() const { return this; }
  __device__ __host__ bool IsValid() const { return win != 0; }

// Device-only: ccoGetLocalPtr / ccoGetLsaPeerPtr are themselves declared inside
// cco.hpp's `#if defined(__HIPCC__)`, so a plain C++ TU cannot name them. Host
// code only ever fills offsets, never dereferences through them.
#if defined(__HIPCC__) || defined(__CUDACC__)
  template <typename T>
  __device__ __forceinline__ T GetAs() const {
    return reinterpret_cast<T>(::mori::cco::ccoGetLocalPtr(
        reinterpret_cast<::mori::cco::ccoWindow_t>(win), static_cast<size_t>(off)));
  }

  // The caller must already have established that `worldPe` is on this node --
  // every call site in the kernel sits behind a `destNode != myNode` skip, and
  // cross-node traffic goes out as (window, offset) through the RDMA path
  // instead. A cross-node rank here does not fault, it lands somewhere else in
  // the flat window, which is why that skip is load-bearing.
  template <typename T>
  __device__ __forceinline__ T GetAs(int worldPe) const {
    return reinterpret_cast<T>(
        ::mori::cco::ccoGetLsaPeerPtr(reinterpret_cast<::mori::cco::ccoWindow_t>(win),
                                      worldPe - lsaBase, static_cast<size_t>(off)));
  }

  __device__ __forceinline__ void* Get() const { return GetAs<void*>(); }
  __device__ __forceinline__ void* Get(int worldPe) const { return GetAs<void*>(worldPe); }
#endif
};

// ---------------------------------------------------------------------------
// The shape the kernel reads as `args.config`.
//
// Thirteen scalars and the derived counts the bodies call: exactly what the
// `config.` sites touch. EpDispatchCombineConfig has 21 fields and a dozen more
// helpers; carrying them would re-import the coupling this file removes.
//
// Geometry is deliberately NOT here. blockNum and warpNumPerBlock come from
// gridDim/blockDim, and rdmaBlockNum is a per-launch field of the args below --
// not for tidiness, but because dispatch and combine are tuned to DIFFERENT
// values for the same op (internode_tuning_configs: at <=8 tokens dispatch runs
// rdma=32/warp=8 while combine runs rdma=21/warp=6, and the table comment
// records that the coupling is deliberate). Folding geometry into the config,
// and thus later into the NTTP, would give the two phases disagreeing configs
// over one shared arena.
// ---------------------------------------------------------------------------
struct EpInterNodeDeviceCfg {
  // No `rank`. Every field here is a compiled-in shape constant, so the whole
  // struct is constexpr and each read folds to a literal. Carrying the rank
  // would make it a runtime object the compiler has to keep live in scalar
  // registers across every inlined helper -- measured as dispatch_ll's SGPR
  // spill count going 1 -> 15 against the pre-refactor kernel. The rank is
  // `myPe` in DEF_COMMON_VARS, straight off args.
  int worldSize{8};
  int hiddenDim{4096};
  int scaleDim{0};
  int scaleTypeSize{0};
  int maxTokenTypeSize{4};
  int maxNumInpTokenPerRank{128};
  int numExpertPerRank{1};
  int numExpertPerToken{2};
  int maxTotalRecvTokens{0};
  int gpuPerNode{8};
  int numQpPerPe{1};
  EpQuantType quantType{EpQuantType::None};

  __host__ __device__ int MaxNumTokensToSendPerRank() const { return maxNumInpTokenPerRank; }

  __host__ __device__ int MaxNumTokensToSend() const {
    return worldSize * MaxNumTokensToSendPerRank();
  }

  __host__ __device__ int MaxNumTokensToRecvPerRank() const {
    if (maxTotalRecvTokens > 0) {
      int perRank = (maxTotalRecvTokens + worldSize - 1) / worldSize;
      return perRank < maxNumInpTokenPerRank ? perRank : maxNumInpTokenPerRank;
    }
    return maxNumInpTokenPerRank;
  }

  __host__ __device__ int MaxNumTokensToRecv() const {
    return worldSize * MaxNumTokensToRecvPerRank();
  }

  // Byte accounting for one transported token. size_t on purpose: these are
  // multiplied by token counts and the int forms overflow at shapes this kernel
  // is expected to run.
  __host__ __device__ size_t HiddenDimSz() const { return static_cast<size_t>(hiddenDim); }
  __host__ __device__ size_t HiddenBytes(size_t tokenTypeSize) const {
    return tokenTypeSize * static_cast<size_t>(hiddenDim);
  }
  __host__ __device__ size_t IndexBytes() const {
    return static_cast<size_t>(numExpertPerToken) * sizeof(ep_index_t);
  }
  __host__ __device__ size_t WeightBytes() const {
    return static_cast<size_t>(numExpertPerToken) * sizeof(float);
  }
  __host__ __device__ size_t SrcTokenIdBytes() const { return sizeof(ep_index_t); }
  __host__ __device__ size_t ScaleBytes() const {
    return static_cast<size_t>(scaleDim) * static_cast<size_t>(scaleTypeSize);
  }
  __host__ __device__ size_t XferBytesPerToken(size_t tokenTypeSize) const {
    return HiddenBytes(tokenTypeSize) + IndexBytes() + WeightBytes() + SrcTokenIdBytes() +
           ScaleBytes();
  }
  __host__ __device__ size_t MaxXferBytesPerToken() const {
    return XferBytesPerToken(static_cast<size_t>(maxTokenTypeSize));
  }
};

// ---------------------------------------------------------------------------
// Flat token indices and send-slot offsets. Pure index arithmetic over config
// fields; ported from v1's common.hpp with only the config type changed.
// ---------------------------------------------------------------------------
__device__ inline int FlatTokenIndex(const EpInterNodeDeviceCfg& config, int pe, int localTokId) {
  return pe * config.MaxNumTokensToSend() + localTokId;
}
__device__ inline int PeFromFlatTokenIndex(const EpInterNodeDeviceCfg& config, int flatIdx) {
  return flatIdx / config.MaxNumTokensToSend();
}
__device__ inline int LocalTokIdFromFlatTokenIndex(const EpInterNodeDeviceCfg& config,
                                                   int flatIdx) {
  return flatIdx % config.MaxNumTokensToSend();
}
__device__ inline int NullFlatTokenIndex(const EpInterNodeDeviceCfg& config) {
  return config.worldSize * config.MaxNumTokensToSend();
}

// Offset into a per-PE send staging buffer; stride is MaxNumTokensToSendPerRank,
// which is what the host allocated per PE.
__device__ inline int SendBufSlotOffset(const EpInterNodeDeviceCfg& config, int pe, int slotId) {
  return pe * config.MaxNumTokensToSendPerRank() + slotId;
}
__device__ inline int PeFromSendBufSlotOffset(const EpInterNodeDeviceCfg& config, int flatIdx) {
  return flatIdx / config.MaxNumTokensToSendPerRank();
}
__device__ inline int SlotIdFromSendBufSlotOffset(const EpInterNodeDeviceCfg& config, int flatIdx) {
  return flatIdx % config.MaxNumTokensToSendPerRank();
}
__device__ inline int NullSendBufSlotOffset(const EpInterNodeDeviceCfg& config) {
  return config.worldSize * config.MaxNumTokensToSendPerRank();
}

// ---------------------------------------------------------------------------
// The by-value kernel argument.
//
// One `window`, seventeen offsets into it, the local (non-symmetric) buffers as
// pointers, and the per-launch scalars. `reg(off)` is how a body turns an offset
// into something addressable; it is built per access and optimises away.
//
// The field list is an X-macro so the schema cannot drift from the declaration:
// the schema string, the offsetof table and the ascending-order assert below are
// all generated from this one list.
// ---------------------------------------------------------------------------
#define MORI_EP_INTERNODE_ARGS_FIELDS(X) \
  X(window, "u64")                       \
  X(offDispatchInp, "u64")               \
  X(offCombineInp, "u64")                \
  X(offStaging, "u64")                   \
  X(offDispatchOut, "u64")               \
  X(offCombineOut, "u64")                \
  X(offDispatchStaging, "u64")           \
  X(offInpWeights, "u64")                \
  X(offDispatchOutWeights, "u64")        \
  X(offCombineOutWeights, "u64")         \
  X(offOutIndices, "u64")                \
  X(offRecvTokenNum, "u64")              \
  X(offNodeRecvTokenNum, "u64")          \
  X(offDispTokOffset, "u64")             \
  X(offDispTokIdToSrcTokId, "u64")       \
  X(offCrossDeviceBarrier, "u64")        \
  X(offChunkFlag, "u64")                 \
  X(offOutScales, "u64")                 \
  X(rank, "i32")                         \
  X(lsaBase, "i32")                      \
  X(rdmaBlockNum, "i32")                 \
  X(replayMode, "i32")                   \
  X(curRankNumToken, "i32")              \
  X(tokenIndices, "p")                   \
  X(inpTokenBuf, "p")                    \
  X(weightsBuf, "p")                     \
  X(scalesBuf, "p")                      \
  X(dispDestTokIdMap, "p")               \
  X(interNodeDispDestTokIdMap, "p")      \
  X(interNodeDispSendMap, "p")           \
  X(interNodeChunkFlagCombine, "p")      \
  X(destPeTokenCounter, "p")             \
  X(blockFlagCounter, "p")               \
  X(totalRecvTokenNum, "p")              \
  X(dispTokIdToSrcTokIdLocal, "p")       \
  X(dispatchGridBarrier, "p")            \
  X(combineGridBarrier, "p")             \
  X(interNodeBlocksBarrier, "p")         \
  X(crossDeviceBarrierFlag, "p")

#define MORI_EP_INTERNODE_ARGS_SCHEMA_ENTRY(name, tag) #name ":" tag ","
// Trailing comma: the binding skips empty items, so there is no last-element
// special case to get wrong.
#define MORI_EP_INTERNODE_ARGS_SCHEMA \
  MORI_EP_INTERNODE_ARGS_FIELDS(MORI_EP_INTERNODE_ARGS_SCHEMA_ENTRY)

struct EpInterNodeArgs {
  uint64_t window{0};

  uint64_t offDispatchInp{0};
  uint64_t offCombineInp{0};
  uint64_t offStaging{0};
  uint64_t offDispatchOut{0};
  uint64_t offCombineOut{0};
  uint64_t offDispatchStaging{0};
  uint64_t offInpWeights{0};
  uint64_t offDispatchOutWeights{0};
  uint64_t offCombineOutWeights{0};
  uint64_t offOutIndices{0};
  uint64_t offRecvTokenNum{0};
  uint64_t offNodeRecvTokenNum{0};
  uint64_t offDispTokOffset{0};
  uint64_t offDispTokIdToSrcTokId{0};
  uint64_t offCrossDeviceBarrier{0};
  uint64_t offChunkFlag{0};
  // Only read when config.scaleDim > 0; the arena does not carry the region
  // otherwise, so this stays 0 and nothing dereferences it.
  uint64_t offOutScales{0};

  // Which world rank this is. Runtime, not Cfg, for the reason EpArgs gives:
  // as a compiled-in constant every rank on a node builds its own copy of an
  // identical kernel.
  int32_t rank{0};
  int32_t lsaBase{0};
  int32_t rdmaBlockNum{-1};
  // i32 rather than bool: the schema's scalar tags are fixed-width, and a
  // one-byte field would put every pointer after it at a different offset than
  // the binding computes.
  int32_t replayMode{0};
  int32_t curRankNumToken{0};

  ep_index_t* tokenIndices{nullptr};
  // void*, like the intranode EpArgs: callers cast at the use site. Keeping it
  // T* made this struct a template for one member, which then needed an
  // untyped alias, layout asserts across T, and a memcpy to pun between them.
  const void* inpTokenBuf{nullptr};
  float* weightsBuf{nullptr};
  uint8_t* scalesBuf{nullptr};

  // Local, non-symmetric. interNodeChunkFlagCombine is NOT the local half of
  // offChunkFlag despite the name: the symmetric one is written by dispatch and
  // read+cleared by combine (combine enumerates its work from dispatch's
  // leftover flags and has no producer of its own), while this one counts
  // completions. Different types, different lifetimes; pairing them by name
  // leaves combine with an all-zero work list and a hang.
  ep_index_t* dispDestTokIdMap{nullptr};
  ep_index_t* interNodeDispDestTokIdMap{nullptr};
  ep_index_t* interNodeDispSendMap{nullptr};
  ep_index_t* interNodeChunkFlagCombine{nullptr};
  ep_index_t* destPeTokenCounter{nullptr};
  ep_index_t* blockFlagCounter{nullptr};
  ep_index_t* totalRecvTokenNum{nullptr};
  ep_index_t* dispTokIdToSrcTokIdLocal{nullptr};
  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};
  uint32_t* interNodeBlocksBarrier{nullptr};
  uint64_t* crossDeviceBarrierFlag{nullptr};

  // An offset as something addressable. Built per access; the three members are
  // all the accessors need, so this costs nothing at -O2.
  __device__ __host__ EpInterNodeRegion reg(uint64_t off) const {
    return EpInterNodeRegion{window, off, lsaBase};
  }
};

namespace detail {

#define MORI_EP_INTERNODE_ARGS_OFFSET(name, tag) \
  offsetof(::mori::ops::v2::EpInterNodeArgs, name),
inline constexpr size_t kEpInterNodeArgsOffsets[] = {
    MORI_EP_INTERNODE_ARGS_FIELDS(MORI_EP_INTERNODE_ARGS_OFFSET)};
#undef MORI_EP_INTERNODE_ARGS_OFFSET

constexpr size_t kEpInterNodeArgsFieldCount =
    sizeof(kEpInterNodeArgsOffsets) / sizeof(kEpInterNodeArgsOffsets[0]);

constexpr bool EpInterNodeArgsOffsetsAscend() {
  for (size_t i = 1; i < kEpInterNodeArgsFieldCount; ++i)
    if (kEpInterNodeArgsOffsets[i] <= kEpInterNodeArgsOffsets[i - 1]) return false;
  return true;
}

}  // namespace detail

static_assert(detail::kEpInterNodeArgsFieldCount == 39,
              "added an EpInterNodeArgs field -- add it to MORI_EP_INTERNODE_ARGS_FIELDS in "
              "the same position and bump this count");
static_assert(detail::EpInterNodeArgsOffsetsAscend(),
              "MORI_EP_INTERNODE_ARGS_FIELDS is not in declaration order -- the binding "
              "would write each argument into the wrong slot");

// What crosses into a JIT module: the arguments plus the communicator. Passing
// the comm by value is the point of the CCO port; mori-shmem needed a device
// global filled by the host after every hipModuleLoad. It is the one opaque
// member, because it is cco's struct and not EP's to name.
struct EpInterNodeCcoArgs {
  EpInterNodeArgs args;
  ::mori::cco::ccoDevComm devComm;
};

static_assert(sizeof(EpInterNodeCcoArgs) ==
                  sizeof(EpInterNodeArgs) + sizeof(::mori::cco::ccoDevComm),
              "EpInterNodeCcoArgs has interior padding -- the schema describes it as the "
              "argument fields followed by one byte range and would place devComm wrong");

// Partitions a loop over (numItems x dimSize) work across globalWarpNum warps.
// When there are more warps than items, multiple warps collaborate on a single item
// by splitting dimSize; when there are fewer warps, each warp handles multiple items.
struct MultiWarpIter {
  int warpsPerItem;
  size_t dimPerWarp;
  size_t dimSize;

  // dimGranularity rounds dimPerWarp up to a multiple of itself, so callers doing
  // vectorized loads get every warp's slice starting on a vector boundary and
  // sized in whole vector steps.
  inline __device__ MultiWarpIter(int globalWarpNum, int numItems, size_t dimSize_,
                                  size_t dimGranularity = 1)
      : dimSize(dimSize_) {
    warpsPerItem = (globalWarpNum + numItems - 1) / numItems;
    dimPerWarp = (dimSize + warpsPerItem - 1) / warpsPerItem;
    if (dimGranularity > 1) {
      dimPerWarp = ((dimPerWarp + dimGranularity - 1) / dimGranularity) * dimGranularity;
      // A coarser slice means fewer warps are actually needed; keep warpsPerItem
      // consistent with dimPerWarp or the tail warps decode to empty ranges.
      warpsPerItem = static_cast<int>((dimSize + dimPerWarp - 1) / dimPerWarp);
    }
  }

  inline __device__ void Decode(int i, int& itemId, int& inItemPartId, size_t& dimOffset,
                                size_t& dimChunk) const {
    itemId = i / warpsPerItem;
    inItemPartId = i % warpsPerItem;
    dimOffset = (size_t)inItemPartId * dimPerWarp;
    dimChunk = (dimOffset < dimSize) ? std::min(dimSize - dimOffset, dimPerWarp) : size_t{0};
  }
};

}  // namespace v2
}  // namespace ops
}  // namespace mori
