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

#include <hip/hip_bfloat16.h>
#include <hip/library_types.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "mori/application/application.hpp"
#include "mori/core/profiler/constants.hpp"
#include "mori/core/profiler/kernel_profiler.hpp"
#include "mori/hip_compat.hpp"

// data_types.hpp and hip_fp8.h contain device builtins on ROCm <=6.x;
// the template args / variant that need them are also hipcc-only.
#if defined(__HIPCC__) || defined(__CUDACC__)
#include <hip/hip_fp8.h>

#include "mori/utils/data_types.hpp"
#endif

namespace mori {
namespace moe {

enum KernelType {
  IntraNode = 0,
  InterNode = 1,
  InterNodeV1 = 2,
  InterNodeV1LL = 3,
  AsyncLL = 4,
  IntraNodeLL = 5
};
enum class QuantType { None = 0, Fp8DirectCast = 1, Fp8BlockwiseQuant = 2 };

inline const char* HipDataTypeToString(hipDataType dtype) {
  switch (dtype) {
    case HIP_R_16F:
      return "HIP_R_16F";
    case HIP_R_32F:
      return "HIP_R_32F";
    case HIP_R_16BF:
      return "HIP_R_16BF";
    case HIP_R_8F_E4M3:
      return "HIP_R_8F_E4M3";
    case HIP_R_8F_E4M3_FNUZ:
      return "HIP_R_8F_E4M3_FNUZ";
    default:
      return "Unknown";
  }
}

#if defined(__HIPCC__) || defined(__CUDACC__)
inline size_t GetHipDataTypeSize(hipDataType dtype) {
  switch (dtype) {
    case HIP_R_32F:
      return sizeof(float);
    case HIP_R_16BF:
      return sizeof(hip_bfloat16);
    case HIP_R_8F_E4M3:
      return sizeof(__hip_fp8_e4m3);
    case HIP_R_8F_E4M3_FNUZ:
      return sizeof(__hip_fp8_e4m3_fnuz);
    default:
      throw std::runtime_error("Unknown hipDataType");
  }
}
#endif

using index_t = int32_t;

#define MAX_EXPERTS_PER_TOKEN (9)
struct EpDispatchCombineConfig {
  constexpr static size_t kPackedI32Len = 19;

  int rank{0};
  int worldSize{0};
  int hiddenDim{4096};
  int scaleDim{32};
  int scaleTypeSize{1};
  int maxTokenTypeSize{4};
  int maxNumInpTokenPerRank{128};
  int numExpertPerRank{1};
  int numExpertPerToken{2};
  int maxTotalRecvTokens{0};
  int warpNumPerBlock{1};
  int blockNum{1};
  // If true, use external buffer which incurs extra copy overhead; otherwise, the kernel assumes
  // the provided buffer is shmemInpTokMemObj
  bool useExternalInpBuffer{true};
  KernelType kernelType{KernelType::IntraNode};
  int gpuPerNode{8};
  int rdmaBlockNum{1};
  int numQpPerPe{1};
  QuantType quantType{QuantType::None};
  bool enableSdma{false};

  inline __host__ __device__ int MaxNumTokensToSendPerRank() const { return maxNumInpTokenPerRank; }

  inline __host__ __device__ int MaxNumTokensToSend() const {
    return worldSize * MaxNumTokensToSendPerRank();
  }

  inline __host__ __device__ int MaxNumTokensToRecvPerRank() const {
    if (maxTotalRecvTokens > 0) {
      int perRank = (maxTotalRecvTokens + worldSize - 1) / worldSize;
      return perRank < maxNumInpTokenPerRank ? perRank : maxNumInpTokenPerRank;
    }
    return maxNumInpTokenPerRank;
  }

  inline __host__ __device__ int MaxNumTokensToRecv() const {
    return worldSize * MaxNumTokensToRecvPerRank();
  }

  std::vector<int32_t> ToPackedI32Array() const;
  static EpDispatchCombineConfig FromPackedI32Array(const int32_t* packed, size_t size);
  inline __host__ __device__ size_t HiddenBytes(size_t tokenTypeSize) const {
    return tokenTypeSize * hiddenDim;
  }
  inline __host__ __device__ size_t MaxHiddenBytes() const { return HiddenBytes(maxTokenTypeSize); }

  inline __host__ __device__ size_t IndexBytes() const {
    return numExpertPerToken * sizeof(index_t);
  }
  inline __host__ __device__ size_t WeightBytes() const {
    return numExpertPerToken * sizeof(float);
  }
  inline __host__ __device__ size_t SrcTokenIdBytes() const { return sizeof(index_t); }
  inline __host__ __device__ size_t ScaleBytes() const { return scaleDim * scaleTypeSize; }
  // Size_t accessors for fields used in token-offset arithmetic.
  // Use these instead of the raw int members to avoid int32 overflow when
  // multiplying by token counts (e.g. tokenId * HiddenDimSz() is size_t * size_t).
  inline __host__ __device__ size_t HiddenDimSz() const { return (size_t)hiddenDim; }

  inline __host__ __device__ size_t XferBytesPerToken(size_t tokenTypeSize) const {
    return HiddenBytes(tokenTypeSize) + IndexBytes() + WeightBytes() + SrcTokenIdBytes() +
           ScaleBytes();
  }
  inline __host__ __device__ size_t MaxXferBytesPerToken() const {
    return XferBytesPerToken(maxTokenTypeSize);
  }
};

// Per-kernel-type token buffer groups.
// Used both as host-side allocation holders (via variant in EpDispatchCombineHandle)
// and embedded by value in EpDispatchCombineArgs / EpDispatchCombineArgsRaw for kernel launch.

// IntraNode: no RDMA path, staging buffer not needed.
struct ShmemBufsIntraNode {
  mori::application::SymmMemObjPtr dispatchOut;
  mori::application::SymmMemObjPtr combineInp;
  mori::application::SymmMemObjPtr combineOut;
};

// InterNodeV1 / InterNodeV1LL: full 5-buffer set used by the V1 RDMA path.
struct ShmemBufsInterNodeV1 {
  mori::application::SymmMemObjPtr dispatchInp;
  mori::application::SymmMemObjPtr combineInp;
  mori::application::SymmMemObjPtr dispatchOut;
  mori::application::SymmMemObjPtr combineOut;
  mori::application::SymmMemObjPtr staging;
  // Dispatch send source, separate from `staging` so combine can't overwrite it.
  mori::application::SymmMemObjPtr dispatchStaging;
};

// InterNode / AsyncLL: full 5-buffer set used by the non-V1 RDMA paths.
struct ShmemBufsInterNode {
  mori::application::SymmMemObjPtr dispatchInp;
  mori::application::SymmMemObjPtr combineInp;
  mori::application::SymmMemObjPtr dispatchOut;
  mori::application::SymmMemObjPtr combineOut;
  mori::application::SymmMemObjPtr staging;
};

class EpDispatchCombineHandle {
 public:
  EpDispatchCombineHandle(EpDispatchCombineConfig config);
  ~EpDispatchCombineHandle();

  void PrepareInference(hipDataType inputType, void* input, void* output, float* weights,
                        index_t* tokenIndices, index_t numToken) {
    this->inputType = inputType;
    this->inpTokenBuf = input;
    this->outTokenBuf = output;
    this->weightsBuf = weights;
    this->tokenIndices = tokenIndices;
    this->curRankNumToken = numToken;
  }

  void PrepareInference(hipDataType inputType, void* input, void* output, float* weights,
                        uint8_t* scales, index_t* tokenIndices, index_t numToken) {
    this->inputType = inputType;
    this->inpTokenBuf = input;
    this->outTokenBuf = output;
    this->weightsBuf = weights;
    this->scalesBuf = scales;
    this->tokenIndices = tokenIndices;
    this->curRankNumToken = numToken;
  }

#ifdef ENABLE_STANDARD_MOE_ADAPT
  void SetStandardMoeOutputBuffers(void* packedRecvX, int* packedRecvCount, int* packedRecvSrcInfo,
                                   int64_t* packedRecvLayoutRange) {
    enableStandardMoeOutput = true;
    standardPackedRecvX = packedRecvX;
    // standardPackedRecvCount = packedRecvCount;
    standardPackedRecvSrcInfo = packedRecvSrcInfo;
    standardPackedRecvLayoutRange = packedRecvLayoutRange;
  }

  void ClearStandardMoeOutputBuffers() {
    enableStandardMoeOutput = false;
    standardPackedRecvX = nullptr;
    // standardPackedRecvCount = nullptr;
    standardPackedRecvSrcInfo = nullptr;
    standardPackedRecvLayoutRange = nullptr;
  }
#endif

  void LaunchReset(hipStream_t = 0);

  // Release every buffer the constructor allocated (shmem symmetric buffers,
  // plain hipMalloc scratch, barriers). Idempotent: a second call is a no-op,
  // and the destructor will not double-free after an explicit Finalize().
  //
  // Collective: the symmetric heap is allocated in lockstep across the PE
  // group, so every rank must call this the same number of times, in the same
  // order, with the op idle (no in-flight dispatch/combine).
  void Finalize();

  // Free all buffers and re-allocate them for `newConfig`. Used to resize the
  // all-to-all dispatch/combine buffers in place when the owning process
  // changes role (e.g. an sglang PD prefill<->decode role switch changes
  // max_num_inp_token_per_rank) without tearing down the shmem context.
  //
  // Only capacity-like fields may change; anything that would alter the
  // symmetric layout agreed with peers (rank/worldSize/gpuPerNode/kernelType/
  // hiddenDim/numExpertPerRank/numExpertPerToken/quantType/scale geometry/
  // maxTokenTypeSize) must be identical or this throws std::runtime_error.
  //
  // Collective: same ordering requirement as Finalize(); all ranks must call
  // Reconfigure with configs that agree on the invariant fields.
  void Reconfigure(const EpDispatchCombineConfig& newConfig);

  bool IsInitialized() const { return buffersInitialized; }

  // Re-seed the cross-device barrier generation to the value a freshly built
  // handle starts at, and zero the symmetric barrier buffer with it.
  //
  // `crossDeviceBarrierFlag` is a monotonically increasing GENERATION counter,
  // not a scratch flag: every barrier does atomicAdd(flag, 1) and then spins
  // until its peers publish the same generation (intranode.hpp:69,73). So the
  // group only makes progress while all ranks hold the SAME value, and
  // InitializeBarrier() re-seeds it. Reconfigure() therefore re-seeds it on
  // every rank that resizes -- but a rank that REJECTS the new config (or that
  // the group agreed had nothing to do) never reaches InitializeBarrier and
  // keeps its old, higher generation. The next collective then has that rank
  // waiting on generation N while its peers publish 1: an unkillable spin
  // inside the kernel on every rank.
  //
  // Exposed so those non-resizing paths can converge on the seed explicitly.
  // Deliberately NOT a "restore the old value" snapshot: that direction is
  // wrong for the InterNode families, where the peer buffer is an ACCUMULATOR
  // (AMO_ADD, spun on as flag*numQps -- internode.hpp:357-360) that a rebuild
  // re-allocates zeroed. Restoring a high generation there would wait for a
  // total that can never accumulate. Converging DOWN on the seed is correct for
  // both families because it matches what a rebuild produces.
  //
  // Collective, and it must be ordered against dispatch/combine like any other
  // barrier operation: call it with the op idle and on every rank.
  void ResetBarrierGeneration();

  // Read-only snapshot of the state a cross-device barrier actually depends on,
  // so a test can MEASURE which part of it diverges across ranks instead of
  // inferring it from a hang.
  //
  // Why this exists: the asymmetric-reject hang has now survived two fixes
  // aimed at hypotheses derived from the parent process's view (a peer-VA
  // offset story, refuted by the heap-symmetry probe; a generation-counter
  // story, refuted at world 8 when ResetBarrierGeneration did not help). Both
  // were plausible and both were wrong, and each cost a turn. A barrier's
  // progress depends on exactly three things -- the generation every rank
  // holds, the local address of the symmetric flag buffer, and the address
  // each rank believes its PEERS' copies live at -- and none of them was ever
  // read. A rank that does no heap work keeps peer pointers derived from the
  // heap layout its peers have since torn down and rebuilt twice, and
  // RegisterStaticHeapSubRegion (symmetric_memory.cpp:392) computes those as
  // `peerPtrs[i] + offset` with no allgather to check. Equal heap ACCOUNTING
  // (total_free_space, block counts) does not imply equal ADDRESSES, so the
  // heap-symmetry probe could not have caught that and its green is not
  // evidence against it.
  //
  // Rank-local, read-only, no collective, safe on a finalized handle (reports
  // initialized=false and zeros). The caller gathers it and compares.
  struct BarrierProbe {
    bool initialized{false};
    // Generation this rank would enter its next barrier with.
    uint64_t generation{0};
    // Seed a freshly built handle of this config would use.
    uint64_t seed{0};
    // Local address + size of the symmetric barrier buffer.
    uintptr_t localPtr{0};
    size_t size{0};
    // What THIS rank believes each peer's copy of that buffer is at. The
    // symmetric-heap contract is that every rank agrees, i.e. peerPtrs[i] as
    // seen by rank j must equal localPtr as seen by rank i, for all i, j.
    std::vector<uintptr_t> peerPtrs;
    // Current contents of the local barrier slots (what a spin reads).
    std::vector<uint64_t> slots;
    // EVERY symmetric object the handle owns, not just the barrier.
    //
    // The barrier buffer alone is worldSize*2*sizeof(uint32_t)*sizeof(uint64_t)
    // -- 512 B at world 8 and CAPACITY-INDEPENDENT -- and it is allocated LAST
    // (InitializeAll: Shmem -> TokenNumSignal -> OrderMap -> Barrier), so under
    // a first-fit VA manager it is the object most likely to land back exactly
    // where it started across a resize. Probing only it is why the earlier
    // revision of this diagnostic came back green: it cannot move, so it cannot
    // show a stale peer mapping either.
    //
    // The objects that DO resize with maxNumInpTokenPerRank -- dispatchOut /
    // combineInp / combineOut / staging, recvTokenNumMemObj, dispTokOffsetMemObj,
    // dispTokIdToSrcTokIdMemObj, the weight/scale/index buffers -- are the ones
    // a non-participating rank derives stale peer addresses for, and the
    // intranode dispatch protocol spins on exactly those (intranode.hpp:183-186
    // waits on a signal then stores into the peer's recvTokenNumMemObj).
    struct SymmObjProbe {
      std::string name;
      uintptr_t localPtr{0};
      size_t size{0};
      std::vector<uintptr_t> peerPtrs;
    };
    std::vector<SymmObjProbe> objects;
    // DEVICE-RESIDENT CONTENTS of the plain-device scalars a kernel spins on.
    //
    // Everything else in this struct is an ADDRESS. Five host-visible
    // mechanisms have now each been measured across an asymmetric reject and
    // none diverges (generation, heap accounting, instantaneous peer
    // addressing, barrier-object staleness, all-object staleness/size/offset),
    // so the address space is exhausted and the remaining suspect is state
    // that lives in device memory and is not an address.
    //
    // These three are the spin predicates:
    //   dispatchGridBarrier -- intranode.hpp:181 `WaitUntilEquals(.., gridDim.x)`
    //   combineGridBarrier  -- intranode.hpp:60   same
    //   totalRecvTokenNum   -- accumulated at :201, read by the host after
    // and they are the ONLY buffers in the op that a reconfigure treats
    // asymmetrically in a way no address check can see: a rank that rebuilds
    // gets them freshly hipMalloc'd and memset to 0, while a rank refused
    // inside ValidateReconfigurable keeps whatever the last kernel left in
    // them. If that residue is ever nonzero, the non-rebuilding rank enters
    // its next dispatch with a grid-barrier count that is already offset, so
    // `WaitUntilEquals(gridDim.x)` is unsatisfiable -- an in-kernel spin, on
    // one rank, with every address on every rank perfectly symmetric. That is
    // the exact signature of the wedge.
    //
    // Recorded as a name-keyed map for the same reason `objects` is: the set
    // is not fixed across kernel types and a positional diff would pair up
    // unrelated counters.
    std::vector<std::pair<std::string, uint64_t>> counters;
  };
  BarrierProbe ProbeBarrierState() const;

  index_t GetCurRankNumToken() const { return curRankNumToken; }
  int Fp8BlockwiseCombineScaleDim() const { return fp8BlockwiseCombineScaleDim; }
  int Fp8BlockwiseCombineScaleTypeSize() const { return fp8BlockwiseCombineScaleTypeSize; }

  mori::application::SymmMemObjPtr GetShmemDispatchOutTokMemObj() const {
    if (config.kernelType == KernelType::IntraNode || config.kernelType == KernelType::IntraNodeLL)
      return std::get<ShmemBufsIntraNode>(shmemTokBufs).dispatchOut;
    if (config.kernelType == KernelType::InterNodeV1 ||
        config.kernelType == KernelType::InterNodeV1LL)
      return std::get<ShmemBufsInterNodeV1>(shmemTokBufs).dispatchOut;
    return std::get<ShmemBufsInterNode>(shmemTokBufs).dispatchOut;
  }
  mori::application::SymmMemObjPtr GetShmemCombineOutTokMemObj() const {
    if (config.kernelType == KernelType::IntraNode || config.kernelType == KernelType::IntraNodeLL)
      return std::get<ShmemBufsIntraNode>(shmemTokBufs).combineOut;
    if (config.kernelType == KernelType::InterNodeV1 ||
        config.kernelType == KernelType::InterNodeV1LL)
      return std::get<ShmemBufsInterNodeV1>(shmemTokBufs).combineOut;
    return std::get<ShmemBufsInterNode>(shmemTokBufs).combineOut;
  }
  mori::application::SymmMemObjPtr GetShmemCombineInpTokMemObj() const {
    if (config.kernelType == KernelType::IntraNode || config.kernelType == KernelType::IntraNodeLL)
      return std::get<ShmemBufsIntraNode>(shmemTokBufs).combineInp;
    if (config.kernelType == KernelType::InterNodeV1 ||
        config.kernelType == KernelType::InterNodeV1LL)
      return std::get<ShmemBufsInterNodeV1>(shmemTokBufs).combineInp;
    return std::get<ShmemBufsInterNode>(shmemTokBufs).combineInp;
  }

 private:
  // Allocate/free every buffer group; Initialize/FinalizeAll are the shared
  // bodies used by the ctor, dtor, Finalize() and Reconfigure().
  void InitializeAll();
  void FinalizeAll();
  // Normalize config-derived fields (numQpPerPe clamp, SDMA flag,
  // maxTotalRecvTokens clamp, fp8 blockwise scale dim) before allocating.
  void NormalizeConfig();
  // Throw if `newConfig` changes a field that the symmetric layout / peers
  // depend on. Called by Reconfigure() before any buffer is freed.
  void ValidateReconfigurable(const EpDispatchCombineConfig& newConfig) const;

  bool buffersInitialized{false};

  void InitializeShmemBuf();
  void FinalizeShmemBuf();

  void InitializeTokenNumSignalBuf();
  void FinalizeTokenNumSignalBuf();

  void InitializeOrderMapBuf();
  void FinalizeOrderMapBuf();

  void InitializeBarrier();
  void FinalizeBarrier();
  // The barrier generation a freshly built handle starts at (kernel-type
  // dependent). Single source of truth for InitializeBarrier() and
  // ResetBarrierGeneration().
  uint64_t BarrierGenerationSeed() const;

 public:
  // Updated at each round of inference
  index_t curRankNumToken{0};
  int curHiddenDim{-1};

  index_t multiProcessorCount{0};
  index_t maxThreads{0};

 public:
  // Config
  EpDispatchCombineConfig config;
  int fp8BlockwiseCombineScaleDim{0};
  int fp8BlockwiseCombineScaleTypeSize{0};
  // Routed expert indices for tokens
  index_t* tokenIndices{nullptr};

  // Kernel input/output buffer
  void* inpTokenBuf{nullptr};
  void* outTokenBuf{nullptr};
  hipDataType inputType;
  float* weightsBuf{nullptr};
  uint8_t* scalesBuf{nullptr};

  // Registered buffers for tokens — allocated according to kernelType.
  std::variant<ShmemBufsIntraNode, ShmemBufsInterNodeV1, ShmemBufsInterNode> shmemTokBufs;

  // Registered buffer used for weights, indices and scales
  mori::application::SymmMemObjPtr shmemInpWeightsMemObj;
  mori::application::SymmMemObjPtr shmemDispatchOutWeightsMemObj;
  mori::application::SymmMemObjPtr shmemCombineOutWeightsMemObj;
  mori::application::SymmMemObjPtr shmemInpScalesMemObj;
  mori::application::SymmMemObjPtr shmemOutScalesMemObj;
  mori::application::SymmMemObjPtr shmemInpIndicesMemObj;
  mori::application::SymmMemObjPtr shmemOutIndicesMemObj;

  // Record number of tokens that will be received from other PE
  mori::application::SymmMemObjPtr recvTokenNumMemObj;
  mori::application::SymmMemObjPtr sendTokenNumMemObj;
  mori::application::SymmMemObjPtr sendAtomicSignalMemObj;

  // Barrier for intra-grid synchronization
  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};

  // Map dispatch input token index to staging buffer index, saved at dispatch send phase and used
  // at combine recv phase
  index_t* dispSenderIdxMap{nullptr};
  // Map dispatch staging buffer index to output buffer index, saved at dispatch recv phase and used
  // at combine send phase
  index_t* dispReceiverIdxMap{nullptr};

#ifdef ENABLE_STANDARD_MOE_ADAPT
  // Map dispatch token to expert slot index (size: MaxNumTokensToRecv * numExpertPerToken), saved
  // at ConvertDispatchOutput and used at ConvertCombineInput
  uint64_t* dispTokToEpSlotMap{nullptr};

  // Standard MoE output buffers (set per-dispatch when enabled).
  bool enableStandardMoeOutput{false};
  void* standardPackedRecvX{nullptr};
  int* standardPackedRecvCount{nullptr};
  int* standardPackedRecvSrcInfo{nullptr};
  int64_t* standardPackedRecvLayoutRange{nullptr};
#endif

  // Map staging buffer index to dispatch input token index, saved at dispatch init phase and used
  // at dispatch send phase
  index_t* destPeTokenIdxMap{nullptr};
  // Map output buffer index to combine input token index, saved at dispatch recv phase and used at
  // combine send phase
  index_t* srcPeTokenIdxMap{nullptr};

  // Count the number of tokens sent to destination pe
  index_t* destPeTokenCounter{nullptr};
  // Count the number of tokens sent to local pe
  index_t* localPeTokenCounter{nullptr};

  // Intra-node kernel parameters
  mori::application::SymmMemObjPtr dispTokOffsetMemObj;
  mori::application::SymmMemObjPtr dispTokIdToSrcTokIdMemObj;
  index_t* dispDestTokIdMap{nullptr};
  index_t* totalRecvTokenNum{nullptr};
  mori::application::SymmMemObjPtr crossDeviceBarrierMemObj;
  uint64_t* crossDeviceBarrierFlag{nullptr};

  // Inter-node v1 kernel parameters
  // Signal the completion of inter-node token transfer
  mori::application::SymmMemObjPtr interNodeChunkFlagMemObj;
  // Signal the number of tokens transferred from other nodes
  mori::application::SymmMemObjPtr nodeRecvTokenNumMemObj;
  // Count the number of tokens sent to other nodes
  index_t* destNodeTokenCounter{nullptr};
  // Counter that is used to sort the ordering of inter-node token chunk transfer
  index_t* blockFlagCounter{nullptr};
  // Barrier blocks that do inter node rdma transfer
  uint32_t* interNodeBlocksBarrier{nullptr};
  // Map dispatch token idx for inter-node tokens
  index_t* interNodeDispDestTokIdMap{nullptr};
  // Barrier rdma block group
  index_t* interNodeChunkFlagCombine{nullptr};
  // Map dispatched rdma token chunk index
  index_t* interNodeDispSendMap{nullptr};
#ifdef ENABLE_PROFILER
  mori::core::profiler::ProfilerConfig profilerConfig;
#endif
};

// Template args struct and helpers require HIP types (hip_bfloat16, fp8, fp4).
// Only available under hipcc; CXX code uses EpDispatchCombineArgsRaw instead.
#if defined(__HIPCC__) || defined(__CUDACC__)

template <typename T>
struct EpDispatchCombineArgs {
  using data_type = T;
  EpDispatchCombineConfig config;
  int fp8BlockwiseCombineScaleDim{0};
  int rdmaBlockNum{-1};
  index_t curRankNumToken{0};
  index_t* tokenIndices{nullptr};
  T* inpTokenBuf{nullptr};
  T* outTokenBuf{nullptr};
  float* weightsBuf{nullptr};
  uint8_t* scalesBuf{nullptr};
  ShmemBufsIntraNode intraNodeTokBufs;
  ShmemBufsInterNodeV1 interNodeV1TokBufs;
  ShmemBufsInterNode interNodeTokBufs;
  mori::application::SymmMemObjPtr shmemInpWeightsMemObj;
  mori::application::SymmMemObjPtr shmemDispatchOutWeightsMemObj;
  mori::application::SymmMemObjPtr shmemCombineOutWeightsMemObj;
  mori::application::SymmMemObjPtr shmemInpScalesMemObj;
  mori::application::SymmMemObjPtr shmemOutScalesMemObj;
  mori::application::SymmMemObjPtr shmemInpIndicesMemObj;
  mori::application::SymmMemObjPtr shmemOutIndicesMemObj;
  mori::application::SymmMemObjPtr recvTokenNumMemObj;
  mori::application::SymmMemObjPtr sendTokenNumMemObj;
  mori::application::SymmMemObjPtr sendAtomicSignalMemObj;
  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};
  index_t* destPeTokenCounter{nullptr};
  index_t* localPeTokenCounter{nullptr};
  index_t* dispReceiverIdxMap{nullptr};
  index_t* dispSenderIdxMap{nullptr};
  index_t* destPeTokenIdxMap{nullptr};
  index_t* srcPeTokenIdxMap{nullptr};
  mori::application::SymmMemObjPtr dispTokOffsetMemObj;
  mori::application::SymmMemObjPtr dispTokIdToSrcTokIdMemObj;
  index_t* dispDestTokIdMap{nullptr};
  index_t* totalRecvTokenNum{nullptr};
  mori::application::SymmMemObjPtr crossDeviceBarrierMemObj;
  uint64_t* crossDeviceBarrierFlag{nullptr};
  mori::application::SymmMemObjPtr interNodeChunkFlagMemObj;
  index_t* destNodeTokenCounter{nullptr};
  mori::application::SymmMemObjPtr nodeRecvTokenNumMemObj;
  index_t* blockFlagCounter{nullptr};
  uint32_t* interNodeBlocksBarrier{nullptr};
  index_t* interNodeDispDestTokIdMap{nullptr};
  index_t* interNodeChunkFlagCombine{nullptr};
  index_t* interNodeDispSendMap{nullptr};
#ifdef ENABLE_PROFILER
  mori::core::profiler::ProfilerConfig profilerConfig;
#endif

#ifdef ENABLE_STANDARD_MOE_ADAPT
  bool enableStandardMoeOutput{false};
  void* standardPackedRecvX{nullptr};
  int* standardPackedRecvCount{nullptr};
  int* standardPackedRecvSrcInfo{nullptr};
  int64_t* standardPackedRecvLayoutRange{nullptr};
  uint64_t* dispTokToEpSlotMap{nullptr};
#endif
};

#endif  // __HIPCC__ || __CUDACC__  (template args)

// Non-template args struct: identical binary layout to EpDispatchCombineArgs<T> (T* → void*).
// Used by Python-side kernel launch where the type is erased.
struct EpDispatchCombineArgsRaw {
  EpDispatchCombineConfig config;
  int fp8BlockwiseCombineScaleDim{0};
  int rdmaBlockNum{-1};
  index_t curRankNumToken{0};
  index_t* tokenIndices{nullptr};
  void* inpTokenBuf{nullptr};
  void* outTokenBuf{nullptr};
  float* weightsBuf{nullptr};
  uint8_t* scalesBuf{nullptr};
  ShmemBufsIntraNode intraNodeTokBufs;
  ShmemBufsInterNodeV1 interNodeV1TokBufs;
  ShmemBufsInterNode interNodeTokBufs;
  mori::application::SymmMemObjPtr shmemInpWeightsMemObj;
  mori::application::SymmMemObjPtr shmemDispatchOutWeightsMemObj;
  mori::application::SymmMemObjPtr shmemCombineOutWeightsMemObj;
  mori::application::SymmMemObjPtr shmemInpScalesMemObj;
  mori::application::SymmMemObjPtr shmemOutScalesMemObj;
  mori::application::SymmMemObjPtr shmemInpIndicesMemObj;
  mori::application::SymmMemObjPtr shmemOutIndicesMemObj;
  mori::application::SymmMemObjPtr recvTokenNumMemObj;
  mori::application::SymmMemObjPtr sendTokenNumMemObj;
  mori::application::SymmMemObjPtr sendAtomicSignalMemObj;
  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};
  index_t* destPeTokenCounter{nullptr};
  index_t* localPeTokenCounter{nullptr};
  index_t* dispReceiverIdxMap{nullptr};
  index_t* dispSenderIdxMap{nullptr};
  index_t* destPeTokenIdxMap{nullptr};
  index_t* srcPeTokenIdxMap{nullptr};
  mori::application::SymmMemObjPtr dispTokOffsetMemObj;
  mori::application::SymmMemObjPtr dispTokIdToSrcTokIdMemObj;
  index_t* dispDestTokIdMap{nullptr};
  index_t* totalRecvTokenNum{nullptr};
  mori::application::SymmMemObjPtr crossDeviceBarrierMemObj;
  uint64_t* crossDeviceBarrierFlag{nullptr};
  mori::application::SymmMemObjPtr interNodeChunkFlagMemObj;
  index_t* destNodeTokenCounter{nullptr};
  mori::application::SymmMemObjPtr nodeRecvTokenNumMemObj;
  index_t* blockFlagCounter{nullptr};
  uint32_t* interNodeBlocksBarrier{nullptr};
  index_t* interNodeDispDestTokIdMap{nullptr};
  index_t* interNodeChunkFlagCombine{nullptr};
  index_t* interNodeDispSendMap{nullptr};
#ifdef ENABLE_PROFILER
  mori::core::profiler::ProfilerConfig profilerConfig;
#endif

#ifdef ENABLE_STANDARD_MOE_ADAPT
  bool enableStandardMoeOutput{false};
  void* standardPackedRecvX{nullptr};
  int* standardPackedRecvCount{nullptr};
  int* standardPackedRecvSrcInfo{nullptr};
  int64_t* standardPackedRecvLayoutRange{nullptr};
  uint64_t* dispTokToEpSlotMap{nullptr};
#endif
};

#if defined(__HIPCC__) || defined(__CUDACC__)
static_assert(sizeof(EpDispatchCombineArgsRaw) == sizeof(EpDispatchCombineArgs<hip_bfloat16>),
              "EpDispatchCombineArgsRaw must have identical layout to EpDispatchCombineArgs<T>");
#endif

EpDispatchCombineArgsRaw GetEpDispatchCombineArgsRaw(const EpDispatchCombineHandle& handle,
                                                     int rdmaBlockNum);

struct LocalExpertCountArgs {
  const index_t* indices;
  const index_t* totalRecvTokenNum;
  int rank;
  int numExpertPerRank;
  int numExpertPerToken;
  int* localExpertCount;
};

#ifdef ENABLE_STANDARD_MOE_ADAPT
struct ConvertDispatchOutputArgs {
  EpDispatchCombineConfig config;
  const void* dispatchOutX{nullptr};
  const void* dispatchOutTopkIdx{nullptr};
  const index_t* dispatchSrcTokenPos{nullptr};
  const index_t* totalRecvTokenNum{nullptr};
  uint32_t* dispatchGridBarrier{nullptr};
  void* packedRecvX{nullptr};
  int* packedRecvCount{nullptr};
  int* packedRecvSrcInfo{nullptr};
  int64_t* packedRecvLayoutRange{nullptr};
  uint64_t* dispTokToEpSlotMap{nullptr};
};

struct ConvertCombineInputArgs {
  EpDispatchCombineConfig config;
  const void* packedRecvX{nullptr};
  const void* topkIdx{nullptr};
  const void* topkWeights{nullptr};
  const void* packedRecvSrcInfo{nullptr};
  const void* packedRecvLayoutRange{nullptr};
  const index_t* totalRecvTokenNum{nullptr};
  void* combineInput{nullptr};
  uint64_t* dispTokToEpSlotMap{nullptr};
  int* packedRecvCount{nullptr};
  mori::application::SymmMemObjPtr shmemCombineInpTokMemObj;
  mori::application::SymmMemObjPtr dispTokIdToSrcTokIdMemObj;
};
#endif

}  // namespace moe
}  // namespace mori

namespace std {

static std::ostream& operator<<(std::ostream& s, mori::moe::EpDispatchCombineConfig config) {
  std::stringstream ss;
  ss << "EpDispatchCombineConfig: " << std::endl
     << "  WorldSize: " << config.worldSize << std::endl
     << "  hiddenDim: " << config.hiddenDim << std::endl
     << "  scaleDim: " << config.scaleDim << std::endl
     << "  scaleTypeSize: " << config.scaleTypeSize << std::endl
     << "  maxTokenTypeSize: " << config.maxTokenTypeSize << std::endl
     << "  maxNumInpTokenPerRank: " << config.maxNumInpTokenPerRank << std::endl
     << "  numExpertPerRank: " << config.numExpertPerRank << std::endl
     << "  numExpertPerToken: " << config.numExpertPerToken << std::endl
     << "  warpNumPerBlock: " << config.warpNumPerBlock << std::endl
     << "  blockNum: " << config.blockNum;
  s << ss.str();
  return s;
}

}  // namespace std
