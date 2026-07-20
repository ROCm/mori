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
#include <variant>
#include <vector>

// This header is compiled both into host TUs and into the device .hip kernels.
// The device side only needs application::SymmMemObjPtr (a device-safe POD from
// application_device_types.hpp); the full application.hpp pulls the host RDMA
// stack -> system mlx5dv.h/verbs.h, which must stay out of device compiles.
#if !defined(__HIPCC__) && !defined(__CUDACC__)
#include "mori/application/application.hpp"
#else
#include "mori/application/application_device_types.hpp"
#endif
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
enum class QuantType { None = 0, Fp8DirectCast = 1, Fp8BlockwiseQuant = 2, Fp4BlockwiseQuant = 3 };

// Blockwise combine quant types share the same staging/scale layout (per-block float scales);
// they differ only in the transported element codec: FP8 = 1 byte/elem, FP4 (E2M1) = 0.5 byte/elem
// (2 values packed per byte), so an FP4 token region is half the size of an FP8 one.
inline __host__ __device__ bool IsBlockwiseCombineQuant(QuantType q) {
  return q == QuantType::Fp8BlockwiseQuant || q == QuantType::Fp4BlockwiseQuant;
}

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

// Caller-owned routing pointers for cached/replay routing dispatch/combine.
// All fields must be non-null when passed to GetEpDispatchCombineArgsRaw(..., routing, ...).
struct EpDispatchCombineRoutingPtrs {
  index_t* dispDestTokIdMap{nullptr};
  index_t* interNodeDispDestTokIdMap{nullptr};
  index_t* interNodeDispSendMap{nullptr};
  index_t* totalRecvTokenNum{nullptr};
  index_t* dispTokIdToSrcTokIdLocal{nullptr};

  bool IsValid() const {
    return dispDestTokIdMap != nullptr && interNodeDispDestTokIdMap != nullptr &&
           interNodeDispSendMap != nullptr && totalRecvTokenNum != nullptr &&
           dispTokIdToSrcTokIdLocal != nullptr;
  }

  // Throws std::invalid_argument listing any null required pointer.
  void Validate() const;
};

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

  // Per-token combine token-region bytes (the quantized payload, excluding scales/weights/index).
  // FP4 blockwise packs two E2M1 values per byte, so its slot is half the FP8 (1 byte/elem) size;
  // FP8 blockwise and other paths keep the existing element sizing. Used by both the host staging
  // allocator and the intra-node kernel so the slot stride stays consistent.
  inline __host__ __device__ size_t CombineTokenRegionBytes() const {
    if (quantType == QuantType::Fp4BlockwiseQuant) return (static_cast<size_t>(hiddenDim) + 1) / 2;
    return HiddenBytes(maxTokenTypeSize);
  }

  inline __host__ __device__ size_t IndexBytes() const {
    return numExpertPerToken * sizeof(index_t);
  }
  inline __host__ __device__ size_t WeightBytes() const {
    return numExpertPerToken * sizeof(float);
  }
  inline __host__ __device__ size_t SrcTokenIdBytes() const { return sizeof(index_t); }
  inline __host__ __device__ size_t ScaleBytes() const {
    return static_cast<size_t>(scaleDim) * scaleTypeSize;
  }
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
  // ccoCommPtr (raw ccoComm*, as uintptr_t) selects the CCO backend for symmetric
  // buffers; 0 keeps the legacy shmem heap. Reuses a caller-owned cco communicator
  // (e.g. the one created by the Python mori.cco.Communicator, via comm.ptr).
  EpDispatchCombineHandle(EpDispatchCombineConfig config, uintptr_t ccoCommPtr = 0);
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
  void InitializeShmemBuf();
  void FinalizeShmemBuf();

  void InitializeTokenNumSignalBuf();
  void FinalizeTokenNumSignalBuf();

  void InitializeOrderMapBuf();
  void FinalizeOrderMapBuf();

  void InitializeBarrier();
  void FinalizeBarrier();

  // Symmetric allocation router: a CCO window (when useCcoComm) or the shmem heap.
  // Returns a SymmMemObjPtr whose device p2pPeerPtrs[pe] address peer pe directly,
  // so the existing kernels (GetAs<T*>(pe)) work unchanged on either backend.
  mori::application::SymmMemObjPtr MallocSymm(size_t size, unsigned int flags);
  void FreeSymmByLocalPtr(void* localPtr);
  void FinalizeCcoAllocs();

 public:
  // Updated at each round of inference
  index_t curRankNumToken{0};
  int curHiddenDim{-1};

  index_t multiProcessorCount{0};
  index_t maxThreads{0};

 public:
  // Config
  EpDispatchCombineConfig config;

  // CCO backend: when ccoCommPtr != nullptr, symmetric buffers are carved from a
  // cco window (LSA flat-VA) instead of the shmem heap. Lets intra-node EP run on
  // archs where mori-shmem is unavailable (e.g. gfx1250). ccoAllocs tracks the
  // per-buffer resources so the handle can release them on teardown.
  void* ccoCommPtr{nullptr};
  bool useCcoComm{false};
  struct CcoSymmAlloc {
    void* win{nullptr};  // ccoWindow_t
    void* localPtr{nullptr};
    mori::application::SymmMemObj* cpuObj{nullptr};
    mori::application::SymmMemObj* gpuObj{nullptr};
    uintptr_t* p2pDev{nullptr};
  };
  std::vector<CcoSymmAlloc> ccoAllocs;
  // Single CCO arena carved into sub-regions by bump offset (mirrors FlyDSL's
  // SymmArena). gfx942/ROCm fails hipMemSetAccess after a few *separate* VMM
  // allocations, so all symmetric buffers share ONE window.
  void* ccoArenaWin{nullptr};
  void* ccoArenaLocalPtr{nullptr};
  size_t ccoArenaSize{0};
  size_t ccoArenaBump{0};

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
  // NOTIFY-based slot pre-assignment (MORI_DISP_NOTIFY): symmetric count matrix
  // M[src][dest] (worldSize*worldSize index_t). Each rank writes its row (counts
  // to each dest PE) to every peer; after a barrier every rank has the full matrix
  // and computes contiguous per-source slot regions, so payload send needs only a
  // LOCAL atomic within its region instead of a cross-GPU remote atomic.
  mori::application::SymmMemObjPtr dispCountMatrixMemObj;
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
  bool replayMode{false};
  // BW diagnostic (MORI_BW_DUMMY_ATOMICS): extra remote atomicAdds injected per
  // sent token in intra-node dispatch to test the remote-atomic bottleneck.
  int bwDummyAtomics{0};
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
  mori::application::SymmMemObjPtr dispCountMatrixMemObj;
  index_t* dispDestTokIdMap{nullptr};
  index_t* totalRecvTokenNum{nullptr};
  index_t* dispTokIdToSrcTokIdLocal{nullptr};
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
  bool replayMode{false};
  // BW diagnostic (MORI_BW_DUMMY_ATOMICS): keep in sync with EpDispatchCombineArgs<T>.
  int bwDummyAtomics{0};
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
  mori::application::SymmMemObjPtr dispCountMatrixMemObj;
  index_t* dispDestTokIdMap{nullptr};
  index_t* totalRecvTokenNum{nullptr};
  index_t* dispTokIdToSrcTokIdLocal{nullptr};
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

// Routing-handle overload: routing pointers come from caller-owned tensors;
// `replayMode` selects cache vs replay routing dispatch (combine always passes false).
EpDispatchCombineArgsRaw GetEpDispatchCombineArgsRaw(const EpDispatchCombineHandle& handle,
                                                     int rdmaBlockNum,
                                                     const EpDispatchCombineRoutingPtrs* routing,
                                                     bool replayMode);

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
