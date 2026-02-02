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
#include "src/pybind/mori.hpp"

#include <ATen/hip/HIPContext.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <torch/extension.h>

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "mori/application/application.hpp"
#include "mori/core/profiler/constants.hpp"
#include "mori/io/io.hpp"
#include "mori/ops/ops.hpp"
#include "mori/pybind/profiler_registry.hpp"
#include "mori/shmem/shmem.hpp"
#include "mori/utils/hip_helper.hpp"
#include "src/pybind/torch_utils.hpp"

/* ---------------------------------------------------------------------------------------------- */
/*                                            Ops APIs                                            */
/* ---------------------------------------------------------------------------------------------- */
namespace {

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, torch::Tensor,
           torch::Tensor>
LaunchDispatch(mori::moe::EpDispatchCombineHandle& handle, int kernelType,
               const torch::Tensor& input, const std::optional<torch::Tensor>& weights,
               const std::optional<torch::Tensor>& scales, const torch::Tensor& topkIds,
               int blockNum = -1, int warpPerBlock = -1) {
  assert(input.is_contiguous() && topkIds.is_contiguous());

  float* weightPtr = nullptr;
  if (weights.has_value()) {
    assert(weights->is_contiguous() && weights->element_size() == sizeof(float));
    weightPtr = weights->data_ptr<float>();
  }

  uint8_t* scalePtr = nullptr;
  if (scales.has_value() && (handle.config.scaleDim > 0)) {
    assert(scales->is_contiguous() && scales->element_size() == handle.config.scaleTypeSize);
    scalePtr = reinterpret_cast<uint8_t*>(scales->data_ptr());
  }

  handle.PrepareInference(mori::ScalarTypeToHipDataType(input.scalar_type()), input.data_ptr(),
                          nullptr, weightPtr, scalePtr, topkIds.data_ptr<mori::moe::index_t>(),
                          input.size(0));
  handle.LaunchDispatch((mori::moe::KernelType)kernelType, blockNum, warpPerBlock,
                        at::cuda::getCurrentHIPStream(), /*enableStandardMoeOutput=*/false);

  torch::Tensor out =
      torch::from_blob(handle.shmemDispatchOutTokMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), handle.config.hiddenDim},
                       torch::TensorOptions().dtype(input.scalar_type()).device(torch::kCUDA));

  torch::Tensor outWeights = torch::from_blob(
      handle.shmemDispatchOutWeightsMemObj->Get(),
      {handle.config.MaxNumTokensToRecv(), handle.config.numExpertPerToken},
      torch::TensorOptions().dtype(mori::GetTorchDataType<float>()).device(torch::kCUDA));

  std::optional<torch::Tensor> outScales{std::nullopt};
  if (scales.has_value() && (handle.config.scaleDim > 0)) {
    outScales =
        torch::from_blob(handle.shmemOutScalesMemObj->Get(),
                         {handle.config.MaxNumTokensToRecv(), handle.config.scaleDim},
                         torch::TensorOptions().dtype(scales->scalar_type()).device(torch::kCUDA));
  }

  torch::Tensor outIndices =
      torch::from_blob(handle.shmemOutIndicesMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), handle.config.numExpertPerToken},
                       torch::TensorOptions()
                           .dtype(mori::GetTorchDataType<mori::moe::index_t>())
                           .device(torch::kCUDA));

  torch::Tensor totalRecvTokenNum =
      torch::from_blob(handle.totalRecvTokenNum, {1},
                       torch::TensorOptions()
                           .dtype(mori::GetTorchDataType<mori::moe::index_t>())
                           .device(torch::kCUDA));
  return {out, outWeights, outScales, outIndices, totalRecvTokenNum};
}

// TODO: translate data type
// template <typename T>
std::tuple<torch::Tensor, std::optional<torch::Tensor>> LaunchCombine(
    mori::moe::EpDispatchCombineHandle& handle, int kernelType, const torch::Tensor& input,
    const std::optional<torch::Tensor>& weights, const torch::Tensor& topkIds, int blockNum,
    int warpPerBlock, int useExternalInpBuf = -1) {
  assert(input.is_contiguous() && topkIds.is_contiguous());

  float* weightsPtr = nullptr;
  if (weights.has_value() && weights->size(0) != 0) {
    assert(weights->is_contiguous());
    weightsPtr = weights->data_ptr<float>();
  }

  handle.PrepareInference(mori::ScalarTypeToHipDataType(input.scalar_type()), input.data_ptr(),
                          nullptr, weightsPtr, topkIds.data_ptr<mori::moe::index_t>(),
                          handle.curRankNumToken);
  handle.LaunchCombine((mori::moe::KernelType)kernelType, blockNum, warpPerBlock,
                       at::cuda::getCurrentHIPStream(), /*enableStandardMoeInput=*/false,
                       useExternalInpBuf);

  auto options = torch::TensorOptions().dtype(input.scalar_type()).device(torch::kCUDA);
  torch::Tensor out =
      torch::from_blob(handle.shmemCombineOutTokMemObj->Get(),
                       {handle.config.maxNumInpTokenPerRank, handle.config.hiddenDim}, options);

  std::optional<torch::Tensor> outWeights{std::nullopt};
  if (weightsPtr) {
    outWeights =
        torch::from_blob(handle.shmemCombineOutWeightsMemObj->Get(),
                         {handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
                         torch::TensorOptions().dtype(weights->scalar_type()).device(torch::kCUDA));
  }

  return {out, outWeights};
}

// Standard MoE 3D output: packedRecvX/Count/SrcInfo/LayoutRange
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LaunchDispatchForStandardMoE(
    mori::moe::EpDispatchCombineHandle& handle, int kernelType, const torch::Tensor& input,
    const std::optional<torch::Tensor>& weights, const std::optional<torch::Tensor>& scales,
    const torch::Tensor& topkIds, int blockNum = -1, int warpPerBlock = -1) {
  assert(input.is_contiguous() && topkIds.is_contiguous());

  float* weightPtr = nullptr;
  if (weights.has_value()) {
    assert(weights->is_contiguous() && weights->element_size() == sizeof(float));
    weightPtr = weights->data_ptr<float>();
  }

  uint8_t* scalePtr = nullptr;
  if (scales.has_value() && (handle.config.scaleDim > 0)) {
    assert(scales->is_contiguous() && scales->element_size() == handle.config.scaleTypeSize);
    scalePtr = reinterpret_cast<uint8_t*>(scales->data_ptr());
  }

  handle.PrepareInference(mori::ScalarTypeToHipDataType(input.scalar_type()), input.data_ptr(),
                          nullptr, weightPtr, scalePtr, topkIds.data_ptr<mori::moe::index_t>(),
                          input.size(0));

  const int64_t numLocalExperts = handle.config.numExpertPerRank;
  const int64_t maxTokensPerExpert =
      static_cast<int64_t>(handle.config.worldSize) * handle.config.maxNumInpTokenPerRank;
  const int64_t hidden = input.size(1);

  torch::Tensor packedRecvX =
      torch::empty({numLocalExperts, maxTokensPerExpert, hidden}, input.options());
  auto packedRecvSrcInfo = torch::empty({numLocalExperts, maxTokensPerExpert},
                                        torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto packedRecvLayoutRange = torch::empty({numLocalExperts, handle.config.worldSize},
                                            torch::dtype(torch::kInt64).device(torch::kCUDA));

  handle.SetStandardMoeOutputBuffers(packedRecvX.data_ptr(), handle.standardPackedRecvCount,
                                     packedRecvSrcInfo.data_ptr<int>(),
                                     packedRecvLayoutRange.data_ptr<int64_t>());

  handle.LaunchDispatch((mori::moe::KernelType)kernelType, blockNum, warpPerBlock,
                        at::cuda::getCurrentHIPStream(), /*enableStandardMoeOutput=*/true);

  torch::Tensor packedRecvCount =
      torch::from_blob(handle.standardPackedRecvCount, {numLocalExperts},
                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

  handle.ClearStandardMoeOutputBuffers();

  return {packedRecvX, packedRecvCount, packedRecvSrcInfo, packedRecvLayoutRange};
}

// Standard MoE combine: takes expert output in packed format and combines back
std::tuple<torch::Tensor, std::optional<torch::Tensor>> LaunchCombineForStandardMoE(
    mori::moe::EpDispatchCombineHandle& handle, int kernelType,
    const torch::Tensor& expertOutput,  // [numLocalExperts, maxTokensPerExpert, hidden]
    const std::optional<torch::Tensor>& weights,
    const torch::Tensor& topkIds,
    int blockNum = -1, int warpPerBlock = -1) {
  assert(expertOutput.is_contiguous() && topkIds.is_contiguous());

  float* weightsPtr = nullptr;
  if (weights.has_value() && weights->numel() > 0) {
    assert(weights->is_contiguous() && weights->element_size() == sizeof(float));
    weightsPtr = weights->data_ptr<float>();
  }

  // Prepare inference with expert output as input
  handle.PrepareInference(mori::ScalarTypeToHipDataType(expertOutput.scalar_type()),
                          nullptr,  // inpTokenBuf not used for standard moe combine
                          nullptr,  // outTokenBuf
                          weightsPtr,
                          topkIds.data_ptr<mori::moe::index_t>(),
                          handle.curRankNumToken);

  // Set standard MoE input buffers (reusing output buffer fields)
  handle.SetStandardMoeOutputBuffers(
      expertOutput.data_ptr(),
      handle.standardPackedRecvCount,
      handle.standardPackedRecvSrcInfo,
      handle.standardPackedRecvLayoutRange);

  // Launch combine with enableStandardMoeInput=true
  handle.LaunchCombine((mori::moe::KernelType)kernelType, blockNum, warpPerBlock,
                       at::cuda::getCurrentHIPStream(), /*enableStandardMoeInput=*/true);

  // Get output tensor from shmem buffer
  auto options = torch::TensorOptions().dtype(expertOutput.scalar_type()).device(torch::kCUDA);
  torch::Tensor out =
      torch::from_blob(handle.shmemCombineOutTokMemObj->Get(),
                       {handle.config.maxNumInpTokenPerRank, handle.config.hiddenDim}, options);

  std::optional<torch::Tensor> outWeights{std::nullopt};
  // TODO: do not support weights for standard MoE now
  // if (weightsPtr) {
  //   outWeights =
  //       torch::from_blob(handle.shmemCombineOutWeightsMemObj->Get(),
  //                        {handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
  //                        torch::TensorOptions().dtype(weights->scalar_type()).device(torch::kCUDA));
  // }

  handle.ClearStandardMoeOutputBuffers();

  return {out, outWeights};
}

void LaunchReset(mori::moe::EpDispatchCombineHandle& handle) {
  handle.LaunchReset(at::cuda::getCurrentHIPStream());
}

torch::Tensor GetDispatchSrcTokenId(mori::moe::EpDispatchCombineHandle& handle) {
  auto options = torch::TensorOptions()
                     .dtype(mori::GetTorchDataType<mori::moe::index_t>())
                     .device(torch::kCUDA);
  torch::Tensor tensor =
      torch::from_blob(handle.dispTokIdToSrcTokIdMemObj->template GetAs<mori::moe::index_t*>(),
                       {*handle.totalRecvTokenNum}, options);
  return tensor;
}

#if 0
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ConvertDispatchOutput(
    mori::moe::EpDispatchCombineHandle& handle, const torch::Tensor& dispatchOutX,
    const torch::Tensor& dispatchOutTopkIdx,
    int blockNum = -1, int warpPerBlock = -1) {
  TORCH_CHECK(dispatchOutX.is_cuda() && dispatchOutTopkIdx.is_cuda(),
              "dispatchOutX/dispatchOutTopkIdx must be CUDA tensors");
  TORCH_CHECK(dispatchOutX.dim() == 2 && dispatchOutTopkIdx.dim() == 2,
              "dispatchOutX/dispatchOutTopkIdx must be 2D");
  TORCH_CHECK(dispatchOutX.size(0) == dispatchOutTopkIdx.size(0),
              "dispatchOutX and dispatchOutTopkIdx must have the same first dimension");

  using torch::indexing::Slice;

  const int64_t numLocalExperts = handle.config.numExpertPerRank;
  const int64_t maxTokensPerExpert =
      static_cast<int64_t>(handle.config.worldSize) * handle.config.maxNumInpTokenPerRank;
  const int64_t hidden = dispatchOutX.size(1);
  auto totalTokensTensor = torch::from_blob(
      handle.totalRecvTokenNum, {1},
      torch::TensorOptions()
          .dtype(mori::GetTorchDataType<mori::moe::index_t>())
          .device(torch::kCUDA));
  const int64_t totalTokens = totalTokensTensor.item<int64_t>();
  TORCH_CHECK(totalTokens >= 0, "totalRecvTokenNum must be non-negative");
  TORCH_CHECK(totalTokens <= dispatchOutX.size(0), "totalRecvTokenNum exceeds dispatchOutX size");

  auto options = dispatchOutX.options();
  auto countOptions =
      torch::TensorOptions().dtype(torch::kInt32).device(dispatchOutX.device());
  auto layoutOptions =
      torch::TensorOptions().dtype(torch::kInt64).device(dispatchOutX.device());

  torch::Tensor packedRecvX =
      torch::zeros({numLocalExperts, maxTokensPerExpert, hidden}, options);
  torch::Tensor packedRecvCount = torch::zeros({numLocalExperts}, countOptions);
  torch::Tensor packedRecvSrcInfo =
      torch::zeros({numLocalExperts, maxTokensPerExpert}, countOptions);
  torch::Tensor packedRecvLayoutRange =
      torch::zeros({numLocalExperts, handle.config.worldSize}, layoutOptions);

  TORCH_CHECK(totalTokens <= dispatchOutX.size(0),
              "totalTokens exceeds dispatchOutX size");
  TORCH_CHECK(totalTokens <= dispatchOutTopkIdx.size(0),
              "totalTokens exceeds dispatchOutTopkIdx size");
  torch::Tensor slicedX = dispatchOutX.index({Slice(0, totalTokens)});
  torch::Tensor slicedTopkIdx = dispatchOutTopkIdx.index({Slice(0, totalTokens)});
  torch::Tensor slicedSrcTokenPos = GetDispatchSrcTokenId(handle);
  TORCH_CHECK(slicedSrcTokenPos.size(0) == totalTokens,
              "dispatch src token pos size mismatch");
  const int64_t maxNumTokenPerRank = handle.config.maxNumInpTokenPerRank;
  auto slicedSrcRank = slicedSrcTokenPos.to(torch::kInt64).div(maxNumTokenPerRank, "floor");

  for (int64_t localExpert = 0; localExpert < numLocalExperts; ++localExpert) {
    const int64_t globalExpertId = handle.config.rank * numLocalExperts + localExpert;
    auto mask = slicedTopkIdx.eq(globalExpertId).any(1);
    auto indices = torch::nonzero(mask).squeeze(1);
    const int64_t count = indices.size(0);
    TORCH_CHECK(count <= maxTokensPerExpert,
                "Token count for local expert exceeds maxTokensPerExpert");
    auto countTensor = torch::full({}, count, packedRecvCount.options());
    packedRecvCount.index_put_({localExpert}, countTensor);
    if (count == 0) {
      continue;
    }

    auto expertSrcRank = slicedSrcRank.index_select(0, indices);
    auto expertSrcInfo = slicedSrcTokenPos.index_select(0, indices)
                             .remainder(maxNumTokenPerRank);

    int64_t writeOffset = 0;
    for (int64_t srcRank = 0; srcRank < handle.config.worldSize; ++srcRank) {
      auto rankMask = expertSrcRank.eq(srcRank);
      auto rankPositions = torch::nonzero(rankMask).squeeze(1);
      const int64_t rankCount = rankPositions.size(0);
      const int64_t beginIdx = writeOffset;
      const int64_t packedRange =
          (static_cast<int64_t>(beginIdx) << 32) | (static_cast<int64_t>(rankCount) & 0xffffffff);
      packedRecvLayoutRange.index_put_({localExpert, srcRank},
                                      torch::full({}, packedRange, layoutOptions));
      if (rankCount > 0) {
        auto selectedIdx = indices.index_select(0, rankPositions);
        auto selectedX = slicedX.index_select(0, selectedIdx);
        auto selectedSrcInfo = expertSrcInfo.index_select(0, rankPositions);
        packedRecvX.index({localExpert, Slice(writeOffset, writeOffset + rankCount)})
            .copy_(selectedX);
        packedRecvSrcInfo.index({localExpert, Slice(writeOffset, writeOffset + rankCount)})
            .copy_(selectedSrcInfo);
        writeOffset += rankCount;
      }
    }
    TORCH_CHECK(writeOffset == count, "Packed token count mismatch for local expert");
  }

  return {packedRecvX, packedRecvCount, packedRecvSrcInfo, packedRecvLayoutRange};
}

torch::Tensor ConvertCombineInput(mori::moe::EpDispatchCombineHandle& handle,
                                  const torch::Tensor& packedRecvX,
                                  const torch::Tensor& topkIdx,
                                  const std::optional<torch::Tensor>& topkWeights,
                                  const torch::Tensor& packedRecvSrcInfo,
                                  const torch::Tensor& packedRecvLayoutRange) {
  TORCH_CHECK(packedRecvX.is_cuda(), "packedRecvX must be a CUDA tensor");
  TORCH_CHECK(topkIdx.is_cuda(), "topkIdx must be a CUDA tensor");
  TORCH_CHECK(packedRecvSrcInfo.is_cuda(), "packedRecvSrcInfo must be a CUDA tensor");
  TORCH_CHECK(packedRecvLayoutRange.is_cuda(), "packedRecvLayoutRange must be a CUDA tensor");
  TORCH_CHECK(packedRecvX.dim() == 3, "packedRecvX must be 3D");
  TORCH_CHECK(topkIdx.dim() == 2, "topkIdx must be 2D");
  TORCH_CHECK(packedRecvSrcInfo.dim() == 2, "packedRecvSrcInfo must be 2D");
  TORCH_CHECK(packedRecvLayoutRange.dim() == 2, "packedRecvLayoutRange must be 2D");
  if (topkWeights.has_value()) {
    TORCH_CHECK(topkWeights->is_cuda(), "topkWeights must be a CUDA tensor");
    TORCH_CHECK(topkWeights->dim() == 2, "topkWeights must be 2D");
    TORCH_CHECK(topkIdx.size(0) == topkWeights->size(0),
                "topkIdx and topkWeights must have the same first dimension");
    TORCH_CHECK(topkIdx.size(1) == topkWeights->size(1),
                "topkIdx and topkWeights must have the same second dimension");
  }

  using torch::indexing::Slice;

  const int64_t numLocalExperts = handle.config.numExpertPerRank;
  const int64_t numRanks = handle.config.worldSize;
  const int64_t maxTokensPerExpert =
      static_cast<int64_t>(handle.config.worldSize) * handle.config.maxNumInpTokenPerRank;
  const int64_t hidden = packedRecvX.size(2);
  TORCH_CHECK(packedRecvX.size(0) == numLocalExperts,
              "packedRecvX local expert dimension mismatch");
  TORCH_CHECK(packedRecvX.size(1) == maxTokensPerExpert,
              "packedRecvX token dimension mismatch");
  TORCH_CHECK(packedRecvSrcInfo.size(0) == numLocalExperts,
              "packedRecvSrcInfo local expert dimension mismatch");
  TORCH_CHECK(packedRecvSrcInfo.size(1) == maxTokensPerExpert,
              "packedRecvSrcInfo token dimension mismatch");
  TORCH_CHECK(packedRecvLayoutRange.size(0) == numLocalExperts,
              "packedRecvLayoutRange local expert dimension mismatch");
  TORCH_CHECK(packedRecvLayoutRange.size(1) == numRanks,
              "packedRecvLayoutRange rank dimension mismatch");

  torch::Tensor dispatchSrcTokenPos = GetDispatchSrcTokenId(handle);
  const int64_t totalTokens = dispatchSrcTokenPos.size(0);
  TORCH_CHECK(totalTokens >= 0, "totalTokens must be non-negative");
  TORCH_CHECK(topkIdx.size(0) >= totalTokens,
              "topkIdx has fewer rows than totalTokens");
  if (topkWeights.has_value()) {
    TORCH_CHECK(topkWeights->size(0) >= totalTokens,
                "topkWeights has fewer rows than totalTokens");
  }

  auto options = packedRecvX.options();
  auto accOptions = options.dtype(torch::kFloat32);
  torch::Tensor combineInputAcc = torch::zeros({totalTokens, hidden}, accOptions);

  torch::Tensor slicedTopkIdx = topkIdx.index({Slice(0, totalTokens)});
  auto mapOptions =
      torch::TensorOptions().dtype(torch::kInt64).device(packedRecvX.device());
  const int64_t maxSrcTokens =
      static_cast<int64_t>(handle.config.worldSize) * handle.config.maxNumInpTokenPerRank;
  torch::Tensor srcToDispatch = torch::full({maxSrcTokens}, -1, mapOptions);
  torch::Tensor dispatchIdx1d = torch::arange(totalTokens, mapOptions);
  srcToDispatch.index_put_({dispatchSrcTokenPos.to(torch::kInt64)}, dispatchIdx1d);

  for (int64_t localExpert = 0; localExpert < numLocalExperts; ++localExpert) {
    const int64_t globalExpertId = handle.config.rank * numLocalExperts + localExpert;
    std::optional<torch::Tensor> expertWeights{std::nullopt};
    if (topkWeights.has_value()) {
      auto expertMask = slicedTopkIdx.eq(globalExpertId);
      expertWeights =
          (topkWeights.value().index({Slice(0, totalTokens)}) *
           expertMask.to(topkWeights->scalar_type()))
              .sum(1);
    }

    for (int64_t srcRank = 0; srcRank < numRanks; ++srcRank) {
      const int64_t packedRange =
          packedRecvLayoutRange.index({localExpert, srcRank}).item<int64_t>();
      const int64_t beginIdx = packedRange >> 32;
      const int64_t count = packedRange & 0xffffffff;
      if (count <= 0) {
        continue;
      }

      torch::Tensor slotRange =
          torch::arange(beginIdx, beginIdx + count, mapOptions);
      torch::Tensor srcInfo =
          packedRecvSrcInfo.index({localExpert, slotRange}).to(torch::kInt64);
      torch::Tensor srcTokenPos = srcInfo + srcRank * handle.config.maxNumInpTokenPerRank;
      torch::Tensor dispatchIdx = srcToDispatch.index_select(0, srcTokenPos);
      torch::Tensor valid = dispatchIdx.ge(0);
      if (valid.any().item<bool>() == false) {
        continue;
      }
      torch::Tensor tokenIdx = torch::nonzero(valid).squeeze(1);
      torch::Tensor flatDispatchIdx = dispatchIdx.masked_select(valid);

      torch::Tensor tokens = packedRecvX.index({localExpert, slotRange}).to(torch::kFloat32);
      torch::Tensor selectedTokens = tokens.index_select(0, tokenIdx);
      if (expertWeights.has_value()) {
        torch::Tensor weights = expertWeights->index_select(0, flatDispatchIdx).unsqueeze(1);
        weights = weights.to(torch::kFloat32);
        combineInputAcc.index_put_(
            {flatDispatchIdx},
            combineInputAcc.index({flatDispatchIdx}) + selectedTokens * weights);
      } else {
        combineInputAcc.index_put_({flatDispatchIdx},
                                  combineInputAcc.index({flatDispatchIdx}) + selectedTokens);
      }
    }
  }

  return combineInputAcc.to(options.dtype());
}
#else
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ConvertDispatchOutput(
    mori::moe::EpDispatchCombineHandle& handle, const torch::Tensor& dispatchOutX,
    const torch::Tensor& dispatchOutTopkIdx,
    int blockNum = -1, int warpPerBlock = -1) {
  TORCH_CHECK(dispatchOutX.is_cuda(), "dispatchOutX must be a CUDA tensor");
  TORCH_CHECK(dispatchOutTopkIdx.is_cuda(), "dispatchOutTopkIdx must be a CUDA tensor");
  TORCH_CHECK(dispatchOutX.dim() == 2, "dispatchOutX must be 2D");
  TORCH_CHECK(dispatchOutTopkIdx.dim() == 2, "dispatchOutTopkIdx must be 2D");
  TORCH_CHECK(dispatchOutX.size(0) == dispatchOutTopkIdx.size(0),
              "dispatchOutX and dispatchOutTopkIdx must have the same first dimension");

  const int64_t numLocalExperts = handle.config.numExpertPerRank;
  const int64_t maxTokensPerExpert =
      static_cast<int64_t>(handle.config.worldSize) * handle.config.maxNumInpTokenPerRank;
  const int64_t hidden = dispatchOutX.size(1);

  torch::Tensor packedRecvX =
      torch::empty({numLocalExperts, maxTokensPerExpert, hidden}, dispatchOutX.options());
  auto packedRecvSrcInfo =
      torch::empty({numLocalExperts, handle.config.worldSize * handle.config.maxNumInpTokenPerRank},
                   torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto packedRecvLayoutRange = torch::empty({numLocalExperts, handle.config.worldSize},
                                            torch::dtype(torch::kInt64).device(torch::kCUDA));

  handle.LaunchConvertDispatchOutputKernel(dispatchOutX.data_ptr(), dispatchOutTopkIdx.data_ptr(),
                                           packedRecvX.data_ptr(), handle.standardPackedRecvCount,
                                           packedRecvSrcInfo.data_ptr<int>(),
                                           packedRecvLayoutRange.data_ptr<int64_t>(), blockNum,
                                           warpPerBlock, at::cuda::getCurrentHIPStream());

  torch::Tensor packedRecvCount =
      torch::from_blob(handle.standardPackedRecvCount, {numLocalExperts},
                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

  return {packedRecvX, packedRecvCount, packedRecvSrcInfo, packedRecvLayoutRange};
}

torch::Tensor ConvertCombineInput(mori::moe::EpDispatchCombineHandle& handle,
                                  const torch::Tensor& packedRecvX,
                                  const torch::Tensor& packedRecvSrcInfo,
                                  const torch::Tensor& packedRecvLayoutRange,
                                  int blockNum = -1, int warpPerBlock = -1) {
  TORCH_CHECK(packedRecvX.is_cuda() && packedRecvSrcInfo.is_cuda() &&
                  packedRecvLayoutRange.is_cuda(),
              "packedRecvX/packedRecvSrcInfo/packedRecvLayoutRange must be CUDA tensors");
  TORCH_CHECK(packedRecvX.dim() == 3 && packedRecvSrcInfo.dim() == 2 &&
                  packedRecvLayoutRange.dim() == 2,
              "packedRecvX must be 3D; others must be 2D");

  const int64_t numLocalExperts = handle.config.numExpertPerRank;
  const int64_t numRanks = handle.config.worldSize;
  const int64_t maxTokensPerExpert =
      static_cast<int64_t>(handle.config.worldSize) * handle.config.maxNumInpTokenPerRank;
  const int64_t hidden = packedRecvX.size(2);
  TORCH_CHECK(packedRecvX.size(0) == numLocalExperts &&
                  packedRecvSrcInfo.size(0) == numLocalExperts &&
                  packedRecvLayoutRange.size(0) == numLocalExperts,
              "local expert dimension mismatch");
  TORCH_CHECK(packedRecvX.size(1) == maxTokensPerExpert &&
                  packedRecvSrcInfo.size(1) == maxTokensPerExpert &&
                  packedRecvLayoutRange.size(1) == numRanks,
              "token/rank dimension mismatch");

  // totalTokens comes from handle.totalRecvTokenNum inside the kernel

  auto options = packedRecvX.options();
  // torch::Tensor combineInput = torch::empty({handle.config.MaxNumTokensToRecv(), hidden}, options);
  torch::Tensor combineInput =
      torch::from_blob(handle.shmemCombineInpTokMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), hidden}, options);

  handle.LaunchConvertCombineInputKernel(
      packedRecvX.data_ptr(), packedRecvSrcInfo.data_ptr(), packedRecvLayoutRange.data_ptr(),
      combineInput.data_ptr(), handle.shmemCombineInpTokMemObj, blockNum, warpPerBlock,
      at::cuda::getCurrentHIPStream());

  return combineInput;
}
#endif

torch::Tensor GetDispatchSenderTokenIdxMap(mori::moe::EpDispatchCombineHandle& handle) {
  auto options = torch::TensorOptions()
                     .dtype(mori::GetTorchDataType<mori::moe::index_t>())
                     .device(torch::kCUDA);
  torch::Tensor tensor = torch::from_blob(
      handle.dispSenderIdxMap, {handle.curRankNumToken * handle.config.numExpertPerToken}, options);
  return tensor;
}

torch::Tensor GetDispatchReceiverTokenIdxMap(mori::moe::EpDispatchCombineHandle& handle) {
  auto options = torch::TensorOptions()
                     .dtype(mori::GetTorchDataType<mori::moe::index_t>())
                     .device(torch::kCUDA);
  torch::Tensor tensor =
      torch::from_blob(handle.dispReceiverIdxMap, {*handle.localPeTokenCounter}, options);
  return tensor;
}

torch::Tensor GetRegisteredCombineInputBuffer(mori::moe::EpDispatchCombineHandle& handle,
                                              at::ScalarType scalarType) {
  torch::Tensor out =
      torch::from_blob(handle.shmemCombineInpTokMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), handle.config.hiddenDim},
                       torch::TensorOptions().dtype(scalarType).device(torch::kCUDA));
  return out;
}

#ifdef ENABLE_PROFILER
torch::Tensor GetDebugTimeBuf(mori::moe::EpDispatchCombineHandle& handle) {
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  torch::Tensor tensor =
      torch::from_blob(handle.profilerConfig.debugTimeBuf, {MAX_DEBUG_TIME_SLOTS}, options);
  return tensor;
}

torch::Tensor GetDebugTimeOffset(mori::moe::EpDispatchCombineHandle& handle) {
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor tensor =
      torch::from_blob(handle.profilerConfig.debugTimeOffset, {PROFILER_WARPS_PER_RANK}, options);
  return tensor;
}
#endif

int GetCurDeviceWallClockFreqMhz() { return mori::GetCurDeviceWallClockFreqMhz(); }

void DeclareEpDispatchCombineHandle(pybind11::module& m) {
  std::string className = std::string("EpDispatchCombineHandle");
  pybind11::class_<mori::moe::EpDispatchCombineHandle>(m, className.c_str())
      .def(pybind11::init<mori::moe::EpDispatchCombineConfig>(),
           py::arg("config") = mori::moe::EpDispatchCombineConfig{});

  std::string funcName = std::string("launch_dispatch");
  m.def(funcName.c_str(), &LaunchDispatch);

  funcName = std::string("launch_combine");
  m.def(funcName.c_str(), &LaunchCombine);

  funcName = std::string("launch_dispatch_standard_moe");
  m.def(funcName.c_str(), &LaunchDispatchForStandardMoE);

  funcName = std::string("launch_combine_standard_moe");
  m.def(funcName.c_str(), &LaunchCombineForStandardMoE);

  funcName = std::string("convert_dispatch_output");
  m.def(funcName.c_str(), &ConvertDispatchOutput);

  funcName = std::string("convert_combine_input");
  m.def(funcName.c_str(), &ConvertCombineInput);

  funcName = std::string("launch_reset");
  m.def(funcName.c_str(), &LaunchReset);

  funcName = std::string("get_cur_rank_num_token");
  m.def(funcName.c_str(), &mori::moe::EpDispatchCombineHandle::GetCurRankNumToken);

  funcName = std::string("get_dispatch_src_token_pos");
  m.def(funcName.c_str(), &GetDispatchSrcTokenId);

  funcName = std::string("get_dispatch_sender_token_idx_map");
  m.def(funcName.c_str(), &GetDispatchSenderTokenIdxMap);

  funcName = std::string("get_dispatch_receiver_token_idx_map");
  m.def(funcName.c_str(), &GetDispatchReceiverTokenIdxMap);

  funcName = std::string("get_registered_combine_input_buffer");
  m.def(funcName.c_str(), &GetRegisteredCombineInputBuffer);

#ifdef ENABLE_PROFILER
  funcName = std::string("get_debug_time_buf");
  m.def(funcName.c_str(), &GetDebugTimeBuf);

  funcName = std::string("get_debug_time_offset");
  m.def(funcName.c_str(), &GetDebugTimeOffset);
#endif
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                           Shmem APIs                                           */
/* ---------------------------------------------------------------------------------------------- */
namespace {
int64_t ShmemTorchProcessGroupInit(const std::string& groupName) {
  return mori::shmem::ShmemTorchProcessGroupInit(groupName);
}

int64_t ShmemFinalize() { return mori::shmem::ShmemFinalize(); }

int64_t ShmemModuleInit(uint64_t hipModule) {
  return mori::shmem::ShmemModuleInit(reinterpret_cast<void*>(hipModule));
}

int64_t ShmemMyPe() { return mori::shmem::ShmemMyPe(); }

int64_t ShmemNPes() { return mori::shmem::ShmemNPes(); }

// UniqueId-based initialization APIs
py::bytes ShmemGetUniqueId() {
  mori::shmem::mori_shmem_uniqueid_t uid;
  mori::shmem::ShmemGetUniqueId(&uid);
  return py::bytes(reinterpret_cast<const char*>(uid.data()), uid.size());
}

int64_t ShmemInitAttr(unsigned int flags, int32_t rank, int32_t nranks,
                      const py::bytes& uid_bytes) {
  mori::shmem::mori_shmem_init_attr_t attr;
  mori::shmem::mori_shmem_uniqueid_t uid;

  // Convert Python bytes to uniqueid
  Py_ssize_t len = PyBytes_Size(uid_bytes.ptr());
  const char* data = PyBytes_AsString(uid_bytes.ptr());
  if (len != MORI_SHMEM_UNIQUE_ID_BYTES) {
    throw std::runtime_error("Invalid unique ID size");
  }
  std::memcpy(uid.data(), data, MORI_SHMEM_UNIQUE_ID_BYTES);

  // Set attributes
  mori::shmem::ShmemSetAttrUniqueIdArgs(rank, nranks, &uid, &attr);

  return mori::shmem::ShmemInitAttr(flags, &attr);
}

void ShmemBarrierAll() { mori::shmem::ShmemBarrierAll(); }

// Symmetric memory APIs
uintptr_t ShmemMalloc(size_t size) {
  void* ptr = mori::shmem::ShmemMalloc(size);
  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t ShmemMallocAlign(size_t alignment, size_t size) {
  void* ptr = mori::shmem::ShmemMallocAlign(alignment, size);
  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t ShmemExtMallocWithFlags(size_t size, unsigned int flags) {
  void* ptr = mori::shmem::ShmemExtMallocWithFlags(size, flags);
  return reinterpret_cast<uintptr_t>(ptr);
}

void ShmemFree(uintptr_t ptr) { mori::shmem::ShmemFree(reinterpret_cast<void*>(ptr)); }

int64_t ShmemBufferRegister(uintptr_t ptr, size_t size) {
  return mori::shmem::ShmemBufferRegister(reinterpret_cast<void*>(ptr), size);
}

int64_t ShmemBufferDeregister(uintptr_t ptr, size_t size) {
  return mori::shmem::ShmemBufferDeregister(reinterpret_cast<void*>(ptr), size);
}

// P2P address translation
uint64_t ShmemPtrP2p(uint64_t destPtr, int myPe, int destPe) {
  return mori::shmem::ShmemPtrP2p(destPtr, myPe, destPe);
}

int64_t ShmemNumQpPerPe() { return mori::shmem::ShmemNumQpPerPe(); }

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                             IO APIs                                            */
/* ---------------------------------------------------------------------------------------------- */

namespace mori {

void RegisterMoriOps(py::module_& m) {
  pybind11::enum_<mori::moe::KernelType>(m, "EpDispatchCombineKernelType")
      .value("IntraNode", mori::moe::KernelType::IntraNode)
      .value("InterNode", mori::moe::KernelType::InterNode)
      .value("InterNodeV1", mori::moe::KernelType::InterNodeV1)
      .value("InterNodeV1LL", mori::moe::KernelType::InterNodeV1LL)
      .export_values();

  mori::pybind::RegisterAllProfilerSlots(m);

  pybind11::class_<mori::moe::EpDispatchCombineConfig>(m, "EpDispatchCombineConfig")
      .def(pybind11::init<int, int, int, int, int, int, int, int, int, int, int, bool,
                          mori::moe::KernelType, int, int, int>(),
           py::arg("rank") = 0, py::arg("world_size") = 0, py::arg("hidden_dim") = 0,
           py::arg("scale_dim") = 0, py::arg("scale_type_size") = 0,
           py::arg("max_token_type_size") = 0, py::arg("max_num_inp_token_per_rank") = 0,
           py::arg("num_experts_per_rank") = 0, py::arg("num_experts_per_token") = 0,
           py::arg("warp_num_per_block") = 0, py::arg("block_num") = 0,
           py::arg("use_external_inp_buf") = true,
           py::arg("kernel_type") = mori::moe::KernelType::IntraNode, py::arg("gpu_per_node") = 8,
           py::arg("rdma_block_num") = 0, py::arg("num_qp_per_pe") = 1)
      .def_readwrite("rank", &mori::moe::EpDispatchCombineConfig::rank)
      .def_readwrite("world_size", &mori::moe::EpDispatchCombineConfig::worldSize)
      .def_readwrite("hidden_dim", &mori::moe::EpDispatchCombineConfig::hiddenDim)
      .def_readwrite("scale_dim", &mori::moe::EpDispatchCombineConfig::scaleDim)
      .def_readwrite("scale_type_size", &mori::moe::EpDispatchCombineConfig::scaleTypeSize)
      .def_readwrite("max_token_type_size", &mori::moe::EpDispatchCombineConfig::maxTokenTypeSize)
      .def_readwrite("max_num_inp_token_per_rank",
                     &mori::moe::EpDispatchCombineConfig::maxNumInpTokenPerRank)
      .def_readwrite("num_experts_per_rank", &mori::moe::EpDispatchCombineConfig::numExpertPerRank)
      .def_readwrite("num_experts_per_token",
                     &mori::moe::EpDispatchCombineConfig::numExpertPerToken)
      .def_readwrite("warp_num_per_block", &mori::moe::EpDispatchCombineConfig::warpNumPerBlock)
      .def_readwrite("block_num", &mori::moe::EpDispatchCombineConfig::blockNum)
      .def_readwrite("kernel_type", &mori::moe::EpDispatchCombineConfig::kernelType)
      .def_readwrite("gpu_per_node", &mori::moe::EpDispatchCombineConfig::gpuPerNode)
      .def_readwrite("rdma_block_num", &mori::moe::EpDispatchCombineConfig::rdmaBlockNum)
      .def_readwrite("num_qp_per_pe", &mori::moe::EpDispatchCombineConfig::numQpPerPe);

  m.attr("topk_idx_t") = py::reinterpret_borrow<py::object>(
      (PyObject*)torch::getTHPDtype(c10::CppTypeToScalarType<mori::moe::index_t>::value));

  DeclareEpDispatchCombineHandle(m);

  m.def("get_cur_device_wall_clock_freq_mhz", &GetCurDeviceWallClockFreqMhz,
        "Returns clock frequency of current device's wall clock");
}

void RegisterMoriShmem(py::module_& m) {
  // Initialization flags
  m.attr("MORI_SHMEM_INIT_WITH_MPI_COMM") = mori::shmem::MORI_SHMEM_INIT_WITH_MPI_COMM;
  m.attr("MORI_SHMEM_INIT_WITH_UNIQUEID") = mori::shmem::MORI_SHMEM_INIT_WITH_UNIQUEID;

  // Traditional initialization APIs
  m.def("shmem_torch_process_group_init", &ShmemTorchProcessGroupInit, py::arg("group_name"),
        "Initialize shmem from PyTorch process group");

  // UniqueId-based initialization APIs (nvshmem/rocshmem compatible)
  m.def("shmem_get_unique_id", &ShmemGetUniqueId,
        "Get a unique ID for shmem initialization (returns bytes)");

  m.def("shmem_init_attr", &ShmemInitAttr, py::arg("flags"), py::arg("rank"), py::arg("nranks"),
        py::arg("unique_id"),
        "Initialize shmem with attributes (unique_id should be bytes from shmem_get_unique_id)");

  m.def("shmem_finalize", &ShmemFinalize, "Finalize shmem");

  //  Module-specific initialization (for Triton kernels)
  m.def("shmem_module_init", &ShmemModuleInit, py::arg("hip_module"),
        "Initialize globalGpuStates in a specific HIP module (for Triton kernels)");

  // Query APIs
  m.def("shmem_mype", &ShmemMyPe, "Get my PE (process element) ID");

  m.def("shmem_npes", &ShmemNPes, "Get number of PEs");

  // Collective operations
  m.def("shmem_barrier_all", &ShmemBarrierAll, "Global barrier synchronization");

  // Symmetric memory management
  m.def("shmem_malloc", &ShmemMalloc, py::arg("size"),
        "Allocate symmetric memory (returns address as int)");

  m.def("shmem_malloc_align", &ShmemMallocAlign, py::arg("alignment"), py::arg("size"),
        "Allocate aligned symmetric memory (returns address as int)");

  m.def("shmem_ext_malloc_with_flags", &ShmemExtMallocWithFlags, py::arg("size"), py::arg("flags"),
        "Allocate symmetric memory with flags (returns address as int)");

  m.def("shmem_free", &ShmemFree, py::arg("ptr"),
        "Free symmetric memory (ptr should be int address)");

  // Buffer registration
  m.def("shmem_buffer_register", &ShmemBufferRegister, py::arg("ptr"), py::arg("size"),
        "Register an existing buffer for RDMA (ptr should be int address)");

  m.def("shmem_buffer_deregister", &ShmemBufferDeregister, py::arg("ptr"), py::arg("size"),
        "Deregister a buffer from RDMA (ptr should be int address)");

  // P2P address translation
  m.def("shmem_ptr_p2p", &ShmemPtrP2p, py::arg("dest_ptr"), py::arg("my_pe"), py::arg("dest_pe"),
        "Convert local symmetric memory pointer to remote P2P address. "
        "Returns 0 if connection uses RDMA or if pointer is invalid. "
        "Returns P2P accessible address if connection uses P2P transport.");
  m.def("shmem_torch_process_group_init", &ShmemTorchProcessGroupInit);
  m.def("shmem_finalize", &ShmemFinalize);
  m.def("shmem_mype", &ShmemMyPe);
  m.def("shmem_npes", &ShmemNPes);
  m.def("shmem_num_qp_per_pe", &ShmemNumQpPerPe);
}

void RegisterMoriIo(pybind11::module_& m) {
  m.def("set_log_level", &mori::io::SetLogLevel);

  py::enum_<mori::io::BackendType>(m, "BackendType")
      .value("Unknown", mori::io::BackendType::Unknown)
      .value("XGMI", mori::io::BackendType::XGMI)
      .value("RDMA", mori::io::BackendType::RDMA)
      .value("TCP", mori::io::BackendType::TCP)
      .export_values();

  py::enum_<mori::io::MemoryLocationType>(m, "MemoryLocationType")
      .value("Unknown", mori::io::MemoryLocationType::Unknown)
      .value("CPU", mori::io::MemoryLocationType::CPU)
      .value("GPU", mori::io::MemoryLocationType::GPU)
      .export_values();

  py::enum_<mori::io::StatusCode>(m, "StatusCode")
      .value("SUCCESS", mori::io::StatusCode::SUCCESS)
      .value("INIT", mori::io::StatusCode::INIT)
      .value("IN_PROGRESS", mori::io::StatusCode::IN_PROGRESS)
      .value("ERR_INVALID_ARGS", mori::io::StatusCode::ERR_INVALID_ARGS)
      .value("ERR_NOT_FOUND", mori::io::StatusCode::ERR_NOT_FOUND)
      .value("ERR_RDMA_OP", mori::io::StatusCode::ERR_RDMA_OP)
      .value("ERR_BAD_STATE", mori::io::StatusCode::ERR_BAD_STATE)
      .value("ERR_GPU_OP", mori::io::StatusCode::ERR_GPU_OP)
      .export_values();

  py::enum_<mori::io::PollCqMode>(m, "PollCqMode")
      .value("POLLING", mori::io::PollCqMode::POLLING)
      .value("EVENT", mori::io::PollCqMode::EVENT);

  py::class_<mori::io::BackendConfig>(m, "BackendConfig");

  py::class_<mori::io::RdmaBackendConfig, mori::io::BackendConfig>(m, "RdmaBackendConfig")
      .def(py::init<int, int, int, mori::io::PollCqMode, bool>(), py::arg("qp_per_transfer") = 1,
           py::arg("post_batch_size") = -1, py::arg("num_worker_threads") = -1,
           py::arg("poll_cq_mode") = mori::io::PollCqMode::POLLING,
           py::arg("enable_notification") = true)
      .def_readwrite("qp_per_transfer", &mori::io::RdmaBackendConfig::qpPerTransfer)
      .def_readwrite("post_batch_size", &mori::io::RdmaBackendConfig::postBatchSize)
      .def_readwrite("num_worker_threads", &mori::io::RdmaBackendConfig::numWorkerThreads)
      .def_readwrite("poll_cq_mode", &mori::io::RdmaBackendConfig::pollCqMode)
      .def_readwrite("enable_notification", &mori::io::RdmaBackendConfig::enableNotification);

  py::class_<mori::io::XgmiBackendConfig, mori::io::BackendConfig>(m, "XgmiBackendConfig")
      .def(py::init<int, int>(), py::arg("num_streams") = 64, py::arg("num_events") = 64)
      .def_readwrite("num_streams", &mori::io::XgmiBackendConfig::numStreams)
      .def_readwrite("num_events", &mori::io::XgmiBackendConfig::numEvents);

  py::class_<mori::io::IOEngineConfig>(m, "IOEngineConfig")
      .def(py::init<std::string, uint16_t>(), py::arg("host") = "", py::arg("port") = 0)
      .def_readwrite("host", &mori::io::IOEngineConfig::host)
      .def_readwrite("port", &mori::io::IOEngineConfig::port);

  py::class_<mori::io::TransferStatus>(m, "TransferStatus")
      .def(py::init<>())
      .def("Code", &mori::io::TransferStatus::Code)
      .def("Message", &mori::io::TransferStatus::Message)
      .def("Update", &mori::io::TransferStatus::Update)
      .def("Init", &mori::io::TransferStatus::Init)
      .def("InProgress", &mori::io::TransferStatus::InProgress)
      .def("Succeeded", &mori::io::TransferStatus::Succeeded)
      .def("Failed", &mori::io::TransferStatus::Failed)
      .def("SetCode", &mori::io::TransferStatus::SetCode)
      .def("SetMessage", &mori::io::TransferStatus::SetMessage)
      .def("Wait", &mori::io ::TransferStatus::Wait);

  py::class_<mori::io::EngineDesc>(m, "EngineDesc")
      .def_readonly("key", &mori::io::EngineDesc::key)
      .def_readonly("hostname", &mori::io::EngineDesc::hostname)
      .def_readonly("host", &mori::io::EngineDesc::host)
      .def_readonly("port", &mori::io::EngineDesc::port)
      .def(pybind11::self == pybind11::self)
      .def("pack",
           [](const mori::io::EngineDesc& d) {
             msgpack::sbuffer buf;
             msgpack::pack(buf, d);
             return py::bytes(buf.data(), buf.size());
           })
      .def_static("unpack", [](const py::bytes& b) {
        Py_ssize_t len = PyBytes_Size(b.ptr());
        const char* data = PyBytes_AsString(b.ptr());
        auto out = msgpack::unpack(data, len);
        return out.get().as<mori::io::EngineDesc>();
      });

  py::class_<mori::io::MemoryDesc>(m, "MemoryDesc")
      .def(py::init<>())
      .def_readonly("engine_key", &mori::io::MemoryDesc::engineKey)
      .def_readonly("id", &mori::io::MemoryDesc::id)
      .def_readonly("device_id", &mori::io::MemoryDesc::deviceId)
      .def_property_readonly("data",
                             [](const mori::io::MemoryDesc& desc) -> uintptr_t {
                               return reinterpret_cast<uintptr_t>(desc.data);
                             })
      .def_readonly("size", &mori::io::MemoryDesc::size)
      .def_readonly("loc", &mori::io::MemoryDesc::loc)
      .def_property_readonly("ipc_handle",
                             [](const mori::io::MemoryDesc& desc) {
                               return py::bytes(desc.ipcHandle.data(), desc.ipcHandle.size());
                             })
      .def(pybind11::self == pybind11::self)
      .def("pack",
           [](const mori::io::MemoryDesc& d) {
             msgpack::sbuffer buf;
             msgpack::pack(buf, d);
             return py::bytes(buf.data(), buf.size());
           })
      .def_static("unpack", [](const py::bytes& b) {
        Py_ssize_t len = PyBytes_Size(b.ptr());
        const char* data = PyBytes_AsString(b.ptr());
        auto out = msgpack::unpack(data, len);
        return out.get().as<mori::io::MemoryDesc>();
      });

  py::class_<mori::io::IOEngineSession>(m, "IOEngineSession")
      .def("AllocateTransferUniqueId", &mori::io ::IOEngineSession::AllocateTransferUniqueId)
      .def("Read", &mori::io ::IOEngineSession::Read)
      .def("BatchRead", &mori::io ::IOEngineSession::BatchRead)
      .def("Write", &mori::io ::IOEngineSession::Write)
      .def("BatchWrite", &mori::io ::IOEngineSession::BatchWrite)
      .def("Alive", &mori::io ::IOEngineSession::Alive);

  py::class_<mori::io::IOEngine>(m, "IOEngine")
      .def(py::init<const mori::io::EngineKey&, const mori::io::IOEngineConfig&>())
      .def("GetEngineDesc", &mori::io ::IOEngine::GetEngineDesc)
      .def("CreateBackend", &mori::io::IOEngine::CreateBackend)
      .def("RemoveBackend", &mori::io ::IOEngine::RemoveBackend)
      .def("RegisterRemoteEngine", &mori::io ::IOEngine::RegisterRemoteEngine)
      .def("DeregisterRemoteEngine", &mori::io ::IOEngine::DeregisterRemoteEngine)
      .def("RegisterMemory", &mori::io ::IOEngine::RegisterMemory)
      .def("DeregisterMemory", &mori::io ::IOEngine::DeregisterMemory)
      .def("AllocateTransferUniqueId", &mori::io ::IOEngine::AllocateTransferUniqueId)
      .def("Read", &mori::io ::IOEngine::Read)
      .def("BatchRead", &mori::io ::IOEngine::BatchRead)
      .def("Write", &mori::io ::IOEngine::Write)
      .def("BatchWrite", &mori::io ::IOEngine::BatchWrite)
      .def("CreateSession", &mori::io::IOEngine::CreateSession)
      .def("PopInboundTransferStatus", &mori::io::IOEngine::PopInboundTransferStatus);
}

}  // namespace mori
