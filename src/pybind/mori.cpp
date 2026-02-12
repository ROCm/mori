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
#include "src/pybind/mori_core.hpp"

#include <ATen/hip/HIPContext.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/python.h>

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "mori/application/application.hpp"
#include "mori/core/profiler/constants.hpp"
#include "mori/io/io.hpp"
#include "mori/ops/ops.hpp"
#include "mori/pybind/profiler_registry.hpp"
#include "mori/shmem/shmem.hpp"
#include "mori/utils/data_types.hpp"
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
               int blockNum = -1, int rdmaBlockNum = -1, int warpPerBlock = -1) {
  TORCH_CHECK(input.is_contiguous(), "dispatch input must be contiguous");
  TORCH_CHECK(topkIds.is_contiguous(), "dispatch topkIds must be contiguous");
  const int hiddenDim = static_cast<int>(input.size(1));
  TORCH_CHECK(hiddenDim > 0, "dispatch input hidden dim must be > 0");
  TORCH_CHECK(hiddenDim <= handle.config.hiddenDim, "dispatch input hidden dim ", hiddenDim,
              " exceeds config.hidden_dim ", handle.config.hiddenDim);

  float* weightPtr = nullptr;
  if (weights.has_value()) {
    TORCH_CHECK(weights->is_contiguous(), "dispatch weights must be contiguous");
    TORCH_CHECK(weights->element_size() == sizeof(float),
                "dispatch weights must have element size ", sizeof(float), ", got ",
                weights->element_size());
    weightPtr = weights->data_ptr<float>();
  }

  uint8_t* scalePtr = nullptr;
  if (scales.has_value() && (handle.config.scaleDim > 0)) {
    TORCH_CHECK(scales->is_contiguous(), "dispatch scales must be contiguous");
    TORCH_CHECK(scales->element_size() == handle.config.scaleTypeSize,
                "dispatch scales element size mismatch, expected ",
                handle.config.scaleTypeSize, ", got ", scales->element_size());
    scalePtr = reinterpret_cast<uint8_t*>(scales->data_ptr());
  }

  handle.PrepareInference(mori::ScalarTypeToHipDataType(input.scalar_type()), input.data_ptr(),
                          nullptr, weightPtr, scalePtr, topkIds.data_ptr<mori::moe::index_t>(),
                          input.size(0));
  handle.LaunchDispatch((mori::moe::KernelType)kernelType, blockNum, rdmaBlockNum, warpPerBlock,
                        at::cuda::getCurrentHIPStream(), hiddenDim);

  torch::Tensor out =
      torch::from_blob(handle.shmemDispatchOutTokMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), hiddenDim},
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

std::tuple<torch::Tensor, std::optional<torch::Tensor>> LaunchCombine(
    mori::moe::EpDispatchCombineHandle& handle, int kernelType, const torch::Tensor& input,
    const std::optional<torch::Tensor>& weights, const torch::Tensor& topkIds, int blockNum = -1,
    int rdmaBlockNum = -1, int warpPerBlock = -1, int useExternalInpBuf = -1) {
  TORCH_CHECK(input.is_contiguous(), "combine input must be contiguous");
  TORCH_CHECK(topkIds.is_contiguous(), "combine topkIds must be contiguous");
  const int hiddenDim = static_cast<int>(input.size(1));
  TORCH_CHECK(hiddenDim > 0, "combine input hidden dim must be > 0");
  TORCH_CHECK(hiddenDim <= handle.config.hiddenDim, "combine input hidden dim ", hiddenDim,
              " exceeds config.hidden_dim ", handle.config.hiddenDim);

  float* weightsPtr = nullptr;
  if (weights.has_value() && weights->size(0) != 0) {
    TORCH_CHECK(weights->is_contiguous(), "combine weights must be contiguous");
    weightsPtr = weights->data_ptr<float>();
  }

  handle.PrepareInference(mori::ScalarTypeToHipDataType(input.scalar_type()), input.data_ptr(),
                          nullptr, weightsPtr, topkIds.data_ptr<mori::moe::index_t>(),
                          handle.curRankNumToken);
  handle.LaunchCombine((mori::moe::KernelType)kernelType, blockNum, rdmaBlockNum, warpPerBlock,
                       useExternalInpBuf, at::cuda::getCurrentHIPStream(), hiddenDim);

  auto options = torch::TensorOptions().dtype(input.scalar_type()).device(torch::kCUDA);
  torch::Tensor out =
      torch::from_blob(handle.shmemCombineOutTokMemObj->Get(),
                       {handle.config.maxNumInpTokenPerRank, hiddenDim}, options);

  std::optional<torch::Tensor> outWeights{std::nullopt};
  if (weightsPtr) {
    outWeights =
        torch::from_blob(handle.shmemCombineOutWeightsMemObj->Get(),
                         {handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
                         torch::TensorOptions().dtype(weights->scalar_type()).device(torch::kCUDA));
  }

  return {out, outWeights};
}

#ifdef ENABLE_STANDARD_MOE_ADAPT
// Standard MoE 3D output: packedRecvX/Count/SrcInfo/LayoutRange
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LaunchDispatchForStandardMoE(
    mori::moe::EpDispatchCombineHandle& handle, int kernelType, const torch::Tensor& input,
    const std::optional<torch::Tensor>& weights, const std::optional<torch::Tensor>& scales,
    const torch::Tensor& topkIds, int blockNum = -1, int rdmaBlockNum = -1, int warpPerBlock = -1) {
  TORCH_CHECK(input.is_contiguous(), "dispatch_standard_moe input must be contiguous");
  TORCH_CHECK(topkIds.is_contiguous(), "dispatch_standard_moe topkIds must be contiguous");
  const int hiddenDim = static_cast<int>(input.size(1));
  TORCH_CHECK(hiddenDim > 0, "dispatch_standard_moe input hidden dim must be > 0");
  TORCH_CHECK(hiddenDim <= handle.config.hiddenDim, "dispatch_standard_moe input hidden dim ",
              hiddenDim, " exceeds config.hidden_dim ", handle.config.hiddenDim);

  float* weightPtr = nullptr;
  if (weights.has_value()) {
    TORCH_CHECK(weights->is_contiguous(), "dispatch_standard_moe weights must be contiguous");
    TORCH_CHECK(weights->element_size() == sizeof(float),
                "dispatch_standard_moe weights must have element size ", sizeof(float), ", got ",
                weights->element_size());
    weightPtr = weights->data_ptr<float>();
  }

  uint8_t* scalePtr = nullptr;
  if (scales.has_value() && (handle.config.scaleDim > 0)) {
    TORCH_CHECK(scales->is_contiguous(), "dispatch_standard_moe scales must be contiguous");
    TORCH_CHECK(scales->element_size() == handle.config.scaleTypeSize,
                "dispatch_standard_moe scales element size mismatch, expected ",
                handle.config.scaleTypeSize, ", got ", scales->element_size());
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
  // Not sorted by src token blocks, so layout range is unused (return empty tensor).
  auto packedRecvLayoutRange = torch::empty({0}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  handle.SetStandardMoeOutputBuffers(packedRecvX.data_ptr(), handle.standardPackedRecvCount,
                                     packedRecvSrcInfo.data_ptr<int>(), nullptr);

  handle.LaunchDispatchForStandardMoE((mori::moe::KernelType)kernelType, blockNum, rdmaBlockNum,
                                      warpPerBlock, at::cuda::getCurrentHIPStream(), hiddenDim);

  torch::Tensor packedRecvCount =
      torch::from_blob(handle.standardPackedRecvCount, {numLocalExperts},
                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

  // handle.ClearStandardMoeOutputBuffers();

  return {packedRecvX, packedRecvCount, packedRecvSrcInfo, packedRecvLayoutRange};
}

// Standard MoE combine: takes expert output in packed format and combines back
std::tuple<torch::Tensor, std::optional<torch::Tensor>> LaunchCombineForStandardMoE(
    mori::moe::EpDispatchCombineHandle& handle, int kernelType,
    const torch::Tensor& expertOutput,  // [numLocalExperts, maxTokensPerExpert, hidden]
    const std::optional<torch::Tensor>& weights, const torch::Tensor& topkIds, int blockNum = -1,
    int rdmaBlockNum = -1, int warpPerBlock = -1) {
  TORCH_CHECK(expertOutput.is_contiguous(), "combine_standard_moe expertOutput must be contiguous");
  TORCH_CHECK(topkIds.is_contiguous(), "combine_standard_moe topkIds must be contiguous");
  const int hiddenDim = static_cast<int>(expertOutput.size(2));
  TORCH_CHECK(hiddenDim > 0, "combine_standard_moe input hidden dim must be > 0");
  TORCH_CHECK(hiddenDim <= handle.config.hiddenDim, "combine_standard_moe input hidden dim ",
              hiddenDim, " exceeds config.hidden_dim ", handle.config.hiddenDim);

  float* weightsPtr = nullptr;
  if (weights.has_value() && weights->numel() > 0) {
    TORCH_CHECK(weights->is_contiguous(), "combine_standard_moe weights must be contiguous");
    TORCH_CHECK(weights->element_size() == sizeof(float),
                "combine_standard_moe weights must have element size ", sizeof(float), ", got ",
                weights->element_size());
    weightsPtr = weights->data_ptr<float>();
  }

  // Prepare inference with expert output as input
  handle.PrepareInference(mori::ScalarTypeToHipDataType(expertOutput.scalar_type()),
                          nullptr,  // inpTokenBuf not used for standard moe combine
                          nullptr,  // outTokenBuf
                          weightsPtr, topkIds.data_ptr<mori::moe::index_t>(),
                          handle.curRankNumToken);

  // Set standard MoE input buffers (reusing output buffer fields)
  handle.SetStandardMoeOutputBuffers(expertOutput.data_ptr(), handle.standardPackedRecvCount,
                                     handle.standardPackedRecvSrcInfo,
                                     handle.standardPackedRecvLayoutRange);

  // Launch combine for standard MoE
  handle.LaunchCombineForStandardMoE((mori::moe::KernelType)kernelType, blockNum, rdmaBlockNum,
                                     warpPerBlock, at::cuda::getCurrentHIPStream(), hiddenDim);

  // Get output tensor from shmem buffer
  auto options = torch::TensorOptions().dtype(expertOutput.scalar_type()).device(torch::kCUDA);
  torch::Tensor out =
      torch::from_blob(handle.shmemCombineOutTokMemObj->Get(),
                       {handle.config.maxNumInpTokenPerRank, hiddenDim}, options);

  std::optional<torch::Tensor> outWeights{std::nullopt};
  // TODO: do not support weights for standard MoE now
  // if (weightsPtr) {
  //   outWeights =
  //       torch::from_blob(handle.shmemCombineOutWeightsMemObj->Get(),
  //                        {handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
  //                        torch::TensorOptions().dtype(weights->scalar_type()).device(torch::kCUDA));
  // }

  // handle.ClearStandardMoeOutputBuffers();

  return {out, outWeights};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ConvertDispatchOutput(
    mori::moe::EpDispatchCombineHandle& handle, const torch::Tensor& dispatchOutX,
    const torch::Tensor& dispatchOutTopkIdx, int blockNum = -1, int warpPerBlock = -1) {
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
  TORCH_CHECK(hidden > 0, "dispatchOutX hidden dim must be > 0");
  TORCH_CHECK(hidden <= handle.config.hiddenDim, "dispatchOutX hidden dim ", hidden,
              " exceeds config.hidden_dim ", handle.config.hiddenDim);

  torch::Tensor packedRecvX =
      torch::empty({numLocalExperts, maxTokensPerExpert, hidden}, dispatchOutX.options());
  auto packedRecvSrcInfo =
      torch::empty({numLocalExperts, handle.config.worldSize * handle.config.maxNumInpTokenPerRank},
                   torch::dtype(torch::kInt32).device(torch::kCUDA));
  // Not sorted by src token blocks, so layout range is unused.
  auto packedRecvLayoutRange = torch::empty({0}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  handle.LaunchConvertDispatchOutputKernel(dispatchOutX.data_ptr(), dispatchOutTopkIdx.data_ptr(),
                                           packedRecvX.data_ptr(), handle.standardPackedRecvCount,
                                           packedRecvSrcInfo.data_ptr<int>(), nullptr, blockNum,
                                           warpPerBlock, at::cuda::getCurrentHIPStream(), hidden);

  torch::Tensor packedRecvCount =
      torch::from_blob(handle.standardPackedRecvCount, {numLocalExperts},
                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

  return {packedRecvX, packedRecvCount, packedRecvSrcInfo, packedRecvLayoutRange};
}

torch::Tensor ConvertCombineInput(mori::moe::EpDispatchCombineHandle& handle,
                                  const torch::Tensor& packedRecvX,
                                  const torch::Tensor& packedRecvSrcInfo,
                                  const torch::Tensor& packedRecvLayoutRange, int blockNum = -1,
                                  int warpPerBlock = -1) {
  TORCH_CHECK(packedRecvX.is_cuda() && packedRecvSrcInfo.is_cuda(),
              "packedRecvX/packedRecvSrcInfo must be CUDA tensors");
  TORCH_CHECK(packedRecvX.dim() == 3 && packedRecvSrcInfo.dim() == 2,
              "packedRecvX must be 3D; packedRecvSrcInfo must be 2D");

  const int64_t numLocalExperts = handle.config.numExpertPerRank;
  const int64_t maxTokensPerExpert =
      static_cast<int64_t>(handle.config.worldSize) * handle.config.maxNumInpTokenPerRank;
  const int64_t hidden = packedRecvX.size(2);
  TORCH_CHECK(hidden > 0, "packedRecvX hidden dim must be > 0");
  TORCH_CHECK(hidden <= handle.config.hiddenDim, "packedRecvX hidden dim ", hidden,
              " exceeds config.hidden_dim ", handle.config.hiddenDim);
  TORCH_CHECK(
      packedRecvX.size(0) == numLocalExperts && packedRecvSrcInfo.size(0) == numLocalExperts,
      "local expert dimension mismatch");
  TORCH_CHECK(
      packedRecvX.size(1) == maxTokensPerExpert && packedRecvSrcInfo.size(1) == maxTokensPerExpert,
      "token dimension mismatch");

  auto options = packedRecvX.options();
  // torch::Tensor combineInput = torch::empty({handle.config.MaxNumTokensToRecv(), hidden},
  // options);
  torch::Tensor combineInput =
      torch::from_blob(handle.shmemCombineInpTokMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), hidden}, options);

  // Note: packedRecvLayoutRange is not used in current implementation (passed as nullptr)
  handle.LaunchConvertCombineInputKernel(
      packedRecvX.data_ptr(), packedRecvSrcInfo.data_ptr(), nullptr, combineInput.data_ptr(),
      handle.shmemCombineInpTokMemObj, blockNum, warpPerBlock, at::cuda::getCurrentHIPStream(),
      hidden);

  return combineInput;
}
#endif  // ENABLE_STANDARD_MOE_ADAPT

void LaunchDispatchRecv(mori::moe::EpDispatchCombineHandle& handle, int kernelType,
                        int blockNum = -1, int warpPerBlock = -1) {
  handle.LaunchDispatchRecv((mori::moe::KernelType)kernelType, blockNum, warpPerBlock,
                            at::cuda::getCurrentHIPStream());
}

void LaunchCombineRecv(mori::moe::EpDispatchCombineHandle& handle, int kernelType,
                       int blockNum = -1, int warpPerBlock = -1) {
  handle.LaunchCombineRecv((mori::moe::KernelType)kernelType, blockNum, warpPerBlock,
                           at::cuda::getCurrentHIPStream());
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
                                              at::ScalarType scalarType, int hiddenDim = -1) {
  const int actualHiddenDim =
      (hiddenDim > 0) ? hiddenDim : static_cast<int>(handle.config.hiddenDim);
  TORCH_CHECK(actualHiddenDim > 0, "registered combine input hidden dim must be > 0");
  TORCH_CHECK(actualHiddenDim <= handle.config.hiddenDim, "requested hidden dim ", actualHiddenDim,
              " exceeds config.hidden_dim ", handle.config.hiddenDim);
  torch::Tensor out =
      torch::from_blob(handle.shmemCombineInpTokMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), actualHiddenDim},
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

#ifdef ENABLE_STANDARD_MOE_ADAPT
  funcName = std::string("launch_dispatch_standard_moe");
  m.def(funcName.c_str(), &LaunchDispatchForStandardMoE);

  funcName = std::string("launch_combine_standard_moe");
  m.def(funcName.c_str(), &LaunchCombineForStandardMoE);
#endif

#ifdef ENABLE_STANDARD_MOE_ADAPT
  funcName = std::string("convert_dispatch_output");
  m.def(funcName.c_str(), &ConvertDispatchOutput);

  funcName = std::string("convert_combine_input");
  m.def(funcName.c_str(), &ConvertCombineInput);
#endif

  funcName = std::string("launch_dispatch_recv");
  m.def(funcName.c_str(), &LaunchDispatchRecv);

  funcName = std::string("launch_combine_recv");
  m.def(funcName.c_str(), &LaunchCombineRecv);

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
  m.def(funcName.c_str(), &GetRegisteredCombineInputBuffer, py::arg("handle"),
        py::arg("scalar_type"), py::arg("hidden_dim") = -1);

#ifdef ENABLE_PROFILER
  funcName = std::string("get_debug_time_buf");
  m.def(funcName.c_str(), &GetDebugTimeBuf);

  funcName = std::string("get_debug_time_offset");
  m.def(funcName.c_str(), &GetDebugTimeOffset);
#endif
}

void Cast(const torch::Tensor& input, const torch::Tensor& output) {
  TORCH_CHECK(false, "cast is not implemented yet");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "cast input/output must be contiguous");

  // LaunchCast(static_cast<float*>(input.data_ptr()),
  //            static_cast<mori::mori_fp4_e2m1*>(output.data_ptr()), input.numel(),
  //            at::cuda::getCurrentHIPStream());
}

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
      .value("AsyncLL", mori::moe::KernelType::AsyncLL)
      .export_values();
  pybind11::enum_<mori::moe::QuantType>(m, "EpDispatchCombineQuantType")
      .value("None_", mori::moe::QuantType::None)
      .value("Fp8DirectCast", mori::moe::QuantType::Fp8DirectCast)
      .export_values();

  mori::pybind::RegisterAllProfilerSlots(m);

  pybind11::class_<mori::moe::EpDispatchCombineConfig>(m, "EpDispatchCombineConfig")
      .def(pybind11::init<int, int, int, int, int, int, int, int, int, int, int, bool,
                          mori::moe::KernelType, int, int, int, mori::moe::QuantType>(),
           py::arg("rank") = 0, py::arg("world_size") = 0, py::arg("hidden_dim") = 0,
           py::arg("scale_dim") = 0, py::arg("scale_type_size") = 0,
           py::arg("max_token_type_size") = 0, py::arg("max_num_inp_token_per_rank") = 0,
           py::arg("num_experts_per_rank") = 0, py::arg("num_experts_per_token") = 0,
           py::arg("warp_num_per_block") = 0, py::arg("block_num") = 0,
           py::arg("use_external_inp_buf") = true,
           py::arg("kernel_type") = mori::moe::KernelType::IntraNode, py::arg("gpu_per_node") = 8,
           py::arg("rdma_block_num") = 0, py::arg("num_qp_per_pe") = 1,
           py::arg("quant_type") = mori::moe::QuantType::None)
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
      .def_readwrite("num_qp_per_pe", &mori::moe::EpDispatchCombineConfig::numQpPerPe)
      .def_readwrite("quant_type", &mori::moe::EpDispatchCombineConfig::quantType);

  m.attr("topk_idx_t") = py::reinterpret_borrow<py::object>(
      (PyObject*)torch::getTHPDtype(c10::CppTypeToScalarType<mori::moe::index_t>::value));

  DeclareEpDispatchCombineHandle(m);

  m.def("get_cur_device_wall_clock_freq_mhz", &GetCurDeviceWallClockFreqMhz,
        "Returns clock frequency of current device's wall clock");

  m.def("cast", &Cast, "cast a tensor from type A to type B");
}

}  // namespace mori
