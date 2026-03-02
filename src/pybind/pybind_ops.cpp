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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <hip/hip_runtime.h>

#include "mori/ops/ops.hpp"
#include "mori/pybind/profiler_registry.hpp"
#include "mori/utils/hip_helper.hpp"

/* ---------------------------------------------------------------------------------------------- */
/*                                            Ops APIs                                            */
/* ---------------------------------------------------------------------------------------------- */
namespace {

namespace py = pybind11;

hipDataType IntToHipDataType(int dtype) {
  switch (dtype) {
    case 0: return HIP_R_32F;
    case 1: return HIP_R_16BF;
    case 2: return HIP_R_8F_E4M3;
    case 3: return HIP_R_8F_E4M3_FNUZ;
    case 5: return HIP_R_4F_E2M1;
    default: throw std::runtime_error("Unsupported dtype int: " + std::to_string(dtype));
  }
}

py::tuple LaunchDispatch(mori::moe::EpDispatchCombineHandle& handle, int kernelType,
                         int64_t input_ptr, int input_dtype, int64_t num_tokens,
                         int64_t hidden_dim, int64_t weight_ptr, int64_t scale_ptr,
                         int64_t topkIds_ptr, int64_t stream, int blockNum = -1,
                         int rdmaBlockNum = -1, int warpPerBlock = -1) {
  handle.PrepareInference(
      IntToHipDataType(input_dtype), reinterpret_cast<void*>(input_ptr), nullptr,
      weight_ptr ? reinterpret_cast<float*>(weight_ptr) : nullptr,
      scale_ptr ? reinterpret_cast<uint8_t*>(scale_ptr) : nullptr,
      reinterpret_cast<mori::moe::index_t*>(topkIds_ptr), num_tokens);

  handle.LaunchDispatch(static_cast<mori::moe::KernelType>(kernelType), blockNum, rdmaBlockNum,
                        warpPerBlock, reinterpret_cast<hipStream_t>(stream),
                        static_cast<int>(hidden_dim));

  int64_t out_ptr = reinterpret_cast<int64_t>(handle.shmemDispatchOutTokMemObj->Get());
  int64_t outW_ptr = reinterpret_cast<int64_t>(handle.shmemDispatchOutWeightsMemObj->Get());
  int64_t outS_ptr = (scale_ptr && handle.config.scaleDim > 0)
                         ? reinterpret_cast<int64_t>(handle.shmemOutScalesMemObj->Get())
                         : 0;
  int64_t outI_ptr = reinterpret_cast<int64_t>(handle.shmemOutIndicesMemObj->Get());
  int64_t total_ptr = reinterpret_cast<int64_t>(handle.totalRecvTokenNum);

  return py::make_tuple(out_ptr, outW_ptr, outS_ptr, outI_ptr, total_ptr);
}

py::tuple LaunchCombine(mori::moe::EpDispatchCombineHandle& handle, int kernelType,
                        int64_t input_ptr, int input_dtype, int64_t hidden_dim,
                        int64_t weight_ptr, int64_t topkIds_ptr, int64_t stream,
                        int blockNum = -1, int rdmaBlockNum = -1, int warpPerBlock = -1,
                        int useExternalInpBuf = -1) {
  handle.PrepareInference(
      IntToHipDataType(input_dtype), reinterpret_cast<void*>(input_ptr), nullptr,
      weight_ptr ? reinterpret_cast<float*>(weight_ptr) : nullptr,
      reinterpret_cast<mori::moe::index_t*>(topkIds_ptr), handle.curRankNumToken);

  handle.LaunchCombine(static_cast<mori::moe::KernelType>(kernelType), blockNum, rdmaBlockNum,
                       warpPerBlock, useExternalInpBuf, reinterpret_cast<hipStream_t>(stream),
                       static_cast<int>(hidden_dim));

  int64_t out_ptr = reinterpret_cast<int64_t>(handle.shmemCombineOutTokMemObj->Get());
  int64_t outW_ptr =
      weight_ptr ? reinterpret_cast<int64_t>(handle.shmemCombineOutWeightsMemObj->Get()) : 0;

  return py::make_tuple(out_ptr, outW_ptr);
}

#ifdef ENABLE_STANDARD_MOE_ADAPT
int64_t LaunchDispatchForStandardMoE(
    mori::moe::EpDispatchCombineHandle& handle, int kernelType, int64_t input_ptr, int input_dtype,
    int64_t num_tokens, int64_t hidden_dim, int64_t weight_ptr, int64_t scale_ptr,
    int64_t topkIds_ptr, int64_t stream, int blockNum, int rdmaBlockNum, int warpPerBlock,
    int64_t packedRecvX_ptr, int64_t packedRecvSrcInfo_ptr) {
  handle.PrepareInference(
      IntToHipDataType(input_dtype), reinterpret_cast<void*>(input_ptr), nullptr,
      weight_ptr ? reinterpret_cast<float*>(weight_ptr) : nullptr,
      scale_ptr ? reinterpret_cast<uint8_t*>(scale_ptr) : nullptr,
      reinterpret_cast<mori::moe::index_t*>(topkIds_ptr), num_tokens);

  handle.SetStandardMoeOutputBuffers(reinterpret_cast<void*>(packedRecvX_ptr),
                                     handle.standardPackedRecvCount,
                                     reinterpret_cast<int*>(packedRecvSrcInfo_ptr), nullptr);

  handle.LaunchDispatchForStandardMoE(static_cast<mori::moe::KernelType>(kernelType), blockNum,
                                      rdmaBlockNum, warpPerBlock,
                                      reinterpret_cast<hipStream_t>(stream),
                                      static_cast<int>(hidden_dim));

  return reinterpret_cast<int64_t>(handle.standardPackedRecvCount);
}

py::tuple LaunchCombineForStandardMoE(mori::moe::EpDispatchCombineHandle& handle, int kernelType,
                                      int64_t expertOutput_ptr, int expertOutput_dtype,
                                      int64_t hidden_dim, int64_t weight_ptr, int64_t topkIds_ptr,
                                      int64_t stream, int blockNum, int rdmaBlockNum,
                                      int warpPerBlock) {
  handle.PrepareInference(
      IntToHipDataType(expertOutput_dtype), nullptr, nullptr,
      weight_ptr ? reinterpret_cast<float*>(weight_ptr) : nullptr,
      reinterpret_cast<mori::moe::index_t*>(topkIds_ptr), handle.curRankNumToken);

  handle.SetStandardMoeOutputBuffers(reinterpret_cast<void*>(expertOutput_ptr),
                                     handle.standardPackedRecvCount,
                                     handle.standardPackedRecvSrcInfo,
                                     handle.standardPackedRecvLayoutRange);

  handle.LaunchCombineForStandardMoE(static_cast<mori::moe::KernelType>(kernelType), blockNum,
                                     rdmaBlockNum, warpPerBlock,
                                     reinterpret_cast<hipStream_t>(stream),
                                     static_cast<int>(hidden_dim));

  int64_t out_ptr = reinterpret_cast<int64_t>(handle.shmemCombineOutTokMemObj->Get());
  return py::make_tuple(out_ptr, static_cast<int64_t>(0));
}

int64_t ConvertDispatchOutput(mori::moe::EpDispatchCombineHandle& handle,
                              int64_t dispatchOutX_ptr, int64_t dispatchOutTopkIdx_ptr,
                              int64_t hidden_dim, int64_t stream, int blockNum, int warpPerBlock,
                              int64_t packedRecvX_ptr, int64_t packedRecvSrcInfo_ptr) {
  handle.LaunchConvertDispatchOutputKernel(
      reinterpret_cast<void*>(dispatchOutX_ptr),
      reinterpret_cast<void*>(dispatchOutTopkIdx_ptr), reinterpret_cast<void*>(packedRecvX_ptr),
      handle.standardPackedRecvCount, reinterpret_cast<int*>(packedRecvSrcInfo_ptr), nullptr,
      blockNum, warpPerBlock, reinterpret_cast<hipStream_t>(stream),
      static_cast<int>(hidden_dim));

  return reinterpret_cast<int64_t>(handle.standardPackedRecvCount);
}

int64_t ConvertCombineInput(mori::moe::EpDispatchCombineHandle& handle, int64_t packedRecvX_ptr,
                            int64_t packedRecvSrcInfo_ptr, int64_t hidden_dim, int64_t stream,
                            int blockNum, int warpPerBlock) {
  void* combineInput_ptr = handle.shmemCombineInpTokMemObj->Get();

  handle.LaunchConvertCombineInputKernel(
      reinterpret_cast<void*>(packedRecvX_ptr), reinterpret_cast<void*>(packedRecvSrcInfo_ptr),
      nullptr, combineInput_ptr, handle.shmemCombineInpTokMemObj, blockNum, warpPerBlock,
      reinterpret_cast<hipStream_t>(stream), static_cast<int>(hidden_dim));

  return reinterpret_cast<int64_t>(combineInput_ptr);
}
#endif  // ENABLE_STANDARD_MOE_ADAPT

void LaunchDispatchRecv(mori::moe::EpDispatchCombineHandle& handle, int kernelType, int64_t stream,
                        int blockNum = -1, int warpPerBlock = -1) {
  handle.LaunchDispatchRecv(static_cast<mori::moe::KernelType>(kernelType), blockNum, warpPerBlock,
                            reinterpret_cast<hipStream_t>(stream));
}

void LaunchCombineRecv(mori::moe::EpDispatchCombineHandle& handle, int kernelType, int64_t stream,
                       int blockNum = -1, int warpPerBlock = -1) {
  handle.LaunchCombineRecv(static_cast<mori::moe::KernelType>(kernelType), blockNum, warpPerBlock,
                           reinterpret_cast<hipStream_t>(stream));
}

void LaunchReset(mori::moe::EpDispatchCombineHandle& handle, int64_t stream) {
  handle.LaunchReset(reinterpret_cast<hipStream_t>(stream));
}

py::tuple GetDispatchSrcTokenId(mori::moe::EpDispatchCombineHandle& handle) {
  return py::make_tuple(
      reinterpret_cast<int64_t>(
          handle.dispTokIdToSrcTokIdMemObj->template GetAs<mori::moe::index_t*>()),
      static_cast<int64_t>(*handle.totalRecvTokenNum));
}

py::tuple GetDispatchSenderTokenIdxMap(mori::moe::EpDispatchCombineHandle& handle) {
  return py::make_tuple(
      reinterpret_cast<int64_t>(handle.dispSenderIdxMap),
      static_cast<int64_t>(handle.curRankNumToken * handle.config.numExpertPerToken));
}

py::tuple GetDispatchReceiverTokenIdxMap(mori::moe::EpDispatchCombineHandle& handle) {
  return py::make_tuple(reinterpret_cast<int64_t>(handle.dispReceiverIdxMap),
                        static_cast<int64_t>(*handle.localPeTokenCounter));
}

py::tuple GetRegisteredCombineInputBuffer(mori::moe::EpDispatchCombineHandle& handle,
                                          int hidden_dim = -1) {
  const int actual = (hidden_dim > 0) ? hidden_dim : static_cast<int>(handle.config.hiddenDim);
  return py::make_tuple(reinterpret_cast<int64_t>(handle.shmemCombineInpTokMemObj->Get()),
                        static_cast<int64_t>(handle.config.MaxNumTokensToRecv()),
                        static_cast<int64_t>(actual));
}

#ifdef ENABLE_PROFILER
py::tuple GetDebugTimeBuf(mori::moe::EpDispatchCombineHandle& handle) {
  return py::make_tuple(reinterpret_cast<int64_t>(handle.profilerConfig.debugTimeBuf),
                        static_cast<int64_t>(MAX_DEBUG_TIME_SLOTS));
}

py::tuple GetDebugTimeOffset(mori::moe::EpDispatchCombineHandle& handle) {
  return py::make_tuple(reinterpret_cast<int64_t>(handle.profilerConfig.debugTimeOffset),
                        static_cast<int64_t>(PROFILER_WARPS_PER_RANK));
}
#endif

int GetCurDeviceWallClockFreqMhz() { return mori::GetCurDeviceWallClockFreqMhz(); }

void DeclareEpDispatchCombineHandle(pybind11::module& m) {
  pybind11::class_<mori::moe::EpDispatchCombineHandle>(m, "EpDispatchCombineHandle")
      .def(pybind11::init<mori::moe::EpDispatchCombineConfig>(),
           py::arg("config") = mori::moe::EpDispatchCombineConfig{});

  m.def("launch_dispatch", &LaunchDispatch);
  m.def("launch_combine", &LaunchCombine);

#ifdef ENABLE_STANDARD_MOE_ADAPT
  m.def("launch_dispatch_standard_moe", &LaunchDispatchForStandardMoE);
  m.def("launch_combine_standard_moe", &LaunchCombineForStandardMoE);
  m.def("convert_dispatch_output", &ConvertDispatchOutput);
  m.def("convert_combine_input", &ConvertCombineInput);
#endif

  m.def("launch_dispatch_recv", &LaunchDispatchRecv);
  m.def("launch_combine_recv", &LaunchCombineRecv);
  m.def("launch_reset", &LaunchReset);

  m.def("get_cur_rank_num_token", &mori::moe::EpDispatchCombineHandle::GetCurRankNumToken);
  m.def("get_dispatch_src_token_pos", &GetDispatchSrcTokenId);
  m.def("get_dispatch_sender_token_idx_map", &GetDispatchSenderTokenIdxMap);
  m.def("get_dispatch_receiver_token_idx_map", &GetDispatchReceiverTokenIdxMap);
  m.def("get_registered_combine_input_buffer", &GetRegisteredCombineInputBuffer, py::arg("handle"),
        py::arg("hidden_dim") = -1);

#ifdef ENABLE_PROFILER
  m.def("get_debug_time_buf", &GetDebugTimeBuf);
  m.def("get_debug_time_offset", &GetDebugTimeOffset);
#endif
}

}  // namespace

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

  DeclareEpDispatchCombineHandle(m);

  m.def("get_cur_device_wall_clock_freq_mhz", &GetCurDeviceWallClockFreqMhz,
        "Returns clock frequency of current device's wall clock");
}

}  // namespace mori
