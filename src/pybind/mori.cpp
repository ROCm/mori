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

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "mori/application/application.hpp"
#include "mori/io/io.hpp"
#include "mori/ops/ops.hpp"
#include "mori/shmem/shmem.hpp"
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
  if (scales.has_value() && handle.config.scaleDim > 0) {
    assert(scales->is_contiguous() && scales->element_size() == handle.config.scaleTypeSize);
    scalePtr = reinterpret_cast<uint8_t*>(scales->data_ptr());
  }

  handle.PrepareInference(mori::ScalarTypeToHipDataType(input.scalar_type()), input.data_ptr(),
                          nullptr, weightPtr, scalePtr, topkIds.data_ptr<mori::moe::index_t>(),
                          input.size(0));
  handle.LaunchDispatch((mori::moe::KernelType)kernelType, blockNum, warpPerBlock,
                        at::cuda::getCurrentHIPStream());

  torch::Tensor out =
      torch::from_blob(handle.shmemOutTokMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), handle.config.hiddenDim},
                       torch::TensorOptions().dtype(input.scalar_type()).device(torch::kCUDA));

  std::optional<torch::Tensor> outWeights{std::nullopt};
  if (weightPtr) {
    outWeights = torch::from_blob(
        handle.shmemOutWeightsMemObj->Get(),
        {handle.config.MaxNumTokensToRecv(), handle.config.numExpertPerToken},
        torch::TensorOptions().dtype(mori::GetTorchDataType<float>()).device(torch::kCUDA));
  }

  std::optional<torch::Tensor> outScales{std::nullopt};
  if (scales.has_value() && handle.config.scaleDim > 0) {
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
    int warpPerBlock) {
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
                       at::cuda::getCurrentHIPStream());

  auto options = torch::TensorOptions().dtype(input.scalar_type()).device(torch::kCUDA);
  torch::Tensor out =
      torch::from_blob(handle.shmemOutTokMemObj->Get(),
                       {handle.config.maxNumInpTokenPerRank, handle.config.hiddenDim}, options);

  std::optional<torch::Tensor> outWeights{std::nullopt};
  if (weightsPtr) {
    outWeights =
        torch::from_blob(handle.shmemOutWeightsMemObj->Get(),
                         {handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
                         torch::TensorOptions().dtype(weights->scalar_type()).device(torch::kCUDA));
  }

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

void DeclareEpDispatchCombineHandle(pybind11::module& m) {
  std::string className = std::string("EpDispatchCombineHandle");
  pybind11::class_<mori::moe::EpDispatchCombineHandle>(m, className.c_str())
      .def(pybind11::init<mori::moe::EpDispatchCombineConfig>(),
           py::arg("config") = mori::moe::EpDispatchCombineConfig{});

  std::string funcName = std::string("launch_dispatch");
  m.def(funcName.c_str(), &LaunchDispatch);

  funcName = std::string("launch_combine");
  m.def(funcName.c_str(), &LaunchCombine);

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

int64_t ShmemMyPe() { return mori::shmem::ShmemMyPe(); }

int64_t ShmemNPes() { return mori::shmem::ShmemNPes(); }

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                             IO APIs                                            */
/* ---------------------------------------------------------------------------------------------- */
namespace {}

namespace mori {

void RegisterMoriOps(py::module_& m) {
  pybind11::enum_<mori::moe::KernelType>(m, "EpDispatchCombineKernelType")
      .value("IntraNode", mori::moe::KernelType::IntraNode)
      .value("InterNode", mori::moe::KernelType::InterNode)
      .export_values();

  pybind11::class_<mori::moe::EpDispatchCombineConfig>(m, "EpDispatchCombineConfig")
      .def(pybind11::init<int, int, int, int, int, int, int, int, int, int, int, bool>(),
           py::arg("rank") = 0, py::arg("world_size") = 0, py::arg("hidden_dim") = 0,
           py::arg("scale_dim") = 0, py::arg("scale_type_size") = 0,
           py::arg("max_token_type_size") = 0, py::arg("max_num_inp_token_per_rank") = 0,
           py::arg("num_experts_per_rank") = 0, py::arg("num_experts_per_token") = 0,
           py::arg("warp_num_per_block") = 0, py::arg("block_num") = 0,
           py::arg("use_external_inp_buf") = true)
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
      .def_readwrite("block_num", &mori::moe::EpDispatchCombineConfig::blockNum);

  DeclareEpDispatchCombineHandle(m);
}

void RegisterMoriShmem(py::module_& m) {
  m.def("shmem_torch_process_group_init", &ShmemTorchProcessGroupInit);
  m.def("shmem_finalize", &ShmemFinalize);
  m.def("shmem_mype", &ShmemMyPe);
  m.def("shmem_npes", &ShmemNPes);
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
      .export_values();

  py::enum_<mori::io::PollCqMode>(m, "PollCqMode")
      .value("POLLING", mori::io::PollCqMode::POLLING)
      .value("EVENT", mori::io::PollCqMode::EVENT);

  py::class_<mori::io::BackendConfig>(m, "BackendConfig");

  py::class_<mori::io::RdmaBackendConfig, mori::io::BackendConfig>(m, "RdmaBackendConfig")
      .def(py::init<int, int, int, mori::io::PollCqMode>(), py::arg("qp_per_transfer") = 1,
           py::arg("post_batch_size") = -1, py::arg("num_worker_threads") = -1,
           py::arg("poll_cq_mode") = mori::io::PollCqMode::POLLING)
      .def_readwrite("qp_per_transfer", &mori::io::RdmaBackendConfig::qpPerTransfer)
      .def_readwrite("post_batch_size", &mori::io::RdmaBackendConfig::postBatchSize)
      .def_readwrite("num_worker_threads", &mori::io::RdmaBackendConfig::numWorkerThreads)
      .def_readwrite("poll_cq_mode", &mori::io::RdmaBackendConfig::pollCqMode);

  py::class_<mori::io::IOEngineConfig>(m, "IOEngineConfig")
      .def(py::init<std::string, uint16_t>(), py::arg("host") = "", py::arg("port") = 0)
      .def_readwrite("host", &mori::io::IOEngineConfig::host)
      .def_readwrite("port", &mori::io::IOEngineConfig::port);

  py::class_<mori::io::TransferStatus>(m, "TransferStatus")
      .def(py::init<>())
      .def("Code", &mori::io::TransferStatus::Code)
      .def("Message", &mori::io::TransferStatus::Message)
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
