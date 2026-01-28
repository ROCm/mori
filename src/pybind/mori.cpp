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
#include "mori/collective/collective.hpp"

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
                        at::cuda::getCurrentHIPStream());

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

int64_t ShmemNumQpPerPe() { return mori::shmem::ShmemNumQpPerPe(); }

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
      .value("InterNodeV1", mori::moe::KernelType::InterNodeV1)
      .value("InterNodeV1LL", mori::moe::KernelType::InterNodeV1LL)
      .export_values();

  pybind11::class_<mori::moe::EpDispatchCombineConfig>(m, "EpDispatchCombineConfig")
      .def(pybind11::init<int, int, int, int, int, int, int, int, int, int, int, bool,
                          moe::KernelType, int, int, int>(),
           py::arg("rank") = 0, py::arg("world_size") = 0, py::arg("hidden_dim") = 0,
           py::arg("scale_dim") = 0, py::arg("scale_type_size") = 0,
           py::arg("max_token_type_size") = 0, py::arg("max_num_inp_token_per_rank") = 0,
           py::arg("num_experts_per_rank") = 0, py::arg("num_experts_per_token") = 0,
           py::arg("warp_num_per_block") = 0, py::arg("block_num") = 0,
           py::arg("use_external_inp_buf") = true,
           py::arg("kernel_type") = moe::KernelType::IntraNode, py::arg("gpu_per_node") = 8,
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

  DeclareEpDispatchCombineHandle(m);
}

void RegisterMoriShmem(py::module_& m) {
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

void RegisterMoriCcl(pybind11::module_& m) {
  m.def("shmem_mpi_init",
    []() -> int {
      return mori::shmem::ShmemMpiInit(MPI_COMM_WORLD);
    },
    "Initialize SHMEM with MPI"
  );

  m.def("shmem_my_pe",
    []() -> int {
      return mori::shmem::ShmemMyPe();
    },
    "Get SHMEM PE rank"
  );

  m.def("shmem_n_pes",
    []() -> int {
      return mori::shmem::ShmemNPes();
    },
    "Get number of SHMEM PEs"
  );

  // 在现有的绑定后添加

  
  m.def("shmem_malloc",
    [](size_t size) -> uintptr_t {
      void* ptr = mori::shmem::ShmemMalloc(size);
      return reinterpret_cast<uintptr_t>(ptr);
    },
    py::arg("size"),
    "Allocate symmetric memory"
  );
  
  m.def("shmem_free",
    [](uintptr_t ptr) {
      mori::shmem::ShmemFree(reinterpret_cast<void*>(ptr));
    },
    py::arg("ptr"),
    "Free symmetric memory"
  );
    
  // float32
  m.def("all2all_sdma", 
    [](uintptr_t input_ptr, uintptr_t output_ptr, size_t count, uintptr_t stream) {
      return mori::collective::All2all_sdma<float>(
          reinterpret_cast<float*>(input_ptr),
          reinterpret_cast<float*>(output_ptr),
          count,
          reinterpret_cast<hipStream_t>(stream));
    }, 
    py::arg("input_ptr"), 
    py::arg("output_ptr"), 
    py::arg("count"), 
    py::arg("stream") = 0,
    "All2All SDMA operation for float32"
  );
  
// 修改all2all_sdma_int32绑定
// 修改all2all_sdma_int32绑定，添加详细的错误检查
m.def("all2all_sdma_int32", 
  [](uintptr_t input_ptr, uintptr_t output_ptr, size_t count) -> double {
    printf("[PYBIND] === all2all_sdma_int32 START ===\n");
    printf("[PYBIND] input_ptr=%p, output_ptr=%p, count=%zu\n",
           reinterpret_cast<void*>(input_ptr),
           reinterpret_cast<void*>(output_ptr),
           count);
    
    hipStream_t stream = nullptr;
    double result = -999.0;
    void* flags = nullptr;
    
    try {
      // 获取SHMEM信息
      int myPe = mori::shmem::ShmemMyPe();
      int npes = mori::shmem::ShmemNPes();
      printf("[PYBIND] PE %d of %d\n", myPe, npes);
      
      // 1. 创建hip stream
      printf("[PYBIND] Step 1: Creating hip stream...\n");
      hipError_t stream_err = hipStreamCreate(&stream);
      if (stream_err != hipSuccess) {
        printf("[PYBIND_ERROR] hipStreamCreate failed: %s (code: %d)\n", 
               hipGetErrorString(stream_err), stream_err);
        return -1.0;
      }
      printf("[PYBIND] Stream created: %p\n", stream);
      
      size_t dtype_size = sizeof(int32_t);
      size_t input_size = count * dtype_size;
      size_t output_size = count * dtype_size * npes;
      
      printf("[PYBIND] Sizes: dtype=%zu, input=%zu, output=%zu\n",
             dtype_size, input_size, output_size);
      
      void* input = reinterpret_cast<void*>(input_ptr);
      void* output = reinterpret_cast<void*>(output_ptr);
      
      // 2. 注册输入内存
      printf("[PYBIND] Step 2: Registering input memory at %p (size=%zu)...\n", 
             input, input_size);
      auto inPutBuffObj = mori::shmem::ShmemSymmetricRegister(input, input_size);
      if (!inPutBuffObj.IsValid()) {
        printf("[PYBIND_ERROR] ShmemSymmetricRegister for input failed!\n");
        printf("[PYBIND_ERROR] inPutBuffObj.cpu=%p, inPutBuffObj.gpu=%p\n",
               inPutBuffObj.cpu, inPutBuffObj.gpu);
        hipStreamDestroy(stream);
        return -2.0;
      }
      printf("[PYBIND] Input registered successfully\n");
      
      // 3. 注册输出内存
      printf("[PYBIND] Step 3: Registering output memory at %p (size=%zu)...\n", 
             output, output_size);
      auto outPutBuffObj = mori::shmem::ShmemSymmetricRegister(output, output_size);
      if (!outPutBuffObj.IsValid()) {
        printf("[PYBIND_ERROR] ShmemSymmetricRegister for output failed!\n");
        printf("[PYBIND_ERROR] outPutBuffObj.cpu=%p, outPutBuffObj.gpu=%p\n",
               outPutBuffObj.cpu, outPutBuffObj.gpu);
        hipStreamDestroy(stream);
        return -3.0;
      }
      printf("[PYBIND] Output registered successfully\n");
      
      // 4. 分配标志内存
      size_t flagsSize = npes * sizeof(uint64_t);
      printf("[PYBIND] Step 4: Allocating flags memory (%zu bytes)...\n", flagsSize);
      flags = mori::shmem::ShmemMalloc(flagsSize);
      if (flags == nullptr) {
        printf("[PYBIND_ERROR] ShmemMalloc failed!\n");
        hipStreamDestroy(stream);
        return -4.0;
      }
      printf("[PYBIND] Flags allocated at %p\n", flags);
      
      // 5. 初始化标志
      printf("[PYBIND] Step 5: Initializing flags with hipMemset...\n");
      hipError_t memset_err = hipMemset(flags, 0, flagsSize);
      if (memset_err != hipSuccess) {
        printf("[PYBIND_ERROR] hipMemset failed: %s (code: %d)\n", 
               hipGetErrorString(memset_err), memset_err);
        mori::shmem::ShmemFree(flags);
        hipStreamDestroy(stream);
        return -5.0;
      }
      
      // 6. 同步stream
      printf("[PYBIND] Step 6: Synchronizing stream...\n");
      hipError_t sync_err = hipStreamSynchronize(stream);
      if (sync_err != hipSuccess) {
        printf("[PYBIND_ERROR] hipStreamSynchronize failed: %s (code: %d)\n",
               hipGetErrorString(sync_err), sync_err);
        mori::shmem::ShmemFree(flags);
        hipStreamDestroy(stream);
        return -6.0;
      }
      printf("[PYBIND] Stream synchronized\n");
      
      // 7. 获取标志内存对象
      printf("[PYBIND] Step 7: Querying flags memory object...\n");
      auto flagsObj = mori::shmem::ShmemQueryMemObjPtr(flags);
      if (!flagsObj.IsValid()) {
        printf("[PYBIND_ERROR] ShmemQueryMemObjPtr failed!\n");
        printf("[PYBIND_ERROR] flagsObj.cpu=%p, flagsObj.gpu=%p\n",
               flagsObj.cpu, flagsObj.gpu);
        mori::shmem::ShmemFree(flags);
        hipStreamDestroy(stream);
        return -7.0;
      }
      printf("[PYBIND] Flags memory object obtained\n");
      
      // 8. 打印所有参数
      printf("[PYBIND] Step 8: Launching kernel with parameters:\n");
      printf("[PYBIND]   myPe=%d, npes=%d\n", myPe, npes);
      printf("[PYBIND]   inPutBuffObj.cpu=%p, .gpu=%p\n", 
             inPutBuffObj.cpu, inPutBuffObj.gpu);
      printf("[PYBIND]   outPutBuffObj.cpu=%p, .gpu=%p\n", 
             outPutBuffObj.cpu, outPutBuffObj.gpu);
      printf("[PYBIND]   flagsObj.cpu=%p, .gpu=%p\n", 
             flagsObj.cpu, flagsObj.gpu);
      printf("[PYBIND]   elementCount=%zu\n", count);
      
      // 9. 启动内核
      printf("[PYBIND] Step 9: Launching OneShotAll2allSdmaKernel...\n");
      double start = MPI_Wtime();
      mori::collective::OneShotAll2allSdmaKernel<int32_t><<<1, 512, 0, stream>>>(
          myPe, npes, inPutBuffObj, outPutBuffObj, flagsObj, count);
      
      // 10. 同步等待完成
      printf("[PYBIND] Step 10: Synchronizing after kernel launch...\n");
      hipError_t kernel_sync_err = hipStreamSynchronize(stream);
      if (kernel_sync_err != hipSuccess) {
        printf("[PYBIND_ERROR] Kernel synchronization failed: %s (code: %d)\n", 
               hipGetErrorString(kernel_sync_err), kernel_sync_err);
        
        // 检查是否有更详细的错误信息
        hipError_t last_err = hipGetLastError();
        if (last_err != hipSuccess) {
          printf("[PYBIND_ERROR] Last HIP error: %s (code: %d)\n",
                 hipGetErrorString(last_err), last_err);
        }
        
        mori::shmem::ShmemFree(flags);
        hipStreamDestroy(stream);
        return -8.0;
      }
      
      double end = MPI_Wtime();
      result = end - start;
      
      printf("[PYBIND] Step 11: Kernel completed in %.9f seconds\n", result);
      
      // 11. 清理
      printf("[PYBIND] Step 12: Cleaning up...\n");
      mori::shmem::ShmemFree(flags);
      hipStreamDestroy(stream);
      
      printf("[PYBIND] === all2all_sdma_int32 SUCCESS ===\n");
      return result;
      
    } catch (const std::exception& e) {
      printf("[PYBIND_ERROR] Exception: %s\n", e.what());
      if (flags != nullptr) {
        mori::shmem::ShmemFree(flags);
      }
      if (stream != nullptr) {
        hipStreamDestroy(stream);
      }
      return -999.0;
    } catch (...) {
      printf("[PYBIND_ERROR] Unknown exception\n");
      if (flags != nullptr) {
        mori::shmem::ShmemFree(flags);
      }
      if (stream != nullptr) {
        hipStreamDestroy(stream);
      }
      return -999.0;
    }
  }, 
  py::arg("input_ptr"), 
  py::arg("output_ptr"), 
  py::arg("count"),
  "All2All SDMA for int32"
);

}
}  // namespace mori
