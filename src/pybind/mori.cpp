#include "src/pybind/mori.hpp"

#include <ATen/hip/HIPContext.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "mori/application/application.hpp"
#include "mori/ops/ops.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/pybind/torch_utils.hpp"

/* ---------------------------------------------------------------------------------------------- */
/*                                            Ops APIs                                            */
/* ---------------------------------------------------------------------------------------------- */
namespace {

template <typename T>
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor>
LaunchDispatch(mori::moe::EpDispatchCombineHandle<T>& handle, int kernelType,
               const torch::Tensor& input, const torch::Tensor& weights,
               const std::optional<torch::Tensor>& scales, const torch::Tensor& topkIds) {
  assert(input.is_contiguous() && weights.is_contiguous() && topkIds.is_contiguous());
  uint8_t* scalePtr = nullptr;
  if (scales.has_value()) {
    assert(scales.is_contiguous() && scales.element_size() == handle.config.scaleTypeSize);
    scalePtr = reinterpret_cast<uint8_t*>(scales.data_ptr());
  }

  handle.PrepareInference(reinterpret_cast<T*>(input.data_ptr()), nullptr,
                          weights.data_ptr<float>(), scalePtr,
                          topkIds.data_ptr<mori::moe::index_t>(), input.size(0));
  handle.LaunchDispatch((mori::moe::KernelType)kernelType, at::cuda::getCurrentHIPStream());

  torch::Tensor out = torch::from_blob(
      handle.shmemOutTokMemObj->Get(),
      {handle.config.MaxNumTokensToRecv(), handle.config.hiddenDim},
      torch::TensorOptions().dtype(mori::GetTorchDataType<T>()).device(torch::kCUDA));

  torch::Tensor outWeights = torch::from_blob(
      handle.shmemOutWeightsMemObj->Get(),
      {handle.config.MaxNumTokensToRecv(), handle.config.numExpertPerToken},
      torch::TensorOptions().dtype(mori::GetTorchDataType<float>()).device(torch::kCUDA));

  torch::Tensor outScales = std::optional<torch::Tensor>();
  if (scales.has_value()) {
    outScales =
        torch::from_blob(handle.shmemOutScalesMemObj->Get(),
                         {handle.config.MaxNumTokensToRecv(), handle.config.scaleDim},
                         torch::TensorOptions().dtype(scales.scalar_type()).device(torch::kCUDA));
  }

  torch::Tensor outIndicies =
      torch::from_blob(handle.shmemOutIndiciesMemObj->Get(),
                       {handle.config.MaxNumTokensToRecv(), handle.config.numExpertPerToken},
                       torch::TensorOptions()
                           .dtype(mori::GetTorchDataType<mori::moe::index_t>())
                           .device(torch::kCUDA));

  torch::Tensor totalRecvTokenNum = torch::from_blob(
      handle.totalRecvTokenNum, {1},
      torch::TensorOptions().dtype(mori::GetTorchDataType<mori::moe::index_t>()).device(torch::kCUDA));
  return {out, outWeights, outScales, outIndicies, totalRecvTokenNum};
}

// TODO: translate data type
template <typename T>
torch::Tensor LaunchCombine(mori::moe::EpDispatchCombineHandle<T>& handle, int kernelType,
                            const torch::Tensor& input, const torch::Tensor& weights,
                            const torch::Tensor& topkIds) {
  assert(input.is_contiguous() && weights.is_contiguous() && topkIds.is_contiguous());
  handle.PrepareInference(reinterpret_cast<T*>(input.data_ptr()), nullptr,
                          weights.data_ptr<float>(), topkIds.data_ptr<mori::moe::index_t>(),
                          handle.curRankNumToken);
  handle.LaunchCombine((mori::moe::KernelType)kernelType, at::cuda::getCurrentHIPStream());

  auto options = torch::TensorOptions().dtype(mori::GetTorchDataType<T>()).device(torch::kCUDA);
  torch::Tensor out =
      torch::from_blob(handle.shmemOutTokMemObj->Get(),
                       {handle.config.maxNumInpTokenPerRank, handle.config.hiddenDim}, options);
  return out;
}

template <typename T>
void LaunchReset(mori::moe::EpDispatchCombineHandle<T>& handle) {
  handle.LaunchReset(at::cuda::getCurrentHIPStream());
}

template <typename T>
torch::Tensor GetDispatchSrcTokenId(mori::moe::EpDispatchCombineHandle<T>& handle) {
  auto options = torch::TensorOptions()
                     .dtype(mori::GetTorchDataType<mori::moe::index_t>())
                     .device(torch::kCUDA);
  torch::Tensor tensor =
      torch::from_blob(handle.dispTokIdToSrcTokIdMemObj->template GetAs<mori::moe::index_t*>(),
                       {*handle.totalRecvTokenNum}, options);
  return tensor;
}

template <typename T>
torch::Tensor GetDispatchSenderTokenIdxMap(mori::moe::EpDispatchCombineHandle<T>& handle) {
  auto options = torch::TensorOptions()
                     .dtype(mori::GetTorchDataType<mori::moe::index_t>())
                     .device(torch::kCUDA);
  torch::Tensor tensor =
      torch::from_blob(handle.dispSenderIdxMap,
                       {handle.curRankNumToken * handle.config.numExpertPerToken}, options);
  return tensor;
}

template <typename T>
torch::Tensor GetDispatchReceiverTokenIdxMap(mori::moe::EpDispatchCombineHandle<T>& handle) {
  auto options = torch::TensorOptions()
                     .dtype(mori::GetTorchDataType<mori::moe::index_t>())
                     .device(torch::kCUDA);
  torch::Tensor tensor =
      torch::from_blob(handle.dispReceiverIdxMap, {*handle.localPeTokenCounter}, options);
  return tensor;
}

// Defines handle for different template arguments, we use typeStr to avoid function name shadow
template <typename T>
void DeclareEpDispatchCombineHandle(pybind11::module& m, const std::string& typeStr) {
  std::string className = std::string("EpDispatchCombineHandle") + typeStr;
  pybind11::class_<mori::moe::EpDispatchCombineHandle<T>>(m, className.c_str())
      .def(pybind11::init<mori::moe::EpDispatchCombineConfig>(),
           py::arg("config") = mori::moe::EpDispatchCombineConfig{});

  std::string funcName = std::string("launch_dispatch_") + typeStr;
  m.def(funcName.c_str(), &LaunchDispatch<T>);

  funcName = std::string("launch_combine_") + typeStr;
  m.def(funcName.c_str(), &LaunchCombine<T>);

  funcName = std::string("launch_reset_") + typeStr;
  m.def(funcName.c_str(), &LaunchReset<T>);

  funcName = std::string("get_cur_rank_num_token_") + typeStr;
  m.def(funcName.c_str(), &mori::moe::EpDispatchCombineHandle<T>::GetCurRankNumToken);

  funcName = std::string("get_dispatch_src_token_pos_") + typeStr;
  m.def(funcName.c_str(), &GetDispatchSrcTokenId<T>);

  funcName = std::string("get_dispatch_sender_token_idx_map_") + typeStr;
  m.def(funcName.c_str(), &GetDispatchSenderTokenIdxMap<T>);

  funcName = std::string("get_dispatch_receiver_token_idx_map_") + typeStr;
  m.def(funcName.c_str(), &GetDispatchReceiverTokenIdxMap<T>);
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

namespace mori {

void RegisterMoriOps(py::module_& m) {
  pybind11::enum_<mori::moe::KernelType>(m, "EpDispatchCombineKernelType")
      .value("IntraNode", mori::moe::KernelType::IntraNode)
      .value("InterNode", mori::moe::KernelType::InterNode)
      .export_values();

  pybind11::class_<mori::moe::EpDispatchCombineConfig>(m, "EpDispatchCombineConfig")
      .def(pybind11::init<int, int, int, int, int, int, int, int, int, int>(), py::arg("rank") = 0,
           py::arg("world_size") = 0, py::arg("hidden_dim") = 0, py::arg("scale_dim") = 0,
           py::arg("scale_type_size") = 0, py::arg("max_num_inp_token_per_rank") = 0,
           py::arg("num_experts_per_rank") = 0, py::arg("num_experts_per_token") = 0,
           py::arg("warp_num_per_block") = 0, py::arg("block_num") = 0)
      .def_readonly("rank", &mori::moe::EpDispatchCombineConfig::rank)
      .def_readonly("world_size", &mori::moe::EpDispatchCombineConfig::worldSize)
      .def_readonly("hidden_dim", &mori::moe::EpDispatchCombineConfig::hiddenDim)
      .def_readonly("scale_dim", &mori::moe::EpDispatchCombineConfig::scaleDim)
      .def_readonly("scale_type_size", &mori::moe::EpDispatchCombineConfig::scaleTypeSize)
      .def_readonly("max_num_inp_token_per_rank",
                    &mori::moe::EpDispatchCombineConfig::maxNumInpTokenPerRank)
      .def_readonly("num_experts_per_rank", &mori::moe::EpDispatchCombineConfig::numExpertPerRank)
      .def_readonly("num_experts_per_token", &mori::moe::EpDispatchCombineConfig::numExpertPerToken)
      .def_readonly("warp_num_per_block", &mori::moe::EpDispatchCombineConfig::warpNumPerBlock)
      .def_readonly("block_num", &mori::moe::EpDispatchCombineConfig::blockNum);

  DeclareEpDispatchCombineHandle<float>(m, "Fp32");
  DeclareEpDispatchCombineHandle<hip_bfloat16>(m, "Bf16");
  DeclareEpDispatchCombineHandle<__hip_fp8_e4m3_fnuz>(m, "Fp8E4m3Fnuz");
}

void RegisterMoriShmem(py::module_& m) {
  m.def("shmem_torch_process_group_init", &ShmemTorchProcessGroupInit);
  m.def("shmem_finalize", &ShmemFinalize);
  m.def("shmem_mype", &ShmemMyPe);
  m.def("shmem_npes", &ShmemNPes);
}
}  // namespace mori
