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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LaunchIntraNodeDispatch(mori::moe::EpDispatchCombineHandle<T>& handle, const torch::Tensor& input,
                        const torch::Tensor& weights, const torch::Tensor& scales,
                        const torch::Tensor& topkIds) {
  assert(input.is_contiguous() && weights.is_contiguous() && scales.is_contiguous() &&
         topkIds.is_contiguous());
  assert(scales.element_size() == handle.config.scaleTypeSize);

  handle.PrepareInference(reinterpret_cast<T*>(input.data_ptr()), nullptr,
                          weights.data_ptr<float>(), reinterpret_cast<uint8_t*>(scales.data_ptr()),
                          topkIds.data_ptr<uint32_t>(), input.size(0));
  handle.LaunchIntraNodeDispatch(at::cuda::getCurrentHIPStream());

  torch::Tensor out = torch::from_blob(
      handle.shmemOutTokMemObj->Get(),
      {handle.config.MaxNumTokensToRecvPerRank(), handle.config.hiddenDim},
      torch::TensorOptions().dtype(mori::GetTorchDataType<T>()).device(torch::kCUDA));

  torch::Tensor outWeights = torch::from_blob(
      handle.shmemWeightsMemObj->Get(),
      {handle.config.MaxNumTokensToRecvPerRank(), handle.config.numExpertPerToken},
      torch::TensorOptions().dtype(mori::GetTorchDataType<float>()).device(torch::kCUDA));

  torch::Tensor outScales =
      torch::from_blob(handle.shmemScalesMemObj->Get(),
                       {handle.config.MaxNumTokensToRecvPerRank(), handle.config.scaleDim},
                       torch::TensorOptions().dtype(scales.scalar_type()).device(torch::kCUDA));

  torch::Tensor outIndicies = torch::from_blob(
      handle.shmemIndiciesMemObj->Get(),
      {handle.config.MaxNumTokensToRecvPerRank(), handle.config.numExpertPerToken},
      torch::TensorOptions().dtype(mori::GetTorchDataType<uint32_t>()).device(torch::kCUDA));

  torch::Tensor totalRecvTokenNum = torch::from_blob(
      handle.totalRecvTokenNum, {1},
      torch::TensorOptions().dtype(mori::GetTorchDataType<size_t>()).device(torch::kCUDA));
  return {out, outWeights, outScales, outIndicies, totalRecvTokenNum};
}

// TODO: translate data type
template <typename T>
torch::Tensor LaunchIntraNodeCombine(mori::moe::EpDispatchCombineHandle<T>& handle,
                                     const torch::Tensor& input, const torch::Tensor& weights,
                                     const torch::Tensor& topkIds) {
  assert(input.is_contiguous() && weights.is_contiguous() && topkIds.is_contiguous());
  handle.PrepareInference(reinterpret_cast<T*>(input.data_ptr()), nullptr,
                          weights.data_ptr<float>(), topkIds.data_ptr<uint32_t>(),
                          handle.curRankNumToken);
  handle.LaunchIntraNodeCombine(at::cuda::getCurrentHIPStream());

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
  auto options =
      torch::TensorOptions().dtype(mori::GetTorchDataType<uint32_t>()).device(torch::kCUDA);
  torch::Tensor tensor =
      torch::from_blob(handle.dispTokIdToSrcTokIdMemObj->template GetAs<uint32_t*>(),
                       {int(*handle.totalRecvTokenNum)}, options);
  return tensor;
}

template <typename T>
void DeclareEpDispatchCombineHandle(pybind11::module& m, const std::string& typeStr) {
  std::string className = std::string("EpDispatchCombineHandle") + typeStr;
  pybind11::class_<mori::moe::EpDispatchCombineHandle<T>>(m, className.c_str())
      .def(pybind11::init<mori::moe::EpDispatchCombineConfig>(),
           py::arg("config") = mori::moe::EpDispatchCombineConfig{});

  std::string funcName = std::string("launch_intra_node_dispatch_") + typeStr;
  m.def(funcName.c_str(), &LaunchIntraNodeDispatch<T>);

  funcName = std::string("launch_intra_node_combine_") + typeStr;
  m.def(funcName.c_str(), &LaunchIntraNodeCombine<T>);

  funcName = std::string("launch_reset_") + typeStr;
  m.def(funcName.c_str(), &LaunchReset<T>);

  funcName = std::string("get_dispatch_src_token_pos_") + typeStr;
  m.def(funcName.c_str(), &GetDispatchSrcTokenId<T>);
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
