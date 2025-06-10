#include "src/pybind/mori.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "mori/application/application.hpp"
#include "mori/ops/ops.hpp"
#include "mori/shmem/shmem.hpp"

/* ---------------------------------------------------------------------------------------------- */
/*                                            Ops APIs                                            */
/* ---------------------------------------------------------------------------------------------- */
namespace {

template <typename T>
void LaunchIntraNodeDispatch(mori::moe::EpDispatchCombineHandle<T>& handle,
                             const torch::Tensor& input, const torch::Tensor& output,
                             const torch::Tensor& weights, const torch::Tensor& topkIds) {
  assert(input.is_contiguous() && output.is_contiguous() && weights.is_contiguous() &&
         topkIds.is_contiguous());
  handle.PrepareInference(reinterpret_cast<T*>(input.data_ptr()),
                          reinterpret_cast<T*>(output.data_ptr()), weights.data_ptr<float>(),
                          topkIds.data_ptr<uint32_t>(), input.size(0));
  handle.LaunchIntraNodeDispatch(0);
}

template <typename T>
void LaunchIntraNodeCombine(mori::moe::EpDispatchCombineHandle<T>& handle,
                            const torch::Tensor& input, const torch::Tensor& output,
                            const torch::Tensor& weights, const torch::Tensor& topkIds) {
  assert(input.is_contiguous() && output.is_contiguous() && weights.is_contiguous() &&
         topkIds.is_contiguous());
  handle.PrepareInference(reinterpret_cast<T*>(input.data_ptr()),
                          reinterpret_cast<T*>(output.data_ptr()), weights.data_ptr<float>(),
                          topkIds.data_ptr<uint32_t>(), input.size(0));
  handle.LaunchIntraNodeCombine(0);
}

template <typename T>
void LaunchReset(mori::moe::EpDispatchCombineHandle<T>& handle) {
  handle.LaunchReset(0);
}

template <typename T>
torch::Tensor GetDispatchSrcTokenId(mori::moe::EpDispatchCombineHandle<T>& handle) {
  torch::Tensor tensor =
      torch::from_blob(handle.dispTokIdToSrcTokIdMemObj->template GetAs<uint32_t*>(),
                       {*handle.totalRecvTokenNum}, nullptr, torch::kUInt32);
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
      .def(pybind11::init<int, int, int, int, int, int, int, int>(), py::arg("rank") = 0,
           py::arg("world_size") = 0, py::arg("hidden_dim") = 0,
           py::arg("max_num_inp_token_per_rank") = 0, py::arg("num_expert_per_rank") = 0,
           py::arg("num_expert_per_token") = 0, py::arg("warp_num_per_block") = 0,
           py::arg("block_num") = 0)
      .def_readonly("rank", &mori::moe::EpDispatchCombineConfig::rank)
      .def_readonly("world_size", &mori::moe::EpDispatchCombineConfig::worldSize)
      .def_readonly("hidden_dim", &mori::moe::EpDispatchCombineConfig::hiddenDim)
      .def_readonly("max_num_inp_token_per_rank",
                    &mori::moe::EpDispatchCombineConfig::maxNumInpTokenPerRank)
      .def_readonly("num_experts_per_rank", &mori::moe::EpDispatchCombineConfig::numExpertPerRank)
      .def_readonly("num_experts_per_token", &mori::moe::EpDispatchCombineConfig::numExpertPerToken)
      .def_readonly("warp_num_per_block", &mori::moe::EpDispatchCombineConfig::warpNumPerBlock)
      .def_readonly("block_num", &mori::moe::EpDispatchCombineConfig::blockNum);

  DeclareEpDispatchCombineHandle<float>(m, "Fp32");
  DeclareEpDispatchCombineHandle<hip_bfloat16>(m, "Bf16");
  DeclareEpDispatchCombineHandle<__hip_fp8_e4m3_fnuz>(m, "Fp8E4m3");
}

void RegisterMoriShmem(py::module_& m) {
  m.def("shmem_torch_process_group_init", &ShmemTorchProcessGroupInit);
  m.def("shmem_finalize", &ShmemFinalize);
  m.def("shmem_mype", &ShmemMyPe);
  m.def("shmem_npes", &ShmemNPes);
}
}  // namespace mori
