#include "mori/application/bootstrap/torch_bootstrap.hpp"

#include <mpi.h>

#include <cassert>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace mori {
namespace application {

TorchBootstrapNetwork::TorchBootstrapNetwork(const std::string& groupName) {
  this->group = c10d::resolve_process_group(groupName);
}

TorchBootstrapNetwork::~TorchBootstrapNetwork() { Finalize(); }

void TorchBootstrapNetwork::Initialize() {
  this->worldSize = group->getSize();
  this->localRank = group->getRank();
}

void TorchBootstrapNetwork::Finalize() {}

void TorchBootstrapNetwork::Allgather(void* sendbuf, void* recvbuf, size_t sendcount) {
  std::vector<at::Tensor> inputTensors = {
      at::from_blob(sendbuf, {1, (int)sendcount}, at::TensorOptions().dtype(at::kByte))};

  std::vector<at::Tensor> outputTensors = {
      at::from_blob(recvbuf, {worldSize, (int)sendcount}, at::TensorOptions().dtype(at::kByte))};

  c10d::AllgatherOptions opts;
  auto work = group->allgather_into_tensor_coalesced(outputTensors, inputTensors, opts);
  work->wait();
}

void TorchBootstrapNetwork::AllToAll(void* sendbuf, void* recvbuf, size_t sendcount) {
  at::Tensor inputTensor =
      at::from_blob(sendbuf, {worldSize, (int)sendcount}, at::TensorOptions().dtype(at::kByte));

  at::Tensor outputTensor =
      at::from_blob(recvbuf, {worldSize, (int)sendcount}, at::TensorOptions().dtype(at::kByte));

  std::vector<int64_t> counts(worldSize, 1);

  c10d::AllToAllOptions opts;
  auto work = group->alltoall_base(outputTensor, inputTensor, counts, counts, opts);
  work->wait();
}

void TorchBootstrapNetwork::Barrier() {
  auto work = group->barrier();
  work->wait();
}

}  // namespace application
}  // namespace mori