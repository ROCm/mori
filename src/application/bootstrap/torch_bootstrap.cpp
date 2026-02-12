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
#include "mori/application/bootstrap/torch_bootstrap.hpp"

#include <cassert>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace mori {
namespace application {

TorchBootstrapNetwork::TorchBootstrapNetwork(const std::string& name) : groupName(name) {}

TorchBootstrapNetwork::~TorchBootstrapNetwork() { Finalize(); }

void TorchBootstrapNetwork::Initialize() {
  c10::intrusive_ptr<c10d::ProcessGroup> group = c10d::resolve_process_group(groupName);
  this->worldSize = group->getSize();
  this->localRank = group->getRank();
}

void TorchBootstrapNetwork::Finalize() {}

void TorchBootstrapNetwork::Allgather(void* sendbuf, void* recvbuf, size_t sendcount) {
  c10::intrusive_ptr<c10d::ProcessGroup> group = c10d::resolve_process_group(groupName);

  std::vector<at::Tensor> inputTensors = {
      at::from_blob(sendbuf, {1, (int)sendcount}, at::TensorOptions().dtype(at::kByte))};

  std::vector<at::Tensor> outputTensors = {
      at::from_blob(recvbuf, {worldSize, (int)sendcount}, at::TensorOptions().dtype(at::kByte))};

  c10d::AllgatherOptions opts;
  auto work = group->allgather_into_tensor_coalesced(outputTensors, inputTensors, opts);
  work->wait();
}

void TorchBootstrapNetwork::AllToAll(void* sendbuf, void* recvbuf, size_t sendcount) {
  c10::intrusive_ptr<c10d::ProcessGroup> group = c10d::resolve_process_group(groupName);

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
  c10::intrusive_ptr<c10d::ProcessGroup> group = c10d::resolve_process_group(groupName);

  auto work = group->barrier();
  work->wait();
}

}  // namespace application
}  // namespace mori
