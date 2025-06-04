#include "mori/application/bootstrap/torch_bootstrap.hpp"

#include <mpi.h>

#include <cassert>

namespace mori {
namespace application {

TorchBootstrapNetwork::TorchBootstrapNetwork(const c10::intrusive_ptr<c10d::ProcessGroup> group;)
    : group(group) {
  Initialize();
}

TorchBootstrapNetwork::~TorchBootstrapNetwork() { Finalize(); }

void TorchBootstrapNetwork::Initialize() {
  unsigned n = group->getSize();
  TORCH_CHECK(n == count, "Group size must be equal to count");
}

void TorchBootstrapNetwork::Finalize() {}

void TorchBootstrapNetwork::Allgather(void* sendbuf, void* recvbuf, size_t sendcount) {
  assert(false);
}

void TorchBootstrapNetwork::AllToAll(void* sendbuf, void* recvbuf, size_t sendcount) {
  at::Tensor inputTensor =
      at::from_blob(const_cast<void*>(sendbuf), {(int)world_size, (int)sendcount},
                    at::TensorOptions().dtype(at::kByte));

  at::Tensor outputTensor = at::from_blob(recvbuf, {(int)world_size, (int)sendcount},
                                          at::TensorOptions().dtype(at::kByte));

  std::vector<int64_t> counts(world_size, 1);

  c10d::AllToAllOptions opts;
  auto work = group->alltoall_base(outputTensor, inputTensor, counts, counts, opts);
  work->wait();
}

void TorchBootstrapNetwork::Barrier() { assert(false); }

}  // namespace application
}  // namespace mori