#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "mori/application/application.hpp"

namespace {
void TestTorchBootstrap(std::string groupName) {
  auto group = c10d::resolve_process_group(groupName);
  mori::application::TorchBootstrapNetwork bootNet(group);
}
}  // namespace

namespace mori {
void register_dispatch_combine_ops(torch::Library& m) {
  m.def("test_torch_bootstrap", &TestTorchBootstrap);
}
}  // namespace mori
