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
#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "mori/application/bootstrap/base_bootstrap.hpp"

namespace mori {
namespace application {

class TorchBootstrapNetwork : public BootstrapNetwork {
 public:
  TorchBootstrapNetwork(const std::string& groupName);
  ~TorchBootstrapNetwork();

  void Initialize();
  void Finalize();

  void Allgather(void* sendbuf, void* recvbuf, size_t sendcount);
  void AllToAll(void* sendbuf, void* recvbuf, size_t sendcount);
  void Barrier();

 private:
  c10::intrusive_ptr<c10d::ProcessGroup> group;
};

}  // namespace application
}  // namespace mori
