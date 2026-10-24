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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mori::kiwi::rdma {

// Out-of-band control channel used only during setup and teardown. Every
// method is collective: all ranks must call it in the same order.
class bootstrap_t {
 public:
  virtual ~bootstrap_t() = default;

  virtual int rank() const = 0;
  virtual int size() const = 0;

  // Gathers `bytes` from every rank into `recv`, ordered by rank. `recv` holds
  // size() * bytes bytes.
  virtual void allgather(const void* send, void* recv, size_t bytes) = 0;

  virtual void barrier() {
    uint8_t token = 0;
    std::vector<uint8_t> tokens(static_cast<size_t>(size()));
    allgather(&token, tokens.data(), sizeof(token));
  }
};

// Adapts an allgather implemented elsewhere, for example a Python
// torch.distributed process group, to bootstrap_t.
class callback_bootstrap_t : public bootstrap_t {
 public:
  using allgather_fn_t = std::function<void(const void* send, void* recv, size_t bytes)>;

  callback_bootstrap_t(int rank, int size, allgather_fn_t allgather)
      : rank_(rank), size_(size), allgather_(std::move(allgather)) {}

  int rank() const override { return rank_; }
  int size() const override { return size_; }
  void allgather(const void* send, void* recv, size_t bytes) override {
    allgather_(send, recv, bytes);
  }

 private:
  int rank_;
  int size_;
  allgather_fn_t allgather_;
};

}  // namespace mori::kiwi::rdma
