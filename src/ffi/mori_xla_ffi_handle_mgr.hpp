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

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

namespace mori::ffi {

class HandleManager {
 public:
  static HandleManager& Instance() {
    static HandleManager instance;
    return instance;
  }

  int64_t CreateHandle(moe::EpDispatchCombineConfig config) {
    std::lock_guard<std::mutex> lock(mu_);
    int64_t id = next_id_++;
    handles_.emplace(id, std::make_unique<moe::EpDispatchCombineHandle>(config));
    return id;
  }

  moe::EpDispatchCombineHandle* GetHandle(int64_t id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = handles_.find(id);
    if (it == handles_.end()) {
      throw std::runtime_error("mori FFI: invalid handle id " + std::to_string(id));
    }
    return it->second.get();
  }

  void DestroyHandle(int64_t id) {
    std::lock_guard<std::mutex> lock(mu_);
    handles_.erase(id);
  }

 private:
  HandleManager() = default;
  HandleManager(const HandleManager&) = delete;
  HandleManager& operator=(const HandleManager&) = delete;

  std::mutex mu_;
  int64_t next_id_ = 1;
  std::unordered_map<int64_t, std::unique_ptr<moe::EpDispatchCombineHandle>> handles_;
};

}  // namespace mori::ffi
