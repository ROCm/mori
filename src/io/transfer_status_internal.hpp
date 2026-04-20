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

#include "mori/io/common.hpp"

namespace mori {
namespace io {
namespace internal {

struct TransferStatusAccess {
  static constexpr uint32_t kRdmaFlushCascadeBit = 1u << 0;

  static void MarkRdmaFlushCascade(TransferStatus* status) {
    if (status == nullptr) return;
    status->diagnosticFlags.fetch_or(kRdmaFlushCascadeBit, std::memory_order_relaxed);
  }

  static bool HasRdmaFlushCascade(const TransferStatus& status) {
    return (status.diagnosticFlags.load(std::memory_order_relaxed) & kRdmaFlushCascadeBit) != 0;
  }
};

}  // namespace internal
}  // namespace io
}  // namespace mori
