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

#include "infiniband/verbs.h"
#include "mori/core/transport/rdma/core_device_types.hpp"

namespace mori {
namespace core {

// IBVerbsHandle is host-only: ibv_qp/ibv_cq/etc. are ibverbs objects managed on the host.
// Device code must not access these fields. For device-side RDMA, use WorkQueueHandle /
// CompletionQueueHandle instead (defined in core_device_types.hpp).
struct IBVerbsHandle {
  ibv_qp* qp{nullptr};
  ibv_cq* cq{nullptr};
  ibv_srq* srq{nullptr};
  ibv_comp_channel* compCh{nullptr};
};

}  // namespace core
}  // namespace mori
