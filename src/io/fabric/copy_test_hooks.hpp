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

#include <hip/hip_runtime_api.h>

#include <functional>

namespace mori::io::fabric {
// Internal synchronous-submission test seam. Borrowed, thread-local, disabled by
// default; never exposed by the installed public headers or environment.
struct CopyTestHooks {
  unsigned failLaunch{0};  // 1-based within this transfer; zero disables
  bool failEventRecord{false};
  hipError_t launchError{hipErrorInvalidValue};
  hipError_t eventError{hipErrorInvalidResourceHandle};
  hipError_t drainError{hipSuccess};  // override only AFTER the real drain
  unsigned successfulLaunches{0};
  unsigned drainCalls{0};
  std::function<void(bool afterDrain, hipStream_t)> onDrain;
};
void SetCopyTestHooks(CopyTestHooks* hooks);
}  // namespace mori::io::fabric
