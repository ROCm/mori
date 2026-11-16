// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
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
