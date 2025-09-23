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

#include <mutex>

// Include the new centralized logging system
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace io {

// Legacy SetLogLevel function for backward compatibility
inline void SetLogLevel(const std::string& strLevel) {
  try {
    InitializeLoggingFromEnv();
  } catch (...) {
  }
  
  ForceSetModuleLogLevel(modules::IO, strLevel);
  
  auto logger = mori::ModuleLogger::GetInstance().GetLogger(modules::IO);
  if (logger) {
    logger->info("Set MORI-IO log level to {}", strLevel);
  }
}

// Legacy ScopedTimer - redirect to new implementation
using ScopedTimer = mori::ScopedTimer;

// Legacy ScopedTimer - redirect to new implementation
using ScopedTimer = mori::ScopedTimer;

#define MORI_IO_TIMER(message) MORI_TIMER(message, mori::modules::IO)
#define MORI_IO_FUNCTION_TIMER MORI_FUNCTION_TIMER(mori::modules::IO)

}  // namespace io
}  // namespace mori