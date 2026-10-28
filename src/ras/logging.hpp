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

// RAS logging. Expected conditions, such as a peer without RAS, a connection refused during
// teardown, or a port already in use, log at debug: operators and post-mortem tools read
// warnings as failures.
//
// RAS logs through RasLogger(), not MORI_LOG. MORI_LOG caches each logger in a function-local
// static, and at exit those statics, spdlog's registry with them, can be destroyed while the
// RAS thread is still running, before the atexit handler that stops it. RasLogger() holds the
// "ras" logger for the life of the process, so it is never freed under the RAS thread, and
// returns nullptr once exit begins, so nothing logs on the way out.

#include "spdlog/spdlog.h"

namespace mori {
namespace ras {

// The "ras" module logger, or nullptr once process exit begins.
spdlog::logger* RasLogger();
void DisableLoggingForExit();

}  // namespace ras
}  // namespace mori

#define RAS_LOG_AT(method, ...)                                     \
  do {                                                              \
    if (::spdlog::logger* _ras_logger = ::mori::ras::RasLogger()) { \
      _ras_logger->method(__VA_ARGS__);                             \
    }                                                               \
  } while (0)
#define RAS_LOG_DEBUG(...) RAS_LOG_AT(debug, __VA_ARGS__)
#define RAS_LOG_INFO(...) RAS_LOG_AT(info, __VA_ARGS__)
#define RAS_LOG_WARN(...) RAS_LOG_AT(warn, __VA_ARGS__)
