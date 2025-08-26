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

#include <string>

#include "spdlog/spdlog.h"

namespace mori {
namespace io {

#define MORI_IO_TRACE SPDLOG_TRACE
#define MORI_IO_DEBUG SPDLOG_DEBUG
#define MORI_IO_INFO SPDLOG_INFO
#define MORI_IO_WARN SPDLOG_WARN
#define MORI_IO_ERROR SPDLOG_ERROR
#define MORI_IO_CRITICAL SPDLOG_CRITICAL

// trace / debug / info / warning / error / critical
inline void SetLogLevel(const std::string& strLevel) {
  spdlog::level::level_enum level = spdlog::level::from_str(strLevel);
  spdlog::set_level(level);
  MORI_IO_INFO("Set MORI-IO log level to {}", spdlog::level::to_string_view(level));
}

}  // namespace io
}  // namespace mori
