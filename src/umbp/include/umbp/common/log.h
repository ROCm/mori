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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include <cstdio>
#include <cstdlib>

// Log levels: 0=INFO (verbose), 1=WARN (default), 2=ERROR
// Control via UMBP_LOG_LEVEL env var.  Default is WARN — only warnings and
// errors are printed.  Set UMBP_LOG_LEVEL=0 to see all INFO messages.
inline int UmbpLogLevel() {
  static int level = [] {
    const char* env = std::getenv("UMBP_LOG_LEVEL");
    return env ? std::atoi(env) : 1;
  }();
  return level;
}

#define UMBP_LOG_INFO(fmt, ...)                                                       \
  do {                                                                                \
    if (UmbpLogLevel() <= 0) fprintf(stdout, "[UMBP INFO] " fmt "\n", ##__VA_ARGS__); \
  } while (0)
#define UMBP_LOG_WARN(fmt, ...) fprintf(stderr, "[UMBP WARN] " fmt "\n", ##__VA_ARGS__)
#define UMBP_LOG_ERROR(fmt, ...) fprintf(stderr, "[UMBP ERROR] " fmt "\n", ##__VA_ARGS__)

#define UMBP_CHECK(cond, fmt, ...)                                                         \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      fprintf(stderr, "[UMBP FATAL] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
      std::abort();                                                                        \
    }                                                                                      \
  } while (0)
