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

#include <sys/resource.h>

#include <cerrno>
#include <cstring>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

// Raises this process's soft RLIMIT_NOFILE to its hard limit (no privileges
// needed): the SSD tier keeps every segment file open for its lifetime, and the
// usual soft limit is 1,024.  The limit is process-wide, so only UMBP's own
// executables call this, from main().  The library also runs inside other
// programs (sglang, embedded) and only checks the limit there.
inline void RaiseOpenFileLimit() {
  struct rlimit rl{};
  if (getrlimit(RLIMIT_NOFILE, &rl) != 0 || rl.rlim_cur == rl.rlim_max) return;
  const rlim_t before = rl.rlim_cur;
  rl.rlim_cur = rl.rlim_max;
  if (setrlimit(RLIMIT_NOFILE, &rl) == 0) {
    MORI_UMBP_INFO("[fd-limit] raised RLIMIT_NOFILE soft {} -> {}", before, rl.rlim_cur);
  } else {
    MORI_UMBP_WARN("[fd-limit] could not raise RLIMIT_NOFILE soft {} -> {}: {}", before,
                   rl.rlim_cur, std::strerror(errno));
  }
}

}  // namespace mori::umbp
