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

// Builds a runtime's RasProcessId. Every source is unprivileged and read once at startup:
// process facts from /proc, which are the PID namespace (inode of /proc/self/ns/pid), the
// start time (field 22 of /proc/self/stat), and the boot ID that scopes it.

#include <cstdint>
#include <string>

#include "mori/ras/types.hpp"

namespace mori {
namespace ras {

// Where collection reads from. LiveIdentitySources() is the running system; tests point the
// root at a fixture tree.
struct IdentitySources {
  std::string procRoot{"/proc"};
};

IdentitySources LiveIdentitySources();

struct ProcessFacts {
  uint64_t pidNamespace{0};
  uint64_t startTimeTicks{0};
  std::string bootId;
};
// `pid` is "self" or a decimal PID. Missing facts stay 0 or empty.
ProcessFacts ReadProcessFacts(const std::string& procRoot, const std::string& pid);

RasProcessId CollectIdentity(const std::string& nodeId, const std::string& hostname,
                             const std::string& host, int32_t pid, RuntimeUuid uuid,
                             const IdentitySources& src);

// A random version-4 UUID. Its variant bits keep `lo` nonzero, so it can be a header sender.
RuntimeUuid NewRuntimeUuid();
// 128 random bits as 32 hex digits.
std::string NewToken();

}  // namespace ras
}  // namespace mori
