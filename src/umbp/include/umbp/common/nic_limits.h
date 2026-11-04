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
#include <string_view>

namespace mori::umbp {

// What a NIC can translate of the memory UMBP registers with it.  The limits
// are the NIC's, not UMBP's, so a registrant checks a region here rather than
// knowing them itself.

// AMD Pensando AINIC firmware before 1.117.5-a-190 translates only this many
// page entries of a memory region, and a remote access past them silently
// lands that many pages lower: 1 GiB of 4 KiB pages, 512 GiB of 2 MiB ones.
inline constexpr uint64_t kAinicRegionPages = uint64_t{1} << 18;

// Warns, naming `what`, this host's AINIC firmware and `remedy`, when a region
// of `bytes` on `page_size` pages has more entries than kAinicRegionPages and an
// AINIC here runs firmware before the fix.  `page_size` is what actually backs
// the region: a hugepage request that fell back to 4 KiB pages has 512 times
// the entries.  0 skips the check.
void WarnIfNicRegionExceedsAinicLimit(std::string_view what, uint64_t bytes, uint64_t page_size,
                                      std::string_view remedy);

}  // namespace mori::umbp
