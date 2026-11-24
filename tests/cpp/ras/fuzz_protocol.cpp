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

// Coverage-guided fuzzing of the RAS frame decoder (-DMORI_RAS_FUZZ=ON, clang). Anything it
// accepts must re-encode to a frame that decodes to the same message. Run it with
//   fuzz_ras_protocol -max_total_time=60 <corpus dir>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/ras/protocol.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  using namespace mori::ras::proto;
  RasHeader h;
  (void)DecodeHeader(data, size, kSupportedVersions, &h);
  Frame f;
  if (DecodeFrame(data, size, kSupportedVersions, &f) != DecodeStatus::kOk) return 0;
  std::vector<uint8_t> again = EncodeFrame(f.header, f.message);
  Frame g;
  if (DecodeFrame(again.data(), again.size(), kSupportedVersions, &g) != DecodeStatus::kOk ||
      !(g.message == f.message) || g.header.sequence != f.header.sequence) {
    __builtin_trap();
  }
  return 0;
}
