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

#include <hipfile.h>

namespace mori::umbp {

// hipFile resolved at runtime instead of linked.
//
// WHY.  GdsEngine is one optional transfer path inside umbp_common, and
// umbp_common is linked into libmori_pybinds.so — so a `-lhipfile` here becomes
// a DT_NEEDED on the Python extension, and `import mori` dies with "libhipfile
// .so.0: cannot open shared object file" on every host whose ROCm ships without
// hipFile, whether or not UMBP is used at all.  Building against the header but
// resolving the four entry points with dlopen keeps the capability and makes
// the library a runtime option: present -> GDS works, absent -> GDS is simply
// not offered as an engine.
//
// The signatures come from <hipfile.h> via decltype, so they cannot drift from
// the real ones.
struct HipFileApi {
  decltype(&::hipFileHandleRegister) HandleRegister = nullptr;
  decltype(&::hipFileHandleDeregister) HandleDeregister = nullptr;
  decltype(&::hipFileRead) Read = nullptr;
  decltype(&::hipFileGetOpErrorString) GetOpErrorString = nullptr;
};

// Loads libhipfile on the first call and caches the result (including failure).
// Returns nullptr when the library or any symbol is missing; callers must check
// before every use.  Thread-safe, never throws.
const HipFileApi* HipFile();

}  // namespace mori::umbp
