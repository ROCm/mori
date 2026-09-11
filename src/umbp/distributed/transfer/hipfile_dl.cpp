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
#include "hipfile_dl.h"

#include <dlfcn.h>

#include <mutex>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {

// SONAME first, then the devel symlink: a runtime-only install ships only the
// former.
constexpr const char* kHipFileSonames[] = {"libhipfile.so.0", "libhipfile.so"};

void* OpenHipFile() {
  for (const char* soname : kHipFileSonames) {
    // RTLD_LOCAL: hipFile is ours to call, not something to publish into the
    // process's global namespace.  RTLD_NOW so a partially-resolvable library
    // fails here rather than at the first read.
    if (void* h = ::dlopen(soname, RTLD_NOW | RTLD_LOCAL)) return h;
  }
  return nullptr;
}

template <typename Fn>
bool Bind(void* handle, const char* name, Fn* out) {
  ::dlerror();
  void* sym = ::dlsym(handle, name);
  if (sym == nullptr) {
    const char* err = ::dlerror();
    MORI_UMBP_WARN("[hipFile] symbol {} missing from libhipfile: {}", name,
                   err != nullptr ? err : "unknown");
    return false;
  }
  *out = reinterpret_cast<Fn>(sym);
  return true;
}

}  // namespace

const HipFileApi* HipFile() {
  static HipFileApi api;
  static const HipFileApi* resolved = nullptr;
  static std::once_flag once;

  std::call_once(once, [] {
    void* handle = OpenHipFile();
    if (handle == nullptr) {
      // Not an error: a build that found the headers still runs fine on a host
      // without the library — GDS is just not on offer there.
      const char* err = ::dlerror();
      MORI_UMBP_INFO("[hipFile] libhipfile not loadable ({}) — GDS path disabled",
                     err != nullptr ? err : "no such library");
      return;
    }
    const bool ok = Bind(handle, "hipFileHandleRegister", &api.HandleRegister) &&
                    Bind(handle, "hipFileHandleDeregister", &api.HandleDeregister) &&
                    Bind(handle, "hipFileRead", &api.Read) &&
                    Bind(handle, "hipFileGetOpErrorString", &api.GetOpErrorString);
    if (!ok) {
      // Deliberately not dlclose'd: the handle stays alive for the process, and
      // closing it while another consumer holds symbols would be worse than the
      // leak.
      MORI_UMBP_WARN("[hipFile] libhipfile loaded but incomplete — GDS path disabled");
      return;
    }
    MORI_UMBP_INFO("[hipFile] libhipfile loaded — GDS path available");
    resolved = &api;
  });

  return resolved;
}

}  // namespace mori::umbp
