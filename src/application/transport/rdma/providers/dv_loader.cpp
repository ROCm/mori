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

#include "mori/application/transport/rdma/providers/dv_loader.hpp"

#include <dlfcn.h>

#include "mori/utils/mori_log.hpp"

void* DvLoadLibrary(const char* lib_name) {
  void* handle = dlopen(lib_name, RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    MORI_APP_TRACE("dlopen({}) failed: {}", lib_name, dlerror());
  } else {
    MORI_APP_TRACE("dlopen({}) succeeded", lib_name);
  }
  return handle;
}

void* DvLoadSymbol(void* handle, const char* symbol_name) {
  void* sym = dlsym(handle, symbol_name);
  if (!sym) {
    MORI_APP_WARN("dlsym({}) failed: {}", symbol_name, dlerror());
  }
  return sym;
}
