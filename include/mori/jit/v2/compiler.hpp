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
//
// Content-addressed JIT compile + cache.
//
// Key: name $$ compiler $$ flags $$ include-hash $$ source-text. The
// configuration IS the source text, so every codegen input is in the key by
// construction -- there is no second list to keep in step.

#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mori {
namespace jit {
namespace v2 {

// Fallback entry name. Specs should override it: one symbol for every kernel
// makes a profile unreadable. See KernelSpec::EntryName.
inline constexpr const char* kEntryName = "mori_jit_entry";

class Module {
 public:
  // Loads `hsacoPath` on the current device and resolves `entry`. Throws
  // std::runtime_error on failure.
  explicit Module(const std::string& hsacoPath, std::string entry = kEntryName);
  ~Module();

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  hipFunction_t Entry() const { return entry_; }

 private:
  hipModule_t module_ = nullptr;
  hipFunction_t entry_ = nullptr;
  std::string path_;
  std::string entryName_;
};

// Hash over the kernel header tree. Coarse on purpose: it covers whole
// subdirectories rather than parsing #include graphs, so it can over-invalidate
// but never under-invalidate. Cached per process.
const std::string& IncludeTreeHash(const std::vector<std::string>& relDirs);

// The default dependency set for EP kernels.
const std::vector<std::string>& DefaultSourceDeps();

class Compiler {
 public:
  static Compiler& Instance();

  // Cached module for this (name, source). Compiles and publishes on miss.
  // Throws std::runtime_error if compilation fails.
  std::shared_ptr<Module> Build(const std::string& name, const std::string& code,
                                const std::vector<std::string>& sourceDeps = DefaultSourceDeps(),
                                const std::string& entry = kEntryName);

  // Compile and publish without loading; returns the cache directory. Touches no
  // HIP device, so it runs on a GPU-less build machine and cross-compiles via
  // MORI_JIT_ARCH.
  std::string EnsureCompiled(const std::string& name, const std::string& code,
                             const std::vector<std::string>& sourceDeps = DefaultSourceDeps());

  // Cache directory this (name, source) maps to. No side effects beyond
  // resolving the toolchain.
  std::string CacheDirFor(const std::string& name, const std::string& code,
                          const std::vector<std::string>& sourceDeps = DefaultSourceDeps()) const;

 private:
  Compiler() = default;
  struct Impl;
  Impl& GetImpl() const;
};

}  // namespace v2
}  // namespace jit
}  // namespace mori
