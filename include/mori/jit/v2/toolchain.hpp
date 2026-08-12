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
// Where hipcc is, what arch to target, where the kernel sources are, where the
// cache goes. Resolved once per process.

#pragma once

#include <string>
#include <vector>

namespace mori {
namespace jit {
namespace v2 {

// Wave size is an architecture fact the host needs before any kernel exists
// (it sizes blockDim and the LDS budget), so it lives here rather than in a
// device header. gfx12xx is wave32; everything mori supports today is wave64.
int WaveSizeForArch(const std::string& arch);

struct Toolchain {
  std::string arch;   // "gfx942" / "gfx950" / "gfx1250", no :xnack suffix
  std::string hipcc;  // absolute path
  std::string rocmPath;
  std::string sourceRoot;  // repo root, or the packaged _jit-sources dir
  std::string cacheRoot;   // ~/.mori/jit  (MORI_JIT_CACHE_DIR overrides)
  std::string signature;   // compiler identity; part of every cache key

  // "mlx5"/"bnxt"/"ionic" (default mlx5, matching v1's detect_nic_type() fallback).
  // A device-side GDA build links NIC-specific cco bitcode, so this is a codegen
  // input on the same footing as arch and belongs in every cache key. Not probed
  // in C++: the v2 Python layer resolves it via mori.jit.config.detect_nic_type()
  // and hands it over through MORI_DEVICE_NIC, so there is one detector, not two.
  std::string nic;

  // -I flags, derived from sourceRoot.
  std::vector<std::string> IncludeDirs() const;

  // Flags common to every JIT compile. Anything here that can differ between
  // runs must be in the cache key -- it is, because `Flags()` is hashed whole.
  std::vector<std::string> Flags() const;

  // "<arch>_<nic>": the human-readable cache directory level. Correctness does
  // not depend on it -- both components are inside the digest.
  std::string Tag() const { return arch + "_" + (nic.empty() ? "none" : nic); }

  bool Valid() const { return !arch.empty() && !hipcc.empty() && !sourceRoot.empty(); }

  // Human-readable reason why Valid() is false.
  std::string Diagnose() const;
};

// Process-wide singleton. Throws std::runtime_error if it cannot be resolved.
const Toolchain& GetToolchain();

// Test seam: override the resolved toolchain. Pass an empty arch to reset.
void SetToolchainForTesting(const Toolchain& tc);

}  // namespace v2
}  // namespace jit
}  // namespace mori
