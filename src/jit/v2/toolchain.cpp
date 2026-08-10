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

#include "mori/jit/v2/toolchain.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>

#include "mori/jit/v2/util.hpp"
#include "mori/utils/mori_log.hpp"

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace mori {
namespace jit {
namespace v2 {

namespace fs = std::filesystem;

int WaveSizeForArch(const std::string& arch) {
  // Prefix rather than an enumeration: a new gfx12xx member should not need an
  // edit here. get_warp_size() misreports 64 on gfx1250, so key off the string.
  return arch.rfind("gfx12", 0) == 0 ? 32 : 64;
}

std::vector<std::string> Toolchain::IncludeDirs() const {
  std::vector<std::string> dirs{sourceRoot, sourceRoot + "/include", sourceRoot + "/src"};
  for (const char* sub : {"3rdparty/spdlog/include", "3rdparty/msgpack-c/include"}) {
    std::string p = sourceRoot + "/" + sub;
    if (fs::is_directory(p)) dirs.push_back(p);
  }
  return dirs;
}

std::vector<std::string> Toolchain::Flags() const {
  std::vector<std::string> f{
      "--genco",
      "--offload-arch=" + arch,
      // C++20 is required only here: the Cfg struct is a structural NTTP.
      // Host code stays C++17 (see MORI_JIT_V2_DESIGN §3.5).
      "-std=c++20",
      "-O2",
      "-D__HIP_PLATFORM_AMD__",
      "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
  };
  if (arch.rfind("gfx950", 0) == 0) f.push_back("-DHIP_ENABLE_GFX950_OCP_BUILTINS=1");
  if (const char* extra = std::getenv("MORI_JIT_EXTRA_FLAGS")) {
    for (const std::string& tok : SplitWhitespace(extra)) f.push_back(tok);
  }
  return f;
}

std::string Toolchain::Diagnose() const {
  if (arch.empty()) return "GPU arch not detected (no HIP device? set MORI_JIT_ARCH)";
  if (hipcc.empty()) return "hipcc not found (set ROCM_PATH or MORI_JIT_HIPCC)";
  if (sourceRoot.empty())
    return "mori kernel sources not found (set MORI_SOURCE_ROOT to the repo root)";
  return "ok";
}

namespace {

std::string EnvOr(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::string(v) : fallback;
}

std::string DetectArch() {
  if (const char* v = std::getenv("MORI_JIT_ARCH"); v && *v) return v;
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) return "";
  std::string a(props.gcnArchName);
  auto colon = a.find(':');  // strip ":xnack-" etc; features do not change codegen here
  return colon == std::string::npos ? a : a.substr(0, colon);
}

std::string SelfLibDir() {
#ifdef __linux__
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&DetectArch), &info) && info.dli_fname) {
    return fs::path(info.dli_fname).parent_path().string();
  }
#endif
  return "";
}

// A directory is a usable source root if the headers the generated TU includes
// are actually under it.
bool LooksLikeSourceRoot(const fs::path& p) {
  return fs::is_directory(p / "include" / "mori") && fs::is_directory(p / "src" / "ops");
}

std::string DetectSourceRoot() {
  if (const char* v = std::getenv("MORI_SOURCE_ROOT"); v && *v) {
    if (LooksLikeSourceRoot(v)) return v;
    MORI_WARN(mori::modules::OPS, "MORI_SOURCE_ROOT={} does not look like a mori tree", v);
  }
  // Wheel install: the sources are packaged next to the .so.
  const std::string selfDir = SelfLibDir();
  if (!selfDir.empty()) {
    for (const char* rel : {"_jit-sources", "../_jit-sources", ".."}) {
      fs::path cand = fs::path(selfDir) / rel;
      std::error_code ec;
      cand = fs::weakly_canonical(cand, ec);
      if (!ec && LooksLikeSourceRoot(cand)) return cand.string();
    }
  }
#ifdef MORI_JIT_SOURCE_DIR
  // Baked in by CMake for development builds.
  if (LooksLikeSourceRoot(MORI_JIT_SOURCE_DIR)) return MORI_JIT_SOURCE_DIR;
#endif
  return "";
}

// The NIC is told to us, not probed here: MORI_DEVICE_NIC is the single authority
// (same variable CMake and python/mori/jit/config.py use). The v2 Python layer
// fills it from mori.jit.config.detect_nic_type() (env -> /sys/class/infiniband ->
// lspci -> lib -> mlx5) before the first JIT call, so this normally just reads the
// resolved value. The default matches v1's detect_nic_type() fallback -- mlx5 --
// rather than a bare "none", so a pure-C++ caller still gets a sane NIC instead of
// what reads like "detection failed".
std::string DetectNic() {
  std::string v = EnvOr("MORI_DEVICE_NIC", "mlx5");
  for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return v.empty() ? "mlx5" : v;
}

std::string DetectHipcc(const std::string& rocmPath) {
  if (const char* v = std::getenv("MORI_JIT_HIPCC"); v && *v) return v;
  for (const std::string& cand : {rocmPath + "/bin/hipcc", std::string("/opt/rocm/bin/hipcc")}) {
    if (fs::is_regular_file(cand)) return cand;
  }
  return "";
}

// Compiler identity. Two ROCm versions produce different code from the same
// source, so this belongs in the cache key.
std::string CompilerSignature(const std::string& hipcc) {
  if (hipcc.empty()) return "unknown";
  CommandResult r = RunCommand(Quote(hipcc) + " --version");
  if (r.exitCode != 0) return "unknown";
  // First line is "HIP version: 7.15.0-0000000"; keep it whole.
  auto nl = r.output.find('\n');
  std::string first = nl == std::string::npos ? r.output : r.output.substr(0, nl);
  while (!first.empty() && (first.back() == '\r' || first.back() == ' ')) first.pop_back();
  return first;
}

Toolchain* g_override = nullptr;

}  // namespace

void SetToolchainForTesting(const Toolchain& tc) {
  static Toolchain storage;
  if (tc.arch.empty()) {
    g_override = nullptr;
    return;
  }
  storage = tc;
  g_override = &storage;
}

const Toolchain& GetToolchain() {
  if (g_override) return *g_override;

  static Toolchain tc;
  static std::once_flag once;
  static std::string err;
  std::call_once(once, [] {
    tc.rocmPath = EnvOr("ROCM_PATH", "/opt/rocm");
    tc.arch = DetectArch();
    tc.nic = DetectNic();
    tc.hipcc = DetectHipcc(tc.rocmPath);
    tc.sourceRoot = DetectSourceRoot();
    tc.signature = CompilerSignature(tc.hipcc);

    const char* home = std::getenv("HOME");
    tc.cacheRoot = EnvOr("MORI_JIT_CACHE_DIR", std::string(home ? home : "/tmp") + "/.mori/jit");

    err = tc.Valid() ? "" : tc.Diagnose();
    if (err.empty()) {
      MORI_INFO(mori::modules::OPS, "[jit] arch={} nic={} hipcc={} root={} cache={}", tc.arch,
                tc.nic, tc.hipcc, tc.sourceRoot, tc.cacheRoot);
    }
  });

  if (!err.empty()) throw std::runtime_error("mori jit: " + err);
  return tc;
}

}  // namespace v2
}  // namespace jit
}  // namespace mori
