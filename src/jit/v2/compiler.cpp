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

#include "mori/jit/v2/compiler.hpp"

#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "mori/jit/v2/toolchain.hpp"
#include "mori/jit/v2/util.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace jit {
namespace v2 {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

Module::Module(const std::string& hsacoPath) : path_(hsacoPath) {
  hipError_t err = hipModuleLoad(&module_, hsacoPath.c_str());
  if (err != hipSuccess) {
    throw std::runtime_error("mori jit: hipModuleLoad(" + hsacoPath +
                             ") failed: " + hipGetErrorString(err));
  }
  err = hipModuleGetFunction(&entry_, module_, kEntryName);
  if (err != hipSuccess) {
    (void)hipModuleUnload(module_);
    module_ = nullptr;
    throw std::runtime_error(std::string("mori jit: '") + kEntryName + "' not found in " +
                             hsacoPath + " (" + hipGetErrorString(err) +
                             "). The generated TU must export exactly this symbol.");
  }
}

Module::~Module() {
  if (module_) (void)hipModuleUnload(module_);
}

namespace {

constexpr const char* kSourceFile = "kernel.hip";
constexpr const char* kObjectFile = "kernel.hsaco";

bool IsHeaderLike(const fs::path& p) {
  const std::string ext = p.extension().string();
  return ext == ".hpp" || ext == ".h" || ext == ".cuh" || ext == ".hip" || ext == ".cpp";
}

std::string HashTree(const std::string& root, const std::vector<std::string>& relDirs) {
  // Sorted walk so the digest does not depend on readdir order.
  std::set<fs::path> files;
  for (const auto& rel : relDirs) {
    fs::path base = fs::path(root) / rel;
    std::error_code ec;
    if (!fs::is_directory(base, ec)) continue;
    for (auto it = fs::recursive_directory_iterator(base, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
      if (ec) break;
      if (it->is_regular_file(ec) && IsHeaderLike(it->path())) files.insert(it->path());
    }
  }

  Sha256 h;
  for (const auto& f : files) {
    std::string content;
    if (!ReadFile(f.string(), &content)) continue;
    // Include the path: moving a file changes what the compiler resolves.
    h.Update(fs::relative(f, root).string());
    h.Update("\0", 1);
    h.Update(content);
  }
  return h.HexDigest(16);
}

bool CacheEntryValid(const std::string& dir) {
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return false;
  // The directory is published by rename, so it is complete or absent. A
  // half-populated one means someone hand-edited the cache.
  if (!fs::is_regular_file(fs::path(dir) / kObjectFile, ec)) {
    MORI_WARN(mori::modules::OPS, "[jit] cache entry missing {}: {} -- remove it and rerun",
              kObjectFile, dir);
    return false;
  }
  return true;
}

int CurrentDevice() {
  int dev = 0;
  (void)hipGetDevice(&dev);
  return dev;
}

}  // namespace

const std::vector<std::string>& DefaultSourceDeps() {
  // Directory walks, not prefixes: "src/ops/dispatch_combine" does NOT cover the
  // sibling "src/ops/dispatch_combine_v2", where the v2 kernel bodies live. Both
  // are listed, or editing a kernel body would leave the cache entry valid and
  // the next run would load the stale .hsaco.
  static const std::vector<std::string> deps{"include/mori", "src/ops/dispatch_combine",
                                             "src/ops/dispatch_combine_v2", "src/ops/kernels",
                                             "src/cco"};
  return deps;
}

const std::string& IncludeTreeHash(const std::vector<std::string>& relDirs) {
  static std::mutex mu;
  static std::map<std::string, std::string> cache;

  std::string key;
  for (const auto& d : relDirs) key += d + "|";

  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  const auto t0 = std::chrono::steady_clock::now();
  std::string digest = HashTree(GetToolchain().sourceRoot, relDirs);
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
          .count();
  MORI_DEBUG(mori::modules::OPS, "[jit] include hash {} ({} ms)", digest, ms);
  return cache.emplace(key, std::move(digest)).first->second;
}

struct Compiler::Impl {
  std::mutex mu;
  // Keyed by device as well as directory: hipModuleLoad binds to the calling
  // thread's current device, and one process can drive several GPUs.
  std::map<std::pair<int, std::string>, std::shared_ptr<Module>> modules;
};

Compiler::Impl& Compiler::GetImpl() const {
  static Impl impl;
  return impl;
}

Compiler& Compiler::Instance() {
  static Compiler c;
  return c;
}

std::string Compiler::CacheDirFor(const std::string& name, const std::string& code,
                                  const std::vector<std::string>& sourceDeps) const {
  const Toolchain& tc = GetToolchain();

  Sha256 h;
  h.Update(name);
  h.Update("$$");
  h.Update(tc.signature);
  h.Update("$$");
  // The NIC is not in Flags() today because the intranode LSA path links no NIC
  // bitcode -- but a GDA build does, and a binary that gets linked in is a
  // codegen input like any other. Hashing it here rather than waiting for it to
  // show up in a flag is what keeps "config cannot reach the compiler without
  // reaching the key" true in advance instead of in hindsight.
  h.Update(tc.nic.empty() ? "none" : tc.nic);
  h.Update("$$");
  for (const auto& f : tc.Flags()) {
    h.Update(f);
    h.Update(" ");
  }
  h.Update("$$");
  h.Update(IncludeTreeHash(sourceDeps));
  h.Update("$$");
  h.Update(code);

  // arch and nic are both already inside the digest; Tag() is a directory level
  // purely so a human can find things.
  return tc.cacheRoot + "/" + tc.Tag() + "/kernel." + name + "." + h.HexDigest(24);
}

std::string Compiler::EnsureCompiled(const std::string& name, const std::string& code,
                                     const std::vector<std::string>& sourceDeps) {
  const Toolchain& tc = GetToolchain();
  const std::string dir = CacheDirFor(name, code, sourceDeps);

  if (!CacheEntryValid(dir)) {
    // Build into a private directory, then publish the whole thing by rename.
    const std::string tmp = MakeUniqueTempDir(tc.cacheRoot + "/tmp", name);
    if (tmp.empty())
      throw std::runtime_error("mori jit: cannot create temp dir under " + tc.cacheRoot + "/tmp");

    const std::string srcPath = tmp + "/" + kSourceFile;
    const std::string objPath = tmp + "/" + kObjectFile;
    if (!WriteFileSynced(srcPath, code)) {
      SafeRemoveAll(tmp);
      throw std::runtime_error("mori jit: cannot write " + srcPath);
    }

    std::vector<std::string> argv{tc.hipcc};
    for (const auto& f : tc.Flags()) argv.push_back(f);
    for (const auto& d : tc.IncludeDirs()) {
      argv.push_back("-I");
      argv.push_back(d);
    }
    argv.push_back(srcPath);
    argv.push_back("-o");
    argv.push_back(objPath);

    MORI_INFO(mori::modules::OPS, "[jit] compiling {} for {}", name, tc.arch);
    if (std::getenv("MORI_JIT_VERBOSE")) {
      MORI_INFO(mori::modules::OPS, "[jit] {}", JoinArgv(argv));
    }

    const auto t0 = std::chrono::steady_clock::now();
    CommandResult r = RunProgram(argv);
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
            .count();

    if (r.exitCode != 0 || !fs::is_regular_file(objPath)) {
      // Keep the offending source when asked; it is the only way to see what
      // the renderer actually produced.
      std::string keep;
      if (std::getenv("MORI_JIT_KEEP_FAILED")) {
        keep = tc.cacheRoot + "/failed." + name + "." + std::to_string(::getpid());
        std::error_code ec;
        fs::rename(tmp, keep, ec);
        if (ec) keep.clear();
      }
      if (keep.empty()) SafeRemoveAll(tmp);
      throw std::runtime_error("mori jit: hipcc failed for '" + name + "' (exit " +
                               std::to_string(r.exitCode) + ")\ncommand: " + JoinArgv(argv) + "\n" +
                               r.output +
                               (keep.empty() ? "\n(set MORI_JIT_KEEP_FAILED=1 to retain the source)"
                                             : "\nsource kept at " + keep));
    }

    MORI_INFO(mori::modules::OPS, "[jit] compiled {} in {} ms -> {}", name, ms, dir);

    // Losing the publish race is fine: the winner's directory is byte-equivalent,
    // because the directory name IS the content hash.
    PublishDir(tmp, dir);
    if (!CacheEntryValid(dir)) {
      throw std::runtime_error("mori jit: publish succeeded but cache entry is invalid: " + dir);
    }
  }
  return dir;
}

std::shared_ptr<Module> Compiler::Build(const std::string& name, const std::string& code,
                                        const std::vector<std::string>& sourceDeps) {
  const int dev = CurrentDevice();
  Impl& impl = GetImpl();

  // The memory cache is keyed on the directory, so this needs the key before the
  // compile check. Computing it twice is a hash of a ~400 byte string.
  const std::string probe = CacheDirFor(name, code, sourceDeps);
  {
    std::lock_guard<std::mutex> lock(impl.mu);
    auto it = impl.modules.find({dev, probe});
    if (it != impl.modules.end()) return it->second;
  }

  const std::string dir = EnsureCompiled(name, code, sourceDeps);

  auto mod = std::make_shared<Module>(dir + "/" + kObjectFile);
  std::lock_guard<std::mutex> lock(impl.mu);
  auto [it, inserted] = impl.modules.emplace(std::make_pair(dev, dir), std::move(mod));
  return it->second;
}

}  // namespace v2
}  // namespace jit
}  // namespace mori
