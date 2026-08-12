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
// The backend-wide half of the plan ABI: registry, error channel, arch
// resolution, and the seven extern "C" entry points that serve every kernel.

#include "mori/jit/v2/plan_api.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "mori/jit/v2/toolchain.hpp"

namespace mori {
namespace jit {
namespace v2 {

// ---------------------------------------------------------------------------
// FieldBag
// ---------------------------------------------------------------------------

FieldBag::FieldBag(const char* const* names, const long long* values, int count) {
  entries_.reserve(count > 0 ? static_cast<size_t>(count) : 0);
  for (int i = 0; i < count; ++i) {
    if (!names || !names[i] || !values) continue;
    entries_.push_back(Entry{names[i], values[i], false});
  }
}

bool FieldBag::Has(const char* name) const {
  for (const auto& e : entries_)
    if (e.name == name) return true;
  return false;
}

long long FieldBag::Get(const char* name, long long fallback) const {
  for (const auto& e : entries_) {
    if (e.name == name) {
      e.read = true;
      return e.value;
    }
  }
  return fallback;
}

std::vector<std::string> FieldBag::Unread() const {
  std::vector<std::string> out;
  for (const auto& e : entries_)
    if (!e.read) out.push_back(e.name);
  return out;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

namespace {

std::map<std::string, PlanVTable>& Registry() {
  static std::map<std::string, PlanVTable> r;
  return r;
}

std::mutex& RegistryMutex() {
  static std::mutex m;
  return m;
}

thread_local std::string g_lastError;

}  // namespace

void RegisterPlan(const char* name, const PlanVTable& vt) {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  Registry()[name] = vt;
}

const PlanVTable* FindPlan(const char* name) {
  if (!name) return nullptr;
  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto it = Registry().find(name);
  return it == Registry().end() ? nullptr : &it->second;
}

std::vector<std::string> RegisteredPlans() {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  std::vector<std::string> out;
  for (const auto& kv : Registry()) out.push_back(kv.first);
  return out;
}

void SetPlanError(const std::string& what) { g_lastError = what; }

const char* PlanError() { return g_lastError.c_str(); }

std::string ResolveArch(const char* arch) {
  if (arch && *arch) setenv("MORI_JIT_ARCH", arch, /*overwrite=*/1);
  const std::string resolved = GetToolchain().arch;
  if (arch && *arch && resolved != arch) {
    throw std::runtime_error("toolchain already resolved for '" + resolved +
                             "'; set arch (or MORI_JIT_ARCH) before the first JIT call "
                             "to cross-compile");
  }
  return resolved;
}

namespace detail {

std::string FormatPlanInfo(unsigned grid, unsigned block, unsigned sharedBytes,
                           const std::string& cfgKeyValues, const std::string& cfgText,
                           const std::string& cacheDir) {
  std::ostringstream ss;
  ss << "gridX=" << grid << "\nblockX=" << block << "\nsharedBytes=" << sharedBytes << "\n"
     << cfgKeyValues << "cfgText=" << cfgText << "\ncacheDir=" << cacheDir << "\n";
  return ss.str();
}

}  // namespace detail

}  // namespace v2
}  // namespace jit
}  // namespace mori

// ===========================================================================
// C ABI — seven symbols, for every kernel, forever.
//
// ctypes rather than pybind11 or the CPython C API: this carries only pointers
// and scalars, so there is nothing for a type-conversion layer to do, and it
// keeps torch out of every header on this side (MORI_JIT_V2_DESIGN §3.5).
// ===========================================================================

using mori::jit::v2::FieldBag;
using mori::jit::v2::FindPlan;
using mori::jit::v2::PlanVTable;
using mori::jit::v2::ResolveArch;
using mori::jit::v2::SetPlanError;

namespace {

struct PlanHandle {
  const PlanVTable* vt;
  void* impl;
};

}  // namespace

// EVERY entry below must be noexcept in practice: an exception crossing extern "C"
// into ctypes is undefined behaviour, and in practice terminates the interpreter with
// no traceback. Anything that allocates -- and the schema getters do, if only to build
// an error string -- therefore catches. Failure is reported the way the rest of the
// ABI reports it: a null/negative return plus mori_jit_last_error().
extern "C" {

#define MORI_JIT_API __attribute__((visibility("default")))

MORI_JIT_API const char* mori_jit_last_error() { return mori::jit::v2::PlanError(); }

MORI_JIT_API const char* mori_jit_plan_args_schema(const char* kernel) {
  try {
    const PlanVTable* vt = FindPlan(kernel);
    if (!vt) {
      SetPlanError(std::string("unknown kernel '") + (kernel ? kernel : "(null)") + "'");
      return nullptr;
    }
    return vt->argsSchema;
  } catch (...) {
    return nullptr;
  }
}

// "name:tag=default,..." — the binding builds its constructor from this.
MORI_JIT_API const char* mori_jit_plan_request_schema(const char* kernel) {
  static thread_local std::string s;
  try {
    const PlanVTable* vt = FindPlan(kernel);
    if (!vt) {
      SetPlanError(std::string("unknown kernel '") + (kernel ? kernel : "(null)") + "'");
      return nullptr;
    }
    s = vt->requestSchema();
    return s.c_str();
  } catch (...) {
    return nullptr;
  }
}

MORI_JIT_API int mori_jit_plan_args_size(const char* kernel) {
  try {
    const PlanVTable* vt = FindPlan(kernel);
    if (!vt) {
      SetPlanError(std::string("unknown kernel '") + (kernel ? kernel : "(null)") + "'");
      return -1;
    }
    return static_cast<int>(vt->argsSize);
  } catch (...) {
    return -1;
  }
}

MORI_JIT_API void* mori_jit_plan_create(const char* kernel, const char* arch,
                                        const char* const* names, const long long* values,
                                        int count) {
  try {
    const PlanVTable* vt = FindPlan(kernel);
    if (!vt)
      throw std::runtime_error(std::string("unknown kernel '") + (kernel ? kernel : "(null)") +
                               "'");
    ResolveArch(arch);
    FieldBag bag(names, values, count);
    void* impl = vt->create(bag);
    // A knob nobody read is a knob that silently did nothing.
    const auto unread = bag.Unread();
    if (!unread.empty()) {
      vt->destroy(impl);
      std::string msg = "unknown request field(s):";
      for (const auto& u : unread) msg += " " + u;
      throw std::runtime_error(msg);
    }
    return new PlanHandle{vt, impl};
  } catch (const std::exception& e) {
    SetPlanError(e.what());
    return nullptr;
  }
}

MORI_JIT_API void mori_jit_plan_destroy(void* plan) {
  auto* h = static_cast<PlanHandle*>(plan);
  if (!h) return;
  try {
    h->vt->destroy(h->impl);
  } catch (...) {
    // Swallow: this is the teardown path, the caller has no way to react, and
    // letting it out of extern "C" would turn a leak into a terminate.
  }
  delete h;
}

MORI_JIT_API int mori_jit_plan_launch(void* plan, const void* argBuf, int argSize, void* stream) {
  auto* h = static_cast<PlanHandle*>(plan);
  if (!h) {
    SetPlanError("null plan");
    return -1;
  }
  try {
    h->vt->launch(h->impl, argBuf, static_cast<size_t>(argSize), stream);
    return 0;
  } catch (const std::exception& e) {
    SetPlanError(e.what());
    return -1;
  }
}

// Fills `buf` with "key=value\n..." and returns the number of bytes the full
// text needs (so a short buffer can be retried).
MORI_JIT_API int mori_jit_plan_info(void* plan, char* buf, int len) {
  auto* h = static_cast<PlanHandle*>(plan);
  if (!h) {
    SetPlanError("null plan");
    return -1;
  }
  try {
    const std::string s = h->vt->info(h->impl);
    if (buf && len > 0) {
      const int n = static_cast<int>(s.size()) < len - 1 ? static_cast<int>(s.size()) : len - 1;
      std::memcpy(buf, s.data(), static_cast<size_t>(n));
      buf[n] = '\0';
    }
    return static_cast<int>(s.size()) + 1;
  } catch (const std::exception& e) {
    SetPlanError(e.what());
    return -1;
  }
}

MORI_JIT_API int mori_jit_precompile(const char* kernel, const char* arch) {
  try {
    const PlanVTable* vt = FindPlan(kernel);
    if (!vt)
      throw std::runtime_error(std::string("unknown kernel '") + (kernel ? kernel : "(null)") +
                               "'");
    return vt->precompile(ResolveArch(arch));
  } catch (const std::exception& e) {
    SetPlanError(e.what());
    return -1;
  }
}

// Comma-separated names of every registered kernel.
MORI_JIT_API const char* mori_jit_registered_plans() {
  static thread_local std::string s;
  try {
    s.clear();
    for (const auto& n : mori::jit::v2::RegisteredPlans()) {
      if (!s.empty()) s += ",";
      s += n;
    }
    return s.c_str();
  } catch (...) {
    return nullptr;
  }
}

#undef MORI_JIT_API

}  // extern "C"
