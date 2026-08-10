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
// One C ABI for every JIT kernel, and the registration macro that plugs a Spec
// into it.
//
// Before this existed, each kernel carried ~190 lines of hand-written
// create/destroy/launch/info plus a matching ctypes struct on the Python side --
// and nothing checked that the two declarations of that struct agreed. That is
// the same class of bug as the old EpDispatchCombineArgsRaw layout, reintroduced
// at the language boundary. Two mechanisms remove it:
//
//   * The REQUEST crosses as (name, value) pairs. There is no struct to match,
//     an unknown name is an error, and a missing one takes the C++ default.
//   * The ARGS struct crosses by SCHEMA. C++ publishes its own field list and
//     byte size; Python builds its ctypes type from that string and asserts the
//     size. One declaration, so there is nothing to keep in step.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mori {
namespace jit {
namespace v2 {

// Named request values, as handed over by the binding. Missing keys fall back to
// whatever default the caller passes to Get(), which is the C++-side default --
// so adding a request field never breaks an older binding.
class FieldBag {
 public:
  FieldBag(const char* const* names, const long long* values, int count);

  bool Has(const char* name) const;
  long long Get(const char* name, long long fallback) const;

  // Names supplied by the caller that no Get() asked for. Reported as an error
  // rather than ignored: a typo'd knob that silently does nothing is exactly how
  // a measurement ends up describing the wrong binary.
  std::vector<std::string> Unread() const;

 private:
  struct Entry {
    std::string name;
    long long value;
    mutable bool read = false;
  };
  std::vector<Entry> entries_;
};

// What a kernel must provide to be reachable from the C ABI.
struct PlanVTable {
  // Build and compile. Returns an opaque plan, or throws.
  void* (*create)(const FieldBag&);
  void (*destroy)(void*);
  void (*launch)(void*, const void* argBuf, size_t argSize, void* stream);
  // "key=value\n..." — geometry, the full Cfg, and the cache directory.
  std::string (*info)(void*);
  // Fill the disk cache for the declared coverage; no GPU touched.
  int (*precompile)(const std::string& arch);
  // "name:tag=default,..." for the request; nested flattened as parent.child.
  // The binding generates its constructor from this, so a new request field
  // needs no Python edit at all.
  std::string (*requestSchema)();
  // "name:type,name:type,..."  type in {p,u64,i64,i32,f32}
  const char* argsSchema;
  // sizeof the args struct. The binding builds its own struct from the schema
  // and asserts this -- the check the hand-written ctypes struct never had.
  size_t argsSize;
};

void RegisterPlan(const char* name, const PlanVTable& vt);
const PlanVTable* FindPlan(const char* name);
std::vector<std::string> RegisteredPlans();

// Cross-compiling is a whole-process mode, not a per-call parameter: the
// toolchain resolves its arch once and every compile uses it. An explicit arch
// is applied before that resolution; disagreeing with an already-resolved
// toolchain is an error rather than a silent "rendered for A, compiled for B".
std::string ResolveArch(const char* arch);

void SetPlanError(const std::string& what);
const char* PlanError();

namespace detail {

// Geometry + full Cfg + cache dir, in the key=value form `info` returns.
std::string FormatPlanInfo(unsigned grid, unsigned block, unsigned sharedBytes,
                           const std::string& cfgKeyValues, const std::string& cfgText,
                           const std::string& cacheDir);

}  // namespace detail

// ---------------------------------------------------------------------------
// Registration.
//
// A kernel supplies three small functions and one schema string; everything
// else -- the C entry points, error handling, the plan handle, arch resolution --
// comes from here.
//
//   RequestFn : FieldBag  -> Spec::Cfg          (named, no layout)
//   InfoFn    : Spec::Cfg -> "k=v\n..."         (usually Describe(cfg))
//   PrecompFn : arch      -> kernel count
// ---------------------------------------------------------------------------
#define MORI_JIT_DEFINE_PLAN(NAME, SPEC, REQUEST_FN, REQUEST_SCHEMA_FN, INFO_FN,      \
                             PRECOMPILE_FN, ARGS_T, ARGS_SCHEMA)                                            \
  namespace {                                                                        \
  struct NAME##PlanHandle {                                                          \
    SPEC::Plan plan;                                                                  \
    std::string cfgText;                                                              \
    std::string cacheDir;                                                             \
    std::string cfgKeyValues;                                                         \
  };                                                                                  \
                                                                                      \
  void* NAME##PlanCreate(const ::mori::jit::v2::FieldBag& f) {                            \
    auto* h = new NAME##PlanHandle();                                                  \
    const auto cfg = REQUEST_FN(f);                                                    \
    h->cfgText = Render(cfg);                                                           \
    h->cfgKeyValues = INFO_FN(cfg);                                                     \
    /* Same deps Prepare compiles with. Defaulting them here would report a     \
       directory nobody ever writes for any Spec that overrides SourceDeps. */  \
    h->cacheDir = ::mori::jit::v2::Compiler::Instance().CacheDirFor(                 \
        SPEC::kName, SPEC::RenderSource(cfg), SPEC::SourceDeps());               \
    h->plan = SPEC::Prepare(cfg);                                                        \
    return h;                                                                            \
  }                                                                                      \
  void NAME##PlanDestroy(void* p) { delete static_cast<NAME##PlanHandle*>(p); }           \
  void NAME##PlanLaunch(void* p, const void* buf, size_t size, void* stream) {            \
    auto* h = static_cast<NAME##PlanHandle*>(p);                                           \
    if (size != sizeof(ARGS_T))                                                            \
      throw std::runtime_error("args size " + std::to_string(size) + " != " +              \
                               std::to_string(sizeof(ARGS_T)) +                            \
                               "; the binding's schema is stale");                         \
    SPEC::LaunchRaw(h->plan, buf, size, static_cast<hipStream_t>(stream));                 \
  }                                                                                        \
  std::string NAME##PlanInfo(void* p) {                                                    \
    auto* h = static_cast<NAME##PlanHandle*>(p);                                            \
    return ::mori::jit::v2::detail::FormatPlanInfo(h->plan.geom.gridX, h->plan.geom.blockX,     \
                                               h->plan.geom.sharedBytes, h->cfgKeyValues,   \
                                               h->cfgText, h->cacheDir);                    \
  }                                                                                          \
  const struct NAME##PlanRegistrar {                                                         \
    NAME##PlanRegistrar() {                                                                  \
      ::mori::jit::v2::RegisterPlan(#NAME, ::mori::jit::v2::PlanVTable{                              \
                                           &NAME##PlanCreate, &NAME##PlanDestroy,            \
                                           &NAME##PlanLaunch, &NAME##PlanInfo,               \
                                           &PRECOMPILE_FN, &REQUEST_SCHEMA_FN,               \
                                           ARGS_SCHEMA, sizeof(ARGS_T)});                    \
    }                                                                                         \
  } NAME##PlanRegistrarInstance;                                                              \
  }  // namespace

}  // namespace v2
}  // namespace jit
}  // namespace mori
