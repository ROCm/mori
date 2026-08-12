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
// Cfg -> designated-initialiser text. The rendered text IS the cache key, so a
// field that does not reach here is a field the kernel never sees.
//
// Only non-default fields are emitted: adding a field whose default preserves
// behaviour leaves every existing instance's text (and therefore its cached
// .hsaco) untouched. A field that changes the kernel at its default value is
// caught by the separate include hash, not by this.
//
// HIP-free and attribute-free -- shared by host (C++17) and device TU (C++20).

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

// Field-count guard (merged here from the former field_count.hpp -- both are the
// HIP-free Cfg-introspection header, and a Cfg header always includes both).
// A Cfg field that Render() forgets to emit would silently select the default --
// a wrong kernel, not just a stale cache; pinning the count makes it a compile
// error. Needs C++20 concepts; #include-safe under C++17 (macro is a no-op there,
// and the JIT device TU is C++20, so the assert fires at the latest on first
// compile of the kernel). See docs/MORI_JIT_V2_DESIGN.md §3.5.
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define MORI_JIT_HAS_FIELD_COUNT 1
#else
#define MORI_JIT_HAS_FIELD_COUNT 0
#endif

#if MORI_JIT_HAS_FIELD_COUNT
#define MORI_JIT_ASSERT_FIELD_COUNT(Type, N, Msg) \
  static_assert(::mori::jit::v2::FieldCount<Type>() == (N), Msg)
#else
#define MORI_JIT_ASSERT_FIELD_COUNT(Type, N, Msg) static_assert(true, "")
#endif

namespace mori {
namespace jit {
namespace v2 {

#if MORI_JIT_HAS_FIELD_COUNT
namespace detail {

// Converts to anything, so it can stand in for a member of any type. Declared,
// never defined -- only used in unevaluated context.
struct AnyType {
  template <typename T>
  constexpr operator T() const noexcept;
};

// Largest number of initialisers the aggregate accepts. A member of nested
// aggregate type counts as ONE (AnyType converts directly to it, so brace
// elision never applies) -- FieldCountNestedCountsAsOne pins this.
template <typename T, typename... Args>
constexpr std::size_t TryCount() {
  if constexpr (requires { T{Args{}..., AnyType{}}; })
    return TryCount<T, Args..., AnyType>();
  else
    return sizeof...(Args);
}

}  // namespace detail

// Number of direct members of the aggregate T.
template <typename T>
constexpr std::size_t FieldCount() {
  static_assert(std::is_aggregate_v<T>, "FieldCount requires an aggregate");
  return detail::TryCount<T>();
}
#endif  // MORI_JIT_HAS_FIELD_COUNT

// Scalar renderers. User Cfg structs supply their own RenderValue() found by ADL.
inline std::string RenderValue(bool v) { return v ? "true" : "false"; }
inline std::string RenderValue(int v) { return std::to_string(v); }
inline std::string RenderValue(long v) { return std::to_string(v) + "L"; }

// Accumulates ".name=value" for fields that differ from the default.
class Fields {
 public:
  // Compares the RENDERED forms rather than the values, so nested aggregates
  // need no operator== (which would be a second field list to maintain).
  template <typename T>
  void Put(const char* name, const T& value, const T& defaultValue) {
    using mori::jit::v2::RenderValue;  // ADL for user types, these for scalars
    std::string rendered = RenderValue(value);
    if (rendered == RenderValue(defaultValue)) return;
    items_.emplace_back(std::string(".") + name + "=" + rendered);
  }

  std::string Join() const {
    std::string out;
    for (size_t i = 0; i < items_.size(); ++i) {
      if (i) out += ", ";
      out += items_[i];
    }
    return out;
  }

 private:
  std::vector<std::string> items_;
};

// Wraps the field list in a braced initialiser for `type`.
inline std::string BraceInit(const char* type, const Fields& f) {
  return std::string(type) + "{" + f.Join() + "}";
}

// ---------------------------------------------------------------------------
// Wire schema + generic apply.
//
// Both are driven by the SAME VisitFields walk that Render uses, so the binding
// learns the request shape from the one field list rather than from a second
// declaration someone has to keep in step. That second declaration is what made
// the hand-written ctypes struct a layout hazard.
// ---------------------------------------------------------------------------

// Scalar type tags as they appear on the wire.
inline const char* WireTag(int) { return "i32"; }
inline const char* WireTag(long) { return "i64"; }
inline const char* WireTag(bool) { return "b"; }

inline long long WireValue(int v) { return v; }
inline long long WireValue(long v) { return v; }
inline long long WireValue(bool v) { return v ? 1 : 0; }

inline void WireAssign(int& dst, long long v) { dst = static_cast<int>(v); }
inline void WireAssign(long& dst, long long v) { dst = static_cast<long>(v); }
inline void WireAssign(bool& dst, long long v) { dst = v != 0; }

// An enum crosses as its underlying integer; the binding is told "e" so it can
// apply a name mapping (torch.bfloat16 -> 0) instead of demanding a raw int.
template <typename E, typename = std::enable_if_t<std::is_enum_v<E>>>
inline const char* WireTag(E) {
  return "e";
}
template <typename E, typename = std::enable_if_t<std::is_enum_v<E>>>
inline long long WireValue(E v) {
  return static_cast<long long>(v);
}
template <typename E, typename = std::enable_if_t<std::is_enum_v<E>>>
inline void WireAssign(E& dst, long long v) {
  dst = static_cast<E>(v);
}

// Collects "name:tag=default" for every leaf, flattening nested aggregates as
// "parent.child" so the binding can present them (e.g. off.src) without knowing
// the C++ types.
class SchemaBuilder {
 public:
  void Add(const std::string& name, const char* tag, long long dflt) {
    if (!out_.empty()) out_ += ",";
    out_ += name;
    out_ += ":";
    out_ += tag;
    out_ += "=";
    out_ += std::to_string(dflt);
  }
  const std::string& Str() const { return out_; }

 private:
  std::string out_;
};

}  // namespace v2
}  // namespace jit
}  // namespace mori
