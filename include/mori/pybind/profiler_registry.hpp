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

#include <pybind11/pybind11.h>

#include <functional>
#include <string>
#include <vector>

namespace mori {
namespace pybind {

using SlotBinder = std::function<void(pybind11::module_&)>;

class ProfilerSlotRegistry {
 public:
  static ProfilerSlotRegistry& Get() {
    static ProfilerSlotRegistry instance;
    return instance;
  }

  void Register(SlotBinder binder) { binders_.push_back(binder); }

  void BindAll(pybind11::module_& m) {
    for (auto& binder : binders_) {
      binder(m);
    }
  }

 private:
  std::vector<SlotBinder> binders_;
};

// Helper class for static registration
struct ProfilerSlotRegistration {
  ProfilerSlotRegistration(SlotBinder binder) { ProfilerSlotRegistry::Get().Register(binder); }
};

#define MORI_REGISTER_PROFILER_SLOTS(func) \
  static mori::pybind::ProfilerSlotRegistration __mori_profiler_reg_##func##__LINE__(func)

// Helper function to expose to mori.cpp
inline void RegisterAllProfilerSlots(pybind11::module_& m) {
  ProfilerSlotRegistry::Get().BindAll(m);
}

// Utility to bind slots to module
inline void BindProfilerSlots(pybind11::module_& m, const char* name,
                              const std::vector<std::pair<const char*, int>>& slots) {
  auto sub = m.def_submodule(name, "Auto-generated profiler slots");
  for (const auto& p : slots) {
    sub.attr(p.first) = p.second;
  }
  sub.attr("NUM_SLOTS") = (int)slots.size();
}

}  // namespace pybind
}  // namespace mori
