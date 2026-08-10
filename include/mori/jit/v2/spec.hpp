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
// CRTP skeleton shared by every JIT kernel. A new kernel supplies three things:
// a Cfg type, RenderSource(), and Geometry(). Everything else -- hashing,
// caching, publication, module load, launch -- is here.
//
// Prepare() does all the work and hands back a Plan; Launch(plan, ...) does no
// rendering and no hashing. Steady state is therefore free, and a captured HIP
// graph never sees a compile (which it could not tolerate anyway).

#pragma once

#include <hip/hip_runtime_api.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mori/jit/v2/compiler.hpp"

namespace mori {
namespace jit {
namespace v2 {

struct LaunchGeometry {
  unsigned gridX = 1;
  unsigned blockX = 1;
  unsigned sharedBytes = 0;
};

// CfgT is an explicit parameter rather than `typename Derived::Cfg`: the derived
// class is still incomplete while its own base is being instantiated.
template <typename Derived, typename CfgT>
class KernelSpec {
 public:
  using Cfg = CfgT;

  // A resolved kernel: the loaded module plus the geometry the host derived
  // from the same Cfg the device was compiled with.
  struct Plan {
    std::shared_ptr<Module> module;
    Cfg cfg{};
    LaunchGeometry geom{};

    explicit operator bool() const { return module != nullptr; }
  };

  // Compiles if needed. Call outside HIP graph capture.
  static Plan Prepare(const Cfg& cfg) {
    Plan p;
    p.cfg = cfg;
    p.geom = Derived::Geometry(cfg);
    p.module =
        Compiler::Instance().Build(Derived::kName, Derived::RenderSource(cfg), Derived::SourceDeps());
    return p;
  }

  // Single by-value struct argument, the shape every EP kernel uses.
  template <typename ArgsT>
  static void Launch(const Plan& plan, const ArgsT& args, hipStream_t stream) {
    if (!plan.module) throw std::runtime_error("mori jit: Launch on an unprepared plan");
    LaunchRaw(plan, &args, sizeof(ArgsT), stream);
  }

  static void LaunchRaw(const Plan& plan, const void* argBuf, size_t argSize,
                        hipStream_t stream) {
    // HIP_LAUNCH_PARAM_BUFFER_* is the documented route for a single packed
    // argument buffer; it does not rely on the kernelParams aliasing trick.
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, const_cast<void*>(argBuf),
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize, HIP_LAUNCH_PARAM_END};
    hipError_t err = hipModuleLaunchKernel(plan.module->Entry(), plan.geom.gridX, 1, 1,
                                           plan.geom.blockX, 1, 1, plan.geom.sharedBytes, stream,
                                           nullptr, config);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("mori jit: launch of '") + Derived::kName +
                               "' failed: " + hipGetErrorString(err));
    }
  }

  // Header subtrees whose contents invalidate this kernel. Overridable.
  static const std::vector<std::string>& SourceDeps() { return DefaultSourceDeps(); }
};

}  // namespace v2
}  // namespace jit
}  // namespace mori
