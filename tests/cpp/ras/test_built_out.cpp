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

// The RAS API with RAS built out: compiled without MORI_ENABLE_RAS and linked without
// mori_ras, the way a subsystem builds with BUILD_RAS=OFF. Every call must compile, link, and
// do nothing.
#include <cstdio>

#include "mori/ras/ras.hpp"

#if defined(MORI_ENABLE_RAS)
#error "test_ras_built_out must build without MORI_ENABLE_RAS"
#endif

int main() {
  using namespace mori::ras;
  if (RasRuntime::Get() != nullptr) return 1;
  if (RasRuntime::EnsureStarted(StartParams{"node", "host", "127.0.0.1", 1}) != nullptr) return 2;
  RuntimeStatus st = RasRuntime::Status();
  if (st.state != RunState::kBuiltOut || st.degraded != 0) return 3;
  // How subsystems call it: everything behind the null check.
  if (RasRuntime* r = RasRuntime::Get()) {
    r->Enqueue(PeerAdded{});
    r->SetLabels(Labels{{"role", "x"}});
    (void)r->Endpoint();
    (void)r->Snapshot();
    return 4;
  }
  std::printf("RAS built out: every call is a no-op\n");
  return 0;
}
