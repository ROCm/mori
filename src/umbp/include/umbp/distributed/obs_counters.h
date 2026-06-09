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

// Test/bench-only build switch (renamed from MORI_UMBP_OBS_COUNTERS in v2.1.2).
//
// MORI_UMBP_TESTING covers TWO orthogonal responsibilities, gated by the same
// macro because both are exercised only by the same downstream targets:
//
//   1. Observability counter increments.  Atomic counter members and their
//      public getters are declared unconditionally (ABI stable across
//      test/release builds); only the increment call sites are gated, so
//      release builds pay zero CPU cost and getters return 0.
//
//   2. Test seams.  Selected private members (e.g. PoolClient::IssueBatchWrite)
//      are declared `virtual` only when the macro is defined, allowing test
//      subclasses to inject failures.  Release builds keep them as plain
//      members for inlining.
//
// Build implication: test/release object files MUST NOT be mixed-linked.
// Toggling MORI_UMBP_TESTING changes class layout (vtable presence) and ABI.
// The CI test target builds with -DMORI_UMBP_TESTING=ON unconditionally;
// production release leaves it OFF.
//
// Usage:
//   MORI_UMBP_OBS_INC(some_counter_);
//   MORI_UMBP_OBS_ADD(some_counter_, n);
//   MORI_UMBP_TEST_VIRTUAL void Foo();   // virtual under -DMORI_UMBP_TESTING
#ifdef MORI_UMBP_TESTING
#define MORI_UMBP_OBS_INC(counter) (counter).fetch_add(1, std::memory_order_relaxed)
#define MORI_UMBP_OBS_ADD(counter, n) (counter).fetch_add((n), std::memory_order_relaxed)
#define MORI_UMBP_TEST_VIRTUAL virtual
#else
#define MORI_UMBP_OBS_INC(counter) ((void)0)
#define MORI_UMBP_OBS_ADD(counter, n) ((void)0)
#define MORI_UMBP_TEST_VIRTUAL
#endif
