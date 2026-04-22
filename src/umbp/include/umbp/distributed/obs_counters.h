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

// Observability counter gate.  Release build leaves MORI_UMBP_OBS_COUNTERS
// undefined and increments compile away to (void)0 — zero CPU cost.  Tests
// and benches enable it via target_compile_definitions(MORI_UMBP_OBS_COUNTERS).
//
// Atomic counter members and their public getters are declared
// unconditionally (ABI stable across test/release builds); only the
// increment call sites are gated.  Usage:
//
//   MORI_UMBP_OBS_INC(some_counter_);
//   MORI_UMBP_OBS_ADD(some_counter_, n);
#ifdef MORI_UMBP_OBS_COUNTERS
#define MORI_UMBP_OBS_INC(counter) (counter).fetch_add(1, std::memory_order_relaxed)
#define MORI_UMBP_OBS_ADD(counter, n) (counter).fetch_add((n), std::memory_order_relaxed)
#else
#define MORI_UMBP_OBS_INC(counter) ((void)0)
#define MORI_UMBP_OBS_ADD(counter, n) ((void)0)
#endif
