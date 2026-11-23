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
// Tiny self-contained test harness (no third-party dependency).
#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace nt {

struct Test {
  const char* name;
  std::function<void()> fn;
};
std::vector<Test>& Registry();
extern int g_failures;

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { Registry().push_back({name, fn}); }
};

std::vector<uint8_t> ReadFixture(const std::string& name);

}  // namespace nt

#define TEST(name)                                    \
  static void name();                                 \
  static nt::Registrar name##_registrar(#name, name); \
  static void name()

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) {                                                              \
      fprintf(stderr, "  CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      nt::g_failures++;                                                         \
    }                                                                           \
  } while (0)

#define CHECK_EQ(a, b)                                                                          \
  do {                                                                                          \
    auto va_ = (a);                                                                             \
    auto vb_ = (b);                                                                             \
    if (!(va_ == vb_)) {                                                                        \
      fprintf(stderr, "  CHECK_EQ failed %s:%d: %s == %s (%s vs %s)\n", __FILE__, __LINE__, #a, \
              #b, std::to_string(va_).c_str(), std::to_string(vb_).c_str());                    \
      nt::g_failures++;                                                                         \
    }                                                                                           \
  } while (0)
