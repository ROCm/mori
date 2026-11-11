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
