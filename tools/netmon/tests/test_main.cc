#include <cstring>
#include <fstream>
#include <iterator>

#include "harness.h"

namespace nt {

std::vector<Test>& Registry() {
  static std::vector<Test> r;
  return r;
}
int g_failures = 0;

std::vector<uint8_t> ReadFixture(const std::string& name) {
  std::ifstream f(std::string(NETMON_FIXTURES) + "/" + name, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

}  // namespace nt

int RunRecorderCase(const char* name, const char* out);

int main(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "--recorder-case") == 0) {
    return RunRecorderCase(argv[2], argv[3]);
  }
  const char* only = argc > 1 ? argv[1] : nullptr;
  int ran = 0, failed_tests = 0;
  for (auto& t : nt::Registry()) {
    if (only && std::strstr(t.name, only) == nullptr) continue;
    int before = nt::g_failures;
    fprintf(stderr, "[ RUN  ] %s\n", t.name);
    t.fn();
    bool ok = nt::g_failures == before;
    fprintf(stderr, "[ %s ] %s\n", ok ? " OK " : "FAIL", t.name);
    ran++;
    failed_tests += ok ? 0 : 1;
  }
  fprintf(stderr, "%d tests, %d failed\n", ran, failed_tests);
  return failed_tests ? 1 : 0;
}
