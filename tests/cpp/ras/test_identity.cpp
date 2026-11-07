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
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

#include "src/ras/identity.hpp"

using namespace mori::ras;

namespace {

// A scratch directory tree, removed when the fixture goes away.
class FixtureTree {
 public:
  FixtureTree() {
    char tmpl[] = "/tmp/mori_ras_identity_XXXXXX";
    root_ = mkdtemp(tmpl);
  }
  ~FixtureTree() { std::system(("rm -rf " + root_).c_str()); }
  const std::string& Root() const { return root_; }

  void Write(const std::string& rel, const std::string& content) {
    std::string path = root_ + "/" + rel;
    std::system(("mkdir -p $(dirname " + path + ")").c_str());
    std::ofstream(path) << content;
  }
  void Symlink(const std::string& rel, const std::string& target) {
    std::string path = root_ + "/" + rel;
    std::system(("mkdir -p $(dirname " + path + ")").c_str());
    ASSERT_EQ(symlink(target.c_str(), path.c_str()), 0);
  }

 private:
  std::string root_;
};

}  // namespace

TEST(RasIdentity, ReadsProcessFactsFromAFixture) {
  FixtureTree t;
  // The comm field holds spaces and parentheses; parsing must start after the last ')'.
  std::string stat = "4242 (my (odd) proc) S 1 4242 4242 0 -1 4194560 100 0 0 0 1 2 0 0 20 0 3 0 ";
  stat += "987654321 1000000 100 18446744073709551615 0 0 0 0 0 0 0 0 0 17 5 0 0 0 0 0\n";
  t.Write("proc/4242/stat", stat);
  t.Symlink("proc/4242/ns/pid", "pid:[4026532111]");
  t.Write("proc/sys/kernel/random/boot_id", "1b6c9a52-3f0e-4c8e-9a31-5d2c7e8f9a01\n");
  ProcessFacts f = ReadProcessFacts(t.Root() + "/proc", "4242");
  EXPECT_EQ(f.startTimeTicks, 987654321u);
  EXPECT_EQ(f.pidNamespace, 4026532111u);
  EXPECT_EQ(f.bootId, "1b6c9a52-3f0e-4c8e-9a31-5d2c7e8f9a01");

  ProcessFacts missing = ReadProcessFacts(t.Root() + "/proc", "999");
  EXPECT_EQ(missing.startTimeTicks, 0u);
  EXPECT_EQ(missing.pidNamespace, 0u);
}

TEST(RasIdentity, ReadsTheLiveProcess) {
  ProcessFacts self = ReadProcessFacts("/proc", "self");
  EXPECT_NE(self.pidNamespace, 0u);
  EXPECT_NE(self.startTimeTicks, 0u);
  EXPECT_EQ(self.bootId.size(), 36u);
  ProcessFacts byPid = ReadProcessFacts("/proc", std::to_string(getpid()));
  EXPECT_EQ(byPid.startTimeTicks, self.startTimeTicks);
  EXPECT_EQ(byPid.pidNamespace, self.pidNamespace);
}

TEST(RasIdentity, CollectsTheEngineFieldsAndProcessFacts) {
  FixtureTree t;
  t.Write("proc/self/stat", "77 (x) S 1 1 1 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 555 0 0\n");
  t.Symlink("proc/self/ns/pid", "pid:[4026531836]");
  t.Write("proc/sys/kernel/random/boot_id", "b00710d0-0000-4000-8000-000000000001\n");
  IdentitySources s;
  s.procRoot = t.Root() + "/proc";

  RuntimeUuid uuid{0x1234, 0x8000000000005678ull};
  RasProcessId id = CollectIdentity("node-1", "host-1", "10.0.0.1", 77, uuid, s);
  EXPECT_EQ(id.uuid, uuid);
  EXPECT_EQ(id.nodeId, "node-1");
  EXPECT_EQ(id.hostname, "host-1");
  EXPECT_EQ(id.host, "10.0.0.1");
  EXPECT_EQ(id.pid, 77);
  EXPECT_EQ(id.startTimeTicks, 555u);
  EXPECT_EQ(id.pidNamespace, 4026531836u);
  EXPECT_EQ(id.bootId, "b00710d0-0000-4000-8000-000000000001");
}

TEST(RasIdentity, RuntimeUuidsAreRandomVersion4) {
  std::set<std::pair<uint64_t, uint64_t>> seen;
  for (int i = 0; i < 1000; ++i) {
    RuntimeUuid u = NewRuntimeUuid();
    EXPECT_EQ((u.hi >> 12) & 0xf, 4u);  // version 4
    EXPECT_EQ(u.lo >> 62, 2u);          // RFC 4122 variant
    EXPECT_NE(u.lo, 0u);
    EXPECT_TRUE(seen.insert({u.hi, u.lo}).second);
  }
  RuntimeUuid u{0x0123456789abcdefull, 0x8edcba9876543210ull};
  EXPECT_EQ(u.ToString(), "01234567-89ab-cdef-8edc-ba9876543210");
}

TEST(RasIdentity, TokensAre128RandomBits) {
  std::set<std::string> seen;
  for (int i = 0; i < 1000; ++i) {
    std::string tok = NewToken();
    ASSERT_EQ(tok.size(), 32u);
    EXPECT_EQ(tok.find_first_not_of("0123456789abcdef"), std::string::npos);
    EXPECT_TRUE(seen.insert(tok).second);
  }
}
