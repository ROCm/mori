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

#include <set>
#include <string>

#include "src/ras/identity.hpp"

using namespace mori::ras;

TEST(RasIdentity, ParsesTheStartTimeAfterTheLastParenthesis) {
  // The comm field holds spaces and parentheses; parsing must start after the last ')'.
  std::string stat = "4242 (my (odd) proc) S 1 4242 4242 0 -1 4194560 100 0 0 0 1 2 0 0 20 0 3 0 ";
  stat += "987654321 1000000 100 18446744073709551615 0 0 0 0 0 0 0 0 0 17 5 0 0 0 0 0";
  EXPECT_EQ(StartTimeFromStat(stat), 987654321u);
  EXPECT_EQ(StartTimeFromStat("4242 (short) S 1 2 3"), 0u);
  EXPECT_EQ(StartTimeFromStat(""), 0u);
}

TEST(RasIdentity, ParsesThePidNamespaceInode) {
  EXPECT_EQ(InodeFromNsLink("pid:[4026532111]"), 4026532111u);
  EXPECT_EQ(InodeFromNsLink("net:[4026532111]"), 0u);
  EXPECT_EQ(InodeFromNsLink(""), 0u);
}

TEST(RasIdentity, ReadsTheLiveProcess) {
  ProcessFacts f = ReadProcessFacts();
  struct stat ns{};
  ASSERT_EQ(stat("/proc/self/ns/pid", &ns), 0);
  EXPECT_EQ(f.pidNamespace, ns.st_ino);
  EXPECT_NE(f.startTimeTicks, 0u);
  EXPECT_EQ(f.bootId.size(), 36u);
}

TEST(RasIdentity, CollectsTheEngineFieldsAndProcessFacts) {
  RuntimeUuid uuid{0x1234, 0x8000000000005678ull};
  RasProcessId id = CollectIdentity("node-1", "host-1", "10.0.0.1", 77, uuid);
  EXPECT_EQ(id.uuid, uuid);
  EXPECT_EQ(id.nodeId, "node-1");
  EXPECT_EQ(id.hostname, "host-1");
  EXPECT_EQ(id.host, "10.0.0.1");
  EXPECT_EQ(id.pid, 77);
  ProcessFacts f = ReadProcessFacts();
  EXPECT_EQ(id.startTimeTicks, f.startTimeTicks);
  EXPECT_EQ(id.pidNamespace, f.pidNamespace);
  EXPECT_EQ(id.bootId, f.bootId);
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
