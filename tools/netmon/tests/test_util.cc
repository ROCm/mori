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
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

#include "cli.h"
#include "harness.h"
#include "meta_json.h"
#include "privilege.h"
#include "util.h"

using namespace netmon;

namespace {

bool Parse(std::vector<std::string> args, RecordArgs* a, std::string* err) {
  args.insert(args.begin(), "record");
  std::vector<char*> v;
  for (auto& s : args) v.push_back(s.data());
  v.push_back(nullptr);
  return ParseRecordArgs(static_cast<int>(args.size()), v.data(), a, err);
}

}  // namespace

TEST(Durations) {
  uint64_t ns;
  CHECK(ParseDurationNs("2ms", &ns) && ns == 2000000ull);
  CHECK(ParseDurationNs("500us", &ns) && ns == 500000ull);
  CHECK(ParseDurationNs("1.5s", &ns) && ns == 1500000000ull);
  CHECK(ParseDurationNs("10m", &ns) && ns == 600000000000ull);
  CHECK(ParseDurationNs("0", &ns) && ns == 0);
  CHECK(!ParseDurationNs("5", &ns));
  CHECK(!ParseDurationNs("ms", &ns));
  CHECK(!ParseDurationNs("-1s", &ns));
  CHECK(!ParseDurationNs("2days", &ns));
  CHECK(!ParseDurationNs("1e30s", &ns));  // overflow guard
}

TEST(CpuListsAndIntervals) {
  std::vector<int> c;
  CHECK(ParseCpuList("0-3,8", &c) && c.size() == 5 && c[4] == 8);
  CHECK(!ParseCpuList("3-1", &c));
  CHECK(!ParseCpuList("a", &c));
  CHECK(!ParseCpuList("", &c));
  RecordArgs a;
  std::string e;
  CHECK(ParseIntervals("rdma=1ms,ethtool=0", &a, &e));
  CHECK_EQ(a.interval_rdma_ns, 1000000ull);
  CHECK_EQ(a.interval_ethtool_ns, 0ull);
  CHECK(!ParseIntervals("foo=1ms", &a, &e));
  CHECK(!ParseIntervals("rdma=2m", &a, &e));  // > 60 s
}

TEST(RecordArgs_) {
  RecordArgs a;
  std::string e;
  CHECK(Parse({"-o", "/x/{host}", "--sources", "rdma,ipc", "--tag", "role=D", "--tag", "a=b", "-d",
               "10s", "-q"},
              &a, &e));
  CHECK(!a.sources_auto && a.want_rdma && a.want_ipc && !a.want_ethtool);
  CHECK(a.tags.size() == 2 && a.tags["role"] == "D");
  CHECK(ExpandHost(a.out, "n1") == "/x/n1");
  RecordArgs b;
  CHECK(!Parse({"-o", "a", "-o", "b"}, &b, &e));  // duplicate single-value option
  RecordArgs c;
  CHECK(!Parse({"--nics", "x"}, &c, &e));  // missing -o
  RecordArgs d;
  CHECK(!Parse({"-o", "a", "--sources", "rdma,bogus"}, &d, &e));
  RecordArgs f;
  CHECK(!Parse({"-o", "a", "extra"}, &f, &e));
  RecordArgs g;
  CHECK(!Parse({"-o", "a", "--host-id", "a/b"}, &g, &e));
}

TEST(SudoArgv) {
  auto v =
      BuildSudoArgv("/abs/ainic-netmon", {"record", "-o", "dir with space", "--sudo"}, 1000, 1001);
  std::vector<std::string> want = {"sudo",           "--",      "/abs/ainic-netmon", "record", "-o",
                                   "dir with space", "--owner", "1000:1001"};
  CHECK(v == want);
  // An explicit --owner is kept and not overridden.
  v = BuildSudoArgv("/b", {"record", "--owner", "5:6", "--sudo", "-o", "d"}, 1000, 1001);
  want = {"sudo", "--", "/b", "record", "--owner", "5:6", "-o", "d"};
  CHECK(v == want);
  v = BuildSudoArgv("/b", {"record", "--owner=5:6", "--sudo"}, 1000, 1001);
  CHECK(v.back() == "--owner=5:6");
}

TEST(OwnerParsing) {
  uid_t u;
  gid_t g;
  CHECK(ParseOwner("100428:100428", &u, &g) && u == 100428 && g == 100428);
  CHECK(!ParseOwner("100428", &u, &g));
  CHECK(!ParseOwner("a:b", &u, &g));
  CHECK(!ParseOwner("1:", &u, &g));
  Owner o;
  Status st;
  CHECK(DetermineOwner("7:8", "/nonexistent/x", &o, &st) && o.uid == 7 && o.source == "--owner");
  CHECK(!DetermineOwner("bad", "/x", &o, &st) && !st.ok());
}

TEST(JsonEscaping) {
  JsonWriter j(false);
  j.BeginObject().Key("a\"b").String("x\n\t\\\x01").Key("n").Null();
  j.Key("arr").StringArray({"p", "q"}).Key("d").Double(1.5).EndObject();
  CHECK(j.str() ==
        "{\"a\\\"b\":\"x\\n\\t\\\\\\u0001\",\"n\":null,\"arr\":[\"p\",\"q\"],\"d\":1.5}");
}

TEST(DefaultExperimentName) {
  // same as os.path.basename(os.path.dirname(os.path.abspath(p))) with cwd /w/exp
  CHECK(ParentDirName("./run", "/w/exp") == "exp");
  CHECK(ParentDirName("run/", "/w/exp") == "exp");
  CHECK(ParentDirName("../exp2/n1", "/w/exp") == "exp2");
  CHECK(ParentDirName("/shared/exp42/host1", "/x") == "exp42");
  CHECK(ParentDirName("/shared/exp42/./host1/", "/x") == "exp42");
  CHECK(ParentDirName("/a/b/../c/d", "/x") == "c");
  CHECK(ParentDirName("/run", "/x").empty());  // parent is "/": no name
  CHECK(LexicalAbsPath("../../..", "/a") == "/");
}

TEST(ContainerDetection) {
  char tmpl[] = "/tmp/netmon_ctr_XXXXXX";
  const std::string root = mkdtemp(tmpl);
  auto write = [&](const std::string& rel, const std::string& text) {
    FILE* f = fopen((root + rel).c_str(), "w");
    CHECK(f != nullptr);
    if (f) {
      fputs(text.c_str(), f);
      fclose(f);
    }
  };
  CHECK(mkdir((root + "/proc").c_str(), 0700) == 0 && mkdir((root + "/proc/1").c_str(), 0700) == 0);
  write("/proc/1/cgroup", "0::/init.scope\n");  // a host (or a VM)
  CHECK(!InContainer(root));
  write("/proc/1/cgroup", "0::/system.slice/docker-0123abcd.scope\n");
  CHECK(InContainer(root));
  write("/proc/1/cgroup", "0::/\n");  // cgroup namespace hides the path: markers decide
  CHECK(!InContainer(root));
  CHECK(mkdir((root + "/run").c_str(), 0700) == 0);
  write("/run/.containerenv", "");  // Podman
  CHECK(InContainer(root));
  unlink((root + "/run/.containerenv").c_str());
  write("/.dockerenv", "");
  CHECK(InContainer(root));
  unlink((root + "/.dockerenv").c_str());
  unlink((root + "/proc/1/cgroup").c_str());
  rmdir((root + "/proc/1").c_str());
  rmdir((root + "/proc").c_str());
  rmdir((root + "/run").c_str());
  rmdir(root.c_str());
}
