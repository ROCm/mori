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
// JIT core: no GPU and no hipcc required. These pin the properties the whole
// design rests on -- that a config cannot reach the compiler without also
// reaching the cache key, and that a forgotten field is a compile error.

#include <gtest/gtest.h>

#include <atomic>
#include <cctype>
#include <filesystem>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include "mori/jit/v2/compiler.hpp"
#include "mori/jit/v2/render.hpp"
#include "mori/jit/v2/toolchain.hpp"
#include "mori/jit/v2/util.hpp"
#include "mori/ops/dispatch_combine_v2/ep_cfg.hpp"
#include "mori/ops/dispatch_combine_v2/ep_spec.hpp"

namespace fs = std::filesystem;
using namespace mori::jit::v2;
using namespace mori::ops::v2;

// --------------------------------------------------------------------------
// FieldCount
// --------------------------------------------------------------------------

namespace {
struct Flat3 {
  int a = 0;
  bool b = false;
  long c = 0;
};
struct Nested {
  int a = 0;
  Flat3 inner{};
  bool z = false;
};
struct Empty {};
}  // namespace

#if MORI_JIT_HAS_FIELD_COUNT
TEST(FieldCount, CountsFlatMembers) { EXPECT_EQ(FieldCount<Flat3>(), 3u); }

TEST(FieldCount, EmptyAggregate) { EXPECT_EQ(FieldCount<Empty>(), 0u); }

// The property the Cfg guards depend on: a nested aggregate is ONE field, not
// its own field count. If brace elision ever started applying here, every
// FieldCount assertion in the Cfg headers would silently need a different number.
TEST(FieldCount, NestedCountsAsOne) { EXPECT_EQ(FieldCount<Nested>(), 3u); }

// The Cfg's own count: if this ever disagrees with VisitFields(EpCfg), a field is
// silently missing from the rendered text -- a wrong kernel, not a stale one.
TEST(FieldCount, MatchesEpCfg) { EXPECT_EQ(FieldCount<EpCfg>(), 11u); }
#endif

// --------------------------------------------------------------------------
// EpCfg: rendering and geometry.
//
// The whole design rests on the rendered text being the specialisation AND the
// cache key, so these check that every field reaches the text and that host and
// device compute the launch geometry from the same arithmetic.
// --------------------------------------------------------------------------

TEST(Render, DefaultCfgEmitsNoFields) {
  // Only non-default fields are emitted, so adding a field whose default
  // preserves behaviour leaves every existing instance's text -- and therefore
  // its cached .hsaco -- untouched.
  EXPECT_EQ(Render(EpCfg{}), "EpCfg{}");
}

TEST(Render, OnlyNonDefaultFieldsAppear) {
  EpCfg c;
  c.hiddenDim = 1024;
  const std::string t = Render(c);
  EXPECT_NE(t.find(".hiddenDim=1024"), std::string::npos);
  EXPECT_EQ(t.find("worldSize"), std::string::npos);
}

TEST(Render, EveryScalarFieldReachesTheText) {
  // The load-bearing one: a field that does not reach the text is a field the
  // kernel never sees, which is a silently WRONG kernel rather than a stale
  // cache entry. Perturb each field in turn and require the text to change.
  const EpCfg base;
  auto differs = [&](const EpCfg& c, const char* what) {
    EXPECT_NE(Render(c), Render(base)) << what << " does not reach the rendered text";
  };
  EpCfg c;
  c = base;
  c.worldSize = 4;
  differs(c, "worldSize");
  c = base;
  c.hiddenDim = 1024;
  differs(c, "hiddenDim");
  c = base;
  c.maxTokPerRank = 256;
  differs(c, "maxTokPerRank");
  c = base;
  c.numExpertPerRank = 16;
  differs(c, "numExpertPerRank");
  c = base;
  c.numExpertPerToken = 4;
  differs(c, "numExpertPerToken");
  c = base;
  c.maxRecv = 2048;
  differs(c, "maxRecv");
  c = base;
  c.dtype = EpDType::Fp32;
  differs(c, "dtype");
  c = base;
  c.blockNum = 32;
  differs(c, "blockNum");
  c = base;
  c.warpPerBlock = 4;
  differs(c, "warpPerBlock");
  c = base;
  c.waveSize = 32;
  differs(c, "waveSize");
  c = base;
  c.useWeights = false;
  differs(c, "useWeights");
}

TEST(Geometry, BlockThreadsIsWarpsTimesWave) {
  EpCfg c;
  c.warpPerBlock = 8;
  c.waveSize = 64;
  EXPECT_EQ(EpBlockThreads(c), 512);
  c.waveSize = 32;  // gfx12xx
  EXPECT_EQ(EpBlockThreads(c), 256);
}

TEST(Geometry, CombineSharedMemCoversOnePointerArrayPerWarp) {
  EpCfg c;
  c.warpPerBlock = 8;
  c.numExpertPerToken = 8;
  c.useWeights = true;  // srcPtrs + srcWeightPtrs
  EXPECT_EQ(EpCombineSharedBytes(c), int(sizeof(void*)) * 8 * 8 * 2);
  c.useWeights = false;  // srcPtrs only
  EXPECT_EQ(EpCombineSharedBytes(c), int(sizeof(void*)) * 8 * 8);
}

TEST(Geometry, ValidityRejectsWhatTheKernelCannotRun) {
  EpCfg ok;
  EXPECT_TRUE(EpCfgIsValid(ok));

  EpCfg c = ok;
  c.hiddenDim = 7;  // token bytes not 16 B aligned; WarpCopy moves whole vec4s
  EXPECT_FALSE(EpCfgIsValid(c));

  c = ok;
  c.numExpertPerToken = 64;  // the dedup ballot needs topk < waveSize
  EXPECT_FALSE(EpCfgIsValid(c));

  c = ok;
  c.worldSize = 128;  // the grid-barrier peer loop needs worldSize <= waveSize
  EXPECT_FALSE(EpCfgIsValid(c));

  c = ok;
  // The recv capacity is also the flat-index stride, so an undersized cap does
  // not merely overrun -- it re-encodes to the next peer.
  c.maxRecv = ok.worldSize * ok.maxTokPerRank - 1;
  EXPECT_FALSE(EpCfgIsValid(c));

  c = ok;
  c.warpPerBlock = 64;  // 64 * 64 > 1024 threads
  EXPECT_FALSE(EpCfgIsValid(c));
}

// The gfx125x dispatch quota. The kernel body needs a gfx125x toolchain to
// compile at all, so this is the only place the arithmetic can be checked on an
// ordinary box -- which is why it lives in ep_cfg.hpp rather than inline.
TEST(WarpTokenQuota, NeverStallsTheLoop) {
  // A quota of 0 makes the token loop step by `aWarps * 0` and never advance:
  // an unkillable hang, and no correctness check can report it because the
  // check hangs with it. Every degenerate input must still yield >= 1.
  for (int n : {-1, 0, 1, 3}) {
    for (int aWarps : {-1, 0, 1, 512}) {
      for (int tpi : {1, 4, 8}) {
        EXPECT_GE(EpWarpTokenQuota(n, aWarps, tpi), 1)
            << "n=" << n << " aWarps=" << aWarps << " tpi=" << tpi;
      }
    }
  }
}

TEST(WarpTokenQuota, IsInactiveOnceTheGridIsFull) {
  // Above the threshold the large-token path must be IDENTICAL, not merely
  // close: aWarps * tpi tokens is exactly where the natural quota fills the
  // grid, and anything at or above it keeps tpi untouched.
  const int aWarps = 512, tpi = 4;
  EXPECT_EQ(EpWarpTokenQuota(aWarps * tpi, aWarps, tpi), tpi);
  EXPECT_EQ(EpWarpTokenQuota(aWarps * tpi + 1, aWarps, tpi), tpi);
  EXPECT_EQ(EpWarpTokenQuota(16384, aWarps, tpi), tpi);
}

TEST(WarpTokenQuota, SpreadsAScarceBatchOverEveryWarp) {
  const int aWarps = 512, tpi = 4;
  // 512 tokens over 512 warps: one each, instead of four to the first 128.
  EXPECT_EQ(EpWarpTokenQuota(512, aWarps, tpi), 1);
  EXPECT_EQ(EpWarpTokenQuota(64, aWarps, tpi), 1);
  // ceil, not floor -- flooring would leave the tail tokens to no warp at all.
  EXPECT_EQ(EpWarpTokenQuota(1025, aWarps, tpi), 3);
  // tpi == 1 has nothing to cap.
  EXPECT_EQ(EpWarpTokenQuota(64, aWarps, 1), 1);
}

TEST(Sha256, KnownVectors) {
  EXPECT_EQ(HexDigest(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(HexDigest("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(Sha256, SurvivesBlockBoundaries) {
  // 55/56/64/119/120 straddle the padding cases in Final().
  for (size_t n : {55u, 56u, 57u, 63u, 64u, 65u, 119u, 120u, 128u}) {
    const std::string s(n, 'x');
    Sha256 chunked;
    for (char ch : s) chunked.Update(&ch, 1);
    EXPECT_EQ(chunked.HexDigest(), HexDigest(s)) << "length " << n;
  }
}

TEST(Sha256, TruncationIsAPrefix) {
  const std::string full = HexDigest("mori");
  EXPECT_EQ(HexDigest("mori", 24), full.substr(0, 24));
}

// --------------------------------------------------------------------------
// Atomic publication
// --------------------------------------------------------------------------

class FsUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = fs::temp_directory_path() /
            ("mori_jit_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter_++));
    MakeDirs(root_.string());
  }
  void TearDown() override { SafeRemoveAll(root_.string()); }
  fs::path root_;
  static std::atomic<int> counter_;
};
std::atomic<int> FsUtilTest::counter_{0};

TEST_F(FsUtilTest, PublishMovesTheWholeDirectory) {
  const std::string tmp = MakeUniqueTempDir((root_ / "tmp").string(), "k");
  ASSERT_FALSE(tmp.empty());
  ASSERT_TRUE(WriteFileSynced(tmp + "/kernel.hsaco", "binary"));
  const std::string finalDir = (root_ / "cache" / "kernel.abc").string();

  EXPECT_TRUE(PublishDir(tmp, finalDir));
  EXPECT_TRUE(fs::is_regular_file(finalDir + "/kernel.hsaco"));
  EXPECT_FALSE(fs::exists(tmp));
}

TEST_F(FsUtilTest, LosingTheRaceLeavesTheWinnersCopyAndCleansUp) {
  const std::string finalDir = (root_ / "cache" / "kernel.abc").string();
  const std::string first = MakeUniqueTempDir((root_ / "tmp").string(), "k");
  ASSERT_TRUE(WriteFileSynced(first + "/kernel.hsaco", "winner"));
  ASSERT_TRUE(PublishDir(first, finalDir));

  const std::string second = MakeUniqueTempDir((root_ / "tmp").string(), "k");
  ASSERT_TRUE(WriteFileSynced(second + "/kernel.hsaco", "loser"));
  EXPECT_FALSE(PublishDir(second, finalDir));

  std::string content;
  ASSERT_TRUE(ReadFile(finalDir + "/kernel.hsaco", &content));
  EXPECT_EQ(content, "winner");
  EXPECT_FALSE(fs::exists(second));  // the loser cleans up after itself
}

TEST_F(FsUtilTest, ConcurrentPublishersProduceExactlyOneWinner) {
  const std::string finalDir = (root_ / "cache" / "kernel.race").string();
  constexpr int kThreads = 16;
  std::atomic<int> winners{0};
  std::vector<std::thread> ts;
  for (int i = 0; i < kThreads; ++i) {
    ts.emplace_back([&, i] {
      const std::string tmp = MakeUniqueTempDir((root_ / "tmp").string(), "k");
      if (tmp.empty()) return;
      WriteFileSynced(tmp + "/kernel.hsaco", "content-" + std::to_string(i));
      if (PublishDir(tmp, finalDir)) winners.fetch_add(1);
    });
  }
  for (auto& t : ts) t.join();

  EXPECT_EQ(winners.load(), 1);
  EXPECT_TRUE(fs::is_regular_file(finalDir + "/kernel.hsaco"));
  // Nothing left behind: every loser removed its own temp directory.
  int leftovers = 0;
  std::error_code ec;
  for (auto& e : fs::directory_iterator(root_ / "tmp", ec)) {
    (void)e;
    ++leftovers;
  }
  EXPECT_EQ(leftovers, 0);
}

TEST_F(FsUtilTest, UniqueTempDirsDoNotCollide) {
  std::set<std::string> seen;
  for (int i = 0; i < 64; ++i) {
    const std::string d = MakeUniqueTempDir((root_ / "tmp").string(), "k");
    ASSERT_FALSE(d.empty());
    EXPECT_TRUE(seen.insert(d).second) << d;
  }
}

// --------------------------------------------------------------------------
// Subprocess
// --------------------------------------------------------------------------

TEST(Process, CapturesStdoutAndExitCode) {
  CommandResult r = RunProgram({"/bin/sh", "-c", "printf hello; exit 3"});
  EXPECT_EQ(r.exitCode, 3);
  EXPECT_EQ(r.output, "hello");
}

TEST(Process, CapturesStderr) {
  CommandResult r = RunProgram({"/bin/sh", "-c", "printf oops >&2"});
  EXPECT_EQ(r.exitCode, 0);
  EXPECT_EQ(r.output, "oops");
}

TEST(Process, ArgumentsAreNotShellInterpreted) {
  // A path containing shell metacharacters must reach the program verbatim --
  // this is why the compiler is invoked with an argv vector, not a command line.
  CommandResult r = RunProgram({"/bin/echo", "a b; rm -rf /", "$HOME"});
  EXPECT_EQ(r.exitCode, 0);
  EXPECT_EQ(r.output, "a b; rm -rf / $HOME\n");
}

TEST(Process, MissingProgramIsAnError) {
  CommandResult r = RunProgram({"/nonexistent/mori-not-a-program"});
  EXPECT_NE(r.exitCode, 0);
}

// --------------------------------------------------------------------------
// Two-level dispatch: tuned table, then the shape-agnostic fallback.
// Needs no GPU -- MakeCombineCfg takes the arch as a parameter.
// --------------------------------------------------------------------------

// NIC. A GDA build links NIC-specific cco bitcode, so the NIC is a codegen
// input; the old path recorded it only in a hand-spelled `<arch>_<nic>`
// directory name, which is a key that a rename can lose.
// --------------------------------------------------------------------------

namespace {
// Enough of a toolchain for CacheDirFor; no GPU and no hipcc are touched.
Toolchain FakeToolchain(const std::string& arch, const std::string& nic) {
  Toolchain tc;
  tc.arch = arch;
  tc.nic = nic;
  tc.hipcc = "/nonexistent/hipcc";
  tc.rocmPath = "/opt/rocm";
  tc.sourceRoot = "/nonexistent/src";
  tc.cacheRoot = "/tmp/mori-jit-test";
  tc.signature = "test";
  return tc;
}

struct ToolchainOverride {
  explicit ToolchainOverride(const Toolchain& tc) { SetToolchainForTesting(tc); }
  ~ToolchainOverride() { SetToolchainForTesting(Toolchain{}); }
};

std::string CacheDirWith(const std::string& arch, const std::string& nic) {
  ToolchainOverride guard(FakeToolchain(arch, nic));
  return mori::jit::v2::Compiler::Instance().CacheDirFor("combine_reduce", "constexpr int x = 1;",
                                                         {});
}
}  // namespace

// --------------------------------------------------------------------------
// The exported symbol name. One symbol per kernel used to be the design; it made
// every row in a profile read "mori_jit_entry", so dispatch, combine and each
// tuned geometry of both were indistinguishable.
// --------------------------------------------------------------------------

TEST(EntryName, RenderedSourceExportsTheNameTheLoaderAsksFor) {
  // The invariant that replaces "the name is a fixed literal": whatever EntryName
  // says is what the TU defines, because Prepare() passes the same string to
  // hipModuleGetFunction. If these two ever drift, every launch fails with
  // "not found" -- but only on a machine with a GPU, which is late.
  ToolchainOverride guard(FakeToolchain("gfx942", "mlx5"));
  const EpCfg cfg;
  for (const auto& [entry, src] : {std::pair{mori::ops::v2::EpDispatchSpec::EntryName(cfg),
                                             mori::ops::v2::EpDispatchSpec::RenderSource(cfg)},
                                   std::pair{mori::ops::v2::EpCombineSpec::EntryName(cfg),
                                             mori::ops::v2::EpCombineSpec::RenderSource(cfg)}}) {
    EXPECT_NE(src.find(entry + "(EpArgs args)"), std::string::npos)
        << "rendered TU does not define " << entry;
  }
}

TEST(EntryName, DistinguishesWhatAProfileNeedsToTellApart) {
  ToolchainOverride guard(FakeToolchain("gfx942", "mlx5"));
  const EpCfg base;
  const std::string dispatch = mori::ops::v2::EpDispatchSpec::EntryName(base);
  EXPECT_NE(dispatch, mori::ops::v2::EpCombineSpec::EntryName(base));

  // Geometry is the one that matters most in practice: the tuning schedule picks a
  // different (block, warp) per token count, so a single op contributes several
  // kernels to one trace and they are not interchangeable.
  auto renamed = [&](auto mutate, const char* what) {
    EpCfg c = base;
    mutate(c);
    EXPECT_NE(mori::ops::v2::EpDispatchSpec::EntryName(c), dispatch)
        << what << " is not visible in the kernel name";
  };
  renamed([](EpCfg& c) { c.blockNum *= 2; }, "blockNum");
  renamed([](EpCfg& c) { c.warpPerBlock *= 2; }, "warpPerBlock");
  renamed([](EpCfg& c) { c.worldSize = 4; }, "worldSize");
  renamed([](EpCfg& c) { c.hiddenDim = 1024; }, "hiddenDim");
  renamed([](EpCfg& c) { c.numExpertPerToken = 4; }, "numExpertPerToken");
  renamed([](EpCfg& c) { c.dtype = mori::ops::v2::EpDType::Byte8; }, "dtype");
}

TEST(EntryName, IsALegalSymbol) {
  ToolchainOverride guard(FakeToolchain("gfx942", "mlx5"));
  const std::string n = mori::ops::v2::EpDispatchSpec::EntryName(EpCfg{});
  ASSERT_FALSE(n.empty());
  EXPECT_FALSE(std::isdigit(static_cast<unsigned char>(n[0])));
  for (char ch : n) {
    EXPECT_TRUE(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_')
        << "'" << ch << "' cannot appear in a C identifier: " << n;
  }
}

TEST(Nic, ForksTheCacheKey) {
  // Same source, same arch, different NIC -> different digest, not just a
  // different directory. A digest collision here would mean an mlx5 build could
  // serve a bnxt process.
  const std::string none = CacheDirWith("gfx942", "none");
  const std::string mlx5 = CacheDirWith("gfx942", "mlx5");
  const std::string bnxt = CacheDirWith("gfx942", "bnxt");

  EXPECT_NE(fs::path(none).filename(), fs::path(mlx5).filename());
  EXPECT_NE(fs::path(mlx5).filename(), fs::path(bnxt).filename());
}

TEST(Nic, AppearsInTheDirectoryNameForHumans) {
  EXPECT_EQ(fs::path(CacheDirWith("gfx950", "mlx5")).parent_path().filename(), "gfx950_mlx5");
  EXPECT_EQ(fs::path(CacheDirWith("gfx950", "")).parent_path().filename(), "gfx950_none");
}

TEST(Nic, ArchAndNicAreIndependentComponents) {
  // gfx942+mlx5 must not collide with gfx950+mlx5 or with gfx942+bnxt.
  std::set<std::string> digests;
  for (const char* arch : {"gfx942", "gfx950"}) {
    for (const char* nic : {"none", "mlx5", "bnxt"}) {
      digests.insert(fs::path(CacheDirWith(arch, nic)).filename().string());
    }
  }
  EXPECT_EQ(digests.size(), 6u);
}
