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

// test: cco team descriptors + team-rank→world-rank mappings (pure host).
//
// ccoTeamWorld / ccoTeamLsa / ccoTeamCrossNode / ccoTeamRail and the three
// conversion helpers are CCO_HOST_DEVICE_INLINE plain arithmetic over a
// ccoDevComm's {rank, worldSize, lsaSize, lsaRank, myNodeStart}. No GPU / RDMA /
// MPI is involved, so we synthesize ccoDevComm topologies on the host and assert
// the returned {nRanks, rank, stride} and the world-rank mappings directly —
// including the degenerate single-node case where Rail/CrossNode are empty.

#include <cstdio>

#include "hip/hip_runtime.h"
#include "mori/cco/cco.hpp"

using namespace mori::cco;

// The team helpers under test live inside cco.hpp's device-side block
// (#if defined(__HIPCC__) ...), so this TU must be compiled as HIP for them to
// be visible. The CMake harness (add_cco_test) selects HIP iff the source
// contains "__global__"; this otherwise-unused kernel forces that path. The
// tests themselves run entirely on the host — no kernel launch needed.
__global__ void cco_team_force_hip_compile_dummy() {}

static int g_fails = 0;

#define CHECK_EQ(actual, expected, msg)                                                   \
  do {                                                                                    \
    long _a = (long)(actual), _e = (long)(expected);                                      \
    if (_a != _e) {                                                                       \
      std::printf("  FAIL: %s — got %ld, expected %ld  (%s:%d)\n", msg, _a, _e, __FILE__, \
                  __LINE__);                                                              \
      g_fails++;                                                                          \
    }                                                                                     \
  } while (0)

// Build a synthetic ccoDevComm for a uniform layout: `nNodes` nodes of
// `lsaSize` GPUs each, for the GPU at (node, local).
static ccoDevComm makeComm(int nNodes, int lsaSize, int node, int local) {
  ccoDevComm c{};  // zero-init; only the topology fields matter here
  c.worldSize = nNodes * lsaSize;
  c.lsaSize = lsaSize;
  c.rank = node * lsaSize + local;
  c.lsaRank = local;
  c.myNodeStart = node * lsaSize;
  return c;
}

// ── team descriptor shapes ────────────────────────────────────────────────────

static void test_team_shapes_multinode() {
  // 4 nodes × 2 GPUs = 8 ranks; observer = node 1, local 0 → world rank 2.
  ccoDevComm c = makeComm(/*nNodes=*/4, /*lsaSize=*/2, /*node=*/1, /*local=*/0);

  ccoTeam w = ccoTeamWorld(c);
  CHECK_EQ(w.nRanks, 8, "World.nRanks");
  CHECK_EQ(w.rank, 2, "World.rank");
  CHECK_EQ(w.stride, 1, "World.stride");

  ccoTeam l = ccoTeamLsa(c);
  CHECK_EQ(l.nRanks, 2, "Lsa.nRanks");
  CHECK_EQ(l.rank, 0, "Lsa.rank");
  CHECK_EQ(l.stride, 1, "Lsa.stride");

  ccoTeam x = ccoTeamCrossNode(c);
  CHECK_EQ(x.nRanks, 8 - 2, "CrossNode.nRanks");  // world - my node
  CHECK_EQ(x.rank, -1, "CrossNode.rank(sentinel)");
  CHECK_EQ(x.stride, 1, "CrossNode.stride");

  ccoTeam r = ccoTeamRail(c);
  CHECK_EQ(r.nRanks, 4 - 1, "Rail.nRanks");  // nNodes - 1
  CHECK_EQ(r.rank, -1, "Rail.rank(sentinel)");
  CHECK_EQ(r.stride, 2, "Rail.stride == lsaSize");
}

// ── degenerate single-node case (lsaSize == worldSize, nNodes == 1) ───────────
// This is the common single-host deployment (e.g. 8 GPUs on one box). Rail and
// CrossNode must collapse to EMPTY teams (nRanks == 0); callers gate on that.
static void test_team_shapes_singlenode() {
  ccoDevComm c = makeComm(/*nNodes=*/1, /*lsaSize=*/8, /*node=*/0, /*local=*/3);

  ccoTeam w = ccoTeamWorld(c);
  CHECK_EQ(w.nRanks, 8, "1node World.nRanks");
  CHECK_EQ(w.rank, 3, "1node World.rank");

  ccoTeam l = ccoTeamLsa(c);
  CHECK_EQ(l.nRanks, 8, "1node Lsa.nRanks == worldSize");
  CHECK_EQ(l.rank, 3, "1node Lsa.rank");

  ccoTeam x = ccoTeamCrossNode(c);
  CHECK_EQ(x.nRanks, 0, "1node CrossNode.nRanks == 0 (empty)");

  ccoTeam r = ccoTeamRail(c);
  CHECK_EQ(r.nRanks, 0, "1node Rail.nRanks == 0 (empty)");
}

// ── worldSize == 1 (single rank, no peers) ────────────────────────────────────
static void test_team_shapes_single_rank() {
  ccoDevComm c = makeComm(/*nNodes=*/1, /*lsaSize=*/1, /*node=*/0, /*local=*/0);
  ccoTeam w = ccoTeamWorld(c);
  CHECK_EQ(w.nRanks, 1, "1rank World.nRanks");
  CHECK_EQ(w.rank, 0, "1rank World.rank");
  ccoTeam l = ccoTeamLsa(c);
  CHECK_EQ(l.nRanks, 1, "1rank Lsa.nRanks");
  ccoTeam x = ccoTeamCrossNode(c);
  CHECK_EQ(x.nRanks, 0, "1rank CrossNode.nRanks == 0");
  ccoTeam r = ccoTeamRail(c);
  CHECK_EQ(r.nRanks, 0, "1rank Rail.nRanks == 0");  // nNodes(1) - 1 == 0
}

// ── contiguous mapping: World / Lsa via ccoTeamRankToWorld ────────────────────
static void test_contiguous_mapping() {
  // 4 nodes × 2; observer world rank 2 (node 1, local 0).
  ccoDevComm c = makeComm(4, 2, 1, 0);

  // World team: teamRank == worldRank, so identity.
  ccoTeam w = ccoTeamWorld(c);
  for (int tr = 0; tr < w.nRanks; tr++)
    CHECK_EQ(ccoTeamRankToWorld(c, w, tr), tr, "World teamRank->world identity");

  // Lsa team: ranks on my node are [myNodeStart .. +lsaSize).
  ccoTeam l = ccoTeamLsa(c);
  CHECK_EQ(ccoTeamRankToWorld(c, l, 0), 2, "Lsa tr0 -> myNodeStart");
  CHECK_EQ(ccoTeamRankToWorld(c, l, 1), 3, "Lsa tr1 -> myNodeStart+1");
}

// ── CrossNode mapping: ccoCrossNodeTeamRankToWorld skips my own node ──────────
static void test_crossnode_mapping() {
  // 4 nodes × 2 = 8 ranks; observer node 1 (world rank 2/3), myNodeStart = 2.
  // CrossNode members (world ranks) = {0,1, 4,5, 6,7}; team ranks 0..5 map to
  // those in order, skipping my node's [2,3].
  ccoDevComm c = makeComm(4, 2, 1, 0);
  const int expected[6] = {0, 1, 4, 5, 6, 7};
  for (int tr = 0; tr < 6; tr++)
    CHECK_EQ(ccoCrossNodeTeamRankToWorld(c, tr), expected[tr], "CrossNode tr->world");
}

// ── Rail mapping: same lsaRank on each OTHER node ─────────────────────────────
static void test_rail_mapping() {
  // 4 nodes × 2; observer node 1, local 1 → world rank 3, lsaRank 1.
  // Rail = same-rail (lsaRank==1) GPU on the 3 other nodes {0,2,3}:
  //   node0 -> 0*2+1 = 1, node2 -> 2*2+1 = 5, node3 -> 3*2+1 = 7
  // teamRank 0..2 maps over other nodes in order (skip my node 1).
  ccoDevComm c = makeComm(4, 2, 1, 1);
  CHECK_EQ(ccoRailTeamRankToWorld(c, 0), 1, "Rail tr0 -> node0 same rail");
  CHECK_EQ(ccoRailTeamRankToWorld(c, 1), 5, "Rail tr1 -> node2 same rail");
  CHECK_EQ(ccoRailTeamRankToWorld(c, 2), 7, "Rail tr2 -> node3 same rail");

  // observer at local 0 on node 0 → rail peers are lsaRank 0 on nodes {1,2,3}.
  ccoDevComm c0 = makeComm(4, 2, 0, 0);
  CHECK_EQ(ccoRailTeamRankToWorld(c0, 0), 2, "Rail(node0) tr0 -> node1");
  CHECK_EQ(ccoRailTeamRankToWorld(c0, 1), 4, "Rail(node0) tr1 -> node2");
  CHECK_EQ(ccoRailTeamRankToWorld(c0, 2), 6, "Rail(node0) tr2 -> node3");
}

int main() {
  std::printf("=== CCO team descriptor / mapping UT (host-only) ===\n");

  struct {
    const char* name;
    void (*fn)();
  } cases[] = {
      {"shapes_multinode", test_team_shapes_multinode},
      {"shapes_singlenode", test_team_shapes_singlenode},
      {"shapes_single_rank", test_team_shapes_single_rank},
      {"contiguous_map", test_contiguous_mapping},
      {"crossnode_map", test_crossnode_mapping},
      {"rail_map", test_rail_mapping},
  };

  for (auto& c : cases) {
    int before = g_fails;
    c.fn();
    std::printf("  [%-18s] %s\n", c.name, g_fails == before ? "PASS" : "FAIL");
  }

  const int n = (int)(sizeof(cases) / sizeof(cases[0]));
  std::printf("=== %s (%d failures) ===\n", g_fails == 0 ? "PASSED" : "FAILED", g_fails);
  return g_fails == 0 ? 0 : 1;
}
