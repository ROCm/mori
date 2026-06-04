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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
//
// CCO Team API — logical rank-subset descriptors used by per-backend sessions
// (especially ccoGda) to address peers without leaking topology into kernels.
#pragma once

#include "mori/cco/cco_types.hpp"

namespace mori {
namespace cco {

#if defined(__HIPCC__) || defined(__CUDACC__)
#define CCO_HOST_DEVICE_INLINE __host__ __device__ inline
#else
#define CCO_HOST_DEVICE_INLINE inline
#endif

// All ranks in the comm.
CCO_HOST_DEVICE_INLINE ccoTeam ccoTeamWorld(ccoDevComm const& c) {
  ccoTeam t;
  t.nRanks = c.worldSize;
  t.rank = c.rank;
  t.stride = 1;
  return t;
}

// Ranks on the same node (LSA = Local Symmetric Access).
CCO_HOST_DEVICE_INLINE ccoTeam ccoTeamLsa(ccoDevComm const& c) {
  ccoTeam t;
  t.nRanks = c.lsaSize;
  t.rank = c.lsaRank;
  t.stride = 1;
  return t;
}

// Cross-node ranks: world minus my node. Gappy — caller is not a member, so
// rank=-1 is a sentinel; use ccoCrossNodeTeamRankToWorld() for conversion.
CCO_HOST_DEVICE_INLINE ccoTeam ccoTeamCrossNode(ccoDevComm const& c) {
  ccoTeam t;
  t.nRanks = c.worldSize - c.lsaSize;
  t.rank = -1;
  t.stride = 1;
  return t;
}

// Cross-node ranks sharing my NIC rail (same lsaRank index on each other node).
// Gappy with stride=lsaSize; rank=-1 sentinel as above.
CCO_HOST_DEVICE_INLINE ccoTeam ccoTeamRail(ccoDevComm const& c) {
  ccoTeam t;
  int nNodes = c.worldSize / c.lsaSize;
  t.nRanks = nNodes - 1;
  t.rank = -1;
  t.stride = c.lsaSize;
  return t;
}

// Standard team rank → world rank for contiguous teams (World / Lsa /
// user subset with team.rank >= 0).
CCO_HOST_DEVICE_INLINE int ccoTeamRankToWorld(ccoDevComm const& c, ccoTeam tm, int teamRank) {
  return c.rank + (teamRank - tm.rank) * tm.stride;
}

// CrossNode team: first myNodeStart entries map directly, the rest shift
// past lsaSize to skip my own node.
CCO_HOST_DEVICE_INLINE int ccoCrossNodeTeamRankToWorld(ccoDevComm const& c, int teamRank) {
  return teamRank < c.myNodeStart ? teamRank : teamRank + c.lsaSize;
}

// Rail team: teamRank → world rank of same-rail GPU on the teamRank-th
// other node.
CCO_HOST_DEVICE_INLINE int ccoRailTeamRankToWorld(ccoDevComm const& c, int teamRank) {
  int myNode = c.rank / c.lsaSize;
  int otherNode = (teamRank < myNode) ? teamRank : teamRank + 1;
  return otherNode * c.lsaSize + c.lsaRank;
}

// Resolve (team, teamRank) → QP-array index in ccoIbgdaContext::endpoints.
// All connection types currently use world-rank indexing; intra-node QP
// slots in CROSSNODE/RAIL modes are empty stubs that callers must avoid.
CCO_HOST_DEVICE_INLINE int ccoTeamRankToGdaRank(ccoDevComm const& c, ccoTeam tm, int teamRank) {
  int worldRank;
  if (tm.rank >= 0) {
    worldRank = ccoTeamRankToWorld(c, tm, teamRank);
  } else if (tm.stride == 1) {
    worldRank = ccoCrossNodeTeamRankToWorld(c, teamRank);
  } else {
    worldRank = ccoRailTeamRankToWorld(c, teamRank);
  }
  return worldRank;
}

}  // namespace cco
}  // namespace mori
