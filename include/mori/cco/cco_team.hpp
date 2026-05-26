// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
//
// CCO Team API — logical rank-subset descriptors used by per-backend sessions
// (especially CcoGda) to address peers without leaking topology into kernels.
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
CCO_HOST_DEVICE_INLINE CcoTeam ccoTeamWorld(CcoDevComm const& c) {
  CcoTeam t;
  t.nRanks = c.worldSize;
  t.rank   = c.rank;
  t.stride = 1;
  return t;
}

// Ranks on the same node (LSA = Local Symmetric Access).
CCO_HOST_DEVICE_INLINE CcoTeam ccoTeamLsa(CcoDevComm const& c) {
  CcoTeam t;
  t.nRanks = c.lsaSize;
  t.rank   = c.lsaRank;
  t.stride = 1;
  return t;
}

// Cross-node ranks: world minus my node. Gappy — caller is not a member, so
// rank=-1 is a sentinel; use ccoCrossNodeTeamRankToWorld() for conversion.
CCO_HOST_DEVICE_INLINE CcoTeam ccoTeamCrossNode(CcoDevComm const& c) {
  CcoTeam t;
  t.nRanks = c.worldSize - c.lsaSize;
  t.rank   = -1;
  t.stride = 1;
  return t;
}

// Cross-node ranks sharing my NIC rail (same lsaRank index on each other node).
// Gappy with stride=lsaSize; rank=-1 sentinel as above.
CCO_HOST_DEVICE_INLINE CcoTeam ccoTeamRail(CcoDevComm const& c) {
  CcoTeam t;
  int nNodes = c.worldSize / c.lsaSize;
  t.nRanks = nNodes - 1;
  t.rank   = -1;
  t.stride = c.lsaSize;
  return t;
}

// Standard team rank → world rank for contiguous teams (World / Lsa /
// user subset with team.rank >= 0).
CCO_HOST_DEVICE_INLINE int ccoTeamRankToWorld(CcoDevComm const& c,
                                               CcoTeam tm,
                                               int teamRank) {
  return c.rank + (teamRank - tm.rank) * tm.stride;
}

// CrossNode team: first myNodeStart entries map directly, the rest shift
// past lsaSize to skip my own node.
CCO_HOST_DEVICE_INLINE int ccoCrossNodeTeamRankToWorld(CcoDevComm const& c,
                                                       int teamRank) {
  return teamRank < c.myNodeStart ? teamRank : teamRank + c.lsaSize;
}

// Rail team: teamRank → world rank of same-rail GPU on the teamRank-th
// other node.
CCO_HOST_DEVICE_INLINE int ccoRailTeamRankToWorld(CcoDevComm const& c,
                                                  int teamRank) {
  int myNode = c.rank / c.lsaSize;
  int otherNode = (teamRank < myNode) ? teamRank : teamRank + 1;
  return otherNode * c.lsaSize + c.lsaRank;
}

// Resolve (team, teamRank) → QP-array index in CcoIbgdaContext::endpoints.
// All connection types currently use world-rank indexing; intra-node QP
// slots in CROSSNODE/RAIL modes are empty stubs that callers must avoid.
CCO_HOST_DEVICE_INLINE int ccoTeamRankToGdaRank(CcoDevComm const& c,
                                                 CcoTeam tm,
                                                 int teamRank) {
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
