// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
//
// CCO Team API — logical rank-subset descriptors used by per-backend sessions
// (especially CcoGda) to address peers without leaking topology details into
// the kernel.
//
// Mirrors NCCL's nccl_device/core.h ncclTeamWorld/Lsa/Rail. Adds a CCO-specific
// CcoTeamCrossNode that returns "all cross-node peers, skipping my node",
// pairing with CCO_GDA_CONNECTION_CROSSNODE for QP allocation savings.
//
// All inline funcs are host+device safe (NCCL_HOST_DEVICE_INLINE pattern).
#pragma once

#include "mori/cco/cco_types.hpp"

namespace mori {
namespace cco {

#if defined(__HIPCC__) || defined(__CUDACC__)
#define CCO_HOST_DEVICE_INLINE __host__ __device__ inline
#else
#define CCO_HOST_DEVICE_INLINE inline
#endif

/* ────────────────────────────────────────────────────────────────────────────
 *  Built-in teams
 * ──────────────────────────────────────────────────────────────────────────── */

// All ranks in the comm.
CCO_HOST_DEVICE_INLINE CcoTeam ccoTeamWorld(CcoDevComm const& c) {
  CcoTeam t;
  t.nRanks = c.worldSize;
  t.rank   = c.rank;
  t.stride = 1;
  return t;
}

// Ranks on the same node as me (LSA = Local Symmetric Access).
CCO_HOST_DEVICE_INLINE CcoTeam ccoTeamLsa(CcoDevComm const& c) {
  CcoTeam t;
  t.nRanks = c.lsaSize;
  t.rank   = c.lsaRank;
  t.stride = 1;
  return t;
}

// All cross-node ranks: world minus my node.
// Layout in team-rank order: [0..myNodeStart) ++ [myNodeStart+lsaSize..worldSize).
// This team is NOT a simple {nRanks, rank, stride} arithmetic progression
// because there's a gap at my node — we represent it with stride=1 and let
// teamRankToWorld handle the gap (see below).
CCO_HOST_DEVICE_INLINE CcoTeam ccoTeamCrossNode(CcoDevComm const& c) {
  CcoTeam t;
  t.nRanks = c.worldSize - c.lsaSize;
  // My index in the cross-node team is undefined (I'm not a member).
  // We set rank = -1 as a sentinel so the standard teamRankToWorld formula
  // is not used directly; ccoCrossNodeTeamRankToWorld() must be used instead.
  t.rank   = -1;
  t.stride = 1;
  return t;
}

// Cross-node ranks with the same NIC rail (same lsaRank index) as me.
// In a homogeneous N-nodes × lsaSize-GPUs/node deployment, the team consists
// of the same-rail rank on each *other* node.
// nRanks = (worldSize / lsaSize) - 1
// rank   = -1 (sentinel, I'm not a member of cross-node-only team)
// stride = lsaSize (but with the same gap issue as CrossNode)
CCO_HOST_DEVICE_INLINE CcoTeam ccoTeamRail(CcoDevComm const& c) {
  CcoTeam t;
  int nNodes = c.worldSize / c.lsaSize;
  t.nRanks = nNodes - 1;
  t.rank   = -1;
  t.stride = c.lsaSize;
  return t;
}

/* ────────────────────────────────────────────────────────────────────────────
 *  Rank conversions
 * ──────────────────────────────────────────────────────────────────────────── */

// Standard team rank → world rank conversion (assumes contiguous subset).
// Valid for World, Lsa, and any user-defined subset team with team.rank>=0.
CCO_HOST_DEVICE_INLINE int ccoTeamRankToWorld(CcoDevComm const& c,
                                               CcoTeam tm,
                                               int teamRank) {
  return c.rank + (teamRank - tm.rank) * tm.stride;
}

// Specialised for the "gappy" CrossNode team: world rank order minus my node.
// teamRank ∈ [0..nRanks) — first myNodeStart entries map directly, the rest
// shift past lsaSize to skip my node.
CCO_HOST_DEVICE_INLINE int ccoCrossNodeTeamRankToWorld(CcoDevComm const& c,
                                                       int teamRank) {
  return teamRank < c.myNodeStart ? teamRank : teamRank + c.lsaSize;
}

// Specialised for the Rail team (gappy, stride = lsaSize):
// teamRank → world rank of "same-rail GPU on the teamRank-th other node".
CCO_HOST_DEVICE_INLINE int ccoRailTeamRankToWorld(CcoDevComm const& c,
                                                  int teamRank) {
  int myNode = c.rank / c.lsaSize;
  int otherNode = (teamRank < myNode) ? teamRank : teamRank + 1;
  return otherNode * c.lsaSize + c.lsaRank;
}

/* ────────────────────────────────────────────────────────────────────────────
 *  teamRankToGdaRank — maps a (team, teamRank) pair to a QP-array index in
 *  CcoIbgdaContext::endpoints, taking the connection type into account.
 *
 *  Index layout depends on gdaConnType:
 *    NONE       : no QPs (callers must not invoke gda.put)
 *    FULL       : index = worldRank      (current Context: not yet enforced,
 *                                          intra-node slots are empty stubs)
 *    CROSSNODE  : index = worldRank      (intra-node slots are stubs anyway,
 *                                          callers must avoid same-node peers)
 *    RAIL       : index = worldRank      (TODO: switch to packed rail layout)
 *
 *  For now CROSSNODE/RAIL/FULL all use world-rank indexing — this matches what
 *  Context::CreateAdditionalEndpoints produces (worldSize × numQpPerPe slots,
 *  empty for P2P-reachable peers). Future commits may switch to packed layouts.
 * ──────────────────────────────────────────────────────────────────────────── */

CCO_HOST_DEVICE_INLINE int ccoTeamRankToGdaRank(CcoDevComm const& c,
                                                 CcoTeam tm,
                                                 int teamRank) {
  // Resolve to world rank first based on the team's nature.
  int worldRank;
  if (tm.rank >= 0) {
    // Standard contiguous team (World / Lsa / user subset).
    worldRank = ccoTeamRankToWorld(c, tm, teamRank);
  } else if (tm.stride == 1) {
    // Gappy team with stride 1 → CrossNode.
    worldRank = ccoCrossNodeTeamRankToWorld(c, teamRank);
  } else {
    // Gappy team with stride > 1 → Rail.
    worldRank = ccoRailTeamRankToWorld(c, teamRank);
  }
  // Currently all connection types use world-rank indexing.
  return worldRank;
}

}  // namespace cco
}  // namespace mori
