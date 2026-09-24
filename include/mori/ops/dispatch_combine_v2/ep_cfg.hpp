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
// EP intranode dispatch/combine: the specialisation identity, the runtime
// arguments, and the arithmetic both sides share.
//
// Ported from src/ops/dispatch_combine/intranode.hpp, whose symmetric-memory
// surface is `memObj->GetAs<T*>(pe)`. Here that is
// `ccoGetLsaPeerPtr(win, pe, args.offRegion)`: one arena, one window, one offset
// per region, so 13 SymmMemObjPtr fields become a handle plus eight offsets.
//
// HIP-free and attribute-free: host compiles this with a plain C++ compiler,
// the generated device TU with hipcc. See MORI_JIT_V2_DESIGN.md §3.5.

#pragma once

#include <cstddef>
#include <string>

#include "mori/jit/v2/render.hpp"

namespace mori {
namespace ops {
namespace v2 {

// ---------------------------------------------------------------------------
// dtype tag. The generated source expands it to a real type name; nothing here
// includes a HIP header.
//
// The VALUES are load-bearing: an `e`-tagged field crosses as a bare integer and
// the binding has one dtype name->int table (plan_api.DTYPES) for every kernel,
// so this must agree numerically with mori::ops::v2::DType. Renumbering either
// alone is a silent wrong answer, not a refactor.
// ---------------------------------------------------------------------------
// Byte8 is a TRANSPORT type: dispatch only copies its payload, so fp8 and fp4 (2
// e2m1 per byte, caller halves hiddenDim) both move as bytes. Combine reduces and
// cannot use it -- MakeEpCfg rejects it there.
enum class EpDType : int { Bf16 = 0, Fp32 = 1, Byte8 = 2 };

inline const char* EpDTypeName(EpDType d) {
  switch (d) {
    case EpDType::Fp32:
      return "float";
    case EpDType::Byte8:
      return "unsigned char";
    default:
      return "hip_bfloat16";
  }
}

// Identifier-safe short tag, for the kernel symbol name. Separate from
// EpDTypeName, which yields a C++ type -- "unsigned char" has a space in it and
// would not be a legal symbol.
inline const char* EpDTypeTag(EpDType d) {
  switch (d) {
    case EpDType::Fp32:
      return "fp32";
    case EpDType::Byte8:
      return "byte8";
    default:
      return "bf16";
  }
}

constexpr int EpElemSize(EpDType d) {
  return d == EpDType::Fp32 ? 4 : (d == EpDType::Byte8 ? 1 : 2);
}

inline std::string RenderValue(EpDType d) {
  switch (d) {
    case EpDType::Fp32:
      return "::mori::ops::v2::EpDType::Fp32";
    case EpDType::Byte8:
      return "::mori::ops::v2::EpDType::Byte8";
    default:
      return "::mori::ops::v2::EpDType::Bf16";
  }
}

// ---------------------------------------------------------------------------
// Runtime arguments. One struct for both kernels: the union is small and a
// single published schema keeps the binding generic. Pointers that are NOT in
// the arena stay here (they are plain local buffers no peer reads).
// ---------------------------------------------------------------------------
struct EpArgs {
  // The cco window handle (ccoWindow_t). Runtime, not Cfg: the base address is
  // only known once the arena exists, and it differs per rank.
  unsigned long long window = 0;

  // Arena region byte offsets, matching the region list SymmArena builds. Runtime,
  // not Cfg: as constants they made every arena layout a separate binary AND
  // measured slower on gfx942 (VGPR 9 -> 22 -- the compiler loses "the base is
  // uniform" and rematerialises the address per lane).
  unsigned long long offTokOff = 0;     // index_t[1]            slot allocator
  unsigned long long offRecvNum = 0;    // index_t[worldSize]    recv-count signal
  unsigned long long offRecvToSrc = 0;  // index_t[maxRecv]      slot -> src token
  unsigned long long offOutIdx = 0;     // index_t[maxRecv*topk] forwarded expert ids
  unsigned long long offOutWts = 0;     // float[maxRecv*topk]   forwarded weights
  unsigned long long offDispOut = 0;    // T[maxRecv*hidden]     dispatch landing zone
  unsigned long long offOutTok = 0;     // T[maxRecv*hidden]     combine staging
  unsigned long long offXdb = 0;        // uint64[worldSize]     barrier slots
  // uint8[maxRecv*scaleBytes]. Only read when Cfg.scaleBytes > 0; the arena does
  // not carry the region otherwise, so this stays 0 and nothing dereferences it.
  unsigned long long offOutScales = 0;
  // uint8[worldSize*maxTokPerRank * EpCombinePushSlotBytes]. The PUSH combine's
  // landing zone: a sender writes the peer's slot directly, so unlike offOutTok
  // this is indexed by (SOURCE pe, DESTINATION-local token) and needs the full
  // worldSize*maxTokPerRank slots regardless of the recv cap. Only read when
  // Cfg.combinePush is set; the arena omits the region otherwise.
  unsigned long long offCombPush = 0;
  // uint64[worldSize * EpXdbFlagSlots], indexed (SOURCE pe, SOURCE block). The
  // combine entry barrier's MORI_COMB_BARDIRECT path has every block announce
  // itself to every peer, so a peer needs one slot per remote block rather than
  // the one-per-rank offXdb has. A separate region rather than a wider offXdb:
  // dispatch and the portable combine index offXdb by rank and must keep their
  // offsets, and a region appended at the end moves nothing already laid out.
  unsigned long long offXdbBlk = 0;

  // Which LSA rank this is. Runtime for the same reason: as a Cfg field it made
  // all eight ranks compile their own copy of an identical kernel.
  int rank = 0;

  const int* tokenIndices = nullptr;  // [numTokens * topk] expert ids, <0 drops
  const void* inpTokenBuf = nullptr;  // dispatch: source tokens; combine: post-expert tokens
  const float* weightsBuf = nullptr;  // [numTokens * topk]
  const void* scalesBuf = nullptr;    // [numTokens * scaleBytes] per-token scale rows
  void* outTokenBuf = nullptr;        // combine output, local
  float* outWeightsBuf = nullptr;     // combine weight output, local

  int* dispDestTokIdMap = nullptr;        // [numTokens * topk] flat dest index per (token, k)
  int* destPeTokenCounter = nullptr;      // [worldSize] per-dest send count
  int* totalRecvTokenNum = nullptr;       // [1]
  unsigned int* gridBarrier = nullptr;    // [1] intra-kernel grid rendezvous
  unsigned long long* xdbFlag = nullptr;  // [1] monotone cross-device barrier epoch
  int* combineBarrierFan =
      nullptr;  // [blockNum*16] gfx1250 combine intra-grid fan-out (local scratch)

  int numTokens = 0;  // tokens this rank contributes this call
};

// The wire schema, generated from the field list rather than kept parallel to it.
// The binding builds its ctypes struct from `name:tag` in this order and checks
// sizeof -- which cannot see two same-type fields swapped, and 9 of the 26 are
// bare pointers. So the static_asserts below take the offsets in SCHEMA order:
// any disagreement with the declaration order stops the sequence increasing.
#define MORI_EP_ARGS_FIELDS(X) \
  X(window, "u64")             \
  X(offTokOff, "u64")          \
  X(offRecvNum, "u64")         \
  X(offRecvToSrc, "u64")       \
  X(offOutIdx, "u64")          \
  X(offOutWts, "u64")          \
  X(offDispOut, "u64")         \
  X(offOutTok, "u64")          \
  X(offXdb, "u64")             \
  X(offOutScales, "u64")       \
  X(offCombPush, "u64")        \
  X(offXdbBlk, "u64")          \
  X(rank, "i32")               \
  X(tokenIndices, "p")         \
  X(inpTokenBuf, "p")          \
  X(weightsBuf, "p")           \
  X(scalesBuf, "p")            \
  X(outTokenBuf, "p")          \
  X(outWeightsBuf, "p")        \
  X(dispDestTokIdMap, "p")     \
  X(destPeTokenCounter, "p")   \
  X(totalRecvTokenNum, "p")    \
  X(gridBarrier, "p")          \
  X(xdbFlag, "p")              \
  X(combineBarrierFan, "p")    \
  X(numTokens, "i32")

#define MORI_EP_ARGS_SCHEMA_ENTRY(name, tag) #name ":" tag ","
// Trailing comma: the binding skips empty items, and a separator rule that does
// not special-case the last element is one less thing to get wrong.
#define MORI_EP_ARGS_SCHEMA MORI_EP_ARGS_FIELDS(MORI_EP_ARGS_SCHEMA_ENTRY)

namespace detail {

#define MORI_EP_ARGS_OFFSET(name, tag) offsetof(::mori::ops::v2::EpArgs, name),
inline constexpr size_t kEpArgsOffsets[] = {MORI_EP_ARGS_FIELDS(MORI_EP_ARGS_OFFSET)};
#undef MORI_EP_ARGS_OFFSET

constexpr size_t kEpArgsFieldCount = sizeof(kEpArgsOffsets) / sizeof(kEpArgsOffsets[0]);

constexpr bool EpArgsOffsetsAscend() {
  for (size_t i = 1; i < kEpArgsFieldCount; ++i)
    if (kEpArgsOffsets[i] <= kEpArgsOffsets[i - 1]) return false;
  return true;
}

}  // namespace detail

static_assert(detail::kEpArgsFieldCount == 26,
              "added an EpArgs field -- add it to MORI_EP_ARGS_FIELDS in the same position "
              "and bump this count");
static_assert(detail::EpArgsOffsetsAscend(),
              "MORI_EP_ARGS_FIELDS is not in EpArgs declaration order -- the binding would "
              "write each argument into the wrong slot");

// ---------------------------------------------------------------------------
// Cfg. Shared by dispatch and combine: they run over the same arena and the
// same shape, and only the launch geometry differs. Two Specs, one Cfg.
// ---------------------------------------------------------------------------
struct EpCfg {
  // ---- topology / shape ----
  int worldSize = 8;
  int hiddenDim = 7168;
  int maxTokPerRank = 128;  // per-rank input token capacity
  int numExpertPerRank = 8;
  int numExpertPerToken = 8;  // topk
  int maxRecv = 0;            // 0 = worldSize * maxTokPerRank
  EpDType dtype = EpDType::Bf16;

  // ---- launch geometry (host-derived; see MakeEpCfg) ----
  int blockNum = 64;
  int warpPerBlock = 16;
  int waveSize = 64;

  // ---- algorithm ----
  bool useWeights = true;
  // Combine transport. false = gather: every rank stages its post-expert tokens
  // in its own arena and the token's owner reads the topk sources across the
  // fabric. true = push: the rank holding the expert result writes it straight
  // into the owner's slot, and the owner's reduce is then entirely local.
  //
  // Ported from v1 (src/ops/dispatch_combine/intranode_1250x.hpp, the _nop2p
  // symbols): one TDM load stages the token into a per-warp LDS tile, one TDM
  // store lands it in the peer's slot. It exists because push is where a
  // compress-on-write combine can live -- quantizing into the peer's slot makes
  // the quantization BE the transport, which gather cannot express.
  bool combinePush = false;
  // Compress the combine payload on the wire. 0 = off (send the token in its own
  // dtype), 1 = fp8_direct_cast (cast each element to e4m3 on write, widen back
  // on read, no scale). PUSH-only, and not an arbitrary restriction: the cast has
  // to happen where a rank writes a whole token it owns, and gather has no such
  // writer -- the owner reads its topk sources and never holds one alone.
  //
  // Names and numbering follow the op layer's quant_type, which the FlyDSL
  // backend already implements; 2 (fp8_blockwise, a per-128-element scale) is
  // deliberately left unassigned here rather than reused for something else.
  int combineQuant = 0;
  // Per-token scale row carried alongside the payload, in bytes. 0 = off, and off
  // is free: Render omits default-valued fields, so the Cfg text -- which IS the
  // JIT cache key -- is byte-identical to a build without this feature.
  int scaleBytes = 0;
};

template <typename Self, typename Visit>
inline void VisitFields(Self& c, const EpCfg& d, Visit&& v) {
#define MORI_FIELD(x) v(#x, c.x, d.x)
  MORI_FIELD(worldSize);
  MORI_FIELD(hiddenDim);
  MORI_FIELD(maxTokPerRank);
  MORI_FIELD(numExpertPerRank);
  MORI_FIELD(numExpertPerToken);
  MORI_FIELD(maxRecv);
  MORI_FIELD(dtype);
  MORI_FIELD(blockNum);
  MORI_FIELD(warpPerBlock);
  MORI_FIELD(waveSize);
  MORI_FIELD(useWeights);
  MORI_FIELD(combinePush);
  MORI_FIELD(combineQuant);
  MORI_FIELD(scaleBytes);
#undef MORI_FIELD
}

MORI_JIT_ASSERT_FIELD_COUNT(EpCfg, 14, "added an EpCfg field -- update VisitFields(EpCfg) too");

inline std::string Render(const EpCfg& c) {
  const EpCfg d{};
  mori::jit::v2::Fields f;
  VisitFields(c, d, [&f](const char* name, const auto& value, const auto& dflt) {
    f.Put(name, value, dflt);
  });
  return mori::jit::v2::BraceInit("EpCfg", f);
}

// Every field, default or not -- what the plan reports back through `info`.
inline std::string Describe(const EpCfg& c) {
  std::string out;
  VisitFields(c, c, [&out](const char* name, const auto& value, const auto&) {
    using mori::jit::v2::RenderValue;  // ADL for the Ep types, jit's for scalars
    out += name;
    out += "=";
    out += RenderValue(value);
    out += "\n";
  });
  return out;
}

// ---------------------------------------------------------------------------
// Wire schema + generic apply, driven by the same VisitFields walk Render uses.
// Ep-prefixed rather than overloading combine's EmitSchema / ApplyFields: those
// share this namespace, so an unqualified call would find them by ADL.
// ---------------------------------------------------------------------------
template <typename T>
inline void EpEmitSchema(mori::jit::v2::SchemaBuilder& sb, const std::string& name, const T& v) {
  using mori::jit::v2::WireTag;
  using mori::jit::v2::WireValue;
  sb.Add(name, WireTag(v), WireValue(v));
}
// Apply named request values onto a struct. EpRequest is all scalars: one flat walk.
template <typename T, typename Has, typename Get>
inline void EpApplyFields(T& dst, const std::string& prefix, const Has& has, const Get& get) {
  using mori::jit::v2::WireAssign;
  const T defaults{};
  VisitFields(dst, defaults, [&](const char* n, auto& slot, const auto&) {
    const std::string key = prefix.empty() ? std::string(n) : prefix + "." + n;
    if (has(key)) WireAssign(slot, get(key));
  });
}

// ---------------------------------------------------------------------------
// Shared arithmetic. One definition, used by the host to size the launch and by
// the device as a compile-time constant. Attribute-free constexpr on purpose:
// __host__ __device__ here would drag every host TU through hipcc.
// ---------------------------------------------------------------------------
constexpr int EpBlockThreads(const EpCfg& c) { return c.warpPerBlock * c.waveSize; }

// True when worldSize exceeds a single wavefront. The per-peer loops in
// dispatch Phase 2 and the XDB barrier then iterate more than once per lane
// and need the multi-iteration-safe code path. Compile-time via the Cfg NTTP.
constexpr bool EpIsWideEp(const EpCfg& c) { return c.worldSize > c.waveSize; }

// Recv-slot capacity. The flat token index encodes (pe, localTokId) with this
// stride, so host and device must agree exactly.
constexpr int EpMaxRecv(const EpCfg& c) {
  return c.maxRecv > 0 ? c.maxRecv : c.worldSize * c.maxTokPerRank;
}

// Combine's shared memory: one pointer array per warp for the topk sources,
// plus a second one for the weight pointers when weights are enabled.
constexpr int EpCombineSharedBytes(const EpCfg& c) {
  return static_cast<int>(sizeof(void*)) * c.warpPerBlock * c.numExpertPerToken *
         (c.useWeights ? 2 : 1);
}

constexpr int EpTokenBytes(const EpCfg& c) { return c.hiddenDim * EpElemSize(c.dtype); }

// Element width the combine payload travels in, which is the token dtype's width
// unless combineQuant compresses it. Kept separate from EpElemSize(c.dtype)
// because the two now differ: the LDS tile, the reduce output and the caller's
// buffers stay in the token dtype, and only the slot and the transfer shrink.
constexpr int EpCombineWireElemSize(const EpCfg& c) {
  return c.combineQuant != 0 ? 1 : EpElemSize(c.dtype);
}
// fp4 (combineQuant==3) groups this many elements under one e8m0 scale byte. Must
// match the kernel's encode group kSclG = 8 (ep_intranode_1250x.hpp _cQuantTile4)
// and the decode step, or the scale row would be read with the wrong stride. Set to
// 8 == the native decode intrinsic's pk8 granularity: encode's amax is then per-lane
// (no cross-wave reduce) and the group divides any pipelined chunk, so fp4 can take
// the _cPipe path. (void)c keeps the signature uniform with the other helpers.
constexpr int EpCombineFp4Group(const EpCfg& c) { return (void)c, 8; }
// fp4 wire: hidden/2 payload (e2m1, two per byte) + one e8m0 scale byte per group,
// the scale row padded to 16 B so the topk float weights behind it stay aligned.
constexpr int EpCombineFp4ScaleBytes(const EpCfg& c) {
  const int raw = (c.hiddenDim + EpCombineFp4Group(c) - 1) / EpCombineFp4Group(c);
  return (raw + 15) / 16 * 16;
}
constexpr int EpCombineWireBytes(const EpCfg& c) {
  if (c.combineQuant == 3) return c.hiddenDim / 2 + EpCombineFp4ScaleBytes(c);
  return c.hiddenDim * EpCombineWireElemSize(c);
}

// PUSH slot stride. The weights ride in the slot behind the payload rather than
// being read from offOutWts: the reduce walks the slots by SOURCE pe, while
// offOutWts is indexed by recv slot, and nothing local maps one to the other.
// v1 does the same (intranode_1250x.hpp, combXferBytes).
//
// Padded to 128 B because a TDM store's destination has to start on a 128 B row:
// hidden 7168 bf16 + 8 weights is 14368 B, only 32 B-aligned, so every other slot
// would land off-row. v1 has to fall back to a lane copy when its slot budget
// cannot absorb the pad (combSlotOn128B); here the region is sized from this
// function, so the pad is always affordable and that fallback has no reason to
// exist.
constexpr int EpCombinePushSlotAlign = 128;
constexpr int EpCombinePushSlotBytes(const EpCfg& c) {
  const int packed = EpCombineWireBytes(c) + (c.useWeights ? c.numExpertPerToken * 4 : 0);
  return (packed + EpCombinePushSlotAlign - 1) / EpCombinePushSlotAlign * EpCombinePushSlotAlign;
}
// Slots in the PUSH region: one per (source pe, destination-local token). NOT
// EpMaxRecv -- that is the recv-slot space, and a push slot is addressed by the
// DESTINATION's token id, whose bound is maxTokPerRank.
constexpr long long EpCombinePushSlots(const EpCfg& c) {
  return (long long)c.worldSize * c.maxTokPerRank;
}
constexpr long long EpCombinePushBytes(const EpCfg& c) {
  return c.combinePush ? EpCombinePushSlots(c) * EpCombinePushSlotBytes(c) : 0;
}

// The scale row's SLOT stride: the caller's row padded to 128 B. A transfer is a
// run of consecutive slots, and TdmWholeOrSplit128 only gives a body to the part
// of a run that starts aligned -- at the natural 224 B only every 4th one does.
// gfx1250, 512 tok/rank, hidden 7168, EP4: dispatch 157.7us at 224 B, 94.8 at 256.
//
// Padded on every arch so the destination layout does not fork per arch. The pad
// is unconditional, so small rows amplify: 224->256 is 1.14x, 32->128 is 4x,
// 4->128 is 32x, and the gfx1250 staging pool is worldSize^2 * maxTok * THIS per
// compiled variant. Bounded by the static_assert there, not by this being cheap.
//
// Everything reading the destination uses this, not Cfg.scaleBytes; it assumes
// the arena aligns the region too (SymmArena._ALIGN, checked in hip_backend).
constexpr int EpScaleAlign = 128;
constexpr int EpScaleStride(const EpCfg& c) {
  return c.scaleBytes <= 0 ? 0 : (c.scaleBytes + EpScaleAlign - 1) / EpScaleAlign * EpScaleAlign;
}

// gfx1250 launch LDS. Dispatch stages one token tile per warp through the TDM
// engine; combine reserves the whole budget and sizes its tiles at runtime.
// EpCombine1250xLdsBudget must match MORI_COMB_LDS_BUDGET in ep_intranode_1250x.hpp.
//
// Ep1250xLdsBytes is the physical ceiling; combine's budget happens to be all of
// it. Anything else that needs the ceiling should say so directly rather than
// reach for combine's number -- otherwise retuning combine silently resizes it.
constexpr int Ep1250xLdsBytes = 327680;
constexpr int EpCombine1250xLdsBudget = Ep1250xLdsBytes;
#ifndef MORI_COMB_QPIPE
#define MORI_COMB_QPIPE 4
#endif
// Whether the push send phase gets its TDM tile: one whole token per warp, laid
// after the pointer arrays and the round-robin bookkeeping. gfx1250 combine is
// launched with the ENTIRE budget as dynamic shared memory, so the scheduler's
// bookkeeping has to come out of it too -- the body can declare no static
// __shared__ at all.
//
// When this is false the send falls back to a lane copy. That is correct but is
// the slow path TDM exists to avoid, so hip_backend rejects such a config rather
// than shipping a silent cliff: a config that fell back would look like "push is
// slow here" and nothing would say why.
constexpr int EpCombinePushRRTile = 512;
constexpr long long EpCombinePushLdsNeed(const EpCfg& c) {
  const long long ptr = ((((long long)(1 + (c.useWeights ? 1 : 0)) * c.warpPerBlock *
                           c.numExpertPerToken * (long long)sizeof(void*)) +
                          127) /
                         128) *
                        128;
  // Two arrays of EpCombinePushRRTile: the slot ids, and the recvToSrc entry the
  // grouping pass read for each (MORI_COMB_IDXLDS in the 1250x body). The second
  // is counted even when that gate is off, because this runs on the host before
  // the device build exists and must not size the arena under what the kernel
  // may address.
  const long long rr = (2LL * (long long)EpCombinePushRRTile + 4LL * c.worldSize) * 4;
  // The first tile holds two pipeline chunks, so its size follows the compiled
  // MORI_COMB_QPIPE value. Without quant or chunking it still holds one whole
  // token. Under quant there is also a narrowed tile
  // plus its appended weights: converting in place needs a wave_barrier per pass to keep the
  // reads ahead of the writes, and that barrier measured 10.1us of the send phase
  // at ct=4096 (MORI_COMB_QSKIP intervention). Counted unconditionally whenever
  // combineQuant is on, even though MORI_COMB_QOOP=0 would not use it -- this is
  // the gate that decides whether the TDM path is available at all, and it must
  // not come out smaller than what the kernel might allocate.
  const long long load =
      c.combineQuant && MORI_COMB_QPIPE >= 2
          ? (2LL * (long long)EpTokenBytes(c)) / MORI_COMB_QPIPE
          : (long long)EpTokenBytes(c);
  const long long wire = c.combineQuant ? (long long)EpCombinePushSlotBytes(c) : 0;
  return ((ptr + rr + 127) / 128) * 128 +
         (long long)c.warpPerBlock * (load + wire);
}
constexpr bool EpCombinePushTdmFits(const EpCfg& c) {
  return EpCombinePushLdsNeed(c) <= EpCombine1250xLdsBudget;
}

// xdbFlag slots for the per-block combine entry barrier: one uint64 per block, so a
// block is the only writer of its own epoch. 256 == the CU count, which caps the
// combine block_num; the host allocates this many and every call keeps them in step.
constexpr int EpXdbFlagSlots = 256;

// Per-warp LDS slab. The metadata tile and the payload tile share it (same
// address, different phases), so its size bounds BOTH -- and the metadata batch
// size is (slab - headroom) / bytes-per-token.
//
// Sizing it off the payload dtype alone is fine until a scale row joins the
// metadata: an fp8 payload halves the slab exactly when per-token metadata grows
// from 52 to 276 bytes, and the batch collapses ~11x.
//
// The floor is EMPIRICAL, and from one shape: hidden 7168, topk 6, EP4, 512
// tokens/rank on gfx1250, where it moved dispatch from 93.1us back to 36.7us.
// The mechanism (a narrower slab shrinks the metadata batch) is general; the
// conclusion that the bf16 width is the right target is not necessarily so for a
// very different hidden size, and this is the first thing to re-measure if one
// shows up.
//
// Capped by the physical LDS rather than by combine's budget: the two are the
// same number today, but for a shared physical reason, not because dispatch
// follows combine.
//
// The slab also bounds how many destination slots one payload store may carry.
// The send phase runs worldSize warps as one unit -- warp w sends only peer
// (w % worldSize) -- so the slots one warp owns are consecutive and a single
// store can cover several of them. Four is the cap; the LDS below is what
// actually binds it, and at hidden 7168 only an fp4 payload (hiddenDim already
// halved by the caller, 3584 B a token) leaves room for four.
constexpr int EpDispatch1250xMaxPack = 4;
constexpr int EpDispatch1250xSlabBytes(const EpCfg& c) {
  const int payload = c.hiddenDim * EpElemSize(c.dtype);
  const int wide = c.hiddenDim * EpElemSize(EpDType::Bf16);
  const long long wideTotal = (long long)wide * c.warpPerBlock;
  const int meta = (c.scaleBytes > 0 && wide > payload && wideTotal <= Ep1250xLdsBytes)
                       ? wide
                       : payload;
  // Whole tokens only, and never past the physical LDS: every warp in the block
  // gets a slab, so the block's total is warpPerBlock times this. A kernel that
  // packed more tokens than the host reserved would read its tile's tail back as
  // zero instead of faulting, which is why the count is derived here and not
  // chosen independently on the device.
  const long long perWarp = (long long)c.warpPerBlock * payload;
  const int fits = (perWarp > 0) ? (int)(Ep1250xLdsBytes / perWarp) : 1;
  const int pack =
      fits < 1 ? 1 : (fits > EpDispatch1250xMaxPack ? EpDispatch1250xMaxPack : fits);
  const int packed = pack * payload;
  return packed > meta ? packed : meta;
}
constexpr int EpDispatch1250xLdsBytes(const EpCfg& c) {
  return c.warpPerBlock * EpDispatch1250xSlabBytes(c);
}

// A Cfg that cannot launch is a host-side error, not a kernel that misbehaves.
// rank is not checked: it is a launch argument, so the op layer owns that bound.
constexpr bool EpCfgIsValid(const EpCfg& c) {
  return c.worldSize > 0 && c.hiddenDim > 0 && c.maxTokPerRank > 0 && c.numExpertPerToken > 0 &&
         c.numExpertPerRank > 0 && c.blockNum > 0 && (c.waveSize == 32 || c.waveSize == 64) &&
         c.warpPerBlock > 0 && EpBlockThreads(c) <= 1024 &&
         // The recv capacity must cover the worst case: the device slot counter is
         // unbounded, and since EpMaxRecv is also the flat-index stride an overflow
         // re-encodes to the next peer and combine folds in a stranger's token.
         // Token dropping is not implemented, so reject the cap at construction.
         EpMaxRecv(c) >= c.worldSize * c.maxTokPerRank &&
         // The dedup ballot assumes one lane per top-k slot within a wavefront.
         // worldSize may exceed waveSize (wide EP) but must fit within one
         // block so the combine XDB barrier's thdId < npes poll covers all peers.
         c.numExpertPerToken < c.waveSize && c.worldSize <= EpBlockThreads(c) &&
         // WarpCopy moves whole 16 B chunks.
         (EpTokenBytes(c) % 16) == 0 &&
         // Scale rows are copied as dwords, so the row must be dword-sized.
         c.scaleBytes >= 0 && (c.scaleBytes % 4) == 0;
}

}  // namespace v2
}  // namespace ops
}  // namespace mori
