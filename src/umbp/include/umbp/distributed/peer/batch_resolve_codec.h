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
#pragma once

// Wire codec for the struct-of-arrays BatchResolveKeysResponse.  Shared by the
// peer service (producer) and the pool client (consumer) so the flattened
// layout is defined in exactly one place and the two sides cannot drift.  It is
// header-only and depends only on the generated proto plus the host-side POD
// types, so it is unit-testable without gRPC / RDMA.

#include <cstdint>
#include <string>
#include <vector>

#include "umbp/distributed/types.h"
#include "umbp_peer.pb.h"

namespace mori::umbp {

// Host-side view of one resolved key, independent of the generated proto.
// Used as the encoder input on the peer side.
struct ResolvedKeyEntry {
  bool found = false;
  TierType tier = TierType::UNKNOWN;
  uint64_t size = 0;
  std::vector<PageLocation> pages;
};

// One key decoded out of the SoA response (page_size and descs are batch-level
// and live on DecodedBatchResolve, not here).
struct DecodedResolveKey {
  bool found = false;
  TierType tier = TierType::UNKNOWN;
  uint64_t size = 0;
  std::vector<PageLocation> pages;
};

// Whole-batch decode result.
struct DecodedBatchResolve {
  uint64_t page_size = 0;
  std::vector<BufferMemoryDescBytes> descs;  // batch-level; empty when omitted
  std::vector<DecodedResolveKey> keys;       // parallel to request.keys
};

// proto <-> host tier.  Header-local (inline) to avoid a second definition of
// the anonymous-namespace helpers already living in the .cpp files.
inline TierType BatchResolveTierFromProto(::umbp::TierType t) {
  switch (t) {
    case ::umbp::TIER_HBM:
      return TierType::HBM;
    case ::umbp::TIER_DRAM:
      return TierType::DRAM;
    case ::umbp::TIER_SSD:
      return TierType::SSD;
    default:
      return TierType::UNKNOWN;
  }
}
inline ::umbp::TierType BatchResolveTierToProto(TierType t) {
  switch (t) {
    case TierType::HBM:
      return ::umbp::TIER_HBM;
    case TierType::DRAM:
      return ::umbp::TIER_DRAM;
    case TierType::SSD:
      return ::umbp::TIER_SSD;
    default:
      return ::umbp::TIER_UNKNOWN;
  }
}

// Encode resolved keys into the SoA response.  `descs` are the batch-level
// deduplicated buffer descriptors (by buffer_index); pass an empty vector to
// omit them (honoring BatchResolveKeysRequest.omit_descs).  Clears `resp`
// first.  Not-found keys contribute a found=false slot and no pages.
inline void EncodeBatchResolveResponse(const std::vector<ResolvedKeyEntry>& keys,
                                       uint64_t page_size,
                                       const std::vector<BufferMemoryDescBytes>& descs,
                                       ::umbp::BatchResolveKeysResponse* resp) {
  resp->Clear();
  resp->set_page_size(page_size);

  for (const auto& d : descs) {
    auto* desc = resp->add_descs();
    desc->set_buffer_index(d.buffer_index);
    desc->set_desc(std::string(d.desc_bytes.begin(), d.desc_bytes.end()));
  }

  size_t total_pages = 0;
  for (const auto& k : keys) total_pages += k.pages.size();
  resp->mutable_found()->Reserve(static_cast<int>(keys.size()));
  resp->mutable_tier()->Reserve(static_cast<int>(keys.size()));
  resp->mutable_size()->Reserve(static_cast<int>(keys.size()));
  resp->mutable_page_count()->Reserve(static_cast<int>(keys.size()));
  resp->mutable_buffer_index()->Reserve(static_cast<int>(total_pages));
  resp->mutable_page_index()->Reserve(static_cast<int>(total_pages));

  for (const auto& k : keys) {
    resp->add_found(k.found);
    resp->add_tier(BatchResolveTierToProto(k.tier));
    resp->add_size(k.size);
    resp->add_page_count(static_cast<uint32_t>(k.pages.size()));
    for (const auto& p : k.pages) {
      resp->add_buffer_index(p.buffer_index);
      resp->add_page_index(p.page_index);
    }
  }
}

// Number of keys carried by the response (== request.keys.size() for a
// well-formed response).
inline int BatchResolveKeyCount(const ::umbp::BatchResolveKeysResponse& resp) {
  return resp.found_size();
}

// Decode the whole response.  Page offsets into the flattened arrays are
// resolved internally.  A malformed response (parallel arrays of unequal
// length, or fewer flattened pages than page_count sums to) yields an empty
// `keys` vector so the caller treats the whole batch as failed rather than
// reading past the arrays.
inline DecodedBatchResolve DecodeBatchResolveResponse(
    const ::umbp::BatchResolveKeysResponse& resp) {
  DecodedBatchResolve out;
  out.page_size = resp.page_size();

  out.descs.reserve(resp.descs_size());
  for (const auto& d : resp.descs()) {
    BufferMemoryDescBytes b;
    b.buffer_index = d.buffer_index();
    b.desc_bytes.assign(d.desc().begin(), d.desc().end());
    out.descs.push_back(std::move(b));
  }

  const int n = resp.found_size();
  // All per-key arrays must agree in length, and the flattened page arrays must
  // hold at least the summed page_count.  Anything else is a malformed peer
  // response; refuse to decode it.
  if (resp.tier_size() != n || resp.size_size() != n || resp.page_count_size() != n ||
      resp.buffer_index_size() != resp.page_index_size()) {
    return out;
  }

  size_t total_pages = 0;
  for (int i = 0; i < n; ++i) total_pages += resp.page_count(i);
  if (total_pages > static_cast<size_t>(resp.buffer_index_size())) {
    return out;
  }

  out.keys.reserve(n);
  size_t cursor = 0;
  for (int i = 0; i < n; ++i) {
    DecodedResolveKey k;
    k.found = resp.found(i);
    k.tier = BatchResolveTierFromProto(resp.tier(i));
    k.size = resp.size(i);
    const uint32_t pc = resp.page_count(i);
    k.pages.reserve(pc);
    for (uint32_t p = 0; p < pc; ++p) {
      PageLocation pl;
      pl.buffer_index = resp.buffer_index(static_cast<int>(cursor));
      pl.page_index = resp.page_index(static_cast<int>(cursor));
      k.pages.push_back(pl);
      ++cursor;
    }
    out.keys.push_back(std::move(k));
  }
  return out;
}

}  // namespace mori::umbp
