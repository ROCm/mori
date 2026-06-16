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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mori::umbp {

// One evictable SSD key handed to an SsdEvictStrategy.  The caller
// (PeerSsdManager) filters out keys with an in-flight read or an in-progress
// eviction, so each candidate is eligible at snapshot time; Evict() revalidates
// before any key is actually dropped.
struct SsdEvictCandidate {
  std::string key;
  uint64_t size = 0;
  // Recency rank derived from the manager's LRU list: 0 == oldest (the LRU
  // end), larger == more recently used.  A policy decides how to use it.
  uint64_t lru_rank = 0;
};

// Abstract interface for peer-local SSD victim selection.  Implement this to
// plug in a custom SSD eviction policy.  Mirrors the master-side
// MasterEvictStrategy split: PeerSsdManager owns the tier (capacity, LRU list,
// backend IO, read/evict guards) and delegates only the ranking + budget
// selection here.  Master never evicts SSD; this policy is purely peer-local.
class SsdEvictStrategy {
 public:
  virtual ~SsdEvictStrategy() = default;

  // Pick victims whose total size reaches at least @p bytes_to_free, from the
  // @p candidates (taken by value so an implementation may sort freely).  May
  // return fewer keys if there is not enough free-able data.
  //
  // Candidate-set caveat: PeerSsdManager currently supplies only a
  // recency-ordered prefix — it walks oldest-first and stops once the eligible
  // size covers bytes_to_free, so @p candidates is NOT the full tier.  This is
  // sufficient for LRU / recency-based policies.  A policy that ranks on
  // anything other than recency (e.g. LFU, size) needs the full candidate set
  // and must first lift that early stop in PeerSsdManager::SelectVictims.
  virtual std::vector<std::string> SelectVictims(std::vector<SsdEvictCandidate> candidates,
                                                 size_t bytes_to_free) = 0;
};

// Default policy: pure LRU.  Evict oldest-first (smallest lru_rank) until the
// accumulated size reaches bytes_to_free.
class LruSsdEvictStrategy : public SsdEvictStrategy {
 public:
  std::vector<std::string> SelectVictims(std::vector<SsdEvictCandidate> candidates,
                                         size_t bytes_to_free) override;
};

}  // namespace mori::umbp
