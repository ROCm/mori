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

// GMock mock for IMasterMetadataStore.
//
// Phase 1 use: instantiation gate. If this type compiles and instantiates,
// every pure-virtual on the interface is overridden with a well-typed
// signature — proving the contract has no orphaned/ill-typed methods.
//
// Reused in Phase 3 (consumer-integration) to assert that each rewired
// consumer (Router / EvictionManager / UMBPMasterServiceImpl handlers) calls
// the right store method with correctly-translated arguments.
#pragma once

#include <gmock/gmock.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/master/master_metadata_store.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class MockMasterMetadataStore : public IMasterMetadataStore {
 public:
  // Aliases for types whose commas would otherwise break the MOCK_METHOD macro
  // parser (it splits the argument list on top-level commas).
  using CapsMap = std::map<TierType, TierCapacity>;
  using BudgetMap = std::map<NodeTierKey, uint64_t>;
  using LruResult = std::map<NodeTierKey, std::vector<EvictionCandidate>>;
  using LocationBatch = std::vector<std::vector<Location>>;

  // --- Cross-store writes ---
  MOCK_METHOD(bool, RegisterClient,
              (const ClientRegistration& registration, std::chrono::system_clock::time_point now,
               std::chrono::system_clock::duration stale_after),
              (override));
  MOCK_METHOD(void, UnregisterClient, (const std::string& node_id), (override));
  MOCK_METHOD(HeartbeatResult, ApplyHeartbeat,
              (const std::string& node_id, uint64_t seq, std::chrono::system_clock::time_point now,
               const CapsMap& caps, (const std::vector<KvEvent>&)events, bool is_full_sync),
              (override));
  MOCK_METHOD(std::vector<std::string>, ExpireStaleClients,
              (std::chrono::system_clock::time_point cutoff), (override));

  // --- External-KV writes ---
  MOCK_METHOD(bool, RegisterExternalKvIfAlive,
              (const std::string& node_id, (const std::vector<std::string>&)hashes, TierType tier),
              (override));
  MOCK_METHOD(void, UnregisterExternalKv,
              (const std::string& node_id, (const std::vector<std::string>&)hashes, TierType tier),
              (override));
  MOCK_METHOD(void, UnregisterExternalKvByTier, (const std::string& node_id, TierType tier),
              (override));
  MOCK_METHOD(std::size_t, GarbageCollectHits, (std::chrono::system_clock::time_point cutoff),
              (override));

  // --- Block reads ---
  MOCK_METHOD(std::vector<Location>, LookupBlock, (const std::string& key), (const, override));
  MOCK_METHOD(std::vector<Location>, LookupBlockForRouteGet,
              (const std::string& key, (const std::unordered_set<std::string>&)exclude_nodes,
               std::chrono::system_clock::time_point now,
               std::chrono::system_clock::duration lease_duration),
              (override));
  MOCK_METHOD(LocationBatch, BatchLookupBlockForRouteGet,
              ((const std::vector<std::string>&)keys,
               (const std::unordered_set<std::string>&)exclude_nodes,
               std::chrono::system_clock::time_point now,
               std::chrono::system_clock::duration lease_duration),
              (override));
  MOCK_METHOD(std::vector<bool>, BatchExistsBlock, ((const std::vector<std::string>&)keys),
              (const, override));
  MOCK_METHOD(LruResult, EnumerateLruForEviction,
              (const BudgetMap& bytes_to_free, std::chrono::system_clock::time_point now),
              (const, override));

  // --- Client reads ---
  MOCK_METHOD(std::optional<ClientRecord>, GetClient, (const std::string& node_id),
              (const, override));
  MOCK_METHOD(bool, IsClientAlive, (const std::string& node_id), (const, override));
  MOCK_METHOD(std::optional<std::string>, GetPeerAddress, (const std::string& node_id),
              (const, override));
  MOCK_METHOD(std::vector<ClientRecord>, ListAliveClients, (), (const, override));
  MOCK_METHOD(std::size_t, AliveClientCount, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, GetClientTags, (const std::string& node_id),
              (const, override));

  // --- External-KV reads ---
  MOCK_METHOD(std::vector<NodeMatch>, MatchExternalKv,
              ((const std::vector<std::string>&)hashes, bool count_as_hit,
               std::chrono::system_clock::time_point now),
              (override));
  MOCK_METHOD(std::vector<ExternalKvHitCountEntry>, GetExternalKvHitCounts,
              ((const std::vector<std::string>&)hashes), (const, override));
  MOCK_METHOD(std::size_t, GetExternalKvCount, (const std::string& node_id), (const, override));
};

}  // namespace mori::umbp
