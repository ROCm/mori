// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umbp/engine/engine_types.h"

namespace mori::umbp {

using engine::AllocationHandle;
using engine::CompletionProof;
using engine::OperationId;
using engine::ResidentIdentity;
using engine::ResidentIdentityHash;

struct ReaderIdentity {
  std::string session;
  std::string request;

  bool operator==(const ReaderIdentity& other) const {
    return session == other.session && request == other.request;
  }
};

struct GrantId {
  uint64_t value = 0;
  bool operator==(const GrantId& other) const { return value == other.value; }
};

struct GrantResult {
  bool granted = false;
  GrantId id;
  AllocationHandle source;
};

struct GrantReleaseResult {
  GrantId id;
  bool known = false;
  bool released = false;
  bool quarantined = false;
  std::optional<AllocationHandle> source;
};

struct GrantTableStats {
  size_t active_grants = 0;
  size_t acquire_records = 0;
  size_t release_records = 0;
  size_t closed_sources = 0;
  size_t quarantined_sources = 0;
};

// Explicit multi-reader authorization. TTL expiry closes discovery but never
// proves a DMA stopped: expired/unknown completions are returned to the owner as
// quarantine candidates and their hold remains until physical rebuild.
class GrantTable {
 public:
  using Clock = std::chrono::steady_clock;

  std::vector<GrantResult> Acquire(const ReaderIdentity& reader,
                                   const std::vector<AllocationHandle>& sources,
                                   OperationId operation, Clock::time_point expires_at);
  std::vector<GrantResult> AcquireEligible(const ReaderIdentity& reader,
                                           const std::vector<AllocationHandle>& sources,
                                           const std::vector<bool>& eligible, OperationId operation,
                                           Clock::time_point expires_at);
  std::vector<GrantReleaseResult> Release(const ReaderIdentity& reader,
                                          const std::vector<GrantId>& ids, CompletionProof proof,
                                          OperationId operation);
  std::vector<GrantReleaseResult> Cancel(const ReaderIdentity& reader,
                                         const std::vector<GrantId>& ids, CompletionProof proof,
                                         OperationId operation) {
    return Release(reader, ids, proof, operation);
  }

  void StopNewGrants(const std::vector<AllocationHandle>& sources);
  void AllowNewGrants(const std::vector<AllocationHandle>& sources);
  bool Held(const AllocationHandle& source) const;
  size_t HoldCount(const AllocationHandle& source) const;
  size_t ActiveCount() const;
  std::vector<AllocationHandle> Expire(Clock::time_point now);
  std::vector<AllocationHandle> DrainQuarantineCandidates();
  void AdvanceAckWatermark(OperationId watermark);
  GrantTableStats Stats() const;

 private:
  struct Entry {
    GrantId id;
    ReaderIdentity reader;
    ResidentIdentity source;
    AllocationHandle handle;
    OperationId operation;
    size_t request_index = 0;
    Clock::time_point expires_at;
    bool quarantined = false;
  };

  struct AcquireRecord {
    ReaderIdentity reader;
    std::vector<AllocationHandle> sources;
    std::vector<GrantResult> results;
  };

  struct ReleaseRecord {
    ReaderIdentity reader;
    std::vector<GrantId> ids;
    CompletionProof proof = CompletionProof::kCompleted;
    std::vector<GrantReleaseResult> results;
  };

  bool TooOldLocked(OperationId operation) const;
  void MarkAcquireTerminalLocked(const Entry& entry);

  mutable std::mutex mutex_;
  uint64_t next_id_ = 1;
  std::unordered_map<uint64_t, Entry> grants_;
  std::unordered_map<OperationId, AcquireRecord, engine::OperationIdHash> acquire_ops_;
  std::unordered_map<OperationId, ReleaseRecord, engine::OperationIdHash> release_ops_;
  std::unordered_map<std::string, uint64_t> ack_watermarks_;
  std::unordered_map<ResidentIdentity, size_t, ResidentIdentityHash> holds_;
  std::unordered_set<ResidentIdentity, ResidentIdentityHash> closed_;
  std::unordered_map<ResidentIdentity, AllocationHandle, ResidentIdentityHash> quarantine_;
  std::unordered_set<ResidentIdentity, ResidentIdentityHash> quarantine_reported_;
};

}  // namespace mori::umbp
