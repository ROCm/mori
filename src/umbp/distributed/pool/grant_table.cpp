// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/distributed/pool/grant_table.h"

namespace mori::umbp {
namespace {

std::string SessionKey(const std::array<uint8_t, 16>& session) {
  return std::string(reinterpret_cast<const char*>(session.data()), session.size());
}

}  // namespace

std::vector<GrantResult> GrantTable::Acquire(const ReaderIdentity& reader,
                                             const std::vector<AllocationHandle>& sources,
                                             OperationId operation, Clock::time_point expires_at) {
  return AcquireEligible(reader, sources, std::vector<bool>(sources.size(), true), operation,
                         expires_at);
}

std::vector<GrantResult> GrantTable::AcquireEligible(const ReaderIdentity& reader,
                                                     const std::vector<AllocationHandle>& sources,
                                                     const std::vector<bool>& eligible,
                                                     OperationId operation,
                                                     Clock::time_point expires_at) {
  std::vector<GrantResult> out(sources.size());
  std::lock_guard<std::mutex> lock(mutex_);
  if (TooOldLocked(operation) || eligible.size() != sources.size()) return out;
  auto replay = acquire_ops_.find(operation);
  if (replay != acquire_ops_.end()) {
    return replay->second.reader == reader && replay->second.sources == sources
               ? replay->second.results
               : out;
  }
  std::unordered_set<ResidentIdentity, ResidentIdentityHash> seen;
  for (size_t i = 0; i < sources.size(); ++i) {
    const ResidentIdentity identity(sources[i]);
    out[i].source = sources[i];
    if (!eligible[i] || closed_.count(identity) != 0 || quarantine_.count(identity) != 0 ||
        sources[i].page_id == 0 || !seen.insert(identity).second) {
      continue;
    }
    const GrantId id{next_id_++};
    grants_.emplace(id.value,
                    Entry{id, reader, identity, sources[i], operation, i, expires_at, false});
    ++holds_[identity];
    out[i].granted = true;
    out[i].id = id;
  }
  acquire_ops_.emplace(operation, AcquireRecord{reader, sources, out});
  return out;
}

std::vector<GrantReleaseResult> GrantTable::Release(const ReaderIdentity& reader,
                                                    const std::vector<GrantId>& ids,
                                                    CompletionProof proof, OperationId operation) {
  std::vector<GrantReleaseResult> out(ids.size());
  std::lock_guard<std::mutex> lock(mutex_);
  if (TooOldLocked(operation)) return out;
  auto replay = release_ops_.find(operation);
  if (replay != release_ops_.end()) {
    return replay->second.reader == reader && replay->second.ids == ids &&
                   replay->second.proof == proof
               ? replay->second.results
               : out;
  }
  for (size_t i = 0; i < ids.size(); ++i) {
    out[i].id = ids[i];
    auto it = grants_.find(ids[i].value);
    if (it == grants_.end()) continue;  // duplicate release is inert
    if (!(it->second.reader == reader)) continue;
    out[i].known = true;
    out[i].source = it->second.handle;
    MarkAcquireTerminalLocked(it->second);
    if (proof == CompletionProof::kUnknown || it->second.quarantined) {
      it->second.quarantined = true;
      quarantine_.emplace(it->second.source, it->second.handle);
      out[i].quarantined = true;
      continue;  // retain the hold and entry for process lifetime
    }
    auto held = holds_.find(it->second.source);
    if (held != holds_.end() && --held->second == 0) holds_.erase(held);
    grants_.erase(it);
    out[i].released = true;
  }
  release_ops_.emplace(operation, ReleaseRecord{reader, ids, proof, out});
  return out;
}

void GrantTable::StopNewGrants(const std::vector<AllocationHandle>& sources) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& source : sources) closed_.insert(ResidentIdentity(source));
}

void GrantTable::AllowNewGrants(const std::vector<AllocationHandle>& sources) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& source : sources) closed_.erase(ResidentIdentity(source));
}

bool GrantTable::Held(const AllocationHandle& source) const { return HoldCount(source) != 0; }

size_t GrantTable::HoldCount(const AllocationHandle& source) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = holds_.find(ResidentIdentity(source));
  return it == holds_.end() ? 0 : it->second;
}

std::vector<AllocationHandle> GrantTable::Expire(Clock::time_point now) {
  std::vector<AllocationHandle> expired;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [id, entry] : grants_) {
    (void)id;
    if (!entry.quarantined && entry.expires_at <= now) {
      entry.quarantined = true;
      MarkAcquireTerminalLocked(entry);
      quarantine_.emplace(entry.source, entry.handle);
      expired.push_back(entry.handle);
    }
  }
  return expired;
}

std::vector<AllocationHandle> GrantTable::DrainQuarantineCandidates() {
  std::vector<AllocationHandle> out;
  std::lock_guard<std::mutex> lock(mutex_);
  out.reserve(quarantine_.size());
  for (const auto& [identity, handle] : quarantine_) {
    if (quarantine_reported_.insert(identity).second) out.push_back(handle);
  }
  return out;
}

void GrantTable::AdvanceAckWatermark(OperationId watermark) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string session = SessionKey(watermark.session_id);
  uint64_t& current = ack_watermarks_[session];
  if (watermark.seq <= current) return;
  current = watermark.seq;
  const auto acknowledged = [&](const auto& entry) {
    return SessionKey(entry.first.session_id) == session && entry.first.seq <= current;
  };
  for (auto it = acquire_ops_.begin(); it != acquire_ops_.end();) {
    it = acknowledged(*it) ? acquire_ops_.erase(it) : std::next(it);
  }
  for (auto it = release_ops_.begin(); it != release_ops_.end();) {
    it = acknowledged(*it) ? release_ops_.erase(it) : std::next(it);
  }
}

GrantTableStats GrantTable::Stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return GrantTableStats{grants_.size(), acquire_ops_.size(), release_ops_.size(), closed_.size(),
                         quarantine_.size()};
}

bool GrantTable::TooOldLocked(OperationId operation) const {
  auto watermark = ack_watermarks_.find(SessionKey(operation.session_id));
  return watermark != ack_watermarks_.end() && operation.seq <= watermark->second;
}

void GrantTable::MarkAcquireTerminalLocked(const Entry& entry) {
  auto operation = acquire_ops_.find(entry.operation);
  if (operation == acquire_ops_.end() || entry.request_index >= operation->second.results.size()) {
    return;
  }
  operation->second.results[entry.request_index].granted = false;
}

}  // namespace mori::umbp
