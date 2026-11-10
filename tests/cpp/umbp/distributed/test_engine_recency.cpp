// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/master/in_memory_master_metadata_store.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/transfer/local_copy_engine.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {
namespace {
using namespace std::chrono_literals;

// Observe the exact Touch/unpin boundary without adding production test hooks.
class HandoffPageBackend : public PageBackend {
 public:
  HandoffPageBackend() : PageBackend(TierType::DRAM, 64, Ownership(), 30s, 0ms) {}
  static OwnershipConfig Ownership() {
    OwnershipConfig cfg;
    cfg.buffer_sizes = {64 * 128};
    cfg.use_hugepages = false;
    cfg.prefault = false;
    return cfg;
  }
  void Touch(const std::vector<std::string>& keys) override {
    PageBackend::Touch(keys);
    if (after_touch) after_touch();
  }
  void ReleaseMigrationRead(const std::string& key) override {
    if (before_unpin) before_unpin(key);
    PageBackend::ReleaseMigrationRead(key);
  }
  std::function<void()> after_touch;
  std::function<void(const std::string&)> before_unpin;
};

class EngineRecency : public ::testing::Test {
 protected:
  void SetUp() override { Init(false); }
  void TearDown() override { pool.reset(); }
  void Init(bool dram_first) {
    pool.reset();
    registry = BackendRegistry{};
    auto register_dram = [&] {
      auto backend = std::make_unique<HandoffPageBackend>();
      dram = backend.get();
      ASSERT_TRUE(dram->Init(&transfer));
      dram->TrackEvictionOrder();
      ASSERT_TRUE(registry.Register("dram", std::move(backend)));
      dram_id = registry.BackendId(dram);
    };
    if (dram_first) register_dram();
    for (size_t rank = 0; rank < hbm.size(); ++rank) {
      storage[rank].fill(0x3b);
      engine::PageGeometry geometry;
      geometry.page_tokens = 1;
      geometry.num_pages = 16;
      geometry.object_bytes = 64;
      geometry.regions = {{"kv", TransferRef::HostBytes(storage[rank].data(), storage[rank].size()),
                           64, 1, 0, 0, "kv"}};
      geometry.layout_id = engine::ComputeLayoutId(geometry);
      EngineRegistration registration;
      registration.pool_id = "pool";
      registration.epoch.boot_id[0] = 1;
      registration.owner_rank = rank;
      registration.geometry = geometry;
      auto backend = std::make_unique<EngineHbmBackend>(registration);
      hbm[rank] = backend.get();
      ASSERT_TRUE(backend->Init(nullptr));
      ASSERT_TRUE(registry.Register("hbm" + std::to_string(rank), std::move(backend)));
      hbm_id[rank] = registry.BackendId(hbm[rank]);
    }
    if (!dram_first) register_dram();
    LogicalTierConfig hot;
    hot.name = "hbm";
    hot.entry = true;
    hot.members = {{"hbm0", 1}, {"hbm1", 1}};
    hot.offload_to = {"dram"};
    // Enables the real Pool access index; occupancy stays below the threshold.
    hot.trigger = PoolOffloadTrigger::kWatermark;
    hot.high_watermark = .99;
    hot.low_watermark = .9;
    LogicalTierConfig cold;
    cold.name = "dram";
    cold.members = {{"dram", 1}};
    auto graph = LogicalTierGraph::Compile({hot, cold}, registry);
    ASSERT_TRUE(graph.ok()) << graph.error;
    pool = std::make_unique<PeerPool>(&registry, MakeTieredPlacementPolicy(graph.graph), &transfer);
    pool->SetLocalReadReporter([this](const std::vector<std::string>& keys) {
      // This read takes operation_mutex_: a callback under that lock deadlocks.
      (void)pool->PlacementCount();
      reports.insert(reports.end(), keys.begin(), keys.end());
      master.RecordLocalReads(keys, master_now);
    });
    reports.clear();
  }
  engine::OperationId Op() {
    engine::OperationId op;
    op.session_id[0] = 42;
    op.seq = ++sequence;
    return op;
  }
  engine::AllocationHandle Publish(const std::string& key, size_t rank = 0) {
    auto pages = pool->DelegateEnginePages(hbm_id[rank], rank, 1, Op());
    EXPECT_TRUE(pages.success);
    if (pages.handles.empty()) return {};
    auto page = pages.handles.front();
    ++page.generation;
    EXPECT_TRUE(pool->PublishResident(hbm_id[rank], {{key, {}, page}}, Op()).success);
    return page;
  }
  void PutDram(const std::string& key) {
    auto allocated = pool->BatchAllocate({{key, 64, TierType::DRAM, "dram"}});
    ASSERT_EQ(allocated.size(), 1u);
    ASSERT_EQ(allocated[0].allocation.outcome, AllocateOutcome::kSuccessAllocated);
    ASSERT_TRUE(
        pool->BatchCommit({{{dram_id, allocated[0].allocation.slot_id}, key}})[0].commit.success);
  }
  std::vector<std::string> Lru() {
    std::vector<std::string> keys;
    for (const auto& offer : dram->EvictionCandidates(100)) keys.push_back(offer.key);
    return keys;  // coldest first
  }
  engine::RetireTicket Retire(engine::AllocationHandle page, size_t rank = 0) {
    auto ticket = pool->RetireEnginePages(hbm_id[rank], {page}, Op());
    const auto until = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < until) {
      auto ready = pool->PollReadyEngineRetires(hbm_id[rank]);
      for (const auto& item : ready)
        if (item.id == ticket.id) return item;
      std::this_thread::yield();
    }
    ADD_FAILURE() << "retire never became Ready";
    return ticket;
  }
  void Ack(const engine::RetireTicket& ticket, size_t rank = 0) {
    ASSERT_TRUE(pool->AcknowledgeEngineReleased(hbm_id[rank], ticket.id));
  }
  void ExpectHandoff() {
    EXPECT_EQ(pool->TransitionMetrics().dram_handoff_refreshes, 1u);
    EXPECT_EQ(reports, (std::vector<std::string>{"K"}));
    EXPECT_EQ(pool->AccessOrderForTest().front(), "K");
  }
  LocalCopyEngine transfer;
  BackendRegistry registry;
  HandoffPageBackend* dram = nullptr;
  uint32_t dram_id = 0;
  std::array<EngineHbmBackend*, 2> hbm{};
  std::array<uint32_t, 2> hbm_id{};
  std::array<std::array<uint8_t, 17 * 64>, 2> storage{};
  std::unique_ptr<PeerPool> pool;
  std::vector<std::string> reports;
  InMemoryMasterMetadataStore master;
  std::chrono::system_clock::time_point master_now{100s};
  uint64_t sequence = 0;
};

TEST_F(EngineRecency, ExistingDramHandoffBecomesMruWithoutD2H) {
  PutDram("K");
  auto page = Publish("K");
  PutDram("J");
  ASSERT_EQ(Lru(), (std::vector<std::string>{"K", "J"}));
  auto ticket = Retire(page);
  Ack(ticket);
  EXPECT_EQ(Lru(), (std::vector<std::string>{"J", "K"}));
  EXPECT_EQ(pool->TransitionMetrics().offloaded_bytes, 0u);
  ExpectHandoff();
}

TEST_F(EngineRecency, OtherHbmSuppressesHandoffRegardlessOfRegistryOrder) {
  for (bool dram_first : {false, true}) {
    Init(dram_first);
    PutDram("K");
    auto first = Publish("K", 0);
    auto last = Publish("K", 1);
    PutDram("J");
    Ack(Retire(first), 0);
    EXPECT_EQ(Lru(), (std::vector<std::string>{"K", "J"}));
    EXPECT_TRUE(reports.empty());
    EXPECT_EQ(pool->TransitionMetrics().dram_handoff_refreshes, 0u);
    Ack(Retire(last, 1), 1);
    EXPECT_EQ(Lru(), (std::vector<std::string>{"J", "K"}));
    ExpectHandoff();
  }
}

TEST_F(EngineRecency, NewAndExistingDramHandoffsUpdatePoolAndMaster) {
  for (bool existing : {false, true}) {
    Init(false);
    InMemoryMasterMetadataStore view;
    ClientRegistration registration;
    registration.node_id = "node";
    ASSERT_TRUE(view.RegisterClient(registration, master_now, 30s));
    ASSERT_EQ(view.ApplyHeartbeat("node", 1, master_now, {},
                                  {{KvEvent::Kind::ADD, "K", TierType::HBM, 64}}, false)
                  .status,
              HeartbeatResult::APPLIED);
    pool->SetLocalReadReporter([&](const std::vector<std::string>& keys) {
      (void)pool->PlacementCount();
      reports.insert(reports.end(), keys.begin(), keys.end());
      view.RecordLocalReads(keys, master_now);
    });
    if (existing) PutDram("K");
    auto page = Publish("K");
    auto ticket = Retire(page);
    // The new D2H is committed before ACK; make its recency stale again.
    PutDram("J");
    master_now += 1s;
    ASSERT_EQ(view.ApplyHeartbeat("node", 2, master_now, {},
                                  {{KvEvent::Kind::ADD, "K", TierType::DRAM, 64}}, false)
                  .status,
              HeartbeatResult::APPLIED);
    master_now += 1s;
    Ack(ticket);
    EXPECT_EQ(Lru(), (std::vector<std::string>{"J", "K"}));
    EXPECT_EQ(pool->TransitionMetrics().offloaded_bytes, existing ? 0u : 64u);
    ExpectHandoff();
    const NodeTierKey bucket{"node", TierType::DRAM};
    auto candidates = view.EnumerateEvictionCandidates(
                              {bucket}, EvictionOrder::kLeastRecentlyAccessed, 10, master_now)
                          .at(bucket);
    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].last_accessed_at, master_now);
    pool->SetLocalReadReporter({});
  }
}

TEST_F(EngineRecency, DuplicateAckAndFailedOrQuarantinedRetireDoNotRefresh) {
  PutDram("K");
  auto page = Publish("K");
  PutDram("J");
  auto ticket = Retire(page);
  EXPECT_FALSE(pool->AcknowledgeEngineReleased(hbm_id[0], Op()));
  EXPECT_TRUE(reports.empty());
  Ack(ticket);
  ExpectHandoff();
  dram->Touch({"J"});
  const auto order = Lru();
  (void)pool->AcknowledgeEngineReleased(hbm_id[0], ticket.id);
  EXPECT_EQ(Lru(), order);
  EXPECT_EQ(reports.size(), 1u);
  PutDram("Q");
  auto bad = Publish("Q");
  ASSERT_TRUE(pool->HoldEngineIO(hbm_id[0], {bad}, Op()));
  auto refused = pool->RetireEnginePages(hbm_id[0], {bad}, Op());
  EXPECT_FALSE(pool->AcknowledgeEngineReleased(hbm_id[0], refused.id));
  ASSERT_TRUE(pool->QuarantineEnginePages(hbm_id[0], {bad}));
  EXPECT_FALSE(pool->AcknowledgeEngineReleased(hbm_id[0], refused.id));
  EXPECT_EQ(reports.size(), 1u);
  EXPECT_EQ(pool->TransitionMetrics().dram_handoff_refreshes, 1u);
  PutDram("U");
  auto unknown_page = Publish("U");
  const auto io_hold = Op();
  ASSERT_TRUE(pool->HoldEngineIO(hbm_id[0], {unknown_page}, io_hold));
  auto unknown = pool->RetireEnginePages(hbm_id[0], {unknown_page}, Op());
  const auto before_unknown = Lru();
  ASSERT_TRUE(pool->CompleteEngineIO(hbm_id[0], io_hold, engine::CompletionProof::kUnknown));
  EXPECT_FALSE(pool->AcknowledgeEngineReleased(hbm_id[0], unknown.id));
  EXPECT_EQ(Lru(), before_unknown);
  EXPECT_EQ(reports.size(), 1u);
  // Exhaust DRAM so a genuinely failed F3 takes the bounded safe-refusal path.
  const auto free_bytes = dram->Capacity().available_bytes;
  auto full = dram->BatchAllocate({{"full", free_bytes}});
  ASSERT_EQ(full[0].outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(dram->BatchCommit({{full[0].slot_id, "full"}})[0].success);
  auto no_backing = Publish("F");
  Ack(Retire(no_backing));
  EXPECT_FALSE(dram->Contains("F"));
  EXPECT_GT(pool->TransitionMetrics().f3_rejected, 0u);
  EXPECT_EQ(reports.size(), 1u);
  EXPECT_EQ(pool->TransitionMetrics().dram_handoff_refreshes, 1u);
}

TEST_F(EngineRecency, HandoffRefreshPrecedesLastUnpinAndBlocksEviction) {
  PutDram("K");
  auto page = Publish("K");
  PutDram("J");
  auto ticket = Retire(page);
  EXPECT_EQ(dram->Evict({"K"})[0].bytes_freed, 0u);
  std::promise<void> touched, release;
  auto released = release.get_future().share();
  dram->after_touch = [&] {
    touched.set_value();
    released.wait();
  };
  bool refreshed_before_unpin = false;
  dram->before_unpin = [&](const std::string& key) {
    if (key == "K") refreshed_before_unpin = Lru() == std::vector<std::string>{"J", "K"};
  };
  auto ack = std::async(std::launch::async,
                        [&] { return pool->AcknowledgeEngineReleased(hbm_id[0], ticket.id); });
  const bool saw_touch = touched.get_future().wait_for(300ms) == std::future_status::ready;
  EXPECT_TRUE(saw_touch) << "ACK must Touch before its last backing unpin";
  if (saw_touch) EXPECT_EQ(dram->Evict({"K"})[0].bytes_freed, 0u);
  release.set_value();
  EXPECT_TRUE(ack.get());
  EXPECT_TRUE(refreshed_before_unpin);
  dram->after_touch = {};
  dram->before_unpin = {};
  EXPECT_EQ(dram->Evict({"K"})[0].bytes_freed, 64u);
}

TEST_F(EngineRecency, LocalAndPeerServiceDramGrantsAccountButHbmDoesNotTouchDram) {
  PutDram("K");
  auto page = Publish("K");
  PutDram("J");
  ReaderIdentity reader{"reader", "request"};
  auto grants = pool->AcquireLoadSources(reader, {"K"}, {dram_id}, {TierType::DRAM}, Op(),
                                         GrantTable::Clock::now() + 30s);
  ASSERT_TRUE(grants[0].granted);
  engine::LoadGrantRef local_grant;
  local_grant.id = grants[0].id.value;
  local_grant.backend_id = dram_id;
  local_grant.key = "K";
  ASSERT_TRUE(pool->ReleaseLoadSources(reader, {local_grant}, engine::CompletionProof::kCompleted,
                                       Op())[0]);
  EXPECT_EQ(Lru(), (std::vector<std::string>{"J", "K"}));
  EXPECT_EQ(pool->AccessOrderForTest().front(), "K");
  EXPECT_EQ(reports, (std::vector<std::string>{"K"}));
  dram->Touch({"J"});
  const auto before = Lru();
  auto hbm_grant = pool->AcquireLoadSources(reader, {"K"}, {hbm_id[0]}, {TierType::HBM}, Op(),
                                            GrantTable::Clock::now() + 30s);
  ASSERT_TRUE(hbm_grant[0].granted);
  EXPECT_EQ(Lru(), before);
  EXPECT_EQ(reports.size(), 1u);
  PeerServiceServer server(*pool);
  ASSERT_TRUE(server.Start(0));
  auto stub = ::umbp::UMBPPeer::NewStub(grpc::CreateChannel(
      "localhost:" + std::to_string(server.BoundPort()), grpc::InsecureChannelCredentials()));
  ::umbp::BatchAcquireLoadSourcesRequest request;
  request.set_reader_session("remote");
  request.set_reader_request("request");
  request.mutable_operation()->set_session_id(std::string(16, 'x'));
  request.mutable_operation()->set_seq(1);
  request.set_lease_ms(1000);
  auto* entry = request.add_entries();
  entry->set_key("K");
  entry->set_preferred_backend_id(dram_id);
  entry->set_preferred_tier(::umbp::TIER_DRAM);
  grpc::ClientContext context;
  ::umbp::BatchAcquireLoadSourcesResponse response;
  ASSERT_TRUE(stub->BatchAcquireLoadSources(&context, request, &response).ok());
  ASSERT_EQ(response.grants_size(), 1);
  ASSERT_TRUE(response.grants(0).granted());
  local_grant.id = response.grants(0).grant_id();
  ASSERT_TRUE(pool->ReleaseLoadSources({"remote", "request"}, {local_grant},
                                       engine::CompletionProof::kCompleted, Op())[0]);
  EXPECT_EQ(Lru(), (std::vector<std::string>{"J", "K"}));
  EXPECT_EQ(pool->AccessOrderForTest().front(), "K");
  EXPECT_EQ(reports, (std::vector<std::string>{"K", "K"}));
  EXPECT_EQ(pool->TransitionMetrics().engine_dram_read_grants, 2u);
  EXPECT_EQ(pool->TransitionMetrics().dram_handoff_refreshes, 0u);
  server.Stop();
}

TEST_F(EngineRecency, HandoffUpdatesMasterTimestampAfterExistingKeyAddsDramLocation) {
  ClientRegistration registration;
  registration.node_id = "node";
  ASSERT_TRUE(master.RegisterClient(registration, master_now, 30s));
  auto page = Publish("K");
  ASSERT_EQ(master
                .ApplyHeartbeat("node", 1, master_now, {},
                                {{KvEvent::Kind::ADD, "K", TierType::HBM, 64}}, false)
                .status,
            HeartbeatResult::APPLIED);
  auto allocated = dram->BatchAllocate({{"K", 64}});
  ASSERT_EQ(allocated[0].outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(dram->BatchCommit({{allocated[0].slot_id, "K"}})[0].success);
  pool->RefreshBackendResidencies(hbm_id[0]);
  master_now += 1s;
  ASSERT_EQ(master
                .ApplyHeartbeat("node", 2, master_now, {},
                                {{KvEvent::Kind::ADD, "K", TierType::DRAM, 64}}, false)
                .status,
            HeartbeatResult::APPLIED);
  const std::vector<NodeTierKey> bucket{{"node", TierType::DRAM}};
  auto before = master
                    .EnumerateEvictionCandidates(bucket, EvictionOrder::kLeastRecentlyAccessed, 10,
                                                 master_now)
                    .at(bucket[0]);
  ASSERT_EQ(before.size(), 1u);
  EXPECT_LT(before[0].last_accessed_at, master_now) << "ADD alone keeps key-level heat";
  master_now += 1s;
  Ack(Retire(page));
  auto after = master
                   .EnumerateEvictionCandidates(bucket, EvictionOrder::kLeastRecentlyAccessed, 10,
                                                master_now)
                   .at(bucket[0]);
  ASSERT_EQ(after.size(), 1u);
  EXPECT_EQ(after[0].last_accessed_at, master_now);
  EXPECT_GT(after[0].last_accessed_at, before[0].last_accessed_at);
}

TEST_F(EngineRecency, CopyAndExistsKeepTheirOriginalTouchSemantics) {
  auto page = Publish("K");
  PutDram("J");
  TierTransitionExecutor cleaner(&registry, &transfer);
  auto copied = cleaner.Execute("K", hbm_id[0], {dram_id}, TierTransitionIntent::kCopy);
  ASSERT_TRUE(copied.success);
  EXPECT_TRUE(hbm[0]->IsQualifiedReplica("K"));
  EXPECT_TRUE(reports.empty());
  EXPECT_EQ(pool->TransitionMetrics().engine_dram_read_grants, 0u);
  // Migration reads/pins must not create demand heat.
  dram->Touch({"J"});
  ResolvedEntry held;
  ASSERT_TRUE(dram->AcquireMigrationRead("K", &held, true));
  dram->ReleaseMigrationRead("K");
  EXPECT_EQ(Lru(), (std::vector<std::string>{"K", "J"}));
  EXPECT_EQ(pool->BatchContains({"K"}), (std::vector<bool>{true}));
  // HBM is the placement: Exists touches that copy, not all copies.
  EXPECT_EQ(Lru(), (std::vector<std::string>{"K", "J"}));
  EXPECT_TRUE(reports.empty());
  Ack(Retire(page));
  ExpectHandoff();
  EXPECT_EQ(Lru(), (std::vector<std::string>{"J", "K"}));
  EXPECT_EQ(pool->BatchContains({"J"}), (std::vector<bool>{true}));
  EXPECT_EQ(Lru(), (std::vector<std::string>{"K", "J"}));
  EXPECT_EQ(reports.size(), 1u) << "Exists does not become a demand-read report";
}
}  // namespace
}  // namespace mori::umbp
