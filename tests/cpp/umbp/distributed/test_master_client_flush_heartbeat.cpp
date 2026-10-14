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

// Tests for MasterClient::FlushHeartbeat():
//
//  1. FlushHeartbeat() wakes the sleeping heartbeat thread and fires a
//     heartbeat immediately, well before the configured interval elapses.
//
//  2. FlushHeartbeat() called before StartHeartbeat() is a safe no-op —
//     no crash, no heartbeat sent.
//
//  3. FlushHeartbeat() called while a heartbeat RPC is in-flight sets
//     flush_requested_, so the next loop iteration fires immediately once
//     the in-flight RPC completes (rather than sleeping the full interval).

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "umbp.grpc.pb.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/peer/backend/mock_backend.h"

namespace mori::umbp {
namespace {

// --------------------------------------------------------------------------
// Fake master: records heartbeat arrival times; optionally blocks each
// Heartbeat RPC until ReleaseAll() is called.
// --------------------------------------------------------------------------
class RecordingMasterService final : public ::umbp::UMBPMaster::Service {
 public:
  explicit RecordingMasterService(int interval_ms, bool block_heartbeats = false)
      : interval_ms_(interval_ms), block_heartbeats_(block_heartbeats) {}

  grpc::Status RegisterClient(grpc::ServerContext*, const ::umbp::RegisterClientRequest*,
                              ::umbp::RegisterClientResponse* resp) override {
    resp->set_heartbeat_interval_ms(interval_ms_);
    return grpc::Status::OK;
  }

  grpc::Status Heartbeat(grpc::ServerContext* ctx, const ::umbp::HeartbeatRequest* req,
                         ::umbp::HeartbeatResponse* resp) override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      ++count_;
      last_time_ = std::chrono::steady_clock::now();
      arrival_times_.push_back(last_time_);
      size_t event_count = 0;
      uint64_t highest_seq = 0;
      for (const auto& bundle : req->bundles()) {
        event_count += static_cast<size_t>(bundle.events_size());
        highest_seq = std::max(highest_seq, bundle.seq());
      }
      heartbeat_event_counts_.push_back(event_count);
      resp->set_acked_seq(highest_seq);
      resp->set_status(::umbp::CLIENT_STATUS_ALIVE);
      // A real master re-advertises the effective interval on every response;
      // 0 (the default here) means "no opinion", which is also what a master
      // predating the field sends.
      resp->set_heartbeat_interval_ms(advertised_interval_ms_.load());
      entered_cv_.notify_all();
    }
    if (block_heartbeats_) {
      std::unique_lock<std::mutex> lock(mu_);
      // Poll cancel flag at 25 ms so server shutdown isn't stuck.
      release_cv_.wait_for(lock, std::chrono::milliseconds(25),
                           [&] { return released_ || ctx->IsCancelled(); });
      while (!released_ && !ctx->IsCancelled()) {
        release_cv_.wait_for(lock, std::chrono::milliseconds(25),
                             [&] { return released_ || ctx->IsCancelled(); });
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status UnregisterClient(grpc::ServerContext*, const ::umbp::UnregisterClientRequest*,
                                ::umbp::UnregisterClientResponse*) override {
    return grpc::Status::OK;
  }

  void ReleaseAll() {
    std::lock_guard<std::mutex> lock(mu_);
    released_ = true;
    release_cv_.notify_all();
  }

  // Stand-in for SetRuntimeConfig on a real master: from the next response on,
  // advertise `ms` as the effective interval.  Not guarded by mu_ — it is set
  // from the test thread while the heartbeat handler reads it.
  void SetAdvertisedInterval(uint64_t ms) { advertised_interval_ms_.store(ms); }

  // Arrival timestamps, oldest first; lets a test measure the cadence rather
  // than just the count.
  std::vector<std::chrono::steady_clock::time_point> ArrivalTimes() {
    std::lock_guard<std::mutex> lock(mu_);
    return arrival_times_;
  }

  // Block until at least `target` heartbeats have been received, or `timeout` elapses.
  bool WaitForCount(int target, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    return entered_cv_.wait_for(lock, timeout, [&] { return count_ >= target; });
  }

  // Block until a heartbeat that arrived strictly after `since` is recorded.
  bool WaitForHeartbeatAfter(std::chrono::steady_clock::time_point since,
                             std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    return entered_cv_.wait_for(lock, timeout, [&] { return count_ > 0 && last_time_ > since; });
  }

  int Count() {
    std::lock_guard<std::mutex> lock(mu_);
    return count_;
  }

  std::chrono::steady_clock::time_point LastTime() {
    std::lock_guard<std::mutex> lock(mu_);
    return last_time_;
  }

  std::vector<size_t> HeartbeatEventCounts() {
    std::lock_guard<std::mutex> lock(mu_);
    return heartbeat_event_counts_;
  }

 private:
  const int interval_ms_;
  const bool block_heartbeats_;

  std::atomic<uint64_t> advertised_interval_ms_{0};

  std::mutex mu_;
  std::condition_variable entered_cv_;
  std::condition_variable release_cv_;
  int count_ = 0;
  std::vector<size_t> heartbeat_event_counts_;
  std::vector<std::chrono::steady_clock::time_point> arrival_times_;
  std::chrono::steady_clock::time_point last_time_;
  bool released_ = false;
};

// Upstream drove this through a bespoke OwnedLocationSource holding a vector of
// events. That type is gone — MediumBackend absorbed it — so the backlog is
// built the way a real one arises: commit `count` keys into a backend and let
// its outbox fill. MockBackend is a full MediumBackend, so this exercises the
// same DrainAllBackends path production uses.
//
// The registry must be attached to the client only AFTER the commits, because
// SetBackendRegistry installs the auto-flush hook: with it installed, the
// commits themselves would flush and the backlog would never accumulate.
std::unique_ptr<BackendRegistry> MakeRegistryWithBacklog(size_t count) {
  auto backend = std::make_unique<MockBackend>(TierType::DRAM);

  std::vector<AllocateRequest> allocs;
  allocs.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    allocs.push_back(AllocateRequest{"heartbeat-key-" + std::to_string(i), /*size=*/4096});
  }
  auto allocated = backend->BatchAllocate(allocs);

  std::vector<CommitRequest> commits;
  commits.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(allocated[i].outcome, AllocateOutcome::kSuccessAllocated);
    commits.push_back(CommitRequest{allocated[i].slot_id, allocs[i].key});
  }
  backend->BatchCommit(commits);

  auto registry = std::make_unique<BackendRegistry>();
  registry->Register(std::move(backend));
  return registry;
}

// --------------------------------------------------------------------------
// Test fixture
// --------------------------------------------------------------------------
class FlushHeartbeatTest : public ::testing::Test {
 protected:
  void BuildServer(int interval_ms, bool block = false) {
    service_ = std::make_unique<RecordingMasterService>(interval_ms, block);

    // Port 0 asks the OS for a free one and reports it back. A guessed port
    // collides on a busy host — these tests run in a --network host container
    // alongside everything else on the node — and BuildAndStart signals that
    // by returning nullptr, so the collision surfaces as an unrelated
    // assertion failure rather than as EADDRINUSE.
    int selected_port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
    ASSERT_NE(selected_port, 0);
    address_ = "127.0.0.1:" + std::to_string(selected_port);
  }

  std::unique_ptr<MasterClient> MakeRegisteredClient() {
    UMBPMasterClientConfig cfg;
    cfg.node_id = "flush-hb-test-node";
    cfg.node_address = "127.0.0.1";
    cfg.master_address = address_;
    auto client = std::make_unique<MasterClient>(cfg);
    std::map<TierType, TierCapacity> caps;
    caps[TierType::DRAM] = {1u << 20, 1u << 20};
    auto status = client->RegisterSelf(caps);
    EXPECT_TRUE(status.ok()) << "RegisterSelf failed: " << status.error_message();
    return client;
  }

  void TearDown() override {
    if (service_) service_->ReleaseAll();
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now() + std::chrono::milliseconds(500));
      server_->Wait();
    }
  }

  std::string address_;
  std::unique_ptr<RecordingMasterService> service_;
  std::unique_ptr<grpc::Server> server_;
};

// --------------------------------------------------------------------------
// Test 1: FlushHeartbeat wakes the sleeping heartbeat thread immediately.
//
// With a 10-second interval the first heartbeat would not arrive for ~10s.
// FlushHeartbeat() must deliver it within 500 ms.
// --------------------------------------------------------------------------
TEST_F(FlushHeartbeatTest, WakesHeartbeatThreadImmediately) {
  constexpr int kLongIntervalMs = 10'000;
  constexpr int kFlushDeadlineMs = 500;

  ASSERT_NO_FATAL_FAILURE(BuildServer(kLongIntervalMs));
  auto client = MakeRegisteredClient();

  client->StartHeartbeat();
  ASSERT_EQ(service_->Count(), 0) << "Unexpected heartbeat before FlushHeartbeat()";

  auto t0 = std::chrono::steady_clock::now();
  client->FlushHeartbeat();

  bool fired = service_->WaitForHeartbeatAfter(t0, std::chrono::milliseconds(kFlushDeadlineMs));
  ASSERT_TRUE(fired) << "Heartbeat did not arrive within " << kFlushDeadlineMs
                     << " ms after FlushHeartbeat()";

  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(service_->LastTime() - t0).count();
  EXPECT_LT(elapsed_ms, kFlushDeadlineMs)
      << "Heartbeat arrived after " << elapsed_ms << " ms (budget " << kFlushDeadlineMs << " ms)";
}

// --------------------------------------------------------------------------
// Test 2: FlushHeartbeat() before StartHeartbeat() is a safe no-op.
// --------------------------------------------------------------------------
TEST_F(FlushHeartbeatTest, NoOpBeforeStart) {
  ASSERT_NO_FATAL_FAILURE(BuildServer(5000));

  // Case A: before RegisterSelf and StartHeartbeat.
  {
    UMBPMasterClientConfig cfg;
    cfg.node_id = "flush-noop-node";
    cfg.node_address = "127.0.0.1";
    cfg.master_address = address_;
    MasterClient client(cfg);
    EXPECT_NO_FATAL_FAILURE(client.FlushHeartbeat());
  }

  // Case B: after RegisterSelf but before StartHeartbeat.
  {
    auto client = MakeRegisteredClient();
    EXPECT_NO_FATAL_FAILURE(client->FlushHeartbeat());
    // Destructor runs without starting the heartbeat thread — should be clean.
  }

  EXPECT_EQ(service_->Count(), 0) << "No heartbeat should have been sent";
}

// --------------------------------------------------------------------------
// Test 3: FlushHeartbeat() called while a heartbeat RPC is in-flight.
//
// flush_requested_ persists past the in-flight send; the next wait_for
// iteration sees it as true and fires immediately — gap from RPC release
// to second heartbeat must be well under the 10-second interval.
// --------------------------------------------------------------------------
TEST_F(FlushHeartbeatTest, FlushWhileRPCInFlightFiresNextTickImmediately) {
  constexpr int kLongIntervalMs = 10'000;
  constexpr int kDeadlineMs = 1000;

  ASSERT_NO_FATAL_FAILURE(BuildServer(kLongIntervalMs, /*block=*/true));
  auto client = MakeRegisteredClient();
  client->StartHeartbeat();

  // Kick the first heartbeat and wait for it to enter (and block in) the RPC.
  client->FlushHeartbeat();
  ASSERT_TRUE(service_->WaitForCount(1, std::chrono::milliseconds(500)))
      << "First heartbeat never reached the server";

  // While the RPC is still blocked, request a second flush.
  client->FlushHeartbeat();

  // Release the blocked RPC and measure how quickly the second heartbeat arrives.
  auto t_release = std::chrono::steady_clock::now();
  service_->ReleaseAll();

  bool got_second = service_->WaitForCount(2, std::chrono::milliseconds(kDeadlineMs));
  ASSERT_TRUE(got_second) << "Second heartbeat did not arrive within " << kDeadlineMs
                          << " ms after releasing the blocked RPC";

  auto gap_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(service_->LastTime() - t_release)
          .count();
  EXPECT_LT(gap_ms, kDeadlineMs)
      << "Second heartbeat arrived " << gap_ms << " ms after release (budget " << kDeadlineMs
      << " ms); flush_requested_ may not have been preserved across the in-flight RPC";
}

TEST_F(FlushHeartbeatTest, LargeEventBacklogIsSplitAcrossBoundedImmediateHeartbeats) {
  constexpr size_t kEvents = 20'000;
  constexpr size_t kDefaultMaxEventsPerRpc = 16 * 1024;
  ASSERT_NO_FATAL_FAILURE(BuildServer(/*interval_ms=*/10'000));
  // Declared before the client so it outlives it: MasterClient holds the
  // registry by raw pointer and touches it while shutting the heartbeat down.
  auto registry = MakeRegistryWithBacklog(kEvents);
  auto client = MakeRegisteredClient();
  client->SetBackendRegistry(registry.get());
  client->StartHeartbeat();

  client->FlushHeartbeat();
  ASSERT_TRUE(service_->WaitForCount(2, std::chrono::milliseconds(2000)));
  client->StopHeartbeat();

  const auto counts = service_->HeartbeatEventCounts();
  ASSERT_GE(counts.size(), 2u);
  EXPECT_LE(counts[0], kDefaultMaxEventsPerRpc);
  EXPECT_LE(counts[1], kDefaultMaxEventsPerRpc);
  EXPECT_EQ(counts[0] + counts[1], kEvents);
}

// --------------------------------------------------------------------------
// Test 5: a master-advertised interval change takes effect without a
// re-register.
//
// The peer joins with a 10-second interval, so left alone it would heartbeat
// roughly never on a test's timescale.  The master then starts advertising
// 150 ms on each response (what SetRuntimeConfig does in production).  The
// peer must pick that up and settle into the faster cadence on its own — no
// FlushHeartbeat, no re-register, no restart.
//
// Cadence, not just count, is what is asserted: a count alone would also pass
// if something were firing heartbeats for an unrelated reason.
// --------------------------------------------------------------------------
TEST_F(FlushHeartbeatTest, MasterAdvertisedIntervalChangeTakesEffectWithoutReregister) {
  constexpr int kJoinIntervalMs = 10'000;
  constexpr uint64_t kFastIntervalMs = 150;
  constexpr int kTargetHeartbeats = 5;

  ASSERT_NO_FATAL_FAILURE(BuildServer(kJoinIntervalMs));
  auto client = MakeRegisteredClient();
  client->StartHeartbeat();

  // One flush to deliver the first response; that response is what carries the
  // new interval.  Everything after this must be self-sustaining.
  service_->SetAdvertisedInterval(kFastIntervalMs);
  client->FlushHeartbeat();

  // 5 heartbeats at 150 ms is ~600 ms of cadence after the first; a generous
  // 5 s budget keeps this from flaking on a loaded CI box while still being
  // far below the 10 s join interval — if the change were ignored, only the
  // single flushed heartbeat would ever arrive and this would time out.
  ASSERT_TRUE(service_->WaitForCount(kTargetHeartbeats, std::chrono::milliseconds(5000)))
      << "Only " << service_->Count()
      << " heartbeat(s) arrived; the advertised interval change was not adopted";
  client->StopHeartbeat();

  const auto times = service_->ArrivalTimes();
  ASSERT_GE(times.size(), static_cast<size_t>(kTargetHeartbeats));

  // Skip the first gap: it spans the flush, not the new cadence.
  for (size_t i = 2; i < times.size(); ++i) {
    const auto gap_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(times[i] - times[i - 1]).count();
    EXPECT_LT(gap_ms, kJoinIntervalMs / 2) << "Gap " << i << " was " << gap_ms
                                           << " ms — peer still appears to be on the join interval";
  }
}

// --------------------------------------------------------------------------
// Test 6: an advertised 0 means "no opinion" and must not disturb the peer.
//
// A master that predates the HeartbeatResponse field leaves it at the proto3
// default of 0.  Adopting that literally would collapse the wait to zero and
// spin the heartbeat thread, so 0 has to be ignored.
// --------------------------------------------------------------------------
TEST_F(FlushHeartbeatTest, AdvertisedZeroIntervalIsIgnored) {
  constexpr int kJoinIntervalMs = 400;

  ASSERT_NO_FATAL_FAILURE(BuildServer(kJoinIntervalMs));
  // Default advertised value is already 0; state it for the record.
  service_->SetAdvertisedInterval(0);

  auto client = MakeRegisteredClient();
  client->StartHeartbeat();
  ASSERT_TRUE(service_->WaitForCount(3, std::chrono::milliseconds(5000)));
  client->StopHeartbeat();

  const auto times = service_->ArrivalTimes();
  ASSERT_GE(times.size(), 3u);
  // Still on the join cadence: a spinning thread would have produced gaps near
  // zero and a far higher count in the same window.
  for (size_t i = 1; i < times.size(); ++i) {
    const auto gap_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(times[i] - times[i - 1]).count();
    EXPECT_GT(gap_ms, kJoinIntervalMs / 2)
        << "Gap " << i << " was only " << gap_ms << " ms — an advertised 0 appears to have been "
        << "adopted, collapsing the heartbeat wait";
  }
}

}  // namespace
}  // namespace mori::umbp
