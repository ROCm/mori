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

// BatchRouteGet columnar wire-format round-trip.
//
// The BatchRouteGetResponse was reshaped from a repeated array-of-structs
// (one BatchRouteGetEntry per key, repeating node_id/peer_address strings) to
// a columnar form: a deduplicated `nodes` table plus parallel packed `node_ref`
// / `tier` / `size` arrays, where node_ref == 0 means "not found" (folding away
// the old `found` bool).
//
// This test stands up a mock UMBPMaster service whose BatchRouteGet emits a
// response using the SAME node-table-dedup algorithm as the real
// master_server.cpp producer, then asserts that the real
// MasterClient::BatchRouteGet consumer decodes it correctly.  It therefore
// locks both halves of the contract: that distinct nodes are sent once and
// referenced by a 1-based index, that repeated references to one node resolve
// transparently, that node_ref == 0 maps to a "not found" optional, and that
// the per-key tier/size columns stay aligned with the request keys.
//
// gRPC only, no RDMA — runs unconditionally (not under the "integration" label).

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umbp.grpc.pb.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {
namespace {

static uint16_t AllocPort() {
  static std::atomic<uint16_t> next{0};
  if (next.load() == 0) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    next.store(static_cast<uint16_t>(52000 + (std::rand() % 1500)));
  }
  return next.fetch_add(7);
}

// Canned per-key routing answer the mock server hands back.  A nullopt entry
// (or a key absent from the table) is "not found".
struct Route {
  std::string node_id;
  std::string peer_address;
  ::umbp::TierType tier;
  uint64_t size;
};

// Mock master.  BatchRouteGet builds the columnar response exactly the way the
// real producer (master_server.cpp) does: distinct (node_id, peer_address)
// pairs go into `nodes` once, each key gets a 1-based node_ref (0 = not found),
// and tier/size are parallel per-key columns.
class MockMasterService final : public ::umbp::UMBPMaster::Service {
 public:
  void SetRoute(const std::string& key, Route r) { table_[key] = std::move(r); }

  // Distinct node count emitted by the most recent BatchRouteGet — lets the
  // test assert the response actually deduplicated rather than just that the
  // decoded strings happen to match.
  int LastNodeTableSize() const { return last_nodes_size_.load(); }

  grpc::Status RegisterClient(grpc::ServerContext*, const ::umbp::RegisterClientRequest*,
                              ::umbp::RegisterClientResponse* resp) override {
    resp->set_heartbeat_interval_ms(100000);  // effectively never; no heartbeat in this test
    return grpc::Status::OK;
  }

  grpc::Status BatchRouteGet(grpc::ServerContext*, const ::umbp::BatchRouteGetRequest* req,
                             ::umbp::BatchRouteGetResponse* resp) override {
    std::unordered_map<std::string, uint32_t> node_index;  // "id\0addr" -> 1-based
    for (const auto& key : req->keys()) {
      auto it = table_.find(key);
      if (it == table_.end()) {
        resp->add_node_ref(0);
        resp->add_tier(::umbp::TIER_UNKNOWN);
        resp->add_size(0);
        continue;
      }
      const Route& r = it->second;
      std::string nk = r.node_id;
      nk.push_back('\0');
      nk.append(r.peer_address);
      auto [slot, inserted] = node_index.try_emplace(nk, 0);
      if (inserted) {
        auto* node = resp->add_nodes();
        node->set_node_id(r.node_id);
        node->set_peer_address(r.peer_address);
        slot->second = static_cast<uint32_t>(resp->nodes_size());  // 1-based
      }
      resp->add_node_ref(slot->second);
      resp->add_tier(r.tier);
      resp->add_size(r.size);
    }
    last_nodes_size_.store(resp->nodes_size());
    return grpc::Status::OK;
  }

 private:
  std::unordered_map<std::string, Route> table_;
  std::atomic<int> last_nodes_size_{0};
};

class BatchRouteGetColumnarTest : public ::testing::Test {
 protected:
  void SetUp() override {
    address_ = "127.0.0.1:" + std::to_string(AllocPort());
    service_ = std::make_unique<MockMasterService>();
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);

    UMBPMasterClientConfig cfg;
    cfg.node_id = "caller";
    cfg.node_address = "127.0.0.1";
    cfg.master_address = address_;
    client_ = std::make_unique<MasterClient>(cfg);
  }

  void TearDown() override {
    client_.reset();
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now() + std::chrono::milliseconds(200));
      server_->Wait();
    }
  }

  std::string address_;
  std::unique_ptr<MockMasterService> service_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<MasterClient> client_;
};

// Two distinct nodes, one node referenced by two keys (dedup), and one
// not-found key — the headline case for the columnar reshape.
TEST_F(BatchRouteGetColumnarTest, DedupDistinctNodesAndNotFound) {
  service_->SetRoute("kA1", {"node-a", "10.0.0.1:47071", ::umbp::TIER_DRAM, 100});
  service_->SetRoute("kB", {"node-b", "10.0.0.2:47072", ::umbp::TIER_HBM, 200});
  service_->SetRoute("kA2", {"node-a", "10.0.0.1:47071", ::umbp::TIER_SSD, 300});
  // "kMiss" intentionally has no route -> not found.

  const std::vector<std::string> keys = {"kA1", "kB", "kA2", "kMiss"};
  std::vector<std::optional<RouteGetResult>> out;
  auto status = client_->BatchRouteGet(keys, /*exclude_nodes=*/{}, &out);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // One decoded slot per request key, in request order.
  ASSERT_EQ(out.size(), keys.size());

  // kA1 -> node-a / DRAM / 100
  ASSERT_TRUE(out[0].has_value());
  EXPECT_EQ(out[0]->node_id, "node-a");
  EXPECT_EQ(out[0]->peer_address, "10.0.0.1:47071");
  EXPECT_EQ(out[0]->tier, TierType::DRAM);
  EXPECT_EQ(out[0]->size, 100u);

  // kB -> node-b / HBM / 200
  ASSERT_TRUE(out[1].has_value());
  EXPECT_EQ(out[1]->node_id, "node-b");
  EXPECT_EQ(out[1]->peer_address, "10.0.0.2:47072");
  EXPECT_EQ(out[1]->tier, TierType::HBM);
  EXPECT_EQ(out[1]->size, 200u);

  // kA2 -> node-a again (same strings as out[0]) but its own tier/size column.
  ASSERT_TRUE(out[2].has_value());
  EXPECT_EQ(out[2]->node_id, "node-a");
  EXPECT_EQ(out[2]->peer_address, "10.0.0.1:47071");
  EXPECT_EQ(out[2]->tier, TierType::SSD);
  EXPECT_EQ(out[2]->size, 300u);

  // kMiss -> node_ref 0 -> not found.
  EXPECT_FALSE(out[3].has_value());

  // The two node-a keys must have shared a single node-table entry, so the
  // table holds exactly the 2 distinct nodes, not 3.
  EXPECT_EQ(service_->LastNodeTableSize(), 2);
}

// Every key unresolved: all slots present (parallel to keys) and all empty.
TEST_F(BatchRouteGetColumnarTest, AllNotFound) {
  const std::vector<std::string> keys = {"x", "y", "z"};
  std::vector<std::optional<RouteGetResult>> out;
  auto status = client_->BatchRouteGet(keys, {}, &out);
  ASSERT_TRUE(status.ok()) << status.error_message();

  ASSERT_EQ(out.size(), keys.size());
  for (const auto& o : out) EXPECT_FALSE(o.has_value());
  EXPECT_EQ(service_->LastNodeTableSize(), 0);  // empty node table when nothing resolves
}

// Empty request -> empty result, no error.
TEST_F(BatchRouteGetColumnarTest, EmptyKeys) {
  std::vector<std::optional<RouteGetResult>> out;
  auto status = client_->BatchRouteGet({}, {}, &out);
  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_TRUE(out.empty());
}

}  // namespace
}  // namespace mori::umbp
