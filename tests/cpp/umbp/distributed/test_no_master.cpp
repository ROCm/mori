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

// The distributed client with NO master configured.
//
// An empty master_address is the single-node deployment: the whole data plane
// -- backends, transfer engine, peer service -- is built exactly as it would be
// in a cluster, but nothing is routed, registered or heartbeated, and a local
// miss is the final answer rather than a question for someone else.
//
// The point of these tests is that this is the SAME client, not a fallback to
// local mode: GetDeploymentMode() still reports Distributed, and the config
// alone decides which it is.  No master is started anywhere in this file, so
// any attempt to reach one shows up as a hang or a crash rather than passing
// quietly.

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {
namespace {

constexpr size_t kPageSize = 4096;
constexpr size_t kDramCap = 32ULL * 1024 * 1024;
constexpr size_t kObjectSize = 8192;

uint16_t NextPeerServicePort() {
  static std::atomic<uint16_t> next{
      static_cast<uint16_t>(47000 + (static_cast<unsigned>(::getpid()) % 3000))};
  return next.fetch_add(1);
}

std::vector<char> Pattern(size_t size, int seed) {
  std::vector<char> out(size);
  for (size_t i = 0; i < size; ++i) out[i] = static_cast<char>((i * 37 + seed) & 0xFF);
  return out;
}

class NoMasterTest : public ::testing::Test {
 protected:
  // Deliberately identical to a clustered config except for the one empty
  // field, so a failure here cannot be blamed on some other difference.
  PoolClientConfig Config(const std::string& node_id) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node_id;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = "";  // <-- the whole difference
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.peer_service_port = NextPeerServicePort();
    cfg.dram_page_size = kPageSize;
    cfg.medium = TierType::DRAM;
    cfg.dram.buffer_sizes = {kDramCap};
    cfg.cache_remote_fetches = false;
    return cfg;
  }

  PoolClient* Start(PoolClientConfig cfg) {
    auto client = std::make_unique<PoolClient>(std::move(cfg));
    if (!client->Init()) return nullptr;
    clients_.push_back(std::move(client));
    return clients_.back().get();
  }

  void TearDown() override {
    for (auto& c : clients_) {
      if (c) c->Shutdown();
    }
    clients_.clear();
  }

  std::vector<std::unique_ptr<PoolClient>> clients_;
};

TEST_F(NoMasterTest, InitSucceedsAndReportsNoMaster) {
  auto* node = Start(Config("no-master-init"));
  ASSERT_NE(node, nullptr) << "Init must not require a master";
  EXPECT_FALSE(node->HasMaster());
}

TEST_F(NoMasterTest, PutGetRoundTrip) {
  auto* node = Start(Config("no-master-rt"));
  ASSERT_NE(node, nullptr);

  const std::vector<std::string> keys = {"nm-a", "nm-b", "nm-c"};
  std::vector<std::vector<char>> objects;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < keys.size(); ++i) {
    objects.push_back(Pattern(kObjectSize, static_cast<int>(i)));
    srcs.push_back(objects.back().data());
    sizes.push_back(kObjectSize);
  }
  ASSERT_EQ(node->BatchPut(keys, srcs, sizes), std::vector<bool>(keys.size(), true))
      << "a put with no master must place on this node rather than fail to route";

  std::vector<std::vector<char>> reads(keys.size(), std::vector<char>(kObjectSize, 0));
  std::vector<void*> dsts;
  for (auto& r : reads) dsts.push_back(r.data());
  EXPECT_EQ(node->BatchGet(keys, dsts, sizes), std::vector<bool>(keys.size(), true));
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_EQ(std::memcmp(reads[i].data(), objects[i].data(), kObjectSize), 0)
        << "wrong bytes for " << keys[i];
  }
}

TEST_F(NoMasterTest, ExistsAnswersFromTheLocalMedia) {
  auto* node = Start(Config("no-master-exists"));
  ASSERT_NE(node, nullptr);

  const std::vector<std::string> present = {"nme-a", "nme-b"};
  const auto object = Pattern(kObjectSize, 5);
  ASSERT_EQ(node->BatchPut(present, {object.data(), object.data()}, {kObjectSize, kObjectSize}),
            std::vector<bool>(present.size(), true));

  EXPECT_EQ(node->BatchExists(present), std::vector<bool>(present.size(), true));
  EXPECT_TRUE(node->Exists(present[0]));

  // The half that would otherwise have gone to the master.  With no cluster
  // behind it, "this node does not have it" IS the final answer -- and it must
  // be false, not a fabricated hit and not a hang.
  const std::vector<std::string> mixed = {"nme-a", "nme-never-put"};
  EXPECT_EQ(node->BatchExists(mixed), std::vector<bool>({true, false}));
}

TEST_F(NoMasterTest, MissingKeyIsAMissNotAnError) {
  auto* node = Start(Config("no-master-miss"));
  ASSERT_NE(node, nullptr);

  const auto object = Pattern(kObjectSize, 9);
  ASSERT_EQ(node->BatchPut({"nmm-here"}, {object.data()}, {kObjectSize}),
            std::vector<bool>({true}));

  std::vector<char> hit(kObjectSize, 0);
  std::vector<char> miss(kObjectSize, 0);
  const std::vector<void*> dsts = {hit.data(), miss.data()};
  EXPECT_EQ(node->BatchGet({"nmm-here", "nmm-absent"}, dsts, {kObjectSize, kObjectSize}),
            std::vector<bool>({true, false}));
  EXPECT_EQ(std::memcmp(hit.data(), object.data(), kObjectSize), 0);
}

TEST_F(NoMasterTest, RangedGetServesFromTheLocalMedium) {
  auto* node = Start(Config("no-master-ranged"));
  ASSERT_NE(node, nullptr);

  const std::string key = "nmr-obj";
  const auto object = Pattern(kObjectSize, 13);
  ASSERT_EQ(node->BatchPut({key}, {object.data()}, {kObjectSize}), std::vector<bool>({true}));

  // Two disjoint windows, one at each end of the object.
  const size_t kFront = 512;
  const size_t kTail = 256;
  std::vector<char> front(kFront, 0);
  std::vector<char> tail(kTail, 0);
  const std::vector<std::vector<void*>> dsts = {{front.data(), tail.data()}};
  const std::vector<std::vector<size_t>> sizes = {{kFront, kTail}};
  const std::vector<std::vector<size_t>> offsets = {{0, kObjectSize - kTail}};

  EXPECT_EQ(node->BatchGetRanges({key}, dsts, sizes, offsets), std::vector<bool>({true}));
  EXPECT_EQ(std::memcmp(front.data(), object.data(), kFront), 0);
  EXPECT_EQ(std::memcmp(tail.data(), object.data() + kObjectSize - kTail, kTail), 0);

  // A key nothing here holds cannot be served and has nowhere else to come
  // from; the batch must report it missing rather than route or hang.
  std::vector<char> unused(kFront, 0);
  const std::vector<std::vector<void*>> miss_dsts = {{unused.data()}};
  EXPECT_EQ(node->BatchGetRanges({"nmr-absent"}, miss_dsts, {{kFront}}, {{0}}),
            std::vector<bool>({false}));
}

TEST_F(NoMasterTest, RangedPutAssemblesLocally) {
  auto* node = Start(Config("no-master-ranged-put"));
  ASSERT_NE(node, nullptr);

  const std::string key = "nmrp-obj";
  const auto object = Pattern(kObjectSize, 17);
  const size_t half = kObjectSize / 2;
  const std::vector<std::vector<const void*>> srcs = {{object.data(), object.data() + half}};
  const std::vector<std::vector<size_t>> sizes = {{half, kObjectSize - half}};
  const std::vector<std::vector<size_t>> offsets = {{0, half}};

  ASSERT_EQ(node->BatchPutRanges({key}, {kObjectSize}, srcs, sizes, offsets),
            std::vector<bool>({true}))
      << "a ranged put with no master must place on this node";

  std::vector<char> read(kObjectSize, 0);
  const std::vector<void*> dsts = {read.data()};
  EXPECT_EQ(node->BatchGet({key}, dsts, {kObjectSize}), std::vector<bool>({true}));
  EXPECT_EQ(std::memcmp(read.data(), object.data(), kObjectSize), 0);
}

// External-KV state lives only in the master's index.  With no master these
// must degrade to a clean no-op: reporting succeeds vacuously and a match finds
// nothing.  A crash or a false positive here would break the sglang tree
// connector, which calls them unconditionally.
TEST_F(NoMasterTest, ExternalKvDegradesToNoOp) {
  auto* node = Start(Config("no-master-extkv"));
  ASSERT_NE(node, nullptr);

  const std::vector<std::string> hashes = {"h1", "h2"};
  EXPECT_TRUE(node->ReportExternalKvBlocks(hashes, TierType::DRAM));
  EXPECT_TRUE(node->RevokeExternalKvBlocks(hashes, TierType::DRAM));
  EXPECT_TRUE(node->RevokeAllExternalKvBlocksAtTier(TierType::DRAM));

  std::vector<MasterClient::ExternalKvNodeMatch> matches;
  EXPECT_TRUE(node->MatchExternalKv(hashes, &matches, /*count_as_hit=*/false));
  EXPECT_TRUE(matches.empty());
}

TEST_F(NoMasterTest, ClearAndFlushSucceed) {
  auto* node = Start(Config("no-master-lifecycle"));
  ASSERT_NE(node, nullptr);

  const auto object = Pattern(kObjectSize, 21);
  ASSERT_EQ(node->BatchPut({"nml-a"}, {object.data()}, {kObjectSize}), std::vector<bool>({true}));
  EXPECT_TRUE(node->Clear()) << "Clear must converge vacuously with no master to sync to";
  EXPECT_EQ(node->BatchExists({"nml-a"}), std::vector<bool>({false}));
}

// The same process, the same client class, both deployments -- which is the
// property that makes single-node and clustered a config choice rather than a
// code path.  Runs them in sequence so a leak of one into the other shows up.
TEST_F(NoMasterTest, SameClientClassServesBothDeployments) {
  auto* solo = Start(Config("no-master-mode"));
  ASSERT_NE(solo, nullptr);
  EXPECT_FALSE(solo->HasMaster());

  UMBPConfig cfg;
  cfg.dram.capacity_bytes = kDramCap;
  UMBPDistributedConfig dc;
  dc.master_config.node_id = "no-master-client";
  dc.master_config.node_address = "127.0.0.1";
  dc.master_config.master_address = "";
  dc.io_engine.host = "0.0.0.0";
  dc.io_engine.port = 0;
  dc.peer_service_port = NextPeerServicePort();
  dc.dram_page_size = kPageSize;
  cfg.distributed = dc;

  std::string error;
  ASSERT_TRUE(cfg.Validate(&error)) << "an empty master address must validate: " << error;

  auto client = CreateUMBPClient(cfg);
  ASSERT_NE(client, nullptr);
  EXPECT_EQ(client->GetDeploymentMode(), UMBPDeploymentMode::Distributed)
      << "no master must not silently downgrade to the Local backend";
  EXPECT_TRUE(client->IsDistributed());

  const auto object = Pattern(kObjectSize, 23);
  std::vector<char> read(kObjectSize, 0);
  const auto src = reinterpret_cast<uintptr_t>(object.data());
  const auto dst = reinterpret_cast<uintptr_t>(read.data());
  ASSERT_EQ(client->BatchPut({"nmc-a"}, {src}, {kObjectSize}), std::vector<bool>({true}));
  EXPECT_EQ(client->BatchGet({"nmc-a"}, {dst}, {kObjectSize}), std::vector<bool>({true}));
  EXPECT_EQ(std::memcmp(read.data(), object.data(), kObjectSize), 0);
  EXPECT_EQ(client->BatchExistsConsecutive({"nmc-a", "nmc-absent"}), 1u);
  client->Close();
}

}  // namespace
}  // namespace mori::umbp
