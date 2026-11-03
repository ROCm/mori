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

// EngineDesc across versions: an engine with RAS decodes the EngineDesc of an engine without
// RAS, and the reverse. Engines exchange EngineDesc over framework side channels, so a job can
// mix MORI versions.
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdlib>
#include <msgpack.hpp>
#include <vector>

#include "mori/io/engine.hpp"
#include "mori/ras/ras.hpp"
#include "tests/cpp/ras/ras_test_util.hpp"

using namespace mori::io;

namespace {

// EngineDesc as MORI shipped it before RAS (commit 69776c59).
struct LegacyEngineDesc {
  EngineKey key;
  std::string nodeId;
  std::string hostname;
  std::string host;
  int port;
  int pid{0};
  BackendDescBlobMap backendDescs;
  MSGPACK_DEFINE(key, nodeId, hostname, host, port, pid, backendDescs);
};

// That build's bytes for {"engine-a", "node-1", "host-1", "10.0.0.1", 12345, 4242,
// {RDMA: {1, 2, 3}}}, captured by packing it with the pre-RAS headers.
const std::vector<uint8_t> kLegacyBytes = {
    0x97, 0xa8, 0x65, 0x6e, 0x67, 0x69, 0x6e, 0x65, 0x2d, 0x61, 0xa6, 0x6e, 0x6f, 0x64, 0x65, 0x2d,
    0x31, 0xa6, 0x68, 0x6f, 0x73, 0x74, 0x2d, 0x31, 0xa8, 0x31, 0x30, 0x2e, 0x30, 0x2e, 0x30, 0x2e,
    0x31, 0xcd, 0x30, 0x39, 0xcd, 0x10, 0x92, 0x81, 0x02, 0xc4, 0x03, 0x01, 0x02, 0x03};

template <typename T>
std::vector<uint8_t> Pack(const T& v) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, v);
  return std::vector<uint8_t>(buf.data(), buf.data() + buf.size());
}

template <typename T>
T Unpack(const std::vector<uint8_t>& bytes) {
  msgpack::object_handle oh =
      msgpack::unpack(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  return oh.get().as<T>();
}

EngineDesc Sample() {
  EngineDesc d;
  d.key = "engine-a";
  d.nodeId = "node-1";
  d.hostname = "host-1";
  d.host = "10.0.0.1";
  d.port = 12345;
  d.pid = 4242;
  d.backendDescs[BackendType::RDMA] = {1, 2, 3};
  return d;
}

RasDesc SampleRas() {
  RasDesc r;
  r.host = "10.0.0.1";
  r.port = 39001;
  r.token = "00112233445566778899aabbccddeeff";
  r.uuidHi = 0x0123456789ab4def;
  r.uuidLo = 0x8edcba9876543210;
  return r;
}

void ExpectLegacyFields(const EngineDesc& d) {
  EXPECT_EQ(d.key, "engine-a");
  EXPECT_EQ(d.nodeId, "node-1");
  EXPECT_EQ(d.hostname, "host-1");
  EXPECT_EQ(d.host, "10.0.0.1");
  EXPECT_EQ(d.port, 12345);
  EXPECT_EQ(d.pid, 4242);
  ASSERT_EQ(d.backendDescs.count(BackendType::RDMA), 1u);
  EXPECT_EQ(d.backendDescs.at(BackendType::RDMA), (DescBlob{1, 2, 3}));
}

}  // namespace

TEST(RasEngineDesc, AnEngineWithRasDecodesAnEngineWithoutRas) {
  EngineDesc d = Unpack<EngineDesc>(kLegacyBytes);
  ExpectLegacyFields(d);
  EXPECT_EQ(d.ras, RasDesc{});  // no RAS endpoint: never linked to

  LegacyEngineDesc legacy{"engine-a", "node-1", "host-1", "10.0.0.1", 12345, 4242, {}};
  legacy.backendDescs[BackendType::RDMA] = {1, 2, 3};
  EXPECT_EQ(Pack(legacy), kLegacyBytes);  // the copy above matches what shipped
  ExpectLegacyFields(Unpack<EngineDesc>(Pack(legacy)));
}

TEST(RasEngineDesc, AnEngineWithoutRasDecodesAnEngineWithRas) {
  EngineDesc d = Sample();
  d.ras = SampleRas();
  LegacyEngineDesc legacy = Unpack<LegacyEngineDesc>(Pack(d));
  EXPECT_EQ(legacy.key, "engine-a");
  EXPECT_EQ(legacy.nodeId, "node-1");
  EXPECT_EQ(legacy.hostname, "host-1");
  EXPECT_EQ(legacy.host, "10.0.0.1");
  EXPECT_EQ(legacy.port, 12345);
  EXPECT_EQ(legacy.pid, 4242);
  EXPECT_EQ(legacy.backendDescs.at(BackendType::RDMA), (DescBlob{1, 2, 3}));
  // ...and the legacy engine re-encodes it as exactly what it would have sent itself.
  EXPECT_EQ(Pack(legacy), kLegacyBytes);
}

TEST(RasEngineDesc, TheRasFieldIsAppendedAtTheEnd) {
  EngineDesc d = Sample();
  d.ras = SampleRas();
  EngineDesc back = Unpack<EngineDesc>(Pack(d));
  EXPECT_EQ(back, d);
  EXPECT_EQ(back.ras, SampleRas());

  // The new layout is the legacy one with the array grown by one and the field at its end.
  std::vector<uint8_t> empty = Pack(Sample());
  std::vector<uint8_t> expected = kLegacyBytes;
  expected[0] = 0x98;
  expected.insert(expected.end(), {0x95, 0xa0, 0x00, 0xa0, 0x00, 0x00});  // ["", 0, "", 0, 0]
  EXPECT_EQ(empty, expected);
}

TEST(RasEngineDesc, AnEngineAdvertisesItsRuntimesEndpointOnlyWhenRasRuns) {
  EXPECT_EQ(ras_test::RunInChild([] {
              unsetenv("MORI_RAS_ENABLE");
              IOEngine off("engine-off", IOEngineConfig{"127.0.0.1", 0});
              CHILD_CHECK(off.GetEngineDesc().ras == RasDesc{});
              return 0;
            }),
            0);
  EXPECT_EQ(
      ras_test::RunInChild([] {
        setenv("MORI_RAS_ENABLE", "1", 1);
        IOEngine a("engine-on-a", IOEngineConfig{"127.0.0.1", 0});
        IOEngine b("engine-on-b", IOEngineConfig{"127.0.0.1", 0});
        mori::ras::RasRuntime* rt = mori::ras::RasRuntime::Get();
        CHILD_CHECK(rt != nullptr);
        mori::ras::RasEndpoint ep = rt->Endpoint();
        for (const IOEngine* e : {&a, &b}) {
          const RasDesc& r = e->GetEngineDesc().ras;
          CHILD_CHECK(r.host == "127.0.0.1");
          CHILD_CHECK(r.port == ep.port);
          CHILD_CHECK(r.token == ep.token);
          CHILD_CHECK(r.uuidHi == ep.uuid.hi && r.uuidLo == ep.uuid.lo);
        }
        // Engines in one process are one runtime: registering each other links nothing.
        a.RegisterRemoteEngine(b.GetEngineDesc());
        CHILD_CHECK(ras_test::WaitFor([&] { return rt->Snapshot().sameProcessPeers == 1; }, 2000));
        CHILD_CHECK(rt->Snapshot().links.empty());
        return 0;
      }),
      0);
  EXPECT_EQ(ras_test::RunInChild([] {
              int holder = -1;
              uint16_t port = ras_test::HoldPort("0.0.0.0", &holder);
              setenv("MORI_RAS_ENABLE", "1", 1);
              setenv("MORI_RAS_PORT", std::to_string(port).c_str(), 1);
              IOEngine e("engine-blocked", IOEngineConfig{"127.0.0.1", 0});
              // RAS failed to start, so the engine advertises no endpoint, and works as before.
              CHILD_CHECK(e.GetEngineDesc().ras == RasDesc{});
              CHILD_CHECK(mori::ras::RasRuntime::Status().state == mori::ras::RunState::kFailed);
              return 0;
            }),
            0);
}
