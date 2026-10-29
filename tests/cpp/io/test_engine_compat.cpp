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

#include <gtest/gtest.h>

#include <msgpack.hpp>

#include "mori/io/common.hpp"
#include "mori/io/enum.hpp"

using namespace mori::io;

TEST(EngineDescCompat, DefaultFieldValues) {
  EngineDesc desc;
  EXPECT_EQ(desc.protocolVersion, kMoriIoProtocolVersion);
  EXPECT_EQ(desc.agentMode, static_cast<uint32_t>(AgentMode::READ_WRITE));
  EXPECT_EQ(desc.blockSize, 0u);
}

TEST(EngineDescCompat, MsgpackRoundTrip) {
  EngineDesc orig;
  orig.key = "test-engine";
  orig.nodeId = "node-1";
  orig.hostname = "host-1";
  orig.host = "127.0.0.1";
  orig.port = 12345;
  orig.pid = 42;
  orig.protocolVersion = kMoriIoProtocolVersion;
  orig.agentMode = static_cast<uint32_t>(AgentMode::WRITE);
  orig.blockSize = 65536;

  msgpack::sbuffer buf;
  msgpack::pack(buf, orig);

  auto handle = msgpack::unpack(buf.data(), buf.size());
  auto deserialized = handle.get().as<EngineDesc>();

  EXPECT_EQ(deserialized, orig);
  EXPECT_EQ(deserialized.protocolVersion, kMoriIoProtocolVersion);
  EXPECT_EQ(deserialized.agentMode, static_cast<uint32_t>(AgentMode::WRITE));
  EXPECT_EQ(deserialized.blockSize, 65536u);
}

TEST(EngineDescCompat, LegacyPeerDeserializesToDefaults) {
  // Simulate a legacy peer that serializes only the original 7 fields
  // (key, nodeId, hostname, host, port, pid, backendDescs).
  msgpack::sbuffer buf;
  msgpack::packer<msgpack::sbuffer> pk(&buf);
  pk.pack_array(7);
  pk.pack(std::string("legacy-engine"));
  pk.pack(std::string("node-old"));
  pk.pack(std::string("host-old"));
  pk.pack(std::string("10.0.0.1"));
  pk.pack(9999);
  pk.pack(100);
  pk.pack(BackendDescBlobMap{});

  auto handle = msgpack::unpack(buf.data(), buf.size());
  auto desc = handle.get().as<EngineDesc>();

  EXPECT_EQ(desc.key, "legacy-engine");
  EXPECT_EQ(desc.nodeId, "node-old");
  EXPECT_EQ(desc.port, 9999);
  // New fields should default to 0 when absent from the wire.
  EXPECT_EQ(desc.protocolVersion, 0u);
  EXPECT_EQ(desc.agentMode, 0u);
  EXPECT_EQ(desc.blockSize, 0u);
}

TEST(EngineDescCompat, EqualityIncludesNewFields) {
  EngineDesc a, b;
  a.key = b.key = "e";
  a.nodeId = b.nodeId = "n";
  a.hostname = b.hostname = "h";
  a.host = b.host = "127.0.0.1";
  a.port = b.port = 1;
  a.pid = b.pid = 1;
  a.protocolVersion = b.protocolVersion = 1;
  a.agentMode = b.agentMode = 3;
  a.blockSize = b.blockSize = 4096;
  EXPECT_EQ(a, b);

  b.blockSize = 8192;
  EXPECT_FALSE(a == b);

  b.blockSize = a.blockSize;
  b.protocolVersion = 99;
  EXPECT_FALSE(a == b);
}

TEST(AgentModeEnum, BitfieldValues) {
  EXPECT_EQ(static_cast<uint32_t>(AgentMode::NONE), 0u);
  EXPECT_EQ(static_cast<uint32_t>(AgentMode::READ), 1u);
  EXPECT_EQ(static_cast<uint32_t>(AgentMode::WRITE), 2u);
  EXPECT_EQ(static_cast<uint32_t>(AgentMode::READ_WRITE), 3u);

  uint32_t rw = static_cast<uint32_t>(AgentMode::READ_WRITE);
  EXPECT_TRUE(rw & static_cast<uint32_t>(AgentMode::READ));
  EXPECT_TRUE(rw & static_cast<uint32_t>(AgentMode::WRITE));
}
