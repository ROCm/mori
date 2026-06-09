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

#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/external_kv_block_index.h"
#include "umbp/distributed/master/global_block_index.h"

namespace mori::umbp {

TEST(ClientRegistryExternalKv, UnregisterClientClearsBothIndices) {
  GlobalBlockIndex global_index;
  ExternalKvBlockIndex external_index;
  ClientRegistry registry(ClientRegistryConfig{}, global_index, &external_index);

  ASSERT_TRUE(registry.RegisterClient("node-A", "127.0.0.1:9000", {}, "127.0.0.1:9001"));
  ASSERT_EQ(global_index.ApplyEvents("node-A",
                                     {KvEvent{KvEvent::Kind::ADD, "owned", TierType::DRAM, 128}}),
            1u);
  ASSERT_EQ(external_index.Register("node-A", {"external"}, TierType::DRAM), 1u);

  registry.UnregisterClient("node-A");

  EXPECT_TRUE(global_index.Lookup("owned").empty());
  EXPECT_TRUE(external_index.Match({"external"}).empty());
}

TEST(ClientRegistryExternalKv, UnregisterWithoutExternalIndexDoesNotCrash) {
  GlobalBlockIndex global_index;
  ClientRegistry registry(ClientRegistryConfig{}, global_index);

  ASSERT_TRUE(registry.RegisterClient("node-A", "127.0.0.1:9000", {}, "127.0.0.1:9001"));
  EXPECT_NO_THROW(registry.UnregisterClient("node-A"));
}

}  // namespace mori::umbp
