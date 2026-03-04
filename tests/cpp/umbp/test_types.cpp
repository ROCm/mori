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

#include "umbp/types.h"

namespace mori::umbp {

TEST(TypesTest, TierTypeNameHBM) { EXPECT_STREQ(TierTypeName(TierType::HBM), "HBM"); }

TEST(TypesTest, TierTypeNameDRAM) { EXPECT_STREQ(TierTypeName(TierType::DRAM), "DRAM"); }

TEST(TypesTest, TierTypeNameSSD) { EXPECT_STREQ(TierTypeName(TierType::SSD), "SSD"); }

TEST(TypesTest, TierTypeNameUNKNOWN) { EXPECT_STREQ(TierTypeName(TierType::UNKNOWN), "UNKNOWN"); }

TEST(TypesTest, ClientStatusNameALIVE) {
  EXPECT_STREQ(ClientStatusName(ClientStatus::ALIVE), "ALIVE");
}

TEST(TypesTest, ClientStatusNameEXPIRED) {
  EXPECT_STREQ(ClientStatusName(ClientStatus::EXPIRED), "EXPIRED");
}

TEST(TypesTest, ClientStatusNameUNKNOWN) {
  EXPECT_STREQ(ClientStatusName(ClientStatus::UNKNOWN), "UNKNOWN");
}

TEST(TypesTest, TierCapacityDefaultInit) {
  TierCapacity cap;
  EXPECT_EQ(cap.total_bytes, 0u);
  EXPECT_EQ(cap.available_bytes, 0u);
}

TEST(TypesTest, ClientRecordDefaultInit) {
  ClientRecord record;
  EXPECT_TRUE(record.client_id.empty());
  EXPECT_TRUE(record.node_address.empty());
  EXPECT_EQ(record.status, ClientStatus::UNKNOWN);
  EXPECT_TRUE(record.tier_capacities.empty());
}

}  // namespace mori::umbp
