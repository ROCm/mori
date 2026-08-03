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
#pragma once

#include <memory>

#include "umbp/common/config.h"
#include "umbp/local/tiers/ssd_tier.h"
#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {

// Build the `file`-backend SSD tier from @p ssd_config.  One storage directory
// yields a plain SSDTier; several (comma-separated storage_dir, normally one
// mount per physical drive) yield a ShardedSsdTier that spreads keys across the
// drives and drives their IO in parallel.  capacity_bytes is the total budget
// and is split evenly across the directories.
//
// Both the distributed PeerSsdManager and the standalone LocalStorageManager
// go through here so multi-drive behaves identically in either deployment.
// The SPDK backends are unaffected: they get multi-device support one level
// down, from SpdkEnv's RAID0 over a comma-separated UMBP_SPDK_NVME_PCI.
std::unique_ptr<TierBackend> MakeFileSsdBackend(
    const UMBPSsdConfig& ssd_config, SSDAccessMode access_mode = SSDAccessMode::ReadWrite);

}  // namespace mori::umbp
