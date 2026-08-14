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

// Build the `file`-backend SSD tier: one storage directory yields a plain
// SSDTier, several (comma-separated, one mount per drive) a ShardedSsdTier.
// capacity_bytes is the total budget, split evenly across the directories.
// Both PeerSsdManager and LocalStorageManager go through here, so multi-drive
// behaves identically in either deployment.  SPDK backends are unaffected —
// they get multi-device support from SpdkEnv's RAID0 one level down.
std::unique_ptr<TierBackend> MakeFileSsdBackend(
    const UMBPSsdConfig& ssd_config, SSDAccessMode access_mode = SSDAccessMode::ReadWrite);

}  // namespace mori::umbp
