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
#include <string>
#include <vector>

#include "mori/io/enum.hpp"
#include "umbp/l1/l1_pool.h"
#include "umbp/umbp_client.h"

namespace mori::umbp::standalone {

// The L1 pools' backing store, over the UMBP store behind `client`.
//
// The L1 pool manager manages HBM; this hands what it evicts to the store
// that manages everything below, and adds nothing of its own: no memory, no
// index, no copy. A slot is one stored object, its fragments -- every region
// of every part, on as many GPUs as the pool has parts -- the object's
// ranges, so a write is `BatchPutRanges`, a load `BatchGetRanges` and a
// lookup `BatchExists`. Where the object is placed (this node's medium, or a
// peer the master routes it to), by which transfer engine its bytes move and
// when it is evicted are the store's decisions, made as for any other put.
//
// One object a slot, not one a part: a slot is only useful whole, so it is
// routed, placed, found and evicted as one key (a part of it left behind on
// another node, or evicted alone, would be dead weight), and the master
// indexes one key rather than one per GPU. The store's planner groups the
// ranges by source allocation in either shape, so the copies are the same.
//
// The pools' memory is registered with the store when a pool attaches it,
// `kLocalCopyOnly`: recorded, so that a range is planned against its
// allocation and the store knows which GPU it is on, but neither pinned nor
// exported to the NIC. That is the rule the standalone server follows for
// every allocation it imports over HIP IPC -- the dmabuf fallback is unsafe
// for such a mapping -- and it costs nothing however large the pool is. A put
// the master routes to another node is therefore staged: gathered into the
// store's registered scratch arena, then sent from there.
//
// `location` is where the pools' memory is: GPU in the server, CPU for a test
// that runs the pools over host memory.
//
// `client` must outlive the returned object; calls after the client is closed
// fail and store nothing. No locking is added: the caller serializes against
// the client's lifecycle the way it does for its other data-path calls.
std::shared_ptr<l1::BackingStore> MakeL1BackingStore(
    IUMBPClient* client, mori::io::MemoryLocationType location = mori::io::MemoryLocationType::GPU);

// The key a slot's content is stored under. Store keys are text; the pools'
// are opaque bytes, so one that is not printable is written in hex.
std::string L1StoreKey(const std::string& space, const std::string& key);

}  // namespace mori::umbp::standalone
