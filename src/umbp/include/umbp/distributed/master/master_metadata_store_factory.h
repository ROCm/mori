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

// MakeMasterMetadataStore — selects the master's metadata backend at startup.
//
// Reads UMBP_METADATA_BACKEND ("inmemory" | "redis"; default "inmemory") and
// the UMBP_REDIS_* connection knobs. This is the single production wiring point
// for the Redis backend: MasterServer constructs its store through this factory
// so router / eviction / reaper stay backend-agnostic (they only see
// IMasterMetadataStore&).

#pragma once

#include <memory>

#include "umbp/distributed/master/master_metadata_store.h"

namespace mori::umbp {

// Constructs the metadata store selected by UMBP_METADATA_BACKEND. Falls back
// to the in-memory backend when the env is unset/"inmemory". Throws
// std::runtime_error if "redis" is requested but the Redis backend was not
// compiled in (USE_REDIS_BACKEND=OFF), so a misconfiguration is loud rather
// than silently serving from a wrong backend.
std::unique_ptr<IMasterMetadataStore> MakeMasterMetadataStore();

}  // namespace mori::umbp
