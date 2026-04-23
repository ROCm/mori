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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "umbp/common/config.h"

namespace mori::umbp {

/// Abstract interface for UMBP storage clients.
///
/// Two implementations exist behind this interface:
///   - StandaloneClient: purely local DRAM+SSD storage, no networking.
///   - DistributedClient: master-led global routing + RDMA data plane.
///
/// Use CreateUMBPClient() to obtain the appropriate implementation based on
/// UMBPConfig. All methods are zero-copy and pointer-based, designed for
/// sglang's HostKVCache page buffer model.
class IUMBPClient {
 public:
  virtual ~IUMBPClient() = default;

  // ---- Core KV Operations ----

  /// Write `size` bytes from address `src` into the store under `key`.
  virtual bool Put(const std::string& key, uintptr_t src, size_t size) = 0;

  /// Read stored value for `key` into address `dst`.
  virtual bool Get(const std::string& key, uintptr_t dst, size_t size) = 0;

  /// Check whether `key` exists in the store.
  virtual bool Exists(const std::string& key) const = 0;

  // ---- Batch Operations ----

  virtual std::vector<bool> BatchPut(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& srcs,
                                     const std::vector<size_t>& sizes) = 0;

  /// Depth-aware batch put.  depths[i] is the radix-tree chain depth for
  /// keys[i].  Standalone uses depth for local eviction priority; Distributed
  /// forwards it to the Master for global eviction decisions.
  /// Empty depths vector or depth == -1 falls back to plain LRU.
  virtual std::vector<bool> BatchPutWithDepth(const std::vector<std::string>& keys,
                                              const std::vector<uintptr_t>& srcs,
                                              const std::vector<size_t>& sizes,
                                              const std::vector<int>& depths) = 0;

  virtual std::vector<bool> BatchGet(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dsts,
                                     const std::vector<size_t>& sizes) = 0;

  virtual std::vector<bool> BatchExists(const std::vector<std::string>& keys) const = 0;

  /// Returns the number of keys that exist consecutively from index 0.
  /// Stops at the first key that does not exist (early-stop).
  virtual size_t BatchExistsConsecutive(const std::vector<std::string>& keys) const = 0;

  // ---- Lifecycle ----

  /// Remove all stored entries.
  virtual void Clear() = 0;

  /// Persist all pending write-back data.
  virtual bool Flush() = 0;

  /// Graceful one-time shutdown.  Flushes pending data, stops background
  /// threads, and releases resources.  Also called by the destructor of
  /// concrete implementations.  Calling Close() more than once is safe.
  virtual void Close() = 0;

  // ---- Introspection ----

  /// Returns true when the client operates in distributed (master-led) mode.
  virtual bool IsDistributed() const = 0;

  // ---- Optional zero-copy hooks ----
  //
  // Register a host buffer for zero-copy RDMA transfers.  Standalone
  // mode needs no registration (CPU-local memcpy), so the default
  // implementation is a no-op returning true.  DistributedClient
  // overrides to pin + export the buffer through IOEngine.  Implementers
  // that *do* require registration MUST override; callers may treat a
  // `true` return as "registered or not-needed", and `false` as a hard
  // failure that must be surfaced.
  virtual bool RegisterMemory(uintptr_t /*ptr*/, size_t /*size*/) { return true; }
  virtual void DeregisterMemory(uintptr_t /*ptr*/) {}
};

/// Factory: creates the appropriate IUMBPClient implementation.
/// Creates StandaloneClient when config.distributed is not set,
/// DistributedClient when it is.
std::unique_ptr<IUMBPClient> CreateUMBPClient(const UMBPConfig& config = UMBPConfig{});

}  // namespace mori::umbp
