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

#include "mori/io/engine.hpp"

namespace mori::umbp {

// Phase-2 shim for the type medium_backend.h forward-declares (see design doc
// §4 and §5 Phase 2/6).  A backend's Init(TransferEngine*) only ever calls
// RegisterMemory in this phase — the remote data plane still runs through
// PoolClient directly, not through this object.  Phase 6 turns this into an
// abstract interface with Plan/Submit/Wait and a CompositeTransferEngine that
// multiplexes across transports; backend code does not change when that
// happens, because it never calls anything but RegisterMemory/Deregister.
//
// Concrete (not virtual) on purpose: there is exactly one implementation until
// Phase 6, and PoolClient owns the only instance.
class TransferEngine {
 public:
  explicit TransferEngine(mori::io::IOEngine* io_engine) : io_engine_(io_engine) {}

  // Registers [base, base+size) with the underlying mori-io engine.  `device`
  // is the mori-io device id (-1 for CPU memory).  Returns an invalid
  // (size==0) MemoryDesc if no engine is configured — callers treat that the
  // same as "this node has no RDMA-reachable buffers".
  mori::io::MemoryDesc RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                                      int device) {
    if (io_engine_ == nullptr) return mori::io::MemoryDesc{};
    return io_engine_->RegisterMemory(base, size, device, loc);
  }

  void Deregister(const mori::io::MemoryDesc& desc) {
    if (io_engine_ == nullptr) return;
    io_engine_->DeregisterMemory(desc);
  }

  mori::io::IOEngine* IoEngine() const { return io_engine_; }

 private:
  mori::io::IOEngine* io_engine_;  // not owned; may be null (no RDMA configured)
};

}  // namespace mori::umbp
