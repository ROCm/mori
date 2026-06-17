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

#include <grpcpp/support/status.h>

#include <chrono>
#include <string_view>

namespace mori::umbp {

class MasterClient;

// RAII timer used at the top of every MasterClient RPC method.  It records
// nothing on its own; the caller must invoke SetStatus() once the gRPC call
// returns.  Destruction without a prior SetStatus() is treated as "RPC was
// never issued" (early validation return) and emits no metric, so failing to
// reach the gRPC call site does not pollute the error counter.
//
// Construct it as the first statement of the method (so the timer brackets
// proto building + ctx setup + send + master + recv + parse).  `method_name`
// must outlive the timer; in practice it is a string literal handed in by
// the call site such as "Lookup".
//
// SetStatus() copies the OK bit and the gRPC status code by value rather
// than holding a pointer to the caller's grpc::Status, because the local
// `auto status = stub->...` is typically declared AFTER the timer, which
// means it would destruct first on scope exit and leave the timer with a
// dangling reference.
class ScopedRpcTimer {
 public:
  ScopedRpcTimer(MasterClient* owner, std::string_view method_name) noexcept
      : owner_(owner), method_(method_name), t0_(std::chrono::steady_clock::now()) {}

  ~ScopedRpcTimer();

  ScopedRpcTimer(const ScopedRpcTimer&) = delete;
  ScopedRpcTimer& operator=(const ScopedRpcTimer&) = delete;
  ScopedRpcTimer(ScopedRpcTimer&&) = delete;
  ScopedRpcTimer& operator=(ScopedRpcTimer&&) = delete;

  // Call once the underlying gRPC stub returns, before going out of scope.
  // If never called, the destructor records nothing.
  void SetStatus(const grpc::Status& s) noexcept {
    has_status_ = true;
    ok_ = s.ok();
    code_ = s.error_code();
  }

 private:
  MasterClient* owner_;
  std::string_view method_;
  std::chrono::steady_clock::time_point t0_;
  bool has_status_{false};
  bool ok_{false};
  grpc::StatusCode code_{grpc::StatusCode::UNKNOWN};
};

}  // namespace mori::umbp
