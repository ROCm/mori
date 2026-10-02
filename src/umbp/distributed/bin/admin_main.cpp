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

// umbp_admin — mutate a running master's runtime knobs.
//
// The UMBP_* environment variables that seed these knobs are read once at
// process start and cached in function-local statics, so they cannot be
// changed in a running master.  Restarting one to retune a reporting cadence
// is disproportionate: every peer must re-register and re-ship a full-sync
// snapshot before routing is whole again.  This drives the SetRuntimeConfig
// RPC instead.
//
// Usage:
//   umbp_admin <master_addr> --heartbeat-interval-ms=<N>   set the override
//   umbp_admin <master_addr> --heartbeat-interval-ms=0     clear it
//   umbp_admin <master_addr> --show                        read back, no change

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "umbp.grpc.pb.h"
#include "umbp/common/grpc_limits.h"

namespace {

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " <master_addr> [options]\n"
            << "\nOptions:\n"
            << "  --heartbeat-interval-ms=N  Override the advertised heartbeat interval.\n"
            << "                             0 clears the override (revert to the value\n"
            << "                             derived from UMBP_HEARTBEAT_TTL_SEC /\n"
            << "                             UMBP_HEARTBEAT_INTERVAL_DIVISOR).  Values past\n"
            << "                             the expiry window are clamped by the master.\n"
            << "  --show                     Print the effective config without changing it.\n"
            << "\nExamples:\n"
            << "  " << argv0 << " localhost:50051 --heartbeat-interval-ms=1000\n"
            << "  " << argv0 << " localhost:50051 --show\n";
}

// Returns false when `arg` does not start with `prefix`.
bool ParseValue(std::string_view arg, std::string_view prefix, std::string* out) {
  if (arg.substr(0, prefix.size()) != prefix) return false;
  *out = std::string(arg.substr(prefix.size()));
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 2;
  }

  const std::string master_addr = argv[1];
  bool set_interval = false;
  uint64_t interval_ms = 0;

  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    std::string value;
    if (arg == "--show") {
      continue;  // default behaviour: report whatever comes back
    }
    if (ParseValue(arg, "--heartbeat-interval-ms=", &value)) {
      if (value.empty()) {
        std::cerr << "error: --heartbeat-interval-ms needs a value\n";
        return 2;
      }
      char* end = nullptr;
      const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
      if (end == value.c_str() || *end != '\0') {
        std::cerr << "error: --heartbeat-interval-ms: not a number: " << value << "\n";
        return 2;
      }
      interval_ms = static_cast<uint64_t>(parsed);
      set_interval = true;
      continue;
    }
    std::cerr << "error: unknown argument: " << arg << "\n";
    PrintUsage(argv[0]);
    return 2;
  }

  auto channel = grpc::CreateCustomChannel(master_addr, grpc::InsecureChannelCredentials(),
                                           mori::umbp::GrpcChannelArgs());
  auto stub = ::umbp::UMBPMaster::NewStub(channel);

  ::umbp::SetRuntimeConfigRequest req;
  req.set_set_heartbeat_interval(set_interval);
  req.set_heartbeat_interval_ms(interval_ms);

  ::umbp::SetRuntimeConfigResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
  const grpc::Status status = stub->SetRuntimeConfig(&ctx, req, &resp);
  if (!status.ok()) {
    std::cerr << "SetRuntimeConfig failed: " << status.error_message() << " (code "
              << static_cast<int>(status.error_code()) << ")\n";
    return 1;
  }

  std::cout << "heartbeat_interval_ms=" << resp.heartbeat_interval_ms() << "\n";
  if (!resp.note().empty()) std::cout << "note: " << resp.note() << "\n";
  return 0;
}
