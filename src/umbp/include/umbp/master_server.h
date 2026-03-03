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

#include "umbp/client_registry.h"

namespace grpc_impl {
class Server;
}

namespace mori::umbp {

struct MasterServerConfig {
  std::string listen_address = "0.0.0.0:50051";
  ClientRegistryConfig registry_config;
};

class MasterServer {
 public:
  // PA-7 fix: take by value (not const-ref) for forward-compatibility
  // with unique_ptr members that will be added for Router strategies.
  explicit MasterServer(MasterServerConfig config);
  ~MasterServer();

  MasterServer(const MasterServer&) = delete;
  MasterServer& operator=(const MasterServer&) = delete;

  // Start the gRPC server (blocks until Shutdown is called).
  void Run();

  // Gracefully shut down the server and Reaper.
  void Shutdown();

 private:
  MasterServerConfig config_;
  ClientRegistry registry_;

  std::unique_ptr<grpc_impl::Server> server_;

  // gRPC service implementation (defined in master_server.cpp)
  class UMBPMasterServiceImpl;
  std::unique_ptr<UMBPMasterServiceImpl> service_;
};

}  // namespace mori::umbp
