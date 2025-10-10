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
#include "mori/io/backend.hpp"
#include "mori/io/engine.hpp"
#include "mori/io/fallback_backend.hpp"
#include "mori/io/logging.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "src/io/tcp/backend_impl.hpp"

namespace mori {
namespace io {

// Simple factory to create a single backend or a fallback chain.
// If both rdmaCfg and tcpCfg are provided (non-null ptr), we build RDMA primary + TCP secondary.
std::unique_ptr<Backend> CreateBackendWithOptionalFallback(EngineKey key,
                                                           const IOEngineConfig& engCfg,
                                                           const RdmaBackendConfig* rdmaCfg,
                                                           const TcpBackendConfig* tcpCfg,
                                                           bool enableFallback) {
  if (rdmaCfg && tcpCfg && enableFallback) {
    auto rdma = std::make_unique<RdmaBackend>(key, engCfg, *rdmaCfg);
    auto tcp = std::make_unique<TcpBackend>(key, engCfg, *tcpCfg);
    MORI_IO_INFO("CreateBackend: using FallbackBackend (primary=RDMA, secondary=TCP)");
    return std::make_unique<FallbackBackend>(std::move(rdma), std::move(tcp));
  }
  if (rdmaCfg) {
    MORI_IO_INFO("CreateBackend: using RDMA backend only");
    return std::make_unique<RdmaBackend>(key, engCfg, *rdmaCfg);
  }
  if (tcpCfg) {
    MORI_IO_INFO("CreateBackend: using TCP backend only");
    return std::make_unique<TcpBackend>(key, engCfg, *tcpCfg);
  }
  MORI_IO_ERROR("CreateBackend: no backend config provided");
  return nullptr;
}

}  // namespace io
}  // namespace mori
