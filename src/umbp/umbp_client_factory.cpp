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
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <string>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/distributed_client.h"
#include "umbp/standalone/standalone_process_client.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {

namespace {

// Identity for a client that will never register with anyone.  It still has to
// be unique and stable within the process, because it is what a route result is
// compared against to decide "this key is mine" -- and, more practically,
// because it is the string every log line from this client carries.  Several
// clients per process is the normal case (one per rank), hence the counter.
std::string EmbeddedNodeId() {
  static std::atomic<uint64_t> counter{0};
  char host[256] = {0};
  if (gethostname(host, sizeof(host) - 1) != 0) host[0] = '\0';
  const std::string hostname = host[0] != '\0' ? host : "localhost";
  return "embedded-" + hostname + "-" + std::to_string(static_cast<long>(getpid())) + "-" +
         std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

// Allocation granularity of the embedded DRAM pool.
//
// 2 MiB matches every other UMBP deployment, and matters more here than it
// looks: the pool is paged, so a value smaller than one page still consumes a
// whole one.  A caller storing KV pages sets this to the page's exact byte size
// (which is what the distributed deployments do) and gets one page per key; a
// caller storing small values wants it small.  There is no field on UMBPConfig
// for it -- the knob lives on UMBPDistributedConfig -- so a caller that has not
// configured a deployment at all reaches it through the environment.
uint64_t EmbeddedPageSize() {
  constexpr uint64_t kDefault = 2ULL * 1024 * 1024;
  const char* raw = std::getenv("UMBP_EMBEDDED_DRAM_PAGE_SIZE");
  if (raw == nullptr || raw[0] == '\0') return kDefault;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(raw, &end, 10);
  if (end == raw || parsed == 0) {
    MORI_UMBP_WARN("[UMBP] ignoring UMBP_EMBEDDED_DRAM_PAGE_SIZE='{}' (not a positive integer)",
                   raw);
    return kDefault;
  }
  return static_cast<uint64_t>(parsed);
}

}  // namespace

UMBPConfig WithEmbeddedDefaults(const UMBPConfig& config) {
  // An explicit deployment always wins; this function only fills a vacuum.
  if (config.distributed.has_value() || config.standalone_process.has_value()) return config;

  UMBPConfig out = config;
  UMBPDistributedConfig dist;

  // Everything that would need a peer is left empty, and each omission removes
  // a whole subsystem rather than merely disabling it: no master address means
  // no MasterClient, no registration and no heartbeat; no IO engine host means
  // MoriIoEngine is never constructed; peer_service_port = 0 means no gRPC
  // server is bound.  What remains is one medium plus the local copy engines.
  dist.master_config.master_address.clear();
  dist.master_config.node_id = EmbeddedNodeId();
  // Non-empty because Validate() requires it.  Nothing reads it: it is only
  // ever used to build the peer address this client would advertise, and it
  // advertises nothing.
  dist.master_config.node_address = "127.0.0.1";
  dist.master_config.auto_heartbeat = false;
  dist.io_engine.host.clear();
  dist.peer_service_port = 0;

  // ONE medium, and it is DRAM.  This is the deliberate alignment with the
  // distributed client: UMBP's routing plane does not tier within a node, so a
  // node serves exactly one medium (see UMBPMedium in common/config.h).  DRAM
  // rather than a rule derived from ssd.enabled, because that flag defaults to
  // true and would silently make every unconfigured caller an SSD node.
  // Serving SSD is an explicit choice -- set distributed.medium.
  dist.medium = UMBPMedium::DRAM;
  dist.dram_page_size = EmbeddedPageSize();

  // Nothing can be remote, so every remote-path resource is off: no ranged
  // scratch arenas to allocate or register, and no re-cache of fetches that
  // cannot happen.
  dist.ranged_scratch_size = 0;
  dist.cache_remote_fetches = false;
  dist.ranged_locality_prefetch = false;
  dist.local_first = true;

  if (out.ssd.enabled) {
    MORI_UMBP_WARN(
        "[UMBP] embedded deployment serves DRAM only; the configured SSD tier "
        "(dir='{}', {} MiB) is NOT served. A node serves one medium: set "
        "distributed.medium = SSD for an SSD node, or use a standalone server. "
        "DRAM+SSD tiering on one node returns with the multi-backend work.",
        out.ssd.storage_dir, out.ssd.capacity_bytes / (1024 * 1024));
  }

  MORI_UMBP_INFO("[UMBP] embedded deployment: node_id='{}' DRAM={} MiB page={} KiB (no master)",
                 dist.master_config.node_id, out.dram.capacity_bytes / (1024 * 1024),
                 dist.dram_page_size / 1024);

  out.distributed = dist;
  return out;
}

std::unique_ptr<IUMBPClient> CreateUMBPClient(const UMBPConfig& config) {
  if (config.standalone_process.has_value()) {
    return std::make_unique<standalone::StandaloneProcessClient>(config);
  }
  if (config.distributed.has_value()) {
    return std::make_unique<DistributedClient>(config);
  }
  // No deployment named: an in-process store, private to this client.  It is
  // the same class serving a cluster member -- what makes it embedded is the
  // absence of a master, not a different implementation.
  return std::make_unique<DistributedClient>(WithEmbeddedDefaults(config));
}

}  // namespace mori::umbp
