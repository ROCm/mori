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

// Fallback allocation granularity of the embedded DRAM pool, used only when the
// caller expressed no preference.
//
// This matters more than it looks: the pool is paged, so a value smaller than
// one page still consumes a whole one.  A caller storing KV pages sets the size
// to the page's exact byte size (which is what the distributed deployments do)
// and gets one page per key; a caller storing small values wants it small.  The
// knob is UMBPDistributedConfig::dram_page_size, which WithEmbeddedDefaults
// honours for an embedded pool too, so this is reached only by a caller who set
// neither that field nor the environment override below.
uint64_t EmbeddedPageSize() {
  // 64 KiB rather than the 2 MiB every other deployment uses.  Those are told
  // the KV page's exact byte size and genuinely get one page per key; a caller
  // who named no deployment has told us nothing, and paying 2 MiB for a value
  // that may be a few hundred bytes silently evicts a pool that had ample room.
  // Small enough that a 4 MiB pool holds 64 values instead of 8, large enough
  // that a multi-megabyte value still spans few enough pages for the allocator
  // to find a contiguous run.  Callers storing real KV pages should say so --
  // set distributed.dram_page_size, which this function no longer overrides.
  constexpr uint64_t kDefault = 64ULL * 1024;
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

// Shrink the page to something the pool can actually hold.
//
// The pool is paged and the allocator hands out whole pages, so a page larger
// than the pool means ZERO pages -- and then every put fails with NO_SPACE,
// silently, because a failed put is a legal answer rather than an error.  The
// deleted local backend allocated exact sizes and had no such cliff, so a
// small pool that used to work would simply stop working.  Fit instead, and
// say so.
uint64_t FitPageSizeToPool(uint64_t requested, uint64_t capacity_bytes) {
  constexpr uint64_t kMinPages = 8;  // a pool that holds fewer is not a cache
  constexpr uint64_t kMinPageSize = 4096;
  if (capacity_bytes == 0 || requested <= capacity_bytes / kMinPages) return requested;

  uint64_t fitted = kMinPageSize;
  while (fitted * 2 <= capacity_bytes / kMinPages) fitted *= 2;
  if (fitted >= requested) return requested;
  MORI_UMBP_WARN(
      "[UMBP] embedded page size {} KiB does not fit a {} MiB pool; using {} KiB "
      "so the pool holds at least {} pages. Size the pool for the page (or set "
      "UMBP_EMBEDDED_DRAM_PAGE_SIZE) to choose deliberately.",
      requested / 1024, capacity_bytes / (1024 * 1024), fitted / 1024, kMinPages);
  return fitted;
}

}  // namespace

UMBPConfig WithEmbeddedDefaults(const UMBPConfig& config) {
  // A standalone deployment is a different client entirely -- the pool lives in
  // the server process -- so there is nothing here to fill in for it.
  if (config.standalone_process.has_value()) return config;

  // Whether the store is embedded (in this process) and whether it has a master
  // are independent axes, and DistributedClient already treats them that way:
  // it derives local_only_ from master_address being empty and has no notion of
  // "embedded" at all.  So a caller who supplies distributed config is NOT
  // opting out of in-process operation -- they are only opting out of having
  // every field chosen for them.  Fill the blanks they left rather than
  // refusing to help, which is what forced a caller who wanted to set one field
  // (dram_page_size, say) to hand-build the whole struct.
  //
  // A master address is the one thing that does change the answer: that node is
  // a cluster member whose settings are the deployment's, not ours, so leave it
  // exactly as given.
  const bool synthesized = !config.distributed.has_value();
  UMBPConfig out = config;
  UMBPDistributedConfig dist = synthesized ? UMBPDistributedConfig{} : config.distributed.value();
  if (!dist.master_config.master_address.empty()) return config;

  // Identity, which nothing but this function has any basis to choose.
  if (dist.master_config.node_id.empty()) dist.master_config.node_id = EmbeddedNodeId();
  // Non-empty because Validate() requires it.  Nothing reads it: it is only
  // ever used to build the peer address this client would advertise, and with
  // no master it advertises nothing.
  if (dist.master_config.node_address.empty()) dist.master_config.node_address = "127.0.0.1";
  // Not a preference: with no master there is no MasterClient to beat to, so a
  // heartbeat thread would be pure cost.  The struct default is true because
  // the common case is a cluster member.
  dist.master_config.auto_heartbeat = false;

  // Page size, where 0 means "no opinion".  Both the default and an explicit
  // value are fitted to the pool: the cliff FitPageSizeToPool guards against --
  // a page bigger than the pool means ZERO pages and every put failing with
  // NO_SPACE, silently -- does not care who chose the number, and honouring a
  // deliberate-but-impossible value literally would be a worse answer than
  // fitting it and saying so.  Fitting is a no-op for anything that already
  // leaves the pool at least kMinPages.
  //
  // Only for a pool no peer shares, though: peers in one tier must agree on
  // page_size (see UMBPDistributedConfig::dram_page_size), and a value derived
  // from THIS node's capacity would disagree with a peer sized differently, so
  // an agreed size must survive untouched.  peer_service_port == 0 means no
  // gRPC peer service is bound, hence no peers to disagree with.
  if (dist.peer_service_port == 0) {
    const uint64_t requested = dist.dram_page_size > 0 ? dist.dram_page_size : EmbeddedPageSize();
    dist.dram_page_size = FitPageSizeToPool(requested, out.dram.capacity_bytes);
  }

  if (synthesized) {
    // Only reachable when the caller named no deployment at all, so there is no
    // choice of theirs to overwrite.  Everything that would need a peer is left
    // empty, and each omission removes a whole subsystem rather than merely
    // disabling it: no IO engine host means MoriIoEngine is never constructed;
    // peer_service_port = 0 means no gRPC server is bound.  What remains is one
    // medium plus the local copy engines.
    dist.io_engine.host.clear();
    dist.peer_service_port = 0;

    // ONE medium, and it is DRAM.  This is the deliberate alignment with the
    // distributed client: UMBP's routing plane does not tier within a node, so
    // a node serves exactly one medium (see UMBPMedium in common/config.h).
    // DRAM rather than a rule derived from ssd.enabled, because that flag
    // defaults to true and would silently make every unconfigured caller an SSD
    // node.  Serving SSD is an explicit choice -- set distributed.medium.
    dist.medium = UMBPMedium::DRAM;

    // Nothing can be remote, so every remote-path resource is off: no ranged
    // scratch arenas to allocate or register, and no re-cache of fetches that
    // cannot happen.
    dist.ranged_scratch_size = 0;
    dist.cache_remote_fetches = false;
    dist.ranged_locality_prefetch = false;
    dist.local_first = true;
  }

  if (synthesized && out.ssd.enabled) {
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
  // One class for every in-process store, whether it names a master, names none
  // but configures itself, or names nothing at all.  WithEmbeddedDefaults fills
  // whatever was left blank and returns a cluster member's config untouched.
  return std::make_unique<DistributedClient>(WithEmbeddedDefaults(config));
}

}  // namespace mori::umbp
