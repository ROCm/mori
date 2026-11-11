// RDMA netlink (NLDEV) port hw counters.
//
// RDMA_NLDEV_CMD_STAT_GET with dev+port calls the driver's get_hw_stats directly, bypassing
// the ib_core sysfs `lifespan` cache, so every reply is a fresh snapshot (ionic: one admin
// command, ~1.7 ms). Unprivileged; works in non-privileged containers (rdma netns shared).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "source.h"

namespace netmon {

namespace nl {

// Parse one STAT_GET reply datagram. Pure function (unit tested with real captures).
//  - Skips messages whose nlmsg_seq != expect_seq (*stale = true) and returns Ok with
//    *found = false so the caller keeps receiving.
//  - NLMSG_ERROR becomes an error Status.
//  - If names != nullptr, appends counter names; values go to vals (up to max_vals).
//  - *count is the number of counters in the reply.
Status ParseStatGetReply(const uint8_t* buf, size_t len, uint32_t expect_seq, bool* found,
                         bool* stale, std::vector<std::string>* names, uint64_t* vals,
                         size_t max_vals, size_t* count);

// Build a STAT_GET request for (dev_index, port) with the given sequence number.
std::vector<uint8_t> BuildStatGetRequest(uint32_t dev_index, uint32_t port, uint32_t seq);

}  // namespace nl

class RdmaNetlinkSource : public Source {
 public:
  ~RdmaNetlinkSource() override;
  // Opens the socket only. Resolving the device and the first STAT_GET (which also yields
  // the counter names) happen in Probe(), on the sampler thread under the startup deadline.
  Status Open(const std::string& ibdev, uint32_t port, uint64_t query_timeout_ns);
  Status Probe(std::string* detail) override;

  const char* kind() const override { return "rdma"; }
  const std::vector<std::string>& names() const override { return names_; }
  double snapshot_fraction() const override { return 0.70; }
  Status Query(uint64_t* out) override;

 private:
  Status Exchange(std::vector<std::string>* names, uint64_t* vals, size_t max_vals, size_t* count);
  Status ResolveDevIndex(const std::string& ibdev, uint32_t* index);

  int fd_ = -1;
  std::string ibdev_;
  uint32_t dev_index_ = 0;
  uint32_t port_ = 1;
  uint32_t seq_ = 1;
  uint64_t timeout_ns_ = 2000000000ull;
  std::vector<uint8_t> rx_;
  std::vector<std::string> names_;
};

}  // namespace netmon
