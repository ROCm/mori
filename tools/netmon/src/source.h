// A counter source: one (NIC, kind) pair, queried by exactly one sampler thread.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "util.h"

namespace netmon {

class Source {
 public:
  virtual ~Source() = default;
  virtual const char* kind() const = 0;  // "rdma" | "ipc" | "ethtool"
  virtual const std::vector<std::string>& names() const = 0;
  // Where in [t0, t1] the device snapshot is taken (evidence 02); written to meta.json.
  virtual double snapshot_fraction() const = 0;
  // Startup probe, run on the sampler thread (it may block in the driver). Default: one query.
  virtual Status Probe(std::string* detail);
  // Fill names().size() values. Must not allocate on the hot path.
  virtual Status Query(uint64_t* out) = 0;
};

inline Status Source::Probe(std::string* detail) {
  std::vector<uint64_t> tmp(names().size());
  Status st = Query(tmp.data());
  if (st.ok()) *detail = "ok";
  return st;
}

}  // namespace netmon
