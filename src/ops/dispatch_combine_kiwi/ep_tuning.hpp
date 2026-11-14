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
#include <stdexcept>
#include <string>
#include <string_view>

namespace mori::kiwi::ep {

// RCOMP payload backend. RDMA_ONLY uses RDMA payloads; HYBRID writes
// same-node peer payloads through HIP IPC and retains RDMA payloads
// plus receiver-owned readiness signals for all other peers. AUTO resolves the
// concrete backend independently for each active-token launch.
enum class rcomp_backend_t : int32_t {
  RDMA_ONLY = 0,
  HYBRID = 1,
  AUTO = 2,
};

inline const char* rcomp_backend_name(rcomp_backend_t backend) {
  switch (backend) {
    case rcomp_backend_t::RDMA_ONLY:
      return "rdma-only";
    case rcomp_backend_t::HYBRID:
      return "hybrid";
    case rcomp_backend_t::AUTO:
      return "auto";
  }
  return "unknown";
}

struct runtime_defaults_t {
  int num_blocks;
  rcomp_backend_t rcomp_backend;
};

// One measured launch setting. A rule applies to active token counts above
// the previous rule's max_num_tokens for the same world size, up to its own.
struct runtime_tuning_rule_t {
  int world_size;
  int max_num_tokens;
  int num_blocks;
  rcomp_backend_t rcomp_backend;
};

// Measured launch settings for one GPU architecture and NIC family. Rules are
// grouped by world size and sorted by max_num_tokens. Each table lives in its
// own file under tuning/ and is listed in kRuntimeTuningTables.
struct runtime_tuning_table_t {
  const char* gpu_arch;
  const char* nic_family;
  const runtime_tuning_rule_t* rules;
  size_t num_rules;
};

extern const runtime_tuning_table_t* const kRuntimeTuningTables[];
extern const size_t kNumRuntimeTuningTables;

const runtime_tuning_table_t* find_runtime_tuning_table(std::string_view gpu_arch,
                                                        std::string_view nic_family);

// Return the measured runtime defaults for a positive per-rank token count and
// world size. Context construction uses the maximum configured capacity to
// allocate enough queues and optional hybrid resources. Automatic dispatch
// and combine policy can then select a smaller launch from the active token
// count without changing those allocations.
//
// A tuning table for (gpu_arch, nic_family) takes precedence for the world
// sizes and token counts it measured; a token count between two rows uses the
// larger row. Everything else uses the generic rule, whose block bands combine
// a BF16 single-node (MI300X) one-token result, 1/2/4/8-node small-token
// sweeps, and the existing high-token bands. ws8 uses its measured backend
// threshold; every other topology uses the ws16-derived fallback threshold.
inline runtime_defaults_t recommended_runtime_defaults(int num_tokens, int world_size,
                                                       std::string_view gpu_arch = {},
                                                       std::string_view nic_family = {}) {
  if (num_tokens <= 0) {
    throw std::invalid_argument(
        "kiwi::ep::recommended_runtime_defaults: "
        "num_tokens must be positive");
  }

  if (const runtime_tuning_table_t* table = find_runtime_tuning_table(gpu_arch, nic_family)) {
    for (size_t i = 0; i < table->num_rules; ++i) {
      const runtime_tuning_rule_t& rule = table->rules[i];
      if (rule.world_size == world_size && num_tokens <= rule.max_num_tokens) {
        return {rule.num_blocks, rule.rcomp_backend};
      }
    }
  }

  const int num_blocks = num_tokens == 1    ? 12
                         : num_tokens == 2  ? 20
                         : num_tokens <= 32 ? 32
                         : num_tokens <= 64 ? 64
                                            : 256;

  const int hybrid_min_tokens = world_size == 8 ? 8 : 4;
  return {
      num_blocks,
      num_tokens >= hybrid_min_tokens ? rcomp_backend_t::HYBRID : rcomp_backend_t::RDMA_ONLY,
  };
}

inline int recommended_num_blocks(int num_tokens, int world_size, std::string_view gpu_arch = {},
                                  std::string_view nic_family = {}) {
  return recommended_runtime_defaults(num_tokens, world_size, gpu_arch, nic_family).num_blocks;
}

// gcnArchName of the current HIP device without feature suffixes, e.g.
// "gfx942". Returns an empty string if the device cannot be queried.
std::string current_gpu_arch();

// NIC family of an RDMA device: "pollara" for ionic_*, "cx7" for ConnectX-7,
// otherwise the device-name prefix (e.g. "mlx5"). Empty for an empty name.
std::string nic_family_for_hca(const std::string& hca_name,
                               const std::string& sysfs_root = "/sys/class/infiniband");

}  // namespace mori::kiwi::ep
