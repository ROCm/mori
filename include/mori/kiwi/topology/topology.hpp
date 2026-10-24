#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mori::kiwi::p2p {
namespace topology {

// Return one selected CPU per proxy thread, or -1 when no usable CPU can be
// selected.  CPUs are chosen from the current HIP device's KFD-local CPU slice
// when it intersects the process affinity mask; otherwise they are partitioned
// across the currently allowed mask.
std::vector<int> choose_proxy_thread_cpus(int num_proxy_threads);

void pin_proxy_thread_to_cpu(int thread_id, int num_proxy_threads,
                             int selected_cpu);

// Return the nearest active HCA/NIC name for a HIP GPU id.  The mapping is
// computed across all visible GPUs and balances exact ties so ranks do not all
// bind to the same NIC when several equally-near HCAs are available.
std::string get_hca_name_for_gpu(int gpu_id);
std::string get_hca_name_for_current_gpu();
std::string get_requested_or_auto_hca_name_for_current_gpu();

} // namespace topology
} // namespace kiwi::p2p
