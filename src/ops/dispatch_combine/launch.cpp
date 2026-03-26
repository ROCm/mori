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
//
// C++ launch wrappers for EP dispatch/combine kernels.
// Compiled with a standard C++ compiler (no hipcc needed).
// Loads precompiled .hsaco files via hipModuleLaunchKernel.

#include "mori/ops/dispatch_combine/launch.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <deque>
#include <cassert>
#include <cctype>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>
#endif

#include "mori/utils/mori_log.hpp"

// Forward-declare ShmemModuleInit to avoid pulling in device headers
namespace mori {
namespace shmem {
int64_t ShmemModuleInit(void* hip_module);
}
}  // namespace mori

namespace mori {
namespace moe {

static constexpr int WARP_SIZE = 64;

// -----------------------------------------------------------------------
// KernelRegistry implementation
// -----------------------------------------------------------------------
struct KernelRegistry::Impl {
  std::mutex mu;
  std::vector<hipModule_t> modules;
  std::unordered_map<std::string, hipFunction_t> func_cache;
  bool loaded = false;
};

KernelRegistry::Impl& KernelRegistry::GetImpl() {
#ifdef MORI_MULTITHREAD_SUPPORT 
  static std::mutex mutex_;
  static std::deque<KernelRegistry::Impl> impls_(8); // 8 GPUs per node
  int id = -1;
  HIP_RUNTIME_CHECK(hipGetDevice(&id));
  std::lock_guard<std::mutex> lock(mutex_);
  if (impls_.size() <= static_cast<size_t>(id)) {
    impls_.resize(id + 1);
  }
  return impls_[id];
#else
  static Impl impl;
  return impl;
#endif // MORI_MULTITHREAD_SUPPORT
}

KernelRegistry& KernelRegistry::Instance() {
  static KernelRegistry instance;
  return instance;
}

bool KernelRegistry::IsLoaded() const { return GetImpl().loaded; }

void KernelRegistry::LoadModule(const std::string& hsaco_path) {
  auto& impl = GetImpl();
  std::lock_guard<std::mutex> lock(impl.mu);

  hipModule_t mod;
  hipError_t err = hipModuleLoad(&mod, hsaco_path.c_str());
  if (err != hipSuccess) {
    throw std::runtime_error("Failed to load hsaco: " + hsaco_path + " (" + hipGetErrorString(err) +
                             ")");
  }
  impl.modules.push_back(mod);
  // Initialize shmem globalGpuStates in this module
  mori::shmem::ShmemModuleInit(reinterpret_cast<void*>(mod));
}

void KernelRegistry::LoadFromDirectory(const std::string& dir) {
  auto& impl = GetImpl();
  namespace fs = std::filesystem;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".hsaco") {
      LoadModule(entry.path().string());
      MORI_OPS_INFO("Loaded kernel module: {}", entry.path().string());
    }
  }
  impl.loaded = true;
}

// -----------------------------------------------------------------------
// Hardware detection (cached, mirrors Python detect_nic_type / CMake detect_device_nic)
// -----------------------------------------------------------------------

static std::string s_cached_arch;
static std::string s_cached_nic;
static bool s_hw_detected = false;

static std::string classify_ib_device(const std::string& dev_path) {
  namespace fs = std::filesystem;
  auto driver_link = fs::path(dev_path) / "device" / "driver";
  try {
    std::string driver = fs::read_symlink(driver_link).filename().string();
    if (driver == "bnxt_re" || driver == "bnxt_en") return "bnxt";
    if (driver == "mlx5_core" || driver == "mlx5_ib") return "mlx5";
    if (driver == "ionic_rdma" || driver == "ionic") return "ionic";
  } catch (...) {
  }
  return "";
}

static bool has_nic_lib(const std::string& nic) {
  static const std::unordered_map<std::string, std::string> lib_names = {
      {"mlx5", "libmlx5.so"}, {"bnxt", "libbnxt_re.so"}, {"ionic", "libionic.so"}};
  static const std::vector<std::string> search_paths = {
      "/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"};

  auto it = lib_names.find(nic);
  if (it == lib_names.end()) return false;
  for (const auto& dir : search_paths) {
    if (std::filesystem::exists(std::filesystem::path(dir) / it->second)) return true;
  }
  return false;
}

static void detect_hardware() {
  if (s_hw_detected) return;
  namespace fs = std::filesystem;

  // GPU arch
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
    s_cached_arch = std::string(props.gcnArchName);
    auto colon = s_cached_arch.find(':');
    if (colon != std::string::npos) s_cached_arch = s_cached_arch.substr(0, colon);
  }

  // NIC detection — same priority as Python detect_nic_type() and CMake detect_device_nic()

  // Priority 1: MORI_DEVICE_NIC env override
  const char* env = std::getenv("MORI_DEVICE_NIC");
  if (env) {
    std::string nic(env);
    for (auto& c : nic) c = static_cast<char>(std::tolower(c));
    if (nic == "bnxt" || nic == "ionic" || nic == "mlx5") {
      s_cached_nic = nic;
      s_hw_detected = true;
      return;
    }
  }

  // Priority 2: /sys/class/infiniband/ — count devices by type, verify library
  const std::string ib_dir = "/sys/class/infiniband";
  if (fs::is_directory(ib_dir)) {
    std::unordered_map<std::string, int> counts = {{"mlx5", 0}, {"bnxt", 0}, {"ionic", 0}};
    for (const auto& entry : fs::directory_iterator(ib_dir)) {
      std::string name = entry.path().filename().string();
      if (name.find("bnxt_re") == 0)
        counts["bnxt"]++;
      else if (name.find("ionic") == 0)
        counts["ionic"]++;
      else if (name.find("mlx5") == 0)
        counts["mlx5"]++;
      else {
        std::string nic = classify_ib_device(entry.path().string());
        if (!nic.empty()) counts[nic]++;
      }
    }
    // Pick NIC with most devices + verify lib exists (tie-break: mlx5 > bnxt > ionic)
    std::vector<std::pair<std::string, int>> sorted_nics(counts.begin(), counts.end());
    std::sort(sorted_nics.begin(), sorted_nics.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second > b.second;
      // Tie-break order
      static const std::unordered_map<std::string, int> prio = {
          {"mlx5", 0}, {"bnxt", 1}, {"ionic", 2}};
      return prio.at(a.first) < prio.at(b.first);
    });
    for (const auto& [nic, cnt] : sorted_nics) {
      if (cnt > 0 && has_nic_lib(nic)) {
        s_cached_nic = nic;
        s_hw_detected = true;
        return;
      }
    }
  }

  // Priority 3: userspace library fallback
  for (const auto& nic : {"mlx5", "bnxt", "ionic"}) {
    if (has_nic_lib(nic)) {
      s_cached_nic = nic;
      s_hw_detected = true;
      return;
    }
  }

  // Default
  s_cached_nic = "mlx5";
  s_hw_detected = true;
}

void KernelRegistry::AutoLoad(const std::string& base_dir) {
  namespace fs = std::filesystem;
  detect_hardware();
  const std::string& arch = s_cached_arch;
  const std::string& nic = s_cached_nic;
  std::string arch_nic = arch + "_" + nic;

  if (!arch.empty() && fs::is_directory(base_dir)) {
    // 1. Exact match: <arch>_<nic>
    auto exact = fs::path(base_dir) / arch_nic;
    if (fs::is_directory(exact)) {
      MORI_OPS_INFO("AutoLoad: exact match {}", arch_nic);
      LoadFromDirectory(exact.string());
      return;
    }

    // 2. Arch prefix match: <arch>_*
    for (const auto& entry : fs::directory_iterator(base_dir)) {
      if (entry.is_directory()) {
        std::string name = entry.path().filename().string();
        if (name.find(arch + "_") == 0) {
          MORI_OPS_INFO("AutoLoad: arch match {} (wanted {})", name, arch_nic);
          LoadFromDirectory(entry.path().string());
          return;
        }
      }
    }
  }

  // 3. Fallback: load directly from base_dir (flat layout)
  if (fs::is_directory(base_dir)) {
    LoadFromDirectory(base_dir);
    return;
  }
}

// Resolve the directory containing libmori_ops.so at runtime via dladdr.
static std::string get_self_lib_dir() {
#ifdef __linux__
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&get_self_lib_dir), &info) && info.dli_fname) {
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }
#endif
  return "";
}

void KernelRegistry::AutoLoad() {
  namespace fs = std::filesystem;
  detect_hardware();
  const std::string& arch = s_cached_arch;
  const std::string& nic = s_cached_nic;
  if (arch.empty()) return;

  std::string arch_nic = arch + "_" + nic;

  // 1. MORI_KERNEL_DIR env override
  const char* env_dir = std::getenv("MORI_KERNEL_DIR");
  if (env_dir && fs::is_directory(env_dir)) {
    MORI_OPS_INFO("AutoLoad: using MORI_KERNEL_DIR={}", env_dir);
    AutoLoad(std::string(env_dir));
    if (IsLoaded()) return;
  }

  // 2. Relative to libmori_ops.so: <so_dir>/../lib/<arch>_<nic>/ (build layout)
  //    and <so_dir>/<arch>_<nic>/ (install layout)
  std::string so_dir = get_self_lib_dir();
  if (!so_dir.empty()) {
    auto build_path = fs::path(so_dir) / ".." / "lib" / arch_nic;
    if (fs::is_directory(build_path)) {
      MORI_OPS_INFO("AutoLoad: found build kernels at {}", build_path.string());
      LoadFromDirectory(fs::canonical(build_path).string());
      return;
    }
    auto install_path = fs::path(so_dir) / arch_nic;
    if (fs::is_directory(install_path)) {
      MORI_OPS_INFO("AutoLoad: found installed kernels at {}", install_path.string());
      LoadFromDirectory(install_path.string());
      return;
    }
  }

  // 3. JIT cache: ~/.mori/jit/<arch>_<nic>/latest/
  const char* home = std::getenv("HOME");
  if (home) {
    auto jit_latest = fs::path(home) / ".mori" / "jit" / arch_nic / "latest";
    if (fs::is_directory(jit_latest)) {
      MORI_OPS_INFO("AutoLoad: found JIT cache at {}", jit_latest.string());
      LoadFromDirectory(jit_latest.string());
      return;
    }
  }
}

hipFunction_t KernelRegistry::GetFunction(const std::string& func_name) {
  auto& impl = GetImpl();
  std::lock_guard<std::mutex> lock(impl.mu);

  auto it = impl.func_cache.find(func_name);
  if (it != impl.func_cache.end()) return it->second;

  // Search all loaded modules for the function
  for (auto& mod : impl.modules) {
    hipFunction_t func;
    hipError_t err = hipModuleGetFunction(&func, mod, func_name.c_str());
    if (err == hipSuccess) {
      // Clear any sticky error left by prior failed hipModuleGetFunction calls
      (void)hipGetLastError();
      impl.func_cache[func_name] = func;
      return func;
    }
  }

  throw std::runtime_error("Kernel function not found in any loaded module: " + func_name +
                           ". Ensure BUILD_OPS_DEVICE=ON and .hsaco files are installed.");
}

void KernelRegistry::Launch(const std::string& func_name, unsigned int grid_x, unsigned int block_x,
                            unsigned int shared_mem, hipStream_t stream, void* args,
                            size_t args_size) {
  hipFunction_t func = GetFunction(func_name);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                    HIP_LAUNCH_PARAM_END};
  hipError_t err =
      hipModuleLaunchKernel(func, grid_x, 1, 1, block_x, 1, 1, shared_mem, stream, nullptr, config);
  if (err != hipSuccess) {
    throw std::runtime_error("hipModuleLaunchKernel failed for " + func_name + ": " +
                             hipGetErrorString(err));
  }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static const char* dtype_suffix(hipDataType dtype) {
  switch (dtype) {
    case HIP_R_32F:
      return "f32";
    case HIP_R_16BF:
      return "bf16";
#if HIP_VERSION >= 60000000
    case HIP_R_8F_E4M3_FNUZ:
      return "fp8_fnuz";
    case HIP_R_8F_E4M3:
      return "fp8_ocp";
#endif
#if __has_include(<hip/hip_ext_ocp.h>)
    case HIP_R_4F_E2M1:
      return "fp4";
#endif
    default:
      throw std::runtime_error("Unsupported dtype for kernel launch");
  }
}

static int dispatch_shared_mem(const EpDispatchCombineConfig& cfg, int wpb) {
  return (cfg.worldSize * wpb + cfg.numExpertPerRank * wpb + cfg.numExpertPerRank) *
         static_cast<int>(sizeof(index_t));
}

static int combine_shared_mem(int wpb, int num_experts_per_token) {
  // warpPerBlock * numExpertPerToken * (sizeof(ptr) + sizeof(ptr))
  return wpb * num_experts_per_token * (8 + 8);
}

static void ensure_loaded() {
  if (!KernelRegistry::Instance().IsLoaded()) {
    KernelRegistry::Instance().AutoLoad();

    if (!KernelRegistry::Instance().IsLoaded()) {
      throw std::runtime_error(
          "KernelRegistry: no precompiled kernels found. Either:\n"
          "  1. Build with -DBUILD_OPS_DEVICE=ON and call AutoLoad(\"lib/\")\n"
          "  2. Run 'MORI_PRECOMPILE=1 python -c \"import mori\"' to populate JIT cache\n"
          "  3. Call LoadFromDirectory(path) or LoadModule(path) manually");
    }
  }
}

// -----------------------------------------------------------------------
// LaunchDispatch
// -----------------------------------------------------------------------
void LaunchDispatch(EpDispatchCombineHandle& handle, void* input, void* weights, void* scales,
                    void* indices, int64_t num_tokens, hipDataType dtype, int block_num,
                    int rdma_block_num, int warp_per_block, hipStream_t stream, int hidden_dim) {
  ensure_loaded();

  handle.PrepareInference(dtype, input, nullptr, reinterpret_cast<float*>(weights),
                          reinterpret_cast<uint8_t*>(scales), reinterpret_cast<index_t*>(indices),
                          num_tokens);

  int wpb = (warp_per_block <= 0) ? handle.config.warpNumPerBlock : warp_per_block;
  int bn = (block_num <= 0) ? handle.config.blockNum : block_num;
  int rbn = (rdma_block_num <= 0) ? handle.config.rdmaBlockNum : rdma_block_num;

  EpDispatchCombineArgsRaw args = GetEpDispatchCombineArgsRaw(handle, rbn);
  if (hidden_dim > 0) {
    args.config.hiddenDim = hidden_dim;
    handle.curHiddenDim = hidden_dim;
  }

  unsigned int block_x = WARP_SIZE * wpb;
  int smem = dispatch_shared_mem(handle.config, wpb);
  size_t args_size = sizeof(EpDispatchCombineArgsRaw);
  const char* sfx = dtype_suffix(dtype);
  auto& reg = KernelRegistry::Instance();

  switch (handle.config.kernelType) {
    case KernelType::IntraNode:
      reg.Launch(std::string("EpDispatchIntraNodeKernel_") + sfx, bn, block_x, smem, stream, &args,
                 args_size);
      break;
    case KernelType::InterNode:
      reg.Launch(std::string("EpDispatchInterNodeKernel_") + sfx, bn, block_x, smem, stream, &args,
                 args_size);
      break;
    case KernelType::InterNodeV1:
      reg.Launch(std::string("EpDispatchCopyToStaging_") + sfx, handle.multiProcessorCount, block_x,
                 0, stream, &args, args_size);
      reg.Launch(std::string("EpDispatchInterNodeV1Kernel_") + sfx, bn, block_x, smem, stream,
                 &args, args_size);
      break;
    case KernelType::InterNodeV1LL:
      reg.Launch(std::string("EpDispatchCopyToStaging_") + sfx, handle.multiProcessorCount, block_x,
                 0, stream, &args, args_size);
      reg.Launch(std::string("EpDispatchInterNodeV1KernelLowLatency_") + sfx, bn, block_x, smem,
                 stream, &args, args_size);
      break;
    case KernelType::AsyncLL:
      reg.Launch(std::string("EpDispatchLowLatencyAsyncSend_") + sfx, bn, block_x, smem, stream,
                 &args, args_size);
      break;
    default:
      throw std::runtime_error("Unsupported dispatch kernel_type");
  }
}

// -----------------------------------------------------------------------
// LaunchCombine
// -----------------------------------------------------------------------
void LaunchCombine(EpDispatchCombineHandle& handle, void* input, void* weights, void* indices,
                   int64_t num_tokens, hipDataType dtype, int block_num, int rdma_block_num,
                   int warp_per_block, int use_external_inp_buf, hipStream_t stream,
                   int hidden_dim) {
  ensure_loaded();

  handle.PrepareInference(dtype, input, nullptr, reinterpret_cast<float*>(weights),
                          reinterpret_cast<index_t*>(indices), num_tokens);

  int wpb = (warp_per_block <= 0) ? handle.config.warpNumPerBlock : warp_per_block;
  int bn = (block_num <= 0) ? handle.config.blockNum : block_num;
  int rbn = (rdma_block_num <= 0) ? handle.config.rdmaBlockNum : rdma_block_num;

  EpDispatchCombineArgsRaw args = GetEpDispatchCombineArgsRaw(handle, rbn);
  if (hidden_dim > 0) {
    args.config.hiddenDim = hidden_dim;
    handle.curHiddenDim = hidden_dim;
  }
  if (use_external_inp_buf >= 0) {
    args.config.useExternalInpBuffer = static_cast<bool>(use_external_inp_buf);
  }

  unsigned int block_x = WARP_SIZE * wpb;
  int smem = combine_shared_mem(wpb, handle.config.numExpertPerToken);
  size_t args_size = sizeof(EpDispatchCombineArgsRaw);
  const char* sfx = dtype_suffix(dtype);
  auto& reg = KernelRegistry::Instance();
  int mp = handle.multiProcessorCount;

  switch (handle.config.kernelType) {
    case KernelType::IntraNode:
      if (args.config.useExternalInpBuffer) {
        reg.Launch(std::string("EpCombineIntraNodeKernel_") + sfx + "_nop2p", bn, block_x, smem,
                   stream, &args, args_size);
      } else {
        reg.Launch(std::string("EpCombineIntraNodeKernel_") + sfx + "_p2p", bn, block_x, smem,
                   stream, &args, args_size);
      }
      break;
    case KernelType::InterNode:
      reg.Launch(std::string("EpCombineInterNodeKernel_") + sfx, bn, block_x, smem, stream, &args,
                 args_size);
      break;
    case KernelType::InterNodeV1:
      reg.Launch(std::string("EpCombineSync_") + sfx, mp, block_x, 0, stream, &args, args_size);
      reg.Launch(std::string("EpCombineSyncBarrier_") + sfx, 1, WARP_SIZE, 0, stream, &args,
                 args_size);
      reg.Launch(std::string("EpCombineInterNodeV1Kernel_") + sfx, bn, block_x, smem, stream, &args,
                 args_size);
      reg.Launch(std::string("EpCombineAll_") + sfx, mp, block_x, smem, stream, &args, args_size);
      break;
    case KernelType::InterNodeV1LL:
      reg.Launch(std::string("EpCombineSync_") + sfx, mp, block_x, 0, stream, &args, args_size);
      reg.Launch(std::string("EpCombineSyncBarrier_") + sfx, 1, WARP_SIZE, 0, stream, &args,
                 args_size);
      reg.Launch(std::string("EpCombineInterNodeV1KernelLowLatency_") + sfx, bn, block_x, smem,
                 stream, &args, args_size);
      reg.Launch(std::string("EpCombineAll_") + sfx, mp, block_x, smem, stream, &args, args_size);
      break;
    case KernelType::AsyncLL:
      reg.Launch(std::string("EpCombineLowLatencyAsyncSend_") + sfx, bn, block_x, smem, stream,
                 &args, args_size);
      break;
    default:
      throw std::runtime_error("Unsupported combine kernel_type");
  }
}

// -----------------------------------------------------------------------
// LaunchDispatchRecv
// -----------------------------------------------------------------------
void LaunchDispatchRecv(EpDispatchCombineHandle& handle, int block_num, int warp_per_block,
                        hipStream_t stream) {
  ensure_loaded();

  int wpb = (warp_per_block <= 0) ? handle.config.warpNumPerBlock : warp_per_block;
  int bn = (block_num <= 0) ? handle.config.blockNum : block_num;

  EpDispatchCombineArgsRaw args = GetEpDispatchCombineArgsRaw(handle, 0);
  if (handle.curHiddenDim > 0) args.config.hiddenDim = handle.curHiddenDim;

  unsigned int block_x = WARP_SIZE * wpb;
  int smem = dispatch_shared_mem(handle.config, wpb);
  size_t args_size = sizeof(EpDispatchCombineArgsRaw);
  const char* sfx = dtype_suffix(handle.inputType);

  if (handle.config.kernelType == KernelType::AsyncLL) {
    KernelRegistry::Instance().Launch(std::string("EpDispatchLowLatencyAsyncRecv_") + sfx, bn,
                                      block_x, smem, stream, &args, args_size);
  } else {
    throw std::runtime_error("LaunchDispatchRecv only supported for AsyncLL");
  }
}

// -----------------------------------------------------------------------
// LaunchCombineRecv
// -----------------------------------------------------------------------
void LaunchCombineRecv(EpDispatchCombineHandle& handle, int block_num, int warp_per_block,
                       hipStream_t stream) {
  ensure_loaded();

  int wpb = (warp_per_block <= 0) ? handle.config.warpNumPerBlock : warp_per_block;
  int bn = (block_num <= 0) ? handle.config.blockNum : block_num;

  EpDispatchCombineArgsRaw args = GetEpDispatchCombineArgsRaw(handle, 0);
  if (handle.curHiddenDim > 0) args.config.hiddenDim = handle.curHiddenDim;

  unsigned int block_x = WARP_SIZE * wpb;
  int smem = combine_shared_mem(wpb, handle.config.numExpertPerToken);
  size_t args_size = sizeof(EpDispatchCombineArgsRaw);
  const char* sfx = dtype_suffix(handle.inputType);

  if (handle.config.kernelType == KernelType::AsyncLL) {
    KernelRegistry::Instance().Launch(std::string("EpCombineLowLatencyAsyncRecv_") + sfx, bn,
                                      block_x, smem, stream, &args, args_size);
  } else {
    throw std::runtime_error("LaunchCombineRecv only supported for AsyncLL");
  }
}

// -----------------------------------------------------------------------
// LaunchLocalExpertCount
// -----------------------------------------------------------------------
void LaunchLocalExpertCount(const EpDispatchCombineConfig& config, const index_t* indices,
                            const index_t* total_recv_token_num, int* local_expert_count,
                            int block_num, int warp_per_block, hipStream_t stream) {
  ensure_loaded();

  if (indices == nullptr || total_recv_token_num == nullptr || local_expert_count == nullptr) {
    throw std::runtime_error(
        "LaunchLocalExpertCount requires non-null indices, total_recv_token_num, and output");
  }

  const int wpb = (warp_per_block <= 0) ? config.warpNumPerBlock : warp_per_block;
  const int bn = (block_num <= 0) ? config.blockNum : block_num;
  if (wpb <= 0 || bn <= 0) {
    throw std::runtime_error("LaunchLocalExpertCount requires positive block and warp settings");
  }

  const hipError_t memset_err = hipMemsetAsync(
      local_expert_count, 0, static_cast<size_t>(config.numExpertPerRank) * sizeof(int), stream);
  if (memset_err != hipSuccess) {
    throw std::runtime_error("hipMemsetAsync failed for LaunchLocalExpertCount: " +
                             std::string(hipGetErrorString(memset_err)));
  }

  LocalExpertCountArgs args{indices,
                            total_recv_token_num,
                            config.rank,
                            config.numExpertPerRank,
                            config.numExpertPerToken,
                            local_expert_count};
  KernelRegistry::Instance().Launch("LocalExpertCountKernel", static_cast<unsigned int>(bn),
                                    WARP_SIZE * static_cast<unsigned int>(wpb), 0, stream, &args,
                                    sizeof(args));
}

// -----------------------------------------------------------------------
// LaunchReset
// -----------------------------------------------------------------------
void LaunchReset(EpDispatchCombineHandle& handle, hipStream_t stream) {
  handle.LaunchReset(stream);
}

}  // namespace moe
}  // namespace mori
