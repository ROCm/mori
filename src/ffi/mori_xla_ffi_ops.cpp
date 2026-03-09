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
// Based on PR #173 by Chao Chen <cchen104@amd.com>
// Adapted for the refactored architecture (raw pointer args + hipModuleLaunchKernel).

#include "xla/ffi/api/ffi.h"

#include <hip/hip_runtime.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ffi/mori_xla_ffi_handle_mgr.hpp"

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Kernel module management
// ---------------------------------------------------------------------------

static constexpr int WARP_SIZE = 64;
static constexpr int PTR_SIZE = 8;

class KernelManager {
 public:
  static KernelManager& Instance() {
    static KernelManager instance;
    return instance;
  }

  void RegisterModule(int kernel_type, const std::string& hsaco_path) {
    std::lock_guard<std::mutex> lock(mu_);
    hipModule_t mod;
    hipError_t err = hipModuleLoad(&mod, hsaco_path.c_str());
    if (err != hipSuccess) {
      throw std::runtime_error("Failed to load hsaco: " + hsaco_path +
                               " (" + hipGetErrorString(err) + ")");
    }
    modules_[kernel_type] = mod;
  }

  hipModule_t GetModule(int kernel_type) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = modules_.find(kernel_type);
    if (it == modules_.end()) return nullptr;
    return it->second;
  }

  hipFunction_t GetFunction(int kernel_type, const std::string& name) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = modules_.find(kernel_type);
    if (it == modules_.end()) {
      throw std::runtime_error(
          "No kernel module registered for kernel_type=" +
          std::to_string(kernel_type) +
          ". Call mori_ffi_register_kernel_module() first.");
    }
    auto key = std::make_pair(kernel_type, name);
    auto fit = func_cache_.find(name + "@" + std::to_string(kernel_type));
    if (fit != func_cache_.end()) return fit->second;

    hipFunction_t func;
    hipError_t err = hipModuleGetFunction(&func, it->second, name.c_str());
    if (err != hipSuccess) {
      throw std::runtime_error("Failed to get function '" + name +
                               "': " + hipGetErrorString(err));
    }
    func_cache_[name + "@" + std::to_string(kernel_type)] = func;
    return func;
  }

 private:
  KernelManager() = default;
  std::mutex mu_;
  std::unordered_map<int, hipModule_t> modules_;
  std::unordered_map<std::string, hipFunction_t> func_cache_;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static hipDataType FfiDataTypeToHip(ffi::DataType dtype) {
  switch (dtype) {
    case ffi::DataType::F32:  return HIP_R_32F;
    case ffi::DataType::F16:  return HIP_R_16F;
    case ffi::DataType::BF16: return HIP_R_16BF;
    default:
      throw std::runtime_error("mori FFI: unsupported element type");
  }
}

static const char* HipDataTypeToSuffix(hipDataType dtype) {
  switch (dtype) {
    case HIP_R_32F:          return "f32";
    case HIP_R_16BF:         return "bf16";
    case HIP_R_8F_E4M3:      return "fp8_ocp";
    case HIP_R_8F_E4M3_FNUZ: return "fp8_fnuz";
    default:
      throw std::runtime_error("mori FFI: unsupported dtype for kernel suffix");
  }
}

static int DispatchSharedMem(const mori::moe::EpDispatchCombineConfig& cfg, int warp_per_block) {
  return (cfg.worldSize * warp_per_block +
          cfg.numExpertPerRank * warp_per_block +
          cfg.numExpertPerRank) * 4;
}

static int CombineSharedMem(const mori::moe::EpDispatchCombineConfig& cfg, int warp_per_block) {
  return warp_per_block * cfg.numExpertPerToken * (PTR_SIZE + PTR_SIZE);
}

static void LaunchKernel(int kernel_type, const std::string& func_name,
                         dim3 grid, dim3 block, int shared_mem,
                         hipStream_t stream, void* args_ptr) {
  hipFunction_t func = KernelManager::Instance().GetFunction(kernel_type, func_name);
  void* config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, args_ptr,
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &shared_mem,
      HIP_LAUNCH_PARAM_END};
  size_t args_size = sizeof(mori::moe::EpDispatchCombineArgsRaw);
  void* launch_params[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, args_ptr,
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
      HIP_LAUNCH_PARAM_END};
  hipError_t err = hipModuleLaunchKernel(
      func, grid.x, grid.y, grid.z, block.x, block.y, block.z,
      shared_mem, stream, nullptr, launch_params);
  if (err != hipSuccess) {
    throw std::runtime_error("hipModuleLaunchKernel failed for '" + func_name +
                             "': " + hipGetErrorString(err));
  }
}

// ---------------------------------------------------------------------------
// mori_ep_dispatch
// ---------------------------------------------------------------------------
static ffi::Error MoriEpDispatchImpl(
    ffi::AnyBuffer input, ffi::AnyBuffer topk_ids,
    ffi::Result<ffi::AnyBuffer> out_tokens,
    ffi::Result<ffi::AnyBuffer> out_weights,
    ffi::Result<ffi::AnyBuffer> out_indices,
    ffi::Result<ffi::AnyBuffer> total_recv_token_num,
    int64_t handle_id, int64_t block_num, int64_t rdma_block_num,
    int64_t warp_per_block, int64_t hidden_dim) {
  try {
    auto* handle = mori::ffi::HandleManager::Instance().GetHandle(handle_id);
    auto hip_dtype = FfiDataTypeToHip(input.element_type());
    int num_tokens = static_cast<int>(input.dimensions()[0]);
    int h_dim = (hidden_dim > 0) ? static_cast<int>(hidden_dim)
                                 : static_cast<int>(input.dimensions()[1]);

    handle->PrepareInference(
        hip_dtype, input.untyped_data(), nullptr,
        static_cast<float*>(out_weights->untyped_data()),
        nullptr,
        static_cast<mori::moe::index_t*>(topk_ids.untyped_data()),
        num_tokens);

    auto args = mori::moe::GetEpDispatchCombineArgsRaw(*handle, static_cast<int>(rdma_block_num));
    if (hidden_dim > 0) args.config.hiddenDim = static_cast<int>(hidden_dim);

    hipStream_t stream = nullptr;
    int kt = static_cast<int>(handle->config.kernelType);
    const char* sfx = HipDataTypeToSuffix(hip_dtype);
    int bn = (block_num > 0) ? static_cast<int>(block_num) : handle->config.blockNum;
    int wpb = (warp_per_block > 0) ? static_cast<int>(warp_per_block)
                                   : handle->config.warpNumPerBlock;
    dim3 grid(bn);
    dim3 block(WARP_SIZE * wpb);
    int shmem = DispatchSharedMem(handle->config, wpb);

    using KT = mori::moe::KernelType;
    switch (static_cast<KT>(kt)) {
      case KT::IntraNode:
        LaunchKernel(kt, std::string("EpDispatchIntraNodeKernel_") + sfx,
                     grid, block, shmem, stream, &args);
        break;
      case KT::InterNode:
        LaunchKernel(kt, std::string("EpDispatchInterNodeKernel_") + sfx,
                     grid, block, shmem, stream, &args);
        break;
      case KT::InterNodeV1: {
        dim3 mp_grid(handle->multiProcessorCount);
        LaunchKernel(kt, std::string("EpDispatchCopyToStaging_") + sfx,
                     mp_grid, block, 0, stream, &args);
        LaunchKernel(kt, std::string("EpDispatchInterNodeV1Kernel_") + sfx,
                     grid, block, shmem, stream, &args);
        break;
      }
      case KT::InterNodeV1LL: {
        dim3 mp_grid(handle->multiProcessorCount);
        LaunchKernel(kt, std::string("EpDispatchCopyToStaging_") + sfx,
                     mp_grid, block, 0, stream, &args);
        LaunchKernel(kt, std::string("EpDispatchInterNodeV1KernelLowLatency_") + sfx,
                     grid, block, shmem, stream, &args);
        break;
      }
      case KT::AsyncLL:
        LaunchKernel(kt, std::string("EpDispatchLowLatencyAsyncSend_") + sfx,
                     grid, block, shmem, stream, &args);
        break;
      default:
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "Unsupported dispatch kernel_type");
    }

    return ffi::Error::Success();
  } catch (const std::exception& e) {
    return ffi::Error(ffi::ErrorCode::kInternal, std::string(e.what()));
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mori_ep_dispatch, MoriEpDispatchImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()   // input
        .Arg<ffi::AnyBuffer>()   // topk_ids
        .Ret<ffi::AnyBuffer>()   // out_tokens
        .Ret<ffi::AnyBuffer>()   // out_weights
        .Ret<ffi::AnyBuffer>()   // out_indices
        .Ret<ffi::AnyBuffer>()   // total_recv_token_num
        .Attr<int64_t>("handle_id")
        .Attr<int64_t>("block_num")
        .Attr<int64_t>("rdma_block_num")
        .Attr<int64_t>("warp_per_block")
        .Attr<int64_t>("hidden_dim"));

// ---------------------------------------------------------------------------
// mori_ep_combine
// ---------------------------------------------------------------------------
static ffi::Error MoriEpCombineImpl(
    ffi::AnyBuffer input, ffi::AnyBuffer topk_ids,
    ffi::Result<ffi::AnyBuffer> out_tokens,
    int64_t handle_id, int64_t block_num, int64_t rdma_block_num,
    int64_t warp_per_block, int64_t use_external_inp_buf, int64_t hidden_dim) {
  try {
    auto* handle = mori::ffi::HandleManager::Instance().GetHandle(handle_id);
    auto hip_dtype = FfiDataTypeToHip(input.element_type());
    int h_dim = (hidden_dim > 0) ? static_cast<int>(hidden_dim)
                                 : static_cast<int>(input.dimensions()[1]);

    handle->PrepareInference(
        hip_dtype, input.untyped_data(), nullptr,
        nullptr, nullptr,
        static_cast<mori::moe::index_t*>(topk_ids.untyped_data()),
        handle->curRankNumToken);

    auto args = mori::moe::GetEpDispatchCombineArgsRaw(*handle, static_cast<int>(rdma_block_num));
    if (hidden_dim > 0) args.config.hiddenDim = static_cast<int>(hidden_dim);
    if (use_external_inp_buf >= 0)
      args.config.useExternalInpBuffer = static_cast<bool>(use_external_inp_buf);

    hipStream_t stream = nullptr;
    int kt = static_cast<int>(handle->config.kernelType);
    const char* sfx = HipDataTypeToSuffix(hip_dtype);
    int bn = (block_num > 0) ? static_cast<int>(block_num) : handle->config.blockNum;
    int wpb = (warp_per_block > 0) ? static_cast<int>(warp_per_block)
                                   : handle->config.warpNumPerBlock;
    dim3 grid(bn);
    dim3 block(WARP_SIZE * wpb);
    int shmem = CombineSharedMem(handle->config, wpb);

    bool ext = (use_external_inp_buf >= 0)
                   ? static_cast<bool>(use_external_inp_buf)
                   : handle->config.useExternalInpBuffer;

    using KT = mori::moe::KernelType;
    switch (static_cast<KT>(kt)) {
      case KT::IntraNode:
        if (ext) {
          LaunchKernel(kt, std::string("EpCombineIntraNodeKernel_") + sfx + "_nop2p",
                       grid, block, shmem, stream, &args);
        } else {
          LaunchKernel(kt, std::string("EpCombineIntraNodeKernel_") + sfx + "_p2p",
                       grid, block, shmem, stream, &args);
        }
        break;
      case KT::InterNode:
        LaunchKernel(kt, std::string("EpCombineInterNodeKernel_") + sfx,
                     grid, block, shmem, stream, &args);
        break;
      case KT::InterNodeV1: {
        dim3 mp_grid(handle->multiProcessorCount);
        LaunchKernel(kt, std::string("EpCombineSync_") + sfx,
                     mp_grid, block, 0, stream, &args);
        LaunchKernel(kt, std::string("EpCombineSyncBarrier_") + sfx,
                     dim3(1), dim3(WARP_SIZE), 0, stream, &args);
        LaunchKernel(kt, std::string("EpCombineInterNodeV1Kernel_") + sfx,
                     grid, block, shmem, stream, &args);
        LaunchKernel(kt, std::string("EpCombineAll_") + sfx,
                     mp_grid, block, shmem, stream, &args);
        break;
      }
      case KT::InterNodeV1LL: {
        dim3 mp_grid(handle->multiProcessorCount);
        LaunchKernel(kt, std::string("EpCombineSync_") + sfx,
                     mp_grid, block, 0, stream, &args);
        LaunchKernel(kt, std::string("EpCombineSyncBarrier_") + sfx,
                     dim3(1), dim3(WARP_SIZE), 0, stream, &args);
        LaunchKernel(kt, std::string("EpCombineInterNodeV1KernelLowLatency_") + sfx,
                     grid, block, shmem, stream, &args);
        LaunchKernel(kt, std::string("EpCombineAll_") + sfx,
                     mp_grid, block, shmem, stream, &args);
        break;
      }
      case KT::AsyncLL:
        LaunchKernel(kt, std::string("EpCombineLowLatencyAsyncSend_") + sfx,
                     grid, block, shmem, stream, &args);
        break;
      default:
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "Unsupported combine kernel_type");
    }

    return ffi::Error::Success();
  } catch (const std::exception& e) {
    return ffi::Error(ffi::ErrorCode::kInternal, std::string(e.what()));
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mori_ep_combine, MoriEpCombineImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()   // input
        .Arg<ffi::AnyBuffer>()   // topk_ids
        .Ret<ffi::AnyBuffer>()   // out_tokens
        .Attr<int64_t>("handle_id")
        .Attr<int64_t>("block_num")
        .Attr<int64_t>("rdma_block_num")
        .Attr<int64_t>("warp_per_block")
        .Attr<int64_t>("use_external_inp_buf")
        .Attr<int64_t>("hidden_dim"));

// ---------------------------------------------------------------------------
// mori_ep_dispatch_recv  (AsyncLL only)
// ---------------------------------------------------------------------------
static ffi::Error MoriEpDispatchRecvImpl(
    int64_t handle_id, int64_t block_num, int64_t warp_per_block) {
  try {
    auto* handle = mori::ffi::HandleManager::Instance().GetHandle(handle_id);
    int kt = static_cast<int>(handle->config.kernelType);
    if (static_cast<mori::moe::KernelType>(kt) != mori::moe::KernelType::AsyncLL) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "dispatch_recv only supports AsyncLL");
    }
    auto args = mori::moe::GetEpDispatchCombineArgsRaw(*handle, 0);
    int bn = (block_num > 0) ? static_cast<int>(block_num) : handle->config.blockNum;
    int wpb = (warp_per_block > 0) ? static_cast<int>(warp_per_block)
                                   : handle->config.warpNumPerBlock;
    const char* sfx = HipDataTypeToSuffix(handle->inputType);
    hipStream_t stream = nullptr;
    LaunchKernel(kt, std::string("EpDispatchLowLatencyAsyncRecv_") + sfx,
                 dim3(bn), dim3(WARP_SIZE * wpb),
                 DispatchSharedMem(handle->config, wpb), stream, &args);
    return ffi::Error::Success();
  } catch (const std::exception& e) {
    return ffi::Error(ffi::ErrorCode::kInternal, std::string(e.what()));
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mori_ep_dispatch_recv, MoriEpDispatchRecvImpl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("handle_id")
        .Attr<int64_t>("block_num")
        .Attr<int64_t>("warp_per_block"));

// ---------------------------------------------------------------------------
// mori_ep_combine_recv  (AsyncLL only)
// ---------------------------------------------------------------------------
static ffi::Error MoriEpCombineRecvImpl(
    int64_t handle_id, int64_t block_num, int64_t warp_per_block) {
  try {
    auto* handle = mori::ffi::HandleManager::Instance().GetHandle(handle_id);
    int kt = static_cast<int>(handle->config.kernelType);
    if (static_cast<mori::moe::KernelType>(kt) != mori::moe::KernelType::AsyncLL) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "combine_recv only supports AsyncLL");
    }
    auto args = mori::moe::GetEpDispatchCombineArgsRaw(*handle, 0);
    int bn = (block_num > 0) ? static_cast<int>(block_num) : handle->config.blockNum;
    int wpb = (warp_per_block > 0) ? static_cast<int>(warp_per_block)
                                   : handle->config.warpNumPerBlock;
    const char* sfx = HipDataTypeToSuffix(handle->inputType);
    hipStream_t stream = nullptr;
    LaunchKernel(kt, std::string("EpCombineLowLatencyAsyncRecv_") + sfx,
                 dim3(bn), dim3(WARP_SIZE * wpb),
                 CombineSharedMem(handle->config, wpb), stream, &args);
    return ffi::Error::Success();
  } catch (const std::exception& e) {
    return ffi::Error(ffi::ErrorCode::kInternal, std::string(e.what()));
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mori_ep_combine_recv, MoriEpCombineRecvImpl,
    ffi::Ffi::Bind()
        .Attr<int64_t>("handle_id")
        .Attr<int64_t>("block_num")
        .Attr<int64_t>("warp_per_block"));

// ---------------------------------------------------------------------------
// mori_ep_reset
// ---------------------------------------------------------------------------
static ffi::Error MoriEpResetImpl(int64_t handle_id) {
  try {
    auto* handle = mori::ffi::HandleManager::Instance().GetHandle(handle_id);
    handle->LaunchReset(nullptr);
    return ffi::Error::Success();
  } catch (const std::exception& e) {
    return ffi::Error(ffi::ErrorCode::kInternal, std::string(e.what()));
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mori_ep_reset, MoriEpResetImpl,
    ffi::Ffi::Bind().Attr<int64_t>("handle_id"));

// ---------------------------------------------------------------------------
// Handle lifecycle + kernel module registration (plain C for ctypes)
// ---------------------------------------------------------------------------
extern "C" {

int64_t mori_ffi_create_handle(
    int rank, int world_size, int hidden_dim,
    int scale_dim, int scale_type_size,
    int max_token_type_size, int max_num_inp_token_per_rank,
    int num_experts_per_rank, int num_experts_per_token,
    int warp_num_per_block, int block_num,
    int kernel_type, int gpu_per_node,
    int rdma_block_num, int num_qp_per_pe) {
  mori::moe::EpDispatchCombineConfig config;
  config.rank = rank;
  config.worldSize = world_size;
  config.hiddenDim = hidden_dim;
  config.scaleDim = scale_dim;
  config.scaleTypeSize = scale_type_size;
  config.maxTokenTypeSize = max_token_type_size;
  config.maxNumInpTokenPerRank = max_num_inp_token_per_rank;
  config.numExpertPerRank = num_experts_per_rank;
  config.numExpertPerToken = num_experts_per_token;
  config.warpNumPerBlock = warp_num_per_block;
  config.blockNum = block_num;
  config.kernelType = static_cast<mori::moe::KernelType>(kernel_type);
  config.gpuPerNode = gpu_per_node;
  config.rdmaBlockNum = rdma_block_num;
  config.numQpPerPe = num_qp_per_pe;
  return mori::ffi::HandleManager::Instance().CreateHandle(config);
}

void mori_ffi_destroy_handle(int64_t handle_id) {
  mori::ffi::HandleManager::Instance().DestroyHandle(handle_id);
}

void mori_ffi_register_kernel_module(int kernel_type, const char* hsaco_path) {
  KernelManager::Instance().RegisterModule(kernel_type, hsaco_path);
}

void mori_ffi_shmem_module_init_from_kernel(int kernel_type) {
  hipModule_t mod = KernelManager::Instance().GetModule(kernel_type);
  if (mod) {
    mori::shmem::ShmemModuleInit(reinterpret_cast<void*>(mod));
  }
}

// ---------------------------------------------------------------------------
// Shmem C API (torch-free, for JAX-only usage via ctypes)
// ---------------------------------------------------------------------------

int mori_ffi_shmem_get_unique_id(void* uid_out, int* uid_size) {
  mori::shmem::mori_shmem_uniqueid_t uid;
  mori::shmem::ShmemGetUniqueId(&uid);
  std::memcpy(uid_out, uid.data(), uid.size());
  *uid_size = static_cast<int>(uid.size());
  return 0;
}

int64_t mori_ffi_shmem_init_attr(unsigned int flags, int rank, int nranks,
                                  const void* uid_data, int uid_size) {
  mori::shmem::mori_shmem_init_attr_t attr;
  mori::shmem::mori_shmem_uniqueid_t uid;
  std::memcpy(uid.data(), uid_data, uid_size);
  mori::shmem::ShmemSetAttrUniqueIdArgs(rank, nranks, &uid, &attr);
  return mori::shmem::ShmemInitAttr(flags, &attr);
}

int64_t mori_ffi_shmem_finalize() { return mori::shmem::ShmemFinalize(); }

int64_t mori_ffi_shmem_module_init(uint64_t hip_module) {
  return mori::shmem::ShmemModuleInit(reinterpret_cast<void*>(hip_module));
}

int64_t mori_ffi_load_shmem_module(const char* hsaco_path) {
  return mori::shmem::LoadShmemModule(hsaco_path);
}

int mori_ffi_shmem_mype() { return mori::shmem::ShmemMyPe(); }
int mori_ffi_shmem_npes() { return mori::shmem::ShmemNPes(); }
void mori_ffi_shmem_barrier_all() { mori::shmem::ShmemBarrierAll(); }

void mori_ffi_shmem_barrier_on_stream(int64_t stream) {
  mori::shmem::ShmemBarrierOnStream(reinterpret_cast<hipStream_t>(stream));
}

uint64_t mori_ffi_shmem_malloc(uint64_t size) {
  return reinterpret_cast<uint64_t>(mori::shmem::ShmemMalloc(size));
}

void mori_ffi_shmem_free(uint64_t ptr) {
  mori::shmem::ShmemFree(reinterpret_cast<void*>(ptr));
}

int64_t mori_ffi_shmem_buffer_register(uint64_t ptr, uint64_t size) {
  return mori::shmem::ShmemBufferRegister(reinterpret_cast<void*>(ptr), size);
}

int64_t mori_ffi_shmem_buffer_deregister(uint64_t ptr, uint64_t size) {
  return mori::shmem::ShmemBufferDeregister(reinterpret_cast<void*>(ptr), size);
}

uint64_t mori_ffi_shmem_ptr_p2p(uint64_t dest_ptr, int my_pe, int dest_pe) {
  return mori::shmem::ShmemPtrP2p(dest_ptr, my_pe, dest_pe);
}

int mori_ffi_shmem_num_qp_per_pe() { return mori::shmem::ShmemNumQpPerPe(); }

unsigned int mori_ffi_shmem_init_flag_uniqueid() {
  return mori::shmem::MORI_SHMEM_INIT_WITH_UNIQUEID;
}

}  // extern "C"
