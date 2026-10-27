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
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include "ep_context.hip.hpp"
#include "hip_check.hpp"
#if __has_include(<hip/hip_ext_ocp.h>)
#include <hip/hip_ext_ocp.h>
#endif
#include <cstdio>

// Diagnostic-only compile-time toggles for issue #10 payload-bandwidth
// experiments.  The default build keeps normal correctness-preserving behavior.
// Enabling either toggle is intentionally timing-only: hidden outputs become
// invalid, but slot metadata is still transferred so the benchmark can keep
// exercising the routing/control path without crashing.
#ifndef KIWI_EP_DIAG_SKIP_HIDDEN_COPY
#define KIWI_EP_DIAG_SKIP_HIDDEN_COPY 0
#endif

#ifndef KIWI_EP_DIAG_METADATA_ONLY_PUT
#define KIWI_EP_DIAG_METADATA_ONLY_PUT 0
#endif

namespace mori::kiwi::ep {

__device__ inline char* combine_put_src_ptr(const ep_gcontext_t& gctx, char* slot) {
#if KIWI_EP_DIAG_METADATA_ONLY_PUT
  return slot + gctx.combine_layout.slot.weights_offset;
#else
  return slot;
#endif
}

__device__ inline uintptr_t combine_put_remote_disp(const ep_gcontext_t& gctx, int recv_slot) {
  uintptr_t remote_disp = gctx.combine_remote_disp(recv_slot);
#if KIWI_EP_DIAG_METADATA_ONLY_PUT
  remote_disp += gctx.combine_layout.slot.weights_offset;
#endif
  return remote_disp;
}

__device__ inline size_t combine_put_bytes(const ep_gcontext_t& gctx) {
#if KIWI_EP_DIAG_METADATA_ONLY_PUT
  return gctx.combine_layout.slot.payload_size - gctx.combine_layout.slot.weights_offset;
#else
  return gctx.combine_layout.slot.stride;
#endif
}

__device__ inline uint8_t* dispatch_payload_readiness_flag(const ep_gcontext_t& gctx,
                                                           int source_rank, int source_local_slot) {
  return gctx.dispatch_readiness_flags() +
         gctx.payload_readiness_flag_index(source_rank, source_local_slot);
}

__device__ inline uint8_t* combine_payload_readiness_flag(const ep_gcontext_t& gctx,
                                                          int producer_rank,
                                                          int source_local_slot) {
  return gctx.combine_readiness_flags() +
         gctx.payload_readiness_flag_index(producer_rank, source_local_slot);
}

template <bool SystemScopeStore, bool MetadataSystemScopeLoad = false>
__device__ inline void combine_pack_slot_part(const ep_gcontext_t& gctx, const char* src_hidden,
                                              const float* src_weights, int topk,
                                              size_t hidden_offset, int hidden_bytes,
                                              char* dst_slot, bool copy_weights, int thread_idx,
                                              int thread_count) {
  if (!KIWI_EP_DIAG_SKIP_HIDDEN_COPY) {
    void* dst_hidden =
        static_cast<char*>(gctx.combine_layout.slot.hidden(dst_slot)) + hidden_offset;
    copy_hidden_payload_block<SystemScopeStore>(src_hidden, dst_hidden, hidden_bytes, thread_idx,
                                                thread_count);
  }
  if (copy_weights) {
    copy_bytes_block<SystemScopeStore, MetadataSystemScopeLoad>(
        src_weights, gctx.combine_layout.slot.weights(dst_slot),
        topk * static_cast<int>(sizeof(float)), thread_idx, thread_count);
  }
}

__device__ inline void debug_store_time(uint64_t* debug_time_buf, int idx) {
  if (debug_time_buf != nullptr) {
    debug_time_buf[idx] = static_cast<uint64_t>(wall_clock64());
  }
}

__device__ inline void debug_store_meta(int32_t* debug_time_meta, int idx, int32_t value) {
  if (debug_time_meta != nullptr) {
    debug_time_meta[idx] = value;
  }
}

__device__ inline bool is_aligned_16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xf) == 0;
}

__device__ inline float fp4_e2m1fn_to_float(uint8_t raw) {
#if defined(__HIPCC__) && __has_include(<hip/hip_ext_ocp.h>)
  return fcbx::to_float<float, fcbx::Encoding::E2M1, true>(raw & 0x0f, 0);
#else
  const uint8_t mag = raw & 0x7;
  float value;
  switch (mag) {
    case 0:
      value = 0.0f;
      break;
    case 1:
      value = 0.5f;
      break;
    case 2:
      value = 1.0f;
      break;
    case 3:
      value = 1.5f;
      break;
    case 4:
      value = 2.0f;
      break;
    case 5:
      value = 3.0f;
      break;
    case 6:
      value = 4.0f;
      break;
    default:
      value = 6.0f;
      break;
  }
  return (raw & 0x8) ? -value : value;
#endif
}

__device__ inline uint8_t float_to_fp4_e2m1fn(float value) {
#if defined(__HIPCC__) && __has_include(<hip/hip_ext_ocp.h>)
  return fcbx::from_float<float, fcbx::Encoding::E2M1, true>(value, 0);
#else
  const bool neg = value < 0.0f;
  float a = neg ? -value : value;
  uint8_t mag;
  if (a < 0.25f) {
    mag = 0;
  } else if (a < 0.75f) {
    mag = 1;
  } else if (a < 1.25f) {
    mag = 2;
  } else if (a < 1.75f) {
    mag = 3;
  } else if (a < 2.5f) {
    mag = 4;
  } else if (a < 3.5f) {
    mag = 5;
  } else if (a < 5.0f) {
    mag = 6;
  } else {
    mag = 7;
  }
  return (neg && mag != 0) ? static_cast<uint8_t>(mag | 0x8) : mag;
#endif
}

struct alignas(16) bf16_vec8_storage_t {
  hip_bfloat16 v[8];
};

struct alignas(16) byte_vec16_storage_t {
  uint8_t v[16];
};

__device__ inline float fp8_e4m3_fnuz_to_float(uint8_t raw) {
#if HIP_FP8_TYPE_FNUZ
  __hip_fp8_e4m3_fnuz value;
  value.__x = raw;
  return static_cast<float>(value);
#else
  return ::internal::cast_from_f8<float, true>(static_cast<__hip_fp8_storage_t>(raw), 3, 4);
#endif
}

__device__ inline uint8_t float_to_fp8_e4m3_fnuz(float value) {
#if HIP_FP8_TYPE_FNUZ
  return __hip_fp8_e4m3_fnuz(value).__x;
#else
  return ::internal::cast_to_f8<float, true>(value, 3, 4, true);
#endif
}

__device__ inline void reduce_float4_token(const ep_gcontext_t& gctx, int token,
                                           int combine_hidden_elems, int num_valid,
                                           const int* valid_producer_rank, void* d_output,
                                           int dim_offset, int dim_count, int thread_idx,
                                           int thread_count) {
  float* out =
      reinterpret_cast<float*>(d_output) + static_cast<size_t>(token) * combine_hidden_elems;
  const int range_begin = dim_offset;
  const int range_count = dim_count;
  const int range_end = range_begin + range_count;
  const bool can_vectorize =
      ((range_begin & 0x3) == 0) && ((range_count & 0x3) == 0) && is_aligned_16(out + range_begin);

  if (can_vectorize) {
    const int nvec = range_count / 4;
    float4* out4 = reinterpret_cast<float4*>(out + range_begin);
    for (int vi = thread_idx; vi < nvec; vi += thread_count) {
      float4 acc{0.0f, 0.0f, 0.0f, 0.0f};
      for (int u = 0; u < num_valid; u++) {
        char* slot_ptr = gctx.combine_recv_slot(valid_producer_rank[u], token);
        const float* hidden_data =
            reinterpret_cast<const float*>(gctx.combine_layout.slot.hidden(slot_ptr)) + range_begin;
        const bool remote_slot = valid_producer_rank[u] != gctx.rank_me;
        if (is_aligned_16(hidden_data)) {
          uint4 raw = remote_slot ? ep_load<true>(reinterpret_cast<const uint4*>(hidden_data) + vi)
                                  : reinterpret_cast<const uint4*>(hidden_data)[vi];
          const float4 v = *reinterpret_cast<const float4*>(&raw);
          acc.x += v.x;
          acc.y += v.y;
          acc.z += v.z;
          acc.w += v.w;
        } else {
          const int d = vi * 4;
          acc.x += remote_slot ? ep_load<true>(&hidden_data[d + 0]) : hidden_data[d + 0];
          acc.y += remote_slot ? ep_load<true>(&hidden_data[d + 1]) : hidden_data[d + 1];
          acc.z += remote_slot ? ep_load<true>(&hidden_data[d + 2]) : hidden_data[d + 2];
          acc.w += remote_slot ? ep_load<true>(&hidden_data[d + 3]) : hidden_data[d + 3];
        }
      }
      if (num_valid > 0) {
        out4[vi] = acc;
      }
    }
  } else {
    for (int d = range_begin + thread_idx; d < range_end; d += thread_count) {
      float acc = 0.0f;
      for (int u = 0; u < num_valid; u++) {
        char* slot_ptr = gctx.combine_recv_slot(valid_producer_rank[u], token);
        const float* hidden_data =
            reinterpret_cast<const float*>(gctx.combine_layout.slot.hidden(slot_ptr));
        acc += (valid_producer_rank[u] != gctx.rank_me) ? ep_load<true>(&hidden_data[d])
                                                        : hidden_data[d];
      }
      if (num_valid > 0) {
        out[d] = acc;
      }
    }
  }
}

__device__ inline void reduce_bf16x8_token(const ep_gcontext_t& gctx, int token,
                                           int combine_hidden_elems, int num_valid,
                                           const int* valid_producer_rank, void* d_output,
                                           int dim_offset, int dim_count, int thread_idx,
                                           int thread_count) {
  hip_bfloat16* out16 =
      reinterpret_cast<hip_bfloat16*>(d_output) + static_cast<size_t>(token) * combine_hidden_elems;
  const int range_begin = dim_offset;
  const int range_count = dim_count;
  const int range_end = range_begin + range_count;
  const bool can_vectorize = ((range_begin & 0x7) == 0) && ((range_count & 0x7) == 0) &&
                             is_aligned_16(out16 + range_begin);

  if (can_vectorize) {
    const int nvec = range_count / 8;
    uint4* out4 = reinterpret_cast<uint4*>(out16 + range_begin);
    for (int vi = thread_idx; vi < nvec; vi += thread_count) {
      float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int u = 0; u < num_valid; u++) {
        char* slot_ptr = gctx.combine_recv_slot(valid_producer_rank[u], token);
        const hip_bfloat16* hidden_data =
            reinterpret_cast<const hip_bfloat16*>(gctx.combine_layout.slot.hidden(slot_ptr)) +
            range_begin;
        if (is_aligned_16(hidden_data)) {
          const bool remote_slot = valid_producer_rank[u] != gctx.rank_me;
          uint4 raw = remote_slot ? ep_load<true>(reinterpret_cast<const uint4*>(hidden_data) + vi)
                                  : reinterpret_cast<const uint4*>(hidden_data)[vi];
          const hip_bfloat16* vals = reinterpret_cast<const hip_bfloat16*>(&raw);
#pragma unroll
          for (int i = 0; i < 8; i++) {
            acc[i] += static_cast<float>(vals[i]);
          }
        } else {
          const int d = vi * 8;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            acc[i] += static_cast<float>((valid_producer_rank[u] != gctx.rank_me)
                                             ? ep_load<true>(&hidden_data[d + i])
                                             : hidden_data[d + i]);
          }
        }
      }
      if (num_valid > 0) {
        bf16_vec8_storage_t tmp;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          tmp.v[i] = hip_bfloat16(acc[i]);
        }
        out4[vi] = *reinterpret_cast<const uint4*>(&tmp);
      }
    }
  } else {
    for (int d = range_begin + thread_idx; d < range_end; d += thread_count) {
      float acc = 0.0f;
      for (int u = 0; u < num_valid; u++) {
        char* slot_ptr = gctx.combine_recv_slot(valid_producer_rank[u], token);
        const hip_bfloat16* hidden_data =
            reinterpret_cast<const hip_bfloat16*>(gctx.combine_layout.slot.hidden(slot_ptr));
        acc += static_cast<float>((valid_producer_rank[u] != gctx.rank_me)
                                      ? ep_load<true>(&hidden_data[d])
                                      : hidden_data[d]);
      }
      if (num_valid > 0) {
        out16[d] = hip_bfloat16(acc);
      }
    }
  }
}

__device__ inline void reduce_fp8x16_token(const ep_gcontext_t& gctx, int token,
                                           int combine_hidden_elems, int num_valid,
                                           const int* valid_producer_rank, void* d_output,
                                           int dim_offset, int dim_count, int thread_idx,
                                           int thread_count) {
  uint8_t* out8 =
      reinterpret_cast<uint8_t*>(d_output) + static_cast<size_t>(token) * combine_hidden_elems;
  const int range_begin = dim_offset;
  const int range_count = dim_count;
  const int range_end = range_begin + range_count;
  const bool can_vectorize =
      ((range_begin & 0xf) == 0) && ((range_count & 0xf) == 0) && is_aligned_16(out8 + range_begin);

  if (can_vectorize) {
    const int nvec = range_count / 16;
    uint4* out4 = reinterpret_cast<uint4*>(out8 + range_begin);
    for (int vi = thread_idx; vi < nvec; vi += thread_count) {
      float acc[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int u = 0; u < num_valid; u++) {
        char* slot_ptr = gctx.combine_recv_slot(valid_producer_rank[u], token);
        const uint8_t* hidden_data =
            reinterpret_cast<const uint8_t*>(gctx.combine_layout.slot.hidden(slot_ptr)) +
            range_begin;
        if (is_aligned_16(hidden_data)) {
          const bool remote_slot = valid_producer_rank[u] != gctx.rank_me;
          const uint4 raw = remote_slot
                                ? ep_load<true>(reinterpret_cast<const uint4*>(hidden_data) + vi)
                                : reinterpret_cast<const uint4*>(hidden_data)[vi];
          const uint8_t* vals = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
          for (int i = 0; i < 16; i++) {
            acc[i] += fp8_e4m3_fnuz_to_float(vals[i]);
          }
        } else {
          const int d = vi * 16;
#pragma unroll
          for (int i = 0; i < 16; i++) {
            const uint8_t raw = (valid_producer_rank[u] != gctx.rank_me)
                                    ? ep_load<true>(&hidden_data[d + i])
                                    : hidden_data[d + i];
            acc[i] += fp8_e4m3_fnuz_to_float(raw);
          }
        }
      }
      if (num_valid > 0) {
        byte_vec16_storage_t tmp;
#pragma unroll
        for (int i = 0; i < 16; i++) {
          tmp.v[i] = float_to_fp8_e4m3_fnuz(acc[i]);
        }
        out4[vi] = *reinterpret_cast<const uint4*>(&tmp);
      }
    }
  } else {
    for (int d = range_begin + thread_idx; d < range_end; d += thread_count) {
      float acc = 0.0f;
      for (int u = 0; u < num_valid; u++) {
        char* slot_ptr = gctx.combine_recv_slot(valid_producer_rank[u], token);
        const uint8_t* hidden_data =
            reinterpret_cast<const uint8_t*>(gctx.combine_layout.slot.hidden(slot_ptr));
        const uint8_t raw = (valid_producer_rank[u] != gctx.rank_me)
                                ? ep_load<true>(&hidden_data[d])
                                : hidden_data[d];
        acc += fp8_e4m3_fnuz_to_float(raw);
      }
      if (num_valid > 0) {
        out8[d] = float_to_fp8_e4m3_fnuz(acc);
      }
    }
  }
}

__device__ inline void reduce_fp4x16_token(const ep_gcontext_t& gctx, int token,
                                           int combine_hidden_elems, int num_valid,
                                           const int* valid_producer_rank, void* d_output,
                                           int dim_offset, int dim_count, int thread_idx,
                                           int thread_count) {
  uint8_t* out8 =
      reinterpret_cast<uint8_t*>(d_output) + static_cast<size_t>(token) * combine_hidden_elems;
  const int range_begin = dim_offset;
  const int range_count = dim_count;
  const int range_end = range_begin + range_count;
  const bool can_vectorize =
      ((range_begin & 0xf) == 0) && ((range_count & 0xf) == 0) && is_aligned_16(out8 + range_begin);

  if (can_vectorize) {
    const int nvec = range_count / 16;
    uint4* out4 = reinterpret_cast<uint4*>(out8 + range_begin);
    for (int vi = thread_idx; vi < nvec; vi += thread_count) {
      float lo_acc[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      float hi_acc[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int u = 0; u < num_valid; u++) {
        char* slot_ptr = gctx.combine_recv_slot(valid_producer_rank[u], token);
        const uint8_t* hidden_data =
            reinterpret_cast<const uint8_t*>(gctx.combine_layout.slot.hidden(slot_ptr)) +
            range_begin;
        if (is_aligned_16(hidden_data)) {
          const bool remote_slot = valid_producer_rank[u] != gctx.rank_me;
          const uint4 raw = remote_slot
                                ? ep_load<true>(reinterpret_cast<const uint4*>(hidden_data) + vi)
                                : reinterpret_cast<const uint4*>(hidden_data)[vi];
          const uint8_t* vals = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
          for (int i = 0; i < 16; i++) {
            lo_acc[i] += fp4_e2m1fn_to_float(vals[i] & 0x0f);
            hi_acc[i] += fp4_e2m1fn_to_float((vals[i] >> 4) & 0x0f);
          }
        } else {
          const int d = vi * 16;
#pragma unroll
          for (int i = 0; i < 16; i++) {
            const uint8_t packed = (valid_producer_rank[u] != gctx.rank_me)
                                       ? ep_load<true>(&hidden_data[d + i])
                                       : hidden_data[d + i];
            lo_acc[i] += fp4_e2m1fn_to_float(packed & 0x0f);
            hi_acc[i] += fp4_e2m1fn_to_float((packed >> 4) & 0x0f);
          }
        }
      }
      if (num_valid > 0) {
        byte_vec16_storage_t tmp;
#pragma unroll
        for (int i = 0; i < 16; i++) {
          tmp.v[i] = static_cast<uint8_t>(float_to_fp4_e2m1fn(lo_acc[i]) |
                                          (float_to_fp4_e2m1fn(hi_acc[i]) << 4));
        }
        out4[vi] = *reinterpret_cast<const uint4*>(&tmp);
      }
    }
  } else {
    for (int d = range_begin + thread_idx; d < range_end; d += thread_count) {
      float lo_acc = 0.0f;
      float hi_acc = 0.0f;
      for (int u = 0; u < num_valid; u++) {
        char* slot_ptr = gctx.combine_recv_slot(valid_producer_rank[u], token);
        const uint8_t* hidden_data =
            reinterpret_cast<const uint8_t*>(gctx.combine_layout.slot.hidden(slot_ptr));
        const uint8_t packed = (valid_producer_rank[u] != gctx.rank_me)
                                   ? ep_load<true>(&hidden_data[d])
                                   : hidden_data[d];
        lo_acc += fp4_e2m1fn_to_float(packed & 0x0f);
        hi_acc += fp4_e2m1fn_to_float((packed >> 4) & 0x0f);
      }
      if (num_valid > 0) {
        out8[d] =
            static_cast<uint8_t>(float_to_fp4_e2m1fn(lo_acc) | (float_to_fp4_e2m1fn(hi_acc) << 4));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Single-kernel combine
//
// Grid: num_blocks blocks x EP_THREADS_PER_BLOCK threads. Each block stages/posts
// combine rows with one row per wavefront.  The EP comm context has
// provisioned_num_blocks * EP_WAVES_PER_BLOCK queues, while each launch selects an
// active prefix for its communication wavefronts. Accumulation also assigns
// one output token per wavefront where possible.
// ---------------------------------------------------------------------------
// This template specializes the RCOMP payload backend: RDMA-only or hybrid
// (IPC for same-host peers and RDMA for remote peers).
template <bool UseHybridRcomp>
__global__ void combine_kernel(ep_gcontext_t gctx, const void* d_expert_output,
                               const float* d_expert_weights,
                               void* d_output,  // [num_tokens][combine_hidden_elems], pre-zeroed
                               float* d_output_weights,  // optional [num_tokens][topk]
                               int num_tokens, uint64_t* debug_time_buf, int32_t* debug_time_meta) {
  const ep_device_workspace_t& device_workspace = gctx.device_workspace;
  const int lane = threadIdx.x % EP_WAVE_SIZE;
  const int wf_in_blk = threadIdx.x / EP_WAVE_SIZE;
  const int bid = blockIdx.x;
  const int world_size = gctx.world_size;
  const int rank_me = gctx.rank_me;
  const int combine_hidden_elems = gctx.combine_hidden_elems;
  const int topk = gctx.topk;
  const int combine_elem_size = gctx.combine_layout.slot.elem_size;
  const quant_format_t combine_quant_format = gctx.combine_quant_format;
  const size_t slot_payload_size = gctx.combine_layout.slot.payload_size;
  const int heap_id = gctx.heap_id;
  const int32_t* recv_counts = device_workspace.dispatch_recv_counts();

  const int total_recv = recv_counts[world_size];
  const int combine_hidden_bytes = combine_hidden_elems * combine_elem_size;
  const bool is_first_wf = (wf_in_blk == 0);
  constexpr int kCombineWavesPerBlock = EP_WAVES_PER_BLOCK;
  static_assert(EP_THREADS_PER_BLOCK == EP_WAVE_SIZE * EP_WAVES_PER_BLOCK,
                "EP launch constants are inconsistent");
  const int wave_count = kCombineWavesPerBlock;

  // Acquire each wave's endpoint once. In block 0 / wave 0 this is endpoint
  // 0, reused for the active-count control callback and combine barrier.
  const size_t combine_endpoint_id = static_cast<size_t>(bid * kCombineWavesPerBlock + wf_in_blk);
  auto combine_handle = gctx.comm.get_handle(combine_endpoint_id);

  // invoke invoke is wave-cooperative. Block 0 / wave 0 publishes the
  // active endpoint prefix through endpoint 0 before any EP traffic uses it.
  if (bid == 0 && wf_in_blk == 0) {
    publish_active_queue_count(gctx, combine_handle);
  }

  __shared__ int s_num_valid[kCombineWavesPerBlock];
  __shared__ int s_valid_producer_rank[kCombineWavesPerBlock][MAX_TOPK_SUPPORTED];

  if (bid == 0 && threadIdx.x == 0) {
    debug_store_time(debug_time_buf, 0);  // kernel entry
    debug_store_meta(debug_time_meta, 0, rank_me);
    debug_store_meta(debug_time_meta, 1, world_size);
    debug_store_meta(debug_time_meta, 2, num_tokens);
    debug_store_meta(debug_time_meta, 3, total_recv);
    debug_store_meta(debug_time_meta, 4, gridDim.x);
    debug_store_meta(debug_time_meta, 5, gridDim.x);
    debug_store_meta(debug_time_meta, 6, total_recv - recv_counts[rank_me]);
    debug_store_meta(debug_time_meta, 7, recv_counts[rank_me]);
    debug_store_meta(debug_time_meta, 8, combine_hidden_elems);
    debug_store_meta(debug_time_meta, 9, combine_elem_size);
    debug_store_meta(debug_time_meta, 10, gctx.combine_hidden_elems);
  }

  ep_barrier_ticket_t combine_pack_barrier{};

  if (bid == 0 && is_first_wf) {
    if (lane == 0) {
      debug_store_time(debug_time_buf, 1);  // rcomp combine work start
    }
  }

  // =================================================================
  // Phase 1 + 2: dynamically group waves per incoming row while preserving
  // one communication owner per row and active RCOMP endpoints.
  // =================================================================
  const int raw_pack_rows_per_block = (total_recv + gridDim.x - 1) / gridDim.x;
  int pack_rows_per_block = raw_pack_rows_per_block;
  if (pack_rows_per_block < 1) {
    pack_rows_per_block = 1;
  }
  if (pack_rows_per_block > wave_count) {
    pack_rows_per_block = wave_count;
  }
  const int pack_waves_per_row_candidate = wave_count / pack_rows_per_block;
  const int pack_waves_per_row =
      pack_waves_per_row_candidate > 0 ? pack_waves_per_row_candidate : 1;

  for (int base_flat = bid * pack_rows_per_block; base_flat < total_recv;
       base_flat += gridDim.x * pack_rows_per_block) {
    const int remaining = total_recv - base_flat;
    const int rows_in_block = remaining < pack_rows_per_block ? remaining : pack_rows_per_block;
    const int row_in_block = wf_in_blk / pack_waves_per_row;
    const int part_in_row = wf_in_blk % pack_waves_per_row;
    const bool valid_flat =
        wf_in_blk < pack_rows_per_block * pack_waves_per_row && row_in_block < rows_in_block;
    const int flat_idx = base_flat + row_in_block;
    const bool combine_row_comm_owner = valid_flat && part_in_row == 0;

    int src_rank = -1;
    int local_slot = -1;
    int32_t src_token_id = -1;
    int combine_slot = -1;
    const char* src_hidden = nullptr;
    const float* src_weights = nullptr;
    bool payload_via_rdma = false;
    bool payload_via_ipc = false;
    char* packed_slot = nullptr;
    size_t combine_hidden_offset = 0;
    int combine_hidden_copy_bytes = 0;

    if (valid_flat) {
      if (lane == 0) {
        flat_to_rank_slot_direct(gctx, flat_idx, src_rank, local_slot);
      }
      src_rank = __shfl(src_rank, 0);
      local_slot = __shfl(local_slot, 0);
      char* recv_slot = gctx.dispatch_recv_slot(src_rank, local_slot);
      float* recv_weights = gctx.dispatch_layout.slot.weights(recv_slot);
      const int32_t* src_token_id_ptr = gctx.dispatch_layout.slot.src_id(recv_slot);
      if (lane == 0) {
        src_token_id = (src_rank != rank_me) ? ep_load<true>(src_token_id_ptr) : *src_token_id_ptr;
      }
      src_token_id = __shfl(src_token_id, 0);
      combine_slot = src_token_id;

      const size_t combine_col_begin = (static_cast<size_t>(combine_hidden_elems) * part_in_row) /
                                       static_cast<size_t>(pack_waves_per_row);
      const size_t combine_col_end =
          (static_cast<size_t>(combine_hidden_elems) * (part_in_row + 1)) /
          static_cast<size_t>(pack_waves_per_row);
      const size_t combine_col_count = combine_col_end - combine_col_begin;
      combine_hidden_offset = combine_col_begin * static_cast<size_t>(combine_elem_size);
      combine_hidden_copy_bytes =
          static_cast<int>(combine_col_count * static_cast<size_t>(combine_elem_size));

      src_hidden = reinterpret_cast<const char*>(d_expert_output) +
                   static_cast<size_t>(flat_idx) * combine_hidden_bytes + combine_hidden_offset;
      src_weights = (d_expert_weights != nullptr)
                        ? d_expert_weights + static_cast<size_t>(flat_idx) * topk
                        : recv_weights;
      payload_via_rdma = src_rank != rank_me;
      if constexpr (UseHybridRcomp) {
        payload_via_ipc = src_rank != rank_me && gctx.rcomp_peer_uses_ipc(src_rank);
      }

      packed_slot = gctx.combine_staging_slot(flat_idx);
      if constexpr (UseHybridRcomp) {
        if (payload_via_ipc) {
          packed_slot = gctx.rcomp_peer_combine_recv_slot(src_rank, combine_slot);
        }
      }

      if (payload_via_ipc) {
        combine_pack_slot_part<true, true>(gctx, src_hidden, src_weights, topk,
                                           combine_hidden_offset, combine_hidden_copy_bytes,
                                           packed_slot, combine_row_comm_owner, lane, EP_WAVE_SIZE);
      } else if (payload_via_rdma) {
        combine_pack_slot_part<true>(gctx, src_hidden, src_weights, topk, combine_hidden_offset,
                                     combine_hidden_copy_bytes, packed_slot, combine_row_comm_owner,
                                     lane, EP_WAVE_SIZE);
      } else {
        combine_pack_slot_part<false>(gctx, src_hidden, src_weights, topk, combine_hidden_offset,
                                      combine_hidden_copy_bytes, packed_slot,
                                      combine_row_comm_owner, lane, EP_WAVE_SIZE);
      }
      __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    if (valid_flat) {
      if (payload_via_ipc) {
        if (combine_row_comm_owner && lane == 0) {
          const size_t flag_index = gctx.payload_readiness_flag_index(rank_me, combine_slot);
          payload_readiness_flag_publish(gctx.peer_combine_readiness_flags(src_rank) + flag_index);
        }
      } else if (src_rank == rank_me) {
        const int group_thread_idx = part_in_row * EP_WAVE_SIZE + lane;
        const int group_thread_count = pack_waves_per_row * EP_WAVE_SIZE;
        copy_bytes_block(packed_slot, gctx.combine_recv_slot(rank_me, combine_slot),
                         slot_payload_size, group_thread_idx, group_thread_count);
      } else if (combine_row_comm_owner) {
        uintptr_t remote_disp = combine_put_remote_disp(gctx, combine_slot);
        ep_post_put(gctx, combine_handle, heap_id, src_rank, combine_put_src_ptr(gctx, packed_slot),
                    combine_put_bytes(gctx), remote_disp, gctx.combine_readiness_flags_offset,
                    combine_slot);
      }

      if (src_rank != rank_me && combine_row_comm_owner && lane == 0) {
        payload_readiness_flag_reset(dispatch_payload_readiness_flag(gctx, src_rank, local_slot));
      }
    }
  }

  if (bid == 0 && threadIdx.x == 0) {
    debug_store_time(debug_time_buf, 2);  // block 0 pack/post done
  }

  // Dispatch readiness resets are raw system-scope VMEM stores.  Drain them
  // on every block before the local pack barrier lets block 0 publish the
  // cross-rank pack-release barrier below.
  __builtin_amdgcn_s_waitcnt(0);

  if (bid == 0 && is_first_wf) {
    if (lane == 0) {
      debug_store_time(debug_time_buf, 3);  // comm path wait start
    }
  }

  grid_barrier_ticket_t local_pack_ticket =
      grid_barrier_arrive(device_workspace.grid_barrier(), static_cast<uint32_t>(gridDim.x));
  grid_barrier_wait(device_workspace.grid_barrier(), local_pack_ticket);

  if (bid == 0 && is_first_wf) {
    combine_pack_barrier = post_barrier(gctx, combine_handle);
    if (lane == 0) {
      debug_store_time(debug_time_buf, 4);  // combine-exit barrier posted after pack
    }
  }

  if (bid == 0 && threadIdx.x == 0) {
    debug_store_time(debug_time_buf, 5);  // all blocks completed local pack
  }
  if (bid == 0 && threadIdx.x == 0) {
    debug_store_time(debug_time_buf, 6);  // accumulation start
  }

  // =================================================================
  // Phase 4: accumulation. Distribute all grid-resident wavefronts over
  // output-token ranges. Each wave handles a disjoint hidden range, while
  // the first part owns output-weight writes for its token.
  // =================================================================
  assert((combine_elem_size == 4 || combine_elem_size == 2 || combine_elem_size == 1) &&
         "combine: only combine_elem_size==4 (float), combine_elem_size==2 (bf16), and "
         "combine_elem_size==1 supported");

  const int global_reduce_wave_count = static_cast<int>(gridDim.x) * wave_count;
  const int global_reduce_wave_id = bid * wave_count + wf_in_blk;
  ep_multi_warp_iter_t reduce_iter(global_reduce_wave_count, num_tokens,
                                   static_cast<size_t>(combine_hidden_elems));
  const int total_reduce_parts = reduce_iter.total_parts(num_tokens);
  for (int part_idx = global_reduce_wave_id; part_idx < total_reduce_parts;
       part_idx += global_reduce_wave_count) {
    int token = 0;
    int part_in_token = 0;
    size_t combine_col_begin = 0;
    size_t combine_col_count = 0;
    reduce_iter.decode(part_idx, token, part_in_token, combine_col_begin, combine_col_count);
    const bool valid_token = token < num_tokens;
    const bool token_owner = valid_token && part_in_token == 0;

    if (valid_token) {
      int32_t producer_rank = -1;
      bool valid = false;
      if (lane < topk) {
        producer_rank = gctx.token_to_producer_rank(token, lane);
        valid = (producer_rank >= 0 && producer_rank < world_size);
      }
      const unsigned long long active_topk_mask =
          (topk >= EP_WAVE_SIZE) ? ~0ull : ((1ull << topk) - 1ull);
      const unsigned long long valid_mask = __ballot(valid) & active_topk_mask;
      if (valid) {
        const unsigned long long lower_mask = (lane == 0) ? 0ull : ((1ull << lane) - 1ull);
        const int compact_idx = __popcll(valid_mask & lower_mask);
        s_valid_producer_rank[wf_in_blk][compact_idx] = producer_rank;
      }
      if (lane == 0) {
        s_num_valid[wf_in_blk] = __popcll(valid_mask);
      }
    }
    __builtin_amdgcn_wave_barrier();

    const int num_valid = valid_token ? s_num_valid[wf_in_blk] : 0;

    // Each reduction wave waits only for the remote rows it will read.
    // Extra waves assigned to the same token independently establish this
    // condition before consuming their disjoint hidden ranges.
    if (valid_token && lane < num_valid) {
      const int producer_rank = s_valid_producer_rank[wf_in_blk][lane];
      if (producer_rank != rank_me) {
        payload_readiness_flag_wait(combine_payload_readiness_flag(gctx, producer_rank, token));
      }
    }
    __builtin_amdgcn_wave_barrier();

    // MORI combine does not apply token weights to hidden data; it
    // sums the already-produced expert rows. Grid-resident wavefronts
    // split each token's hidden output range, while compact output weights
    // are written once by the owner wavefront.
    if (token_owner && d_output_weights != nullptr) {
      for (int kk = lane; kk < topk; kk += EP_WAVE_SIZE) {
        float sum = 0.0f;
        for (int u = 0; u < num_valid; u++) {
          char* sp = gctx.combine_recv_slot(s_valid_producer_rank[wf_in_blk][u], token);
          float* wt = gctx.combine_layout.slot.weights(sp);
          sum += (s_valid_producer_rank[wf_in_blk][u] != rank_me) ? ep_load<true>(&wt[kk]) : wt[kk];
        }
        d_output_weights[static_cast<size_t>(token) * topk + kk] = sum;
      }
    }

    if (valid_token && combine_col_count > 0) {
      const int range_offset = static_cast<int>(combine_col_begin);
      const int range_count = static_cast<int>(combine_col_count);
      switch (combine_elem_size) {
        case 4:
          reduce_float4_token(gctx, token, combine_hidden_elems, num_valid,
                              s_valid_producer_rank[wf_in_blk], d_output, range_offset, range_count,
                              lane, EP_WAVE_SIZE);
          break;
        case 2:
          reduce_bf16x8_token(gctx, token, combine_hidden_elems, num_valid,
                              s_valid_producer_rank[wf_in_blk], d_output, range_offset, range_count,
                              lane, EP_WAVE_SIZE);
          break;
        case 1:
          if (combine_quant_format == quant_format_t::FP4) {
            reduce_fp4x16_token(gctx, token, combine_hidden_elems, num_valid,
                                s_valid_producer_rank[wf_in_blk], d_output, range_offset,
                                range_count, lane, EP_WAVE_SIZE);
          } else if (combine_quant_format == quant_format_t::FP8) {
            reduce_fp8x16_token(gctx, token, combine_hidden_elems, num_valid,
                                s_valid_producer_rank[wf_in_blk], d_output, range_offset,
                                range_count, lane, EP_WAVE_SIZE);
          }
          break;
        default:
          break;
      }
    }
    __builtin_amdgcn_wave_barrier();
  }
  if (bid == 0 && threadIdx.x == 0) {
    debug_store_time(debug_time_buf, 7);  // block 0 accumulation done
  }

  // =================================================================
  // Phase 5: every block arrives only after its local accumulation and rcomp
  // combine-flag resets are complete.  Rcomp posted its cross-rank
  // pack-release barrier immediately after all local combine packing, when
  // dispatch payload slots had been consumed and their readiness flags reset.
  // Waiting on that ticket here lets its cross-rank latency overlap local
  // accumulation while still preventing the next dispatch until every rank
  // has released its previous dispatch payload slots.
  // =================================================================
  grid_barrier_ticket_t final_ticket =
      grid_barrier_arrive(device_workspace.grid_barrier(), static_cast<uint32_t>(gridDim.x));

  if (bid == 0) {
    grid_barrier_wait(device_workspace.grid_barrier(), final_ticket);
    // Token parts may span blocks, so no block-local owner can safely
    // release a readiness flag. Reuse the existing final grid rendezvous:
    // once it completes, block 0 can reset every consumed remote slot
    // without adding another grid synchronization.
    for (int entry = threadIdx.x; entry < num_tokens * topk; entry += blockDim.x) {
      const int token = entry / topk;
      const int k = entry - token * topk;
      const int producer_rank = gctx.token_to_producer_rank(token, k);
      if (producer_rank >= 0 && producer_rank < world_size && producer_rank != rank_me) {
        payload_readiness_flag_reset(combine_payload_readiness_flag(gctx, producer_rank, token));
      }
    }
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    if (threadIdx.x == 0) {
      debug_store_time(debug_time_buf, 8);  // all local blocks done
    }
    if (is_first_wf) {
      if (lane == 0) {
        debug_store_time(debug_time_buf, 9);  // wait for combine-exit barrier
      }
      wait_barrier(gctx, combine_pack_barrier);
      if (lane == 0) {
        debug_store_time(debug_time_buf, 10);  // rcomp exit barrier complete
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side combine orchestration
// ---------------------------------------------------------------------------
void combine_enqueue(ep_context_t& ctx, ep_gcontext_t& gctx, const void* d_expert_output,
                     const float* d_expert_weights, const void* d_expert_scales,
                     const int32_t* d_topk_ids, void* d_output, float* d_output_weights,
                     int num_tokens, hipStream_t stream) {
  if (num_tokens < 0 || num_tokens > ctx.config().max_num_inp_token_per_rank) {
    fprintf(stderr,
            "kiwi::ep::combine_enqueue: num_tokens=%d exceeds "
            "max_num_inp_token_per_rank=%d\n",
            num_tokens, ctx.config().max_num_inp_token_per_rank);
    abort();
  }

  const int combine_elem_size = ctx.config().combine_elem_size;
  const int combine_hidden_elems = ctx.config().combine_hidden_elems_effective();
  if (combine_hidden_elems <= 0 || combine_elem_size <= 0) {
    fprintf(stderr,
            "kiwi::ep::combine_enqueue: invalid configured combine row "
            "(hidden_elems=%d elem_size=%d)\n",
            combine_hidden_elems, combine_elem_size);
    abort();
  }
  // Combine polls a receiver-owned readiness flag for each required remote
  // row before accumulation.
  // Keep combine's launch shape simple and aligned with dispatch: one grid
  // size, determined from this call's active token count.
  // Every launched block participates in both communication and accumulation.
  const runtime_defaults_t launch_policy = ctx.config().launch_policy_for(num_tokens);
  const int launch_num_blocks = launch_policy.num_blocks;
  const int provisioned_num_blocks = ctx.config().provisioned_num_blocks();
  if (launch_num_blocks <= 0 || launch_num_blocks > provisioned_num_blocks) {
    fprintf(stderr,
            "kiwi::ep::combine_enqueue: launch_num_blocks=%d exceeds "
            "provisioned_num_blocks=%d\n",
            launch_num_blocks, provisioned_num_blocks);
    abort();
  }

  if (launch_policy.rcomp_backend == rcomp_backend_t::HYBRID) {
    combine_kernel<true><<<launch_num_blocks, EP_THREADS_PER_BLOCK, 0, stream>>>(
        gctx, d_expert_output, d_expert_weights, d_output, d_output_weights, num_tokens,
        ctx.debug_timing_enabled() ? ctx.debug_time_buf() : nullptr,
        ctx.debug_timing_enabled() ? ctx.debug_time_meta() : nullptr);
  } else {
    combine_kernel<false><<<launch_num_blocks, EP_THREADS_PER_BLOCK, 0, stream>>>(
        gctx, d_expert_output, d_expert_weights, d_output, d_output_weights, num_tokens,
        ctx.debug_timing_enabled() ? ctx.debug_time_buf() : nullptr,
        ctx.debug_timing_enabled() ? ctx.debug_time_meta() : nullptr);
  }
  HIP_CHECK(hipGetLastError());
}

void combine(ep_context_t& ctx, ep_gcontext_t& gctx, const void* d_expert_output,
             const float* d_expert_weights, const void* d_expert_scales, const int32_t* d_topk_ids,
             void* d_output, float* d_output_weights, int num_tokens, hipStream_t stream) {
  combine_enqueue(ctx, gctx, d_expert_output, d_expert_weights, d_expert_scales, d_topk_ids,
                  d_output, d_output_weights, num_tokens, stream);
  ctx.stream_sync(stream);
}

}  // namespace mori::kiwi::ep
