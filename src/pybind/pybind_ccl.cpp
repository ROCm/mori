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
#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mori/collective/all2all/oneshot_all2all_sdma_class.hpp"
#include "mori/collective/ccl_kernel_args.hpp"
#include "mori/collective/allgather/allgather_into_tensor.hpp"
#include "mori/collective/allgather/intra_node_subgroup_broadcast_sdma_class.hpp"
#include "mori/collective/allgather/intra_node_subgroup_sdma_class.hpp"
#include "mori/collective/allgather/oneshot_allgather_sdma_class.hpp"
#include "mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp"
#include "mori/collective/inter_node/inter_node_ring_class.hpp"
#include "src/pybind/mori.hpp"

namespace py = pybind11;

namespace {
template <typename T>
void BindAllreduceHandle(py::module_& m, const char* python_name) {
  using Handle = mori::collective::AllreduceSdma<T>;

  py::class_<Handle>(m, python_name)
      .def(py::init<int, int, size_t, size_t, bool, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true, py::arg("use_graph_mode") = false)
      .def(py::init<int, int, size_t, bool, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true, py::arg("use_graph_mode") = false)
      // JIT sync path (two-shot: reduce-scatter kernel, then allgather kernel)
      .def(
          "prepare_reduce_scatter",
          [](Handle& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_reduce_scatter(reinterpret_cast<const T*>(input),
                                               reinterpret_cast<T*>(output), count,
                                               reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "get_reduce_scatter_grid",
          [](Handle& self, size_t count) -> py::tuple {
            auto [blocks, threads] = self.get_reduce_scatter_grid(count);
            return py::make_tuple(blocks, threads);
          },
          py::arg("count"))
      .def(
          "prepare_allgather",
          [](Handle& self, size_t count, int64_t stream) -> int64_t {
            return self.prepare_allgather(count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("count"), py::arg("stream"))
      .def(
          "finish_sync",
          [](Handle& self, uintptr_t output, size_t count, int64_t stream,
             bool force_copy_output_to_user) -> double {
            return self.finish_sync(reinterpret_cast<T*>(output), count,
                                    reinterpret_cast<hipStream_t>(stream),
                                    force_copy_output_to_user);
          },
          py::arg("output_ptr"), py::arg("count"), py::arg("stream"),
          py::arg("force_copy_output_to_user") = false)
      // JIT async path
      .def(
          "prepare_async_reduce_scatter",
          [](Handle& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_async_reduce_scatter(reinterpret_cast<const T*>(input),
                                                     reinterpret_cast<T*>(output), count,
                                                     reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "prepare_async_allgather_put",
          [](Handle& self, size_t count, int64_t stream) -> int64_t {
            return self.prepare_async_allgather_put(count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("count"), py::arg("stream"))
      .def("after_async_start", &Handle::after_async_start)
      .def(
          "prepare_async_wait",
          [](Handle& self, int64_t stream) -> int64_t {
            return self.prepare_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def(
          "finish_async_wait",
          [](Handle& self, int64_t stream) -> double {
            return self.finish_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def("is_async_in_progress", &Handle::is_async_in_progress)
      .def("cancel_async", &Handle::cancel_async)
      .def("reset_flags", &Handle::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](Handle& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer")
      .def("max_blocks", &Handle::max_blocks)
      .def("npes", &Handle::npes);
}
}  // namespace

namespace mori {
void RegisterMoriCcl(pybind11::module_& m) {
  // =========================================================================
  // All2allSdma (uint32_t) — JIT launch path
  // =========================================================================
  using All2allU32 = mori::collective::All2allSdma<uint32_t>;
  py::class_<All2allU32>(m, "All2allSdmaHandle")
      .def(py::init<int, int, size_t, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true)
      .def(py::init<int, int, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true)
      .def(
          "prepare_sync",
          [](All2allU32& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_sync(reinterpret_cast<uint32_t*>(input),
                                     reinterpret_cast<uint32_t*>(output), count,
                                     reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "finish_sync",
          [](All2allU32& self, uintptr_t output, size_t count, int64_t stream) -> double {
            return self.finish_sync(reinterpret_cast<uint32_t*>(output), count,
                                    reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "prepare_async_start",
          [](All2allU32& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_async_start(reinterpret_cast<uint32_t*>(input),
                                            reinterpret_cast<uint32_t*>(output), count,
                                            reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def("after_async_start", &All2allU32::after_async_start)
      .def(
          "prepare_async_wait",
          [](All2allU32& self, int64_t stream) -> int64_t {
            return self.prepare_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def(
          "finish_async_wait",
          [](All2allU32& self, int64_t stream) -> double {
            return self.finish_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def("is_async_in_progress", &All2allU32::is_async_in_progress)
      .def("cancel_async", &All2allU32::cancel_async)
      .def("reset_flags", &All2allU32::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](All2allU32& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer");

  // =========================================================================
  // AllgatherSdma (uint32_t) — JIT launch path
  // =========================================================================
  using AllgatherU32 = mori::collective::AllgatherSdma<uint32_t>;
  py::class_<AllgatherU32>(m, "AllgatherSdmaHandle")
      .def(py::init<int, int, size_t, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true)
      .def(py::init<int, int, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true)
      .def(
          "prepare_sync",
          [](AllgatherU32& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_sync(reinterpret_cast<uint32_t*>(input),
                                     reinterpret_cast<uint32_t*>(output), count,
                                     reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "prepare_sync_param_contiguous",
          [](AllgatherU32& self, uintptr_t input, uintptr_t output, size_t count,
             uintptr_t split_sizes, uintptr_t split_offsets, size_t split_count,
             int64_t stream) -> int64_t {
            return self.prepare_sync_param_contiguous(
                reinterpret_cast<uint32_t*>(input), reinterpret_cast<uint32_t*>(output), count,
                reinterpret_cast<const size_t*>(split_sizes),
                reinterpret_cast<const size_t*>(split_offsets), split_count,
                reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("split_sizes_ptr"),
          py::arg("split_offsets_ptr"), py::arg("split_count"), py::arg("stream"))
      .def(
          "finish_sync",
          [](AllgatherU32& self, uintptr_t output, size_t count, int64_t stream) -> double {
            return self.finish_sync(reinterpret_cast<uint32_t*>(output), count,
                                    reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "prepare_async_start",
          [](AllgatherU32& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_async_start(reinterpret_cast<uint32_t*>(input),
                                            reinterpret_cast<uint32_t*>(output), count,
                                            reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "prepare_async_start_param_contiguous",
          [](AllgatherU32& self, uintptr_t input, uintptr_t output, size_t count,
             uintptr_t split_sizes, uintptr_t split_offsets, size_t split_count,
             int64_t stream) -> int64_t {
            return self.prepare_async_start_param_contiguous(
                reinterpret_cast<uint32_t*>(input), reinterpret_cast<uint32_t*>(output), count,
                reinterpret_cast<const size_t*>(split_sizes),
                reinterpret_cast<const size_t*>(split_offsets), split_count,
                reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("split_sizes_ptr"),
          py::arg("split_offsets_ptr"), py::arg("split_count"), py::arg("stream"))
      .def("after_async_start", &AllgatherU32::after_async_start)
      .def(
          "prepare_async_wait",
          [](AllgatherU32& self, int64_t stream) -> int64_t {
            return self.prepare_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def(
          "finish_async_wait",
          [](AllgatherU32& self, int64_t stream) -> double {
            return self.finish_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def("is_async_in_progress", &AllgatherU32::is_async_in_progress)
      .def("cancel_async", &AllgatherU32::cancel_async)
      .def("reset_flags", &AllgatherU32::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](AllgatherU32& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer")
      .def(
          "register_output_buffer",
          [](AllgatherU32& self, uintptr_t ptr, size_t size) {
            self.register_output_buffer(reinterpret_cast<void*>(ptr), size);
          },
          py::arg("ptr"), py::arg("size"),
          "Register a GPU buffer as direct SDMA output target (collective)")
      .def(
          "deregister_output_buffer",
          [](AllgatherU32& self, uintptr_t ptr) {
            self.deregister_output_buffer(reinterpret_cast<void*>(ptr));
          },
          py::arg("ptr"), "Deregister a previously registered output buffer (collective)")
      .def(
          "is_output_registered",
          [](AllgatherU32& self, uintptr_t ptr) -> bool {
            return self.is_output_registered(reinterpret_cast<void*>(ptr));
          },
          py::arg("ptr"), "Check whether an output buffer is registered for direct SDMA writes");

  // =========================================================================
  // InterNodeRingAllgather — inter-node RDMA ring, JIT launch path
  // =========================================================================
  using InterNodeRing = mori::collective::InterNodeRingAllgather;
  py::class_<InterNodeRing>(m, "InterNodeRingAllgatherHandle")
      .def(py::init<int, int, size_t, int, int, int, int, int, int>(), py::arg("my_pe"),
           py::arg("npes"), py::arg("ring_buffer_bytes") = 512 * 1024 * 1024,
           py::arg("ring_size") = -1, py::arg("ring_pos") = -1, py::arg("pe_base") = 0,
           py::arg("pe_stride") = 1, py::arg("num_qp") = 1, py::arg("num_blocks") = 1)
      .def(
          "prepare_sync",
          [](InterNodeRing& self, uintptr_t input, size_t count, int64_t stream) -> int64_t {
            return self.prepare_sync(input, count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "slot_ptr",
          [](InterNodeRing& self, size_t count) -> uintptr_t { return self.slot_ptr(count); },
          py::arg("count"))
      .def(
          "prepare_sync_in_place",
          [](InterNodeRing& self, size_t count, int64_t stream) -> int64_t {
            return self.prepare_sync_in_place(count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("count"), py::arg("stream"))
      .def(
          "finish_sync",
          [](InterNodeRing& self, uintptr_t output, size_t count, int64_t stream) -> double {
            return self.finish_sync(output, count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "buf_ptr", [](InterNodeRing& self) -> uintptr_t { return self.buf_ptr(); })
      .def(
          "finish_sync_no_copy",
          [](InterNodeRing& self, int64_t stream) -> double {
            return self.finish_sync_no_copy(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      // stream-ordered prepare/finish (ShmemBarrierOnStream
      // instead of host hipStreamSynchronize + host ShmemBarrierAll).
      .def(
          "prepare_stream",
          [](InterNodeRing& self, uintptr_t input, size_t count, int64_t stream) -> int64_t {
            return self.prepare_stream(input, count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "prepare_stream_in_place",
          [](InterNodeRing& self, size_t count, int64_t stream) -> int64_t {
            return self.prepare_stream_in_place(count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("count"), py::arg("stream"))
      .def(
          "finish_stream",
          [](InterNodeRing& self, uintptr_t output, size_t count, int64_t stream,
             bool barrier) -> double {
            return self.finish_stream(output, count, reinterpret_cast<hipStream_t>(stream),
                                      barrier);
          },
          py::arg("output_ptr"), py::arg("count"), py::arg("stream"),
          py::arg("barrier") = true)
      .def(
          "finish_stream_no_copy",
          [](InterNodeRing& self, int64_t stream) -> double {
            return self.finish_stream_no_copy(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def("npes", &InterNodeRing::npes)
      .def("num_blocks", &InterNodeRing::num_blocks);

  // =========================================================================
  // IntraNodeSubGroupAllgatherSdma — intra-node SDMA gather over a sub-group
  //
  // =========================================================================
  using IntraSubGroup = mori::collective::IntraNodeSubGroupAllgatherSdma;
  py::class_<IntraSubGroup>(m, "IntraNodeSubGroupAllgatherSdmaHandle")
      .def(py::init<int, int, size_t, int, int, int, int>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("out_buffer_bytes") = 512 * 1024 * 1024, py::arg("group_size") = -1,
           py::arg("group_pos") = -1, py::arg("pe_base") = 0, py::arg("pe_stride") = 1)
      .def(
          "prepare_sync",
          [](IntraSubGroup& self, uintptr_t input, size_t count, int64_t stream, bool barrier,
             size_t dst_base_offset_bytes, size_t dst_slot_stride_bytes) -> int64_t {
            return self.prepare_sync(input, count, reinterpret_cast<hipStream_t>(stream), barrier,
                                     dst_base_offset_bytes, dst_slot_stride_bytes);
          },
          py::arg("input_ptr"), py::arg("count"), py::arg("stream"), py::arg("barrier") = true,
          py::arg("dst_base_offset_bytes") = 0, py::arg("dst_slot_stride_bytes") = 0)
      .def(
          "finish_sync",
          [](IntraSubGroup& self, uintptr_t output, size_t count, int64_t stream,
             bool barrier) -> double {
            return self.finish_sync(output, count, reinterpret_cast<hipStream_t>(stream), barrier);
          },
          py::arg("output_ptr"), py::arg("count"), py::arg("stream"), py::arg("barrier") = true)
      .def(
          "finish_batch",
          [](IntraSubGroup& self, uintptr_t output, size_t total_count, int64_t stream,
             bool barrier) -> double {
            return self.finish_batch(output, total_count, reinterpret_cast<hipStream_t>(stream),
                                     barrier);
          },
          py::arg("output_ptr"), py::arg("total_count"), py::arg("stream"),
          py::arg("barrier") = true)
      .def(
          "finish_batch_stream",
          [](IntraSubGroup& self, uintptr_t output, size_t total_count, int64_t stream,
             bool barrier) -> double {
            return self.finish_batch_stream(output, total_count,
                                            reinterpret_cast<hipStream_t>(stream), barrier);
          },
          py::arg("output_ptr"), py::arg("total_count"), py::arg("stream"),
          py::arg("barrier") = true)
      .def(
          "register_output_buffer",
          [](IntraSubGroup& self, uintptr_t ptr, size_t size) {
            self.register_output_buffer(ptr, size);
          },
          py::arg("output_ptr"), py::arg("size"))
      .def(
          "deregister_output_buffer",
          [](IntraSubGroup& self, uintptr_t ptr) { self.deregister_output_buffer(ptr); },
          py::arg("output_ptr"))
      .def(
          "is_output_registered",
          [](IntraSubGroup& self, uintptr_t ptr, size_t size) {
            return self.is_output_registered(ptr, size);
          },
          py::arg("output_ptr"), py::arg("size"))
      .def(
          "prepare_sync_direct",
          [](IntraSubGroup& self, uintptr_t input, size_t count, int64_t stream, bool barrier,
             uintptr_t output_ptr, size_t dst_block_offset_bytes,
             size_t dst_slot_stride_bytes) -> int64_t {
            return self.prepare_sync_direct(input, count, reinterpret_cast<hipStream_t>(stream),
                                            barrier, output_ptr, dst_block_offset_bytes,
                                            dst_slot_stride_bytes);
          },
          py::arg("input_ptr"), py::arg("count"), py::arg("stream"), py::arg("barrier") = true,
          py::arg("output_ptr") = 0, py::arg("dst_block_offset_bytes") = 0,
          py::arg("dst_slot_stride_bytes") = 0)
      .def(
          "prepare_sync_direct_param_contiguous",
          [](IntraSubGroup& self, uintptr_t input, int64_t stream, bool barrier,
             uintptr_t output_ptr, size_t block_stride_u32, int num_blocks, size_t world_size,
             uintptr_t split_sizes_ptr, uintptr_t split_offsets_ptr, size_t split_count,
             size_t dst_block_offset_bytes) -> int64_t {
            return self.prepare_sync_direct_param_contiguous(
                input, reinterpret_cast<hipStream_t>(stream), barrier, output_ptr, block_stride_u32,
                num_blocks, world_size, split_sizes_ptr, split_offsets_ptr, split_count,
                dst_block_offset_bytes);
          },
          py::arg("input_ptr"), py::arg("stream"), py::arg("barrier") = true,
          py::arg("output_ptr") = 0, py::arg("block_stride_u32") = 0, py::arg("num_blocks") = 1,
          py::arg("world_size") = 0, py::arg("split_sizes_ptr") = 0,
          py::arg("split_offsets_ptr") = 0, py::arg("split_count") = 0,
          py::arg("dst_block_offset_bytes") = 0)
      .def(
          "finish_direct_stream",
          [](IntraSubGroup& self, int64_t stream, bool barrier) -> double {
            return self.finish_direct_stream(reinterpret_cast<hipStream_t>(stream), barrier);
          },
          py::arg("stream"), py::arg("barrier") = true)
      .def("npes", &IntraSubGroup::npes);

  // =========================================================================
  // IntraNodeSubGroupBroadcastSdma — intra-node SDMA broadcast over a sub-group
  // (this work, M4: the intra-node placement phase of the leader-only hierarchical
  // AllGather). Root (group_pos 0) fans its full buffer to all members via XGMI.
  // =========================================================================
  using IntraBcast = mori::collective::IntraNodeSubGroupBroadcastSdma;
  py::class_<IntraBcast>(m, "IntraNodeSubGroupBroadcastSdmaHandle")
      .def(py::init<int, int, size_t, int, int, int, int>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("out_buffer_bytes") = 512 * 1024 * 1024, py::arg("group_size") = -1,
           py::arg("group_pos") = -1, py::arg("pe_base") = 0, py::arg("pe_stride") = 1)
      .def(
          "prepare_sync",
          [](IntraBcast& self, uintptr_t input, size_t count, int64_t stream) -> int64_t {
            return self.prepare_sync(input, count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "finish_sync",
          [](IntraBcast& self, uintptr_t output, size_t count, int64_t stream) -> double {
            return self.finish_sync(output, count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def("npes", &IntraBcast::npes);

  // =========================================================================
  // DataType enum and size_of
  // =========================================================================
  py::enum_<mori::collective::DataType>(m, "DataType")
      .value("Int8", mori::collective::DataType::kInt8)
      .value("Uint8", mori::collective::DataType::kUint8)
      .value("Int16", mori::collective::DataType::kInt16)
      .value("Uint16", mori::collective::DataType::kUint16)
      .value("Int32", mori::collective::DataType::kInt32)
      .value("Uint32", mori::collective::DataType::kUint32)
      .value("Int64", mori::collective::DataType::kInt64)
      .value("Uint64", mori::collective::DataType::kUint64)
      .value("Float16", mori::collective::DataType::kFloat16)
      .value("BFloat16", mori::collective::DataType::kBFloat16)
      .value("Float32", mori::collective::DataType::kFloat32)
      .value("Float64", mori::collective::DataType::kFloat64);
  m.def("size_of", &mori::collective::SizeOf, py::arg("dtype"),
        "Return element size in bytes for a mori_cpp.DataType value");

  // merge an inter-node ring's jit_args + an intra-node
  // sub-group gather's jit_args into one CclFusedRingLocalGatherArgs for the
  // fused ring||local-gather kernel (the RCCL-parity lever).
  // Takes the two int64 arg pointers the respective prepare_*
  // calls return; returns the fused arg pointer (a static, valid until the next
  // call). Inert until the Python fused launcher is wired.
  m.def("build_fused_ring_local_gather_args",
        &mori::collective::BuildFusedRingLocalGatherArgs, py::arg("ring_args_ptr"),
        py::arg("gather_args_ptr"), py::arg("ring_blocks"),
        "Merge ring + local-gather jit_args into fused-kernel args; returns int64 ptr");

  // =========================================================================
  // AllGatherIntoTensor — REMOVED
  // The underlying AllgatherSdma<uint32_t>::operator() now throws; this
  // wrapper needs the Python JIT launch path to be ported before it can be
  // re-enabled.
  // =========================================================================

  // =========================================================================
  // AllreduceSdma — JIT launch path (typed instantiations)
  // =========================================================================
  BindAllreduceHandle<uint32_t>(m, "AllreduceSdmaHandle");
  BindAllreduceHandle<int32_t>(m, "AllreduceSdmaHandleInt32");
  BindAllreduceHandle<float>(m, "AllreduceSdmaHandleFp32");
  BindAllreduceHandle<half>(m, "AllreduceSdmaHandleFp16");
  BindAllreduceHandle<hip_bfloat16>(m, "AllreduceSdmaHandleBf16");
}
}  // namespace mori
