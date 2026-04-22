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
#include "mori/collective/allgather/oneshot_allgather_sdma_class.hpp"
#include "mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp"
#include "src/pybind/mori.hpp"

namespace py = pybind11;

namespace mori {
void RegisterMoriCcl(pybind11::module_& m) {
  // =========================================================================
  // All2allSdma (uint32_t)
  // =========================================================================
  py::class_<mori::collective::All2allSdma<uint32_t>>(m, "All2allSdmaHandle")
      .def(py::init<int, int, size_t, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true)
      .def(py::init<int, int, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true)
      .def(
          "__call__",
          [](mori::collective::All2allSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> double {
            return self(reinterpret_cast<uint32_t*>(input_ptr),
                        reinterpret_cast<uint32_t*>(output_ptr), count,
                        reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Execute All2all SDMA operation (raw GPU pointers)")
      .def(
          "start_async",
          [](mori::collective::All2allSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> bool {
            return self.start_async(reinterpret_cast<uint32_t*>(input_ptr),
                                    reinterpret_cast<uint32_t*>(output_ptr), count,
                                    reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Start asynchronous All2all SDMA operation (PUT phase)")
      .def(
          "wait_async",
          [](mori::collective::All2allSdma<uint32_t>& self, int64_t stream) -> double {
            return self.wait_async(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream") = 0,
          "Wait for asynchronous All2all SDMA operation to complete (WAIT phase)")
      .def("is_async_in_progress", &mori::collective::All2allSdma<uint32_t>::is_async_in_progress)
      .def("cancel_async", &mori::collective::All2allSdma<uint32_t>::cancel_async)
      .def("reset_flags", &mori::collective::All2allSdma<uint32_t>::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](mori::collective::All2allSdma<uint32_t>& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer");

  // =========================================================================
  // AllgatherSdma (uint32_t)
  // =========================================================================
  py::class_<mori::collective::AllgatherSdma<uint32_t>>(m, "AllgatherSdmaHandle")
      .def(py::init<int, int, size_t, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true)
      .def(py::init<int, int, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true)
      .def(
          "__call__",
          [](mori::collective::AllgatherSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> bool {
            return self(reinterpret_cast<uint32_t*>(input_ptr),
                        reinterpret_cast<uint32_t*>(output_ptr), count,
                        reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Execute Allgather SDMA operation (raw GPU pointers)")
      .def(
          "start_async",
          [](mori::collective::AllgatherSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> bool {
            return self.start_async(reinterpret_cast<uint32_t*>(input_ptr),
                                    reinterpret_cast<uint32_t*>(output_ptr), count,
                                    reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Start asynchronous Allgather SDMA operation (PUT phase)")
      .def(
          "wait_async",
          [](mori::collective::AllgatherSdma<uint32_t>& self, int64_t stream) -> double {
            return self.wait_async(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream") = 0, "Wait for asynchronous Allgather SDMA operation to complete")
      .def("is_async_in_progress", &mori::collective::AllgatherSdma<uint32_t>::is_async_in_progress)
      .def("cancel_async", &mori::collective::AllgatherSdma<uint32_t>::cancel_async)
      .def("reset_flags", &mori::collective::AllgatherSdma<uint32_t>::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](mori::collective::AllgatherSdma<uint32_t>& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer")
      .def(
          "register_output_buffer",
          [](mori::collective::AllgatherSdma<uint32_t>& self, uintptr_t ptr, size_t size) {
            self.register_output_buffer(reinterpret_cast<void*>(ptr), size);
          },
          py::arg("ptr"), py::arg("size"),
          "Register a GPU buffer as direct SDMA output target (collective)")
      .def(
          "deregister_output_buffer",
          [](mori::collective::AllgatherSdma<uint32_t>& self, uintptr_t ptr) {
            self.deregister_output_buffer(reinterpret_cast<void*>(ptr));
          },
          py::arg("ptr"), "Deregister a previously registered output buffer (collective)")
      .def(
          "is_output_registered",
          [](mori::collective::AllgatherSdma<uint32_t>& self, uintptr_t ptr) -> bool {
            return self.is_output_registered(reinterpret_cast<void*>(ptr));
          },
          py::arg("ptr"), "Check whether an output buffer is registered for direct SDMA writes");

  // =========================================================================
  // AllreduceSdma (uint32_t)
  // =========================================================================
  py::class_<mori::collective::AllreduceSdma<uint32_t>>(m, "AllreduceSdmaHandle")
      .def(py::init<int, int, size_t, size_t, bool, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true, py::arg("use_graph_mode") = false)
      .def(py::init<int, int, size_t, bool, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true, py::arg("use_graph_mode") = false)
      .def(
          "__call__",
          [](mori::collective::AllreduceSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> bool {
            return self(reinterpret_cast<uint32_t*>(input_ptr),
                        reinterpret_cast<uint32_t*>(output_ptr), count,
                        reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0)
      .def(
          "pipelined",
          [](mori::collective::AllreduceSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> bool {
            return self.pipelined(reinterpret_cast<uint32_t*>(input_ptr),
                                  reinterpret_cast<uint32_t*>(output_ptr), count,
                                  0, 0, reinterpret_cast<hipStream_t>(stream),
                                  /*external_scatter=*/false);
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Single-kernel pipeline AllReduce (peak bandwidth, no scatter separation)")
      .def(
          "allreduce_inplace",
          [](mori::collective::AllreduceSdma<uint32_t>& self, uintptr_t data_ptr, size_t count,
             int64_t stream) -> bool {
            return self.allreduce_inplace(reinterpret_cast<uint32_t*>(data_ptr), count,
                                          reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("data_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Execute in-place AllReduce SDMA operation")
      .def(
          "start_async",
          [](mori::collective::AllreduceSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> bool {
            return self.start_async(reinterpret_cast<uint32_t*>(input_ptr),
                                    reinterpret_cast<uint32_t*>(output_ptr), count,
                                    reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0)
      .def(
          "wait_async",
          [](mori::collective::AllreduceSdma<uint32_t>& self, int64_t stream) -> double {
            return self.wait_async(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream") = 0)
      .def("is_async_in_progress", &mori::collective::AllreduceSdma<uint32_t>::is_async_in_progress)
      .def("cancel_async", &mori::collective::AllreduceSdma<uint32_t>::cancel_async)
      .def("reset_flags", &mori::collective::AllreduceSdma<uint32_t>::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](mori::collective::AllreduceSdma<uint32_t>& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer")
      .def("enable_phase_timing",
           &mori::collective::AllreduceSdma<uint32_t>::enable_phase_timing,
           py::arg("on"),
           "Enable/disable per-phase timestamp instrumentation for pipelined()")
      .def("get_phase_timestamps",
           &mori::collective::AllreduceSdma<uint32_t>::get_phase_timestamps,
           "Return the most recent pipelined() call's per-phase GPU timestamps "
           "(32 slots, raw s_memtime cycles; slots 0..3+3*numChunks populated)")
      .def("get_last_num_chunks",
           &mori::collective::AllreduceSdma<uint32_t>::get_last_num_chunks,
           "Return numChunks used by the most recent pipelined() call; "
           "needed to parse the phase_timestamps layout")
      .def("enable_copy_timing",
           &mori::collective::AllreduceSdma<uint32_t>::enable_copy_timing,
           py::arg("on"),
           "Enable/disable copy-path timing (hipMemcpyAsync host us + gpu ms)")
      .def("get_copy_timing_ms",
           &mori::collective::AllreduceSdma<uint32_t>::get_copy_timing_ms,
           "Return [host_us, gpu_ms] for the most recent copy (baseline: "
           "single hipMemcpyAsync in copy_output_to_user)")
      .def("get_copy_timing_last_num_chunks",
           &mori::collective::AllreduceSdma<uint32_t>::get_copy_timing_last_num_chunks,
           "Baseline always returns 1 (copy is a single call, not chunked)")
      .def("enable_post_ag_wait",
           &mori::collective::AllreduceSdma<uint32_t>::enable_post_ag_wait,
           py::arg("on"),
           "Stage 1 of E' prototype: compute blocks wait for AG done "
           "instead of exiting; measures CU occupancy cost")
      .def("enable_register_user_output",
           &mori::collective::AllreduceSdma<uint32_t>::enable_register_user_output,
           py::arg("on"),
           "D' prototype: enable lazy-register-user-output fast path")
      .def("register_user_output",
           [](mori::collective::AllreduceSdma<uint32_t>& self,
              uintptr_t ptr, size_t size) {
             return self.register_user_output(reinterpret_cast<void*>(ptr), size);
           },
           py::arg("ptr"), py::arg("size"),
           "Collective call: register a user output buffer as symm memory. "
           "All ranks must call with the same size. Returns true on success.")
      .def("last_register_us",
           &mori::collective::AllreduceSdma<uint32_t>::last_register_us)
      .def("last_register_was_hit",
           &mori::collective::AllreduceSdma<uint32_t>::last_register_was_hit)
      .def("cache_hits",
           &mori::collective::AllreduceSdma<uint32_t>::cache_hits)
      .def("cache_misses",
           &mori::collective::AllreduceSdma<uint32_t>::cache_misses)
      .def("enable_ag_multi_q",
           &mori::collective::AllreduceSdma<uint32_t>::enable_ag_multi_q,
           py::arg("on"),
           "Direction θ: enable multi-qId parallel AG for MULTI_CHUNK. "
           "Must be called BEFORE the first pipelined() call.");

  // =========================================================================
  // AllreduceSdma (fp16)
  // =========================================================================
  py::class_<mori::collective::AllreduceSdma<half>>(m, "AllreduceSdmaHandleFp16")
      .def(py::init<int, int, size_t, size_t, bool, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true, py::arg("use_graph_mode") = false)
      .def(py::init<int, int, size_t, bool, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true, py::arg("use_graph_mode") = false)
      .def(
          "__call__",
          [](mori::collective::AllreduceSdma<half>& self, uintptr_t input_ptr, uintptr_t output_ptr,
             size_t count, int64_t stream) -> bool {
            return self(reinterpret_cast<half*>(input_ptr), reinterpret_cast<half*>(output_ptr),
                        count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Execute AllReduce SDMA operation (fp16)")
      .def(
          "allreduce_inplace",
          [](mori::collective::AllreduceSdma<half>& self, uintptr_t data_ptr, size_t count,
             int64_t stream) -> bool {
            return self.allreduce_inplace(reinterpret_cast<half*>(data_ptr), count,
                                          reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("data_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Execute in-place AllReduce SDMA operation (fp16)")
      .def("reset_flags", &mori::collective::AllreduceSdma<half>::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](mori::collective::AllreduceSdma<half>& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer (fp16)")
      .def("enable_phase_timing",
           &mori::collective::AllreduceSdma<half>::enable_phase_timing, py::arg("on"))
      .def("get_phase_timestamps",
           &mori::collective::AllreduceSdma<half>::get_phase_timestamps)
      .def("get_last_num_chunks",
           &mori::collective::AllreduceSdma<half>::get_last_num_chunks)
      .def("enable_copy_timing",
           &mori::collective::AllreduceSdma<half>::enable_copy_timing, py::arg("on"))
      .def("get_copy_timing_ms",
           &mori::collective::AllreduceSdma<half>::get_copy_timing_ms)
      .def("get_copy_timing_last_num_chunks",
           &mori::collective::AllreduceSdma<half>::get_copy_timing_last_num_chunks)
      .def("enable_post_ag_wait",
           &mori::collective::AllreduceSdma<half>::enable_post_ag_wait, py::arg("on"))
      .def("enable_register_user_output",
           &mori::collective::AllreduceSdma<half>::enable_register_user_output,
           py::arg("on"))
      .def("register_user_output",
           [](mori::collective::AllreduceSdma<half>& self,
              uintptr_t ptr, size_t size) {
             return self.register_user_output(reinterpret_cast<void*>(ptr), size);
           },
           py::arg("ptr"), py::arg("size"))
      .def("last_register_us",
           &mori::collective::AllreduceSdma<half>::last_register_us)
      .def("cache_hits",
           &mori::collective::AllreduceSdma<half>::cache_hits)
      .def("cache_misses",
           &mori::collective::AllreduceSdma<half>::cache_misses)
      .def("enable_ag_multi_q",
           &mori::collective::AllreduceSdma<half>::enable_ag_multi_q,
           py::arg("on"));

  // =========================================================================
  // AllreduceSdma (bf16)
  // =========================================================================
  py::class_<mori::collective::AllreduceSdma<hip_bfloat16>>(m, "AllreduceSdmaHandleBf16")
      .def(py::init<int, int, size_t, size_t, bool, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true, py::arg("use_graph_mode") = false)
      .def(py::init<int, int, size_t, bool, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true, py::arg("use_graph_mode") = false)
      .def(
          "__call__",
          [](mori::collective::AllreduceSdma<hip_bfloat16>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> bool {
            return self(reinterpret_cast<hip_bfloat16*>(input_ptr),
                        reinterpret_cast<hip_bfloat16*>(output_ptr), count,
                        reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Execute AllReduce SDMA operation (bf16)")
      .def(
          "allreduce_inplace",
          [](mori::collective::AllreduceSdma<hip_bfloat16>& self, uintptr_t data_ptr, size_t count,
             int64_t stream) -> bool {
            return self.allreduce_inplace(reinterpret_cast<hip_bfloat16*>(data_ptr), count,
                                          reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("data_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Execute in-place AllReduce SDMA operation (bf16)")
      .def("reset_flags", &mori::collective::AllreduceSdma<hip_bfloat16>::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](mori::collective::AllreduceSdma<hip_bfloat16>& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer (bf16)")
      .def("enable_phase_timing",
           &mori::collective::AllreduceSdma<hip_bfloat16>::enable_phase_timing,
           py::arg("on"))
      .def("get_phase_timestamps",
           &mori::collective::AllreduceSdma<hip_bfloat16>::get_phase_timestamps)
      .def("get_last_num_chunks",
           &mori::collective::AllreduceSdma<hip_bfloat16>::get_last_num_chunks)
      .def("enable_copy_timing",
           &mori::collective::AllreduceSdma<hip_bfloat16>::enable_copy_timing,
           py::arg("on"))
      .def("get_copy_timing_ms",
           &mori::collective::AllreduceSdma<hip_bfloat16>::get_copy_timing_ms)
      .def("get_copy_timing_last_num_chunks",
           &mori::collective::AllreduceSdma<hip_bfloat16>::get_copy_timing_last_num_chunks)
      .def("enable_post_ag_wait",
           &mori::collective::AllreduceSdma<hip_bfloat16>::enable_post_ag_wait,
           py::arg("on"))
      .def("enable_register_user_output",
           &mori::collective::AllreduceSdma<hip_bfloat16>::enable_register_user_output,
           py::arg("on"))
      .def("register_user_output",
           [](mori::collective::AllreduceSdma<hip_bfloat16>& self,
              uintptr_t ptr, size_t size) {
             return self.register_user_output(reinterpret_cast<void*>(ptr), size);
           },
           py::arg("ptr"), py::arg("size"))
      .def("last_register_us",
           &mori::collective::AllreduceSdma<hip_bfloat16>::last_register_us)
      .def("cache_hits",
           &mori::collective::AllreduceSdma<hip_bfloat16>::cache_hits)
      .def("cache_misses",
           &mori::collective::AllreduceSdma<hip_bfloat16>::cache_misses)
      .def("enable_ag_multi_q",
           &mori::collective::AllreduceSdma<hip_bfloat16>::enable_ag_multi_q,
           py::arg("on"));
}
}  // namespace mori
